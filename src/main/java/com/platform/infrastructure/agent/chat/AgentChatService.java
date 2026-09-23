package com.platform.infrastructure.agent.chat;

import com.platform.infrastructure.agent.llm.LlmClient;
import com.platform.infrastructure.agent.llm.LlmMessage;
import com.platform.infrastructure.agent.llm.LlmRequest;
import com.platform.infrastructure.agent.llm.LlmResponse;
import com.platform.infrastructure.agent.mcp.GenericMcpManager;
import com.platform.infrastructure.agent.mcp.McpToolRegistry;
import com.platform.infrastructure.agent.mcp.ToolCallExecutor;
import com.platform.infrastructure.agent.mcp.ToolExecutionConfig;
import com.platform.infrastructure.agent.mcp.ToolExecutionStrategy;
import com.platform.infrastructure.agent.memory.MemoryManager;
import com.platform.infrastructure.agent.skill.SkillRegistry;
import com.platform.application.service.SafetyService;
import com.platform.application.service.TokenUsageService;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import org.springframework.security.core.context.SecurityContextHolder;

import java.util.*;
import java.util.concurrent.*;

@Slf4j
@Service
@RequiredArgsConstructor
public class AgentChatService {

    private final LlmClient llmClient;
    private final McpToolRegistry mcpToolRegistry;
    private final SkillRegistry skillRegistry;
    private final SafetyService safetyService;
    private final ToolCallExecutor toolCallExecutor;
    private final ToolExecutionConfig toolExecutionConfig;
    private final TokenUsageService tokenUsageService;
    private final MemoryManager memoryManager;
    private final ObjectMapper objectMapper = new ObjectMapper();

    private final Map<String, ChatSession> sessions = new ConcurrentHashMap<>();
    private static final int MAX_TOOL_ROUNDS = 5;

    // 线程池: 核心2线程, 最大8线程, 60秒空闲回收, 有界队列防OOM
    private final ExecutorService chatExecutor = new ThreadPoolExecutor(
            2, 8, 60L, TimeUnit.SECONDS,
            new LinkedBlockingQueue<>(16),
            r -> new Thread(r, "agent-chat"),
            new ThreadPoolExecutor.CallerRunsPolicy()
    );

    // 会话Future跟踪, 用于取消
    private final Map<String, Future<?>> sessionFutures = new ConcurrentHashMap<>();

    // ===== 会话管理 =====

    public List<ChatSession> listSessions(String agentName) {
        return sessions.values().stream()
                .filter(s -> agentName == null || agentName.equals(s.getAgentName()))
                .sorted(Comparator.comparing(ChatSession::getUpdatedAt).reversed())
                .toList();
    }

    public ChatSession getSession(String sessionId) { return sessions.get(sessionId); }

    public ChatSession createSession(String agentName, Long userId) {
        ChatSession session = ChatSession.create(agentName, userId);
        sessions.put(session.getId(), session);
        return session;
    }

    public void deleteSession(String sessionId) {
        Future<?> future = sessionFutures.remove(sessionId);
        if (future != null) future.cancel(true);
        sessions.remove(sessionId);
    }

    // ===== 同步聊天 =====

    public Map<String, Object> chatSync(String agentName, String sessionId, String userMessage, List<String> selectedSkills, Long userId) {
        ChatSession session = getOrCreateSession(agentName, sessionId, userId);
        session.addMessage(ChatMessage.user(userMessage));
        // 持久化用户消息到agent_memory（修复#1：原chat链路绕过MemoryManager）
        memoryManager.storeConversation(sessionId, agentName, "user", userMessage);

        String systemPrompt = buildSystemPrompt(agentName, selectedSkills);
        List<LlmMessage> llmMessages = buildLlmMessages(systemPrompt, session.getMessages());
        List<LlmRequest.ToolDefinition> tools = mcpToolRegistry.getToolDefinitions();

        int toolRounds = 0;
        String finalContent = "";
        LlmResponse lastResponse = null;

        while (toolRounds < MAX_TOOL_ROUNDS) {
            log.info("AgentChat同步调用LLM: agent={}, round={}", agentName, toolRounds);
            LlmResponse response = llmClient.chat(llmMessages, tools.isEmpty() ? null : tools,agentName);
            lastResponse = response;
            LlmResponse.Choice choice = response.getChoices().get(0);
            String content = choice.getMessage().getContent();
            List<LlmMessage.ToolCall> toolCalls = choice.getMessage().getToolCalls();

            if (toolCalls != null && !toolCalls.isEmpty()) {
                boolean allToolsFailed = true;
                
                for (LlmMessage.ToolCall tc : toolCalls) {
                    String toolName = tc.getFunction().getName();
                    Map<String, Object> argsMap = parseToolArgs(tc.getFunction().getArguments());
                    log.info("执行MCP工具: {} args={}", toolName, argsMap);
                    session.addMessage(ChatMessage.toolCall(tc.getId(), toolName, argsMap));
                }
                
                List<ToolCallExecutor.ToolExecutionResult> executionResults = toolCallExecutor.execute(
                        toolCalls,
                        (toolName, args) -> executeMcpTool(toolName, args),
                        toolExecutionConfig.getDefaultStrategy(),
                        toolExecutionConfig.getTimeoutSeconds()
                );
                
                for (ToolCallExecutor.ToolExecutionResult result : executionResults) {
                    session.addMessage(ChatMessage.toolResult(result.toolCallId(), result.toolName(), result.result()));
                    if (result.result().isSuccess()) {
                        allToolsFailed = false;
                    }
                }
                llmMessages.add(LlmMessage.builder().role("assistant").content(content != null ? content : "").toolCalls(toolCalls).build());
                for (LlmMessage.ToolCall tc : toolCalls) {
                    String toolCallId = tc.getId();
                    GenericMcpManager.McpToolResult toolResult = session.getMessages().stream()
                        .filter(m -> "tool_result".equals(m.getRole()) && toolCallId.equals(m.getToolCallId()))
                        .map(m -> (GenericMcpManager.McpToolResult) m.getToolResult())
                        .findFirst().orElse(null);
                    String toolResultContent = toolResult != null && toolResult.isSuccess() 
                        ? toolResult.getContent() != null ? toolResult.getContent().toString() : "工具执行成功"
                        : "工具执行失败: " + (toolResult != null && toolResult.getError() != null ? toolResult.getError() : "未知错误");
                    llmMessages.add(LlmMessage.builder().role("tool").toolCallId(toolCallId).content(toolResultContent).build());
                }
                if (allToolsFailed && toolRounds > 0) {
                    log.warn("所有工具调用失败，终止工具调用循环: round={}", toolRounds);
                    finalContent = "工具调用失败，请检查MCP服务是否正常运行。错误信息: " + 
                        session.getMessages().stream()
                            .filter(m -> "tool_result".equals(m.getRole()) && m.getToolResult() instanceof GenericMcpManager.McpToolResult)
                            .map(m -> (GenericMcpManager.McpToolResult) m.getToolResult())
                            .filter(tr -> !tr.isSuccess())
                            .map(GenericMcpManager.McpToolResult::getError)
                            .findFirst().orElse("未知错误");
                    break;
                }
                toolRounds++;
                continue;
            }
            finalContent = content != null ? content : "";
            break;
        }

        // 记录 Token 使用量到数据库
        try {
            if (session.getUserId() != null && lastResponse != null && lastResponse.getUsage() != null) {
                int promptTokens = lastResponse.getUsage().getPromptTokens();
                int completionTokens = lastResponse.getUsage().getCompletionTokens();
                tokenUsageService.recordUsage(
                    session.getUserId(),
                    agentName,
                    sessionId,
                    promptTokens,
                    completionTokens,
                    "deepseek-v4-flash",
                    "chat"
                );
            }
        } catch (Exception e) {
            log.warn("记录Token使用量失败: {}", e.getMessage());
        }

        SafetyService.SafetyCheckResult safetyResult = safetyService.process(finalContent, userMessage, sessionId, agentName);
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("sessionId", session.getId());
        result.put("agentName", agentName);

        if (safetyResult.intercepted) {
            session.addMessage(ChatMessage.system("[安全拦截] " + safetyResult.message));
            result.put("reply", safetyResult.message);
            result.put("safetyCheck", safetyResult.toMap());
        } else {
            session.addMessage(ChatMessage.assistant(finalContent));
            // 持久化助手回复到agent_memory
            memoryManager.storeConversation(sessionId, agentName, "assistant", finalContent);
            result.put("reply", finalContent);
        }
        return result;
    }

    // ===== SSE聊天 =====

    public SseEmitter chat(String agentName, String sessionId, String userMessage, List<String> selectedSkills, Long userId) {
        return chat(agentName, sessionId, userMessage, selectedSkills, userId, null);
    }

    public SseEmitter chat(String agentName, String sessionId, String userMessage, List<String> selectedSkills, Long userId, String authToken) {
        SseEmitter emitter = new SseEmitter(300_000L);
        ChatSession session = getOrCreateSession(agentName, sessionId, userId);
        session.addMessage(ChatMessage.user(userMessage));
        // 持久化用户消息到agent_memory
        memoryManager.storeConversation(sessionId, agentName, "user", userMessage);
        String systemPrompt = buildSystemPrompt(agentName, selectedSkills);
        final ChatSession finalSession = session;
        final String finalAuthToken = authToken;

        emitter.onCompletion(() -> {
            log.info("SSE完成: sessionId={}", finalSession.getId());
            sessionFutures.remove(sessionId);
        });
        emitter.onTimeout(() -> {
            log.warn("SSE超时: sessionId={}", finalSession.getId());
            safeSend(emitter, SseEmitter.event().name("error").data(Map.of("error", "连接超时，请重新发送消息")));
            try { emitter.complete(); } catch (Exception ignored) {}
            sessionFutures.remove(sessionId);
        });
        emitter.onError(e -> {
            log.error("SSE错误: {}", e.getMessage());
            sessionFutures.remove(sessionId);
        });

        // 提交到线程池执行
        Future<?> future = chatExecutor.submit(() -> {
            SecurityContextHolder.setContext(SecurityContextHolder.createEmptyContext());
            try {
                List<ChatMessage> history = finalSession.getMessages();
                List<LlmMessage> llmMessages = buildLlmMessages(systemPrompt, history);
                List<LlmRequest.ToolDefinition> tools = mcpToolRegistry.getToolDefinitions();

                safeSend(emitter, SseEmitter.event().name("thinking").data("正在思考..."));

                int toolRounds = 0;
                String fullResponse = "";

                while (toolRounds < MAX_TOOL_ROUNDS) {
                    log.info("AgentChat调用LLM: agent={}, round={}", agentName, toolRounds);
                    LlmResponse response = llmClient.chat(llmMessages, tools.isEmpty() ? null : tools, agentName);
                    LlmResponse.Choice choice = response.getChoices().get(0);
                    String content = choice.getMessage().getContent();
                    String reasoningContent = choice.getMessage().getReasoningContent();
                    List<LlmMessage.ToolCall> toolCalls = choice.getMessage().getToolCalls();

                    if (reasoningContent != null && !reasoningContent.isEmpty()) {
                        safeSend(emitter, SseEmitter.event().name("reasoning").data(reasoningContent));
                    }

                    if (toolCalls != null && !toolCalls.isEmpty()) {
                        boolean allToolsFailed = true;
                        
                        for (LlmMessage.ToolCall tc : toolCalls) {
                            String toolName = tc.getFunction().getName();
                            String tcId = tc.getId();
                            Map<String, Object> argsMap = parseToolArgs(tc.getFunction().getArguments());
                            finalSession.addMessage(ChatMessage.toolCall(tcId, toolName, argsMap));
                            safeSend(emitter, SseEmitter.event().name("tool").data(Map.of("tool", toolName, "status", "executing")));
                        }
                        
                        List<ToolCallExecutor.ToolExecutionResult> executionResults = toolCallExecutor.execute(
                                toolCalls,
                                (toolName, args) -> executeMcpTool(toolName, args, finalAuthToken),
                                toolExecutionConfig.getDefaultStrategy(),
                                toolExecutionConfig.getTimeoutSeconds()
                        );
                        
                        for (ToolCallExecutor.ToolExecutionResult result : executionResults) {
                            finalSession.addMessage(ChatMessage.toolResult(result.toolCallId(), result.toolName(), result.result()));
                            sendToolResultEvent(emitter, result.toolName(), result.result());
                            if (result.result().isSuccess()) {
                                allToolsFailed = false;
                            }
                        }
                        llmMessages.add(LlmMessage.builder().role("assistant").content(content != null ? content : "").toolCalls(toolCalls).build());
                        for (LlmMessage.ToolCall tc : toolCalls) {
                            String toolCallId = tc.getId();
                            GenericMcpManager.McpToolResult toolResult = finalSession.getMessages().stream()
                                .filter(m -> "tool_result".equals(m.getRole()) && toolCallId.equals(m.getToolCallId()))
                                .map(m -> (GenericMcpManager.McpToolResult) m.getToolResult())
                                .findFirst().orElse(null);
                            String toolResultContent = toolResult != null && toolResult.isSuccess() 
                                ? toolResult.getContent() != null ? toolResult.getContent().toString() : "工具执行成功"
                                : "工具执行失败: " + (toolResult != null && toolResult.getError() != null ? toolResult.getError() : "未知错误");
                            llmMessages.add(LlmMessage.builder().role("tool").toolCallId(toolCallId).content(toolResultContent).build());
                        }
                        if (allToolsFailed && toolRounds > 0) {
                            log.warn("所有工具调用失败，终止工具调用循环: round={}", toolRounds);
                            fullResponse = "工具调用失败，请检查MCP服务是否正常运行。错误信息: " + 
                                finalSession.getMessages().stream()
                                    .filter(m -> "tool_result".equals(m.getRole()) && m.getToolResult() instanceof GenericMcpManager.McpToolResult)
                                    .map(m -> (GenericMcpManager.McpToolResult) m.getToolResult())
                                    .filter(tr -> !tr.isSuccess())
                                    .map(GenericMcpManager.McpToolResult::getError)
                                    .findFirst().orElse("未知错误");
                            break;
                        }
                        toolRounds++;
                        continue;
                    }

                    fullResponse = content != null ? content : "";
                    break;
                }

                // 安全检查：与chatSync()保持一致的安全检查逻辑
                SafetyService.SafetyCheckResult safetyResult = safetyService.process(fullResponse, userMessage, sessionId, agentName);
                if (safetyResult.intercepted) {
                    finalSession.addMessage(ChatMessage.system("[安全拦截] " + safetyResult.message));
                    fullResponse = safetyResult.message;
                }

                sendByParagraphs(emitter, fullResponse);

                finalSession.addMessage(ChatMessage.assistant(fullResponse));
                // 持久化助手回复到agent_memory
                memoryManager.storeConversation(sessionId, agentName, "assistant", fullResponse);
                safeSend(emitter, SseEmitter.event().name("done").data(Map.of("sessionId", finalSession.getId(), "fullResponse", fullResponse)));
                emitter.complete();

            } catch (Exception e) {
                log.error("AgentChat异步执行失败", e);
                safeSend(emitter, SseEmitter.event().name("error").data(Map.of("error", e.getMessage() != null ? e.getMessage() : "未知错误")));
                try { emitter.complete(); } catch (Exception ignored) {}
            }
        });
        sessionFutures.put(sessionId, future);
        return emitter;
    }

    // ===== 确认操作 =====

    public Map<String, Object> handleConfirm(String agentName, String sessionId, String action, String modifiedMessage, Long userId) {
        ChatSession session = sessions.get(sessionId);
        if (session == null) return Map.of("error", "会话不存在");
        return switch (action) {
            case "confirm" -> chatSync(agentName, sessionId, "请继续执行之前的操作", List.of(), userId);
            case "modify" -> modifiedMessage != null && !modifiedMessage.isEmpty()
                    ? chatSync(agentName, sessionId, modifiedMessage, List.of(), userId)
                    : Map.of("error", "修改内容不能为空");
            case "cancel" -> {
                session.addMessage(ChatMessage.system("[用户取消]"));
                yield Map.of("sessionId", session.getId(), "reply", "操作已取消", "agentName", agentName, "tokenUsed", 0);
            }
            case "acknowledge" -> {
                session.addMessage(ChatMessage.system("[用户了解]"));
                yield Map.of("sessionId", session.getId(), "reply", "已记录", "agentName", agentName, "tokenUsed", 0);
            }
            default -> Map.of("error", "未知操作: " + action);
        };
    }

    // ===== 内部方法 =====

    private ChatSession getOrCreateSession(String agentName, String sessionId, Long userId) {
        ChatSession session = sessions.get(sessionId);
        if (session == null) {
            session = ChatSession.create(agentName, userId);
            session.setId(sessionId);
            sessions.put(sessionId, session);
        }
        return session;
    }

    private String buildSystemPrompt(String agentName, List<String> selectedSkills) {
        StringBuilder sb = new StringBuilder();
        sb.append("你是AI测试助手，帮助用户完成测试相关工作。\n\n");
        sb.append("可用的MCP工具：\n");
        mcpToolRegistry.getAllTools().forEach(t -> sb.append("- ").append(t.name()).append(": ").append(t.description()).append("\n"));
        sb.append("\n可用的技能：\n");
        skillRegistry.getAllSkills().forEach(s -> sb.append("- ").append(s.getName()).append(": ").append(s.getDescription()).append("\n"));
        if (selectedSkills != null && !selectedSkills.isEmpty()) {
            sb.append("\n当前已选择的技能：");
            selectedSkills.forEach(name -> sb.append("\n- ").append(name));
        }
        sb.append("\n\n你可以调用MCP工具来获取数据，也可以使用技能来完成复杂任务。\n回复使用中文，格式要求：\n1. 使用标准markdown语法（标题用#，列表用-或数字.，粗体用**）\n2. 禁止使用emoji表情符号和特殊符号\n3. 禁止使用表格，使用列表替代\n4. 回复结构清晰，使用有序列表和无序列表组织内容\n5. 不要使用换行符\\n，段落之间用空行分隔");
        return sb.toString();
    }

    private List<LlmMessage> buildLlmMessages(String systemPrompt, List<ChatMessage> history) {
        List<LlmMessage> messages = new ArrayList<>();
        messages.add(LlmMessage.builder().role("system").content(systemPrompt).build());
        
        List<LlmMessage.ToolCall> pendingToolCalls = new ArrayList<>();
        
        for (ChatMessage msg : history) {
            switch (msg.getRole()) {
                case "user" -> {
                    if (!pendingToolCalls.isEmpty()) {
                        messages.add(LlmMessage.builder().role("assistant").content("").toolCalls(List.copyOf(pendingToolCalls)).build());
                        pendingToolCalls.clear();
                    }
                    messages.add(LlmMessage.builder().role("user").content(msg.getContent()).build());
                }
                case "assistant" -> {
                    if (!pendingToolCalls.isEmpty()) {
                        messages.add(LlmMessage.builder().role("assistant").content("").toolCalls(List.copyOf(pendingToolCalls)).build());
                        pendingToolCalls.clear();
                    }
                    messages.add(LlmMessage.builder().role("assistant").content(msg.getContent()).build());
                }
                case "tool_call" -> {
                    LlmMessage.ToolCall toolCall = LlmMessage.ToolCall.builder()
                            .id(msg.getToolCallId())
                            .type("function")
                            .function(LlmMessage.FunctionCall.builder()
                                    .name(msg.getToolName())
                                    .arguments(msg.getToolArgs() != null ? msg.getToolArgs().toString() : "{}")
                                    .build())
                            .build();
                    pendingToolCalls.add(toolCall);
                }
                case "tool_result" -> {
                    if (!pendingToolCalls.isEmpty()) {
                        messages.add(LlmMessage.builder().role("assistant").content("").toolCalls(List.copyOf(pendingToolCalls)).build());
                        pendingToolCalls.clear();
                    }
                    String result = msg.getToolResult() != null ? msg.getToolResult().toString() : "无结果";
                    String tcId = msg.getToolCallId() != null ? msg.getToolCallId() : msg.getToolName();
                    messages.add(LlmMessage.builder().role("tool").toolCallId(tcId).content("工具 " + msg.getToolName() + " 返回: " + result).build());
                }
                case "system" -> {
                    if (!pendingToolCalls.isEmpty()) {
                        messages.add(LlmMessage.builder().role("assistant").content("").toolCalls(List.copyOf(pendingToolCalls)).build());
                        pendingToolCalls.clear();
                    }
                    messages.add(LlmMessage.builder().role("system").content(msg.getContent()).build());
                }
                default -> {}
            }
        }
        
        if (!pendingToolCalls.isEmpty()) {
            messages.add(LlmMessage.builder().role("assistant").content("").toolCalls(List.copyOf(pendingToolCalls)).build());
        }
        
        return messages;
    }

    private GenericMcpManager.McpToolResult executeMcpTool(String toolName, Map<String, Object> args) {
        return executeMcpTool(toolName, args, null);
    }

    private GenericMcpManager.McpToolResult executeMcpTool(String toolName, Map<String, Object> args, String authToken) {
        try { return mcpToolRegistry.executeTool(toolName, args, authToken); }
        catch (Exception e) {
            log.error("MCP工具调用异常: {} - {}", toolName, e.getMessage());
            return GenericMcpManager.McpToolResult.builder().success(false).error(e.getMessage()).build();
        }
    }

    private void sendToolResultEvent(SseEmitter emitter, String toolName, GenericMcpManager.McpToolResult toolResult) {
        Map<String, Object> data = new LinkedHashMap<>();
        data.put("tool", toolName);
        data.put("success", toolResult.isSuccess());
        if (toolResult.isSuccess()) {
            data.put("result", toolResult.getContent() != null ? toolResult.getContent() : "");
        } else {
            data.put("error", toolResult.getError() != null ? toolResult.getError() : "工具调用失败");
        }
        safeSend(emitter, SseEmitter.event().name("tool_result").data(data));
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> parseToolArgs(String argsJson) {
        try { return (argsJson == null || argsJson.isEmpty()) ? Map.of() : objectMapper.readValue(argsJson, Map.class); }
        catch (Exception e) { return Map.of(); }
    }

    private void safeSend(SseEmitter emitter, SseEmitter.SseEventBuilder event) {
        try { emitter.send(event); }
        catch (IllegalStateException e) { log.debug("SSE emitter已关闭: {}", e.getMessage()); }
        catch (Exception e) { log.debug("SSE发送异常: {}", e.getMessage()); }
    }

    private void sendByParagraphs(SseEmitter emitter, String content) {
        if (content == null || content.isEmpty()) {
            return;
        }
        
        String[] paragraphs = content.split("\n\n");
        for (String paragraph : paragraphs) {
            if (!paragraph.trim().isEmpty()) {
                try {
                    String dataJson = objectMapper.writeValueAsString(Map.of("content", paragraph.trim()));
                    safeSend(emitter, SseEmitter.event().name("token").data(dataJson));
                } catch (Exception e) {
                    safeSend(emitter, SseEmitter.event().name("token").data(paragraph.trim()));
                }
            }
        }
    }
}
