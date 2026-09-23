package com.platform.infrastructure.agent.chat;

import com.platform.application.service.SafetyService;
import com.platform.application.service.TokenUsageService;
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
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/**
 * 验证 AgentChatService 的聊天链路在用户和助手回合后调用 MemoryManager.storeConversation。
 * 对应 issue #1：原 chat 链路绕过 MemoryManager，导致 agent_memory 表为空。
 */
@ExtendWith(MockitoExtension.class)
@DisplayName("AgentChatService 聊天链路 -> MemoryManager 持久化集成测试")
class AgentChatServiceMemoryTest {

    @Mock private LlmClient llmClient;
    @Mock private McpToolRegistry mcpToolRegistry;
    @Mock private SkillRegistry skillRegistry;
    @Mock private SafetyService safetyService;
    @Mock private ToolCallExecutor toolCallExecutor;
    @Mock private ToolExecutionConfig toolExecutionConfig;
    @Mock private TokenUsageService tokenUsageService;
    @Mock private MemoryManager memoryManager;

    private AgentChatService chatService;

    @BeforeEach
    void setUp() {
        when(skillRegistry.getAllSkills()).thenReturn(List.of());
        when(mcpToolRegistry.getAllTools()).thenReturn(List.of());
        when(mcpToolRegistry.getToolDefinitions()).thenReturn(List.of());

        // 构造 LlmResponse（LlmResponse/Choice/Usage 用 @Data，无 @Builder，需用 setter）
        LlmResponse fakeResponse = new LlmResponse();
        LlmResponse.Choice choice = new LlmResponse.Choice();
        choice.setMessage(LlmMessage.builder().role("assistant").content("你好，我是测试助手").build());
        fakeResponse.setChoices(List.of(choice));
        LlmResponse.Usage usage = new LlmResponse.Usage();
        usage.setPromptTokens(10);
        usage.setCompletionTokens(5);
        fakeResponse.setUsage(usage);
        when(llmClient.chat(anyList(), any(), anyString())).thenReturn(fakeResponse);

        // 不拦截
        when(safetyService.process(anyString(), anyString(), anyString(), anyString()))
                .thenReturn(SafetyService.SafetyCheckResult.safe());

        chatService = new AgentChatService(
                llmClient, mcpToolRegistry, skillRegistry, safetyService,
                toolCallExecutor, toolExecutionConfig, tokenUsageService, memoryManager);
    }

    @Test
    @DisplayName("chatSync 后应同时持久化 user 与 assistant 两条消息")
    void testChatSync_PersistsUserAndAssistant() {
        String sessionId = "test-session-1";
        String userMsg = "你好";
        String agentName = "TestAgent";

        Map<String, Object> result = chatService.chatSync(agentName, sessionId, userMsg, List.of(), 1L);

        // 验证：MemoryManager.storeConversation 被调用2次（user + assistant）
        verify(memoryManager, times(2)).storeConversation(eq(sessionId), eq(agentName), anyString(), anyString());
        verify(memoryManager).storeConversation(eq(sessionId), eq(agentName), eq("user"), eq(userMsg));
        verify(memoryManager).storeConversation(eq(sessionId), eq(agentName), eq("assistant"), eq("你好，我是测试助手"));
        assertEquals(sessionId, result.get("sessionId"));
        assertEquals("你好，我是测试助手", result.get("reply"));
    }

    @Test
    @DisplayName("chatSync 在安全拦截时只持久化 user 消息，不持久化被拦截的回复")
    void testChatSync_SafetyIntercept_OnlyPersistsUser() {
        // 让 SafetyService 拦截（使用 intercepted 工厂方法）
        when(safetyService.process(anyString(), anyString(), anyString(), anyString()))
                .thenReturn(SafetyService.SafetyCheckResult.intercepted(
                        1L, "测试规则", "危险词", "KEYWORD", "内容违规，已被拦截", "WARNING"));

        chatService.chatSync("TestAgent", "session-block", "危险内容", List.of(), 1L);

        // 仅 user 被持久化（1次）
        verify(memoryManager, times(1)).storeConversation(anyString(), anyString(), anyString(), anyString());
        verify(memoryManager).storeConversation(eq("session-block"), eq("TestAgent"), eq("user"), eq("危险内容"));
        verify(memoryManager, never())
                .storeConversation(anyString(), anyString(), eq("assistant"), anyString());
    }

    @Test
    @DisplayName("重复调用 chatSync 在同一 session 中累积 storeConversation 调用次数")
    void testChatSync_MultipleRounds_AccumulatesMemoryCalls() {
        String sessionId = "session-multi";
        chatService.chatSync("A", sessionId, "第一轮", List.of(), 1L);
        chatService.chatSync("A", sessionId, "第二轮", List.of(), 1L);

        // 2 轮 × 2 次（user + assistant） = 4 次
        verify(memoryManager, times(4)).storeConversation(eq(sessionId), eq("A"), anyString(), anyString());
    }
}
