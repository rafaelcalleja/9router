/**
 * cursorConnect.js — ConnectRPC client for Cursor API
 *
 * Replaces manual HTTP/2 + ConnectRPC framing with native @connectrpc/connect.
 * Uses generated protobuf stubs from cursor.proto.
 */

import { createClient } from "@connectrpc/connect";
import { createConnectTransport } from "@connectrpc/connect-node";
import { create } from "@bufbuild/protobuf";
import crypto from "crypto";

import {
  ChatService,
  StreamUnifiedChatRequestWithToolsSchema,
  ClientSideToolV2ResultSchema,
  MCPResultSchema,
} from "../gen/cursor_pb.js";

import { buildCursorHeaders } from "./cursorChecksum.js";

// ==================== CONSTANTS ====================

const ROLE = { USER: 1, ASSISTANT: 2 };
const UNIFIED_MODE = { CHAT: 1, AGENT: 2 };
const THINKING_LEVEL = { UNSPECIFIED: 0, MEDIUM: 1, HIGH: 2 };
const CLIENT_SIDE_TOOL_V2_MCP = 19;

// Env-gated debug logging (set CURSOR_CONNECT_DEBUG=1 to enable)
const CURSOR_CONNECT_DEBUG = process.env.CURSOR_CONNECT_DEBUG === "1";
const debugLog = (...args) => CURSOR_CONNECT_DEBUG && console.log(...args);

// ==================== TRANSPORT FACTORY ====================

/**
 * Create a ConnectRPC transport with auth interceptor.
 * Cached per credentials to reuse HTTP/2 connections.
 */
const transportCache = new Map();

function getTransport(baseUrl, credentials, configHeaders = {}) {
  const cacheKey = `${baseUrl}:${credentials.accessToken?.slice(-8)}`;
  if (transportCache.has(cacheKey)) return transportCache.get(cacheKey);

  const accessToken = credentials.accessToken;
  const machineId = credentials.providerSpecificData?.machineId;
  const ghostMode = credentials.providerSpecificData?.ghostMode !== false;
  const headers = buildCursorHeaders(accessToken, machineId, ghostMode, {
    headers: configHeaders
  });

  // Filter headers for ConnectRPC (skip pseudo-headers and te)
  const connectHeaders = {};
  for (const [key, value] of Object.entries(headers)) {
    if (key.startsWith(":") || key === "te") continue;
    connectHeaders[key] = String(value);
  }

  const transport = createConnectTransport({
    baseUrl,
    httpVersion: "2",
    interceptors: [
      (next) => async (req) => {
        for (const [key, value] of Object.entries(connectHeaders)) {
          req.header.set(key, value);
        }
        return next(req);
      },
    ],
  });

  transportCache.set(cacheKey, transport);
  return transport;
}

// ==================== REQUEST BUILDER ====================

/**
 * Build ConnectRPC request objects from OpenAI-format messages.
 *
 * @param {Array} messages - OpenAI-format messages [{role, content, tool_calls?, tool_results?}]
 * @param {string} modelName - Model name
 * @param {Array} tools - Tools [{name, description, input_schema, server_name}]
 * @param {string|null} reasoningEffort - "medium" | "high" | null
 * @param {boolean} maxMode - Enable max mode
 * @returns {{ request: object, toolResultRequests: object[] }}
 */
function buildConnectRequest(messages, modelName, tools = [], reasoningEffort = null, maxMode = false) {
  const hasTools = tools?.length > 0;
  const isAgentic = hasTools;

  // Build conversation messages
  const conversation = [];
  const messageIds = [];
  const toolResultRequests = []; // Separate requests for tool results

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    const role = msg.role === "user" ? ROLE.USER : ROLE.ASSISTANT;
    const msgId = crypto.randomUUID();
    const content = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content || "");

    // Build tool_results embedded in message (for conversation history)
    const toolResults = [];

    // Assistant tool_calls → ToolResultInMessage with tool_call field (ClientSideToolV2Call)
    // This is the native protobuf way to encode what tool the assistant invoked.
    if (msg.role === "assistant" && msg.tool_calls?.length > 0) {
      for (let ti = 0; ti < msg.tool_calls.length; ti++) {
        const tc = msg.tool_calls[ti];
        const fn = tc.function || tc;
        const name = fn.name || tc.name || "tool";
        const args = fn.arguments || tc.rawArgs || "{}";
        toolResults.push({
          toolCallId: tc.id || "",
          toolName: name,
          toolIndex: ti + 1,
          rawArgs: args,
          toolCall: {
            tool: CLIENT_SIDE_TOOL_V2_MCP,
            toolCallId: tc.id || "",
            name: name,
            rawArgs: args,
            mcpParams: { tools: [{ name }] },
          },
        });
      }
    }

    // Existing tool_results from translator (tool result + optional tool_call)
    if (msg.tool_results?.length > 0) {
      for (const tr of msg.tool_results) {
        toolResults.push({
          toolCallId: tr.tool_call_id || "",
          toolName: tr.name || tr.tool_name || "",
          toolIndex: tr.tool_index || 1,
          rawArgs: tr.raw_args || "{}",
          result: {
            tool: CLIENT_SIDE_TOOL_V2_MCP,
            mcpResult: { selectedTool: tr.name || tr.tool_name || "", result: tr.result || tr.content || "" },
            toolCallId: tr.tool_call_id || "",
            modelCallId: tr.model_call_id || "",
          },
          toolCall: tr.tool_call ? {
            tool: CLIENT_SIDE_TOOL_V2_MCP,
            toolCallId: tr.tool_call_id || "",
            name: tr.name || "",
            rawArgs: tr.raw_args || "{}",
          } : undefined,
        });
      }
    }

    conversation.push({
      text: content,
      role,
      messageId: msgId,
      chatModeEnum: 1,
      toolResults,
    });

    messageIds.push({ messageId: msgId, role });

    // If message has tool_results, also build separate tool result requests for bidi
    if (msg.tool_results?.length > 0) {
      for (const tr of msg.tool_results) {
        toolResultRequests.push(buildToolResultRequest(tr));
      }
    }
  }

  // Map reasoning effort
  let thinkingLevel = THINKING_LEVEL.UNSPECIFIED;
  if (reasoningEffort === "medium") thinkingLevel = THINKING_LEVEL.MEDIUM;
  else if (reasoningEffort === "high") thinkingLevel = THINKING_LEVEL.HIGH;

  // Build MCP tools — handle both OpenAI format ({type:"function", function:{...}})
  // and flat format ({name, description, input_schema})
  const mcpTools = tools.map(t => {
    const fn = t.function || t; // unwrap OpenAI format
    const schema = fn.parameters || fn.input_schema || t.input_schema || {};
    return {
      toolName: fn.name || t.name || t.toolName,
      description: fn.description || t.description || "",
      inputSchemaJson: typeof schema === "string" ? schema : JSON.stringify(schema),
      serverName: t.server_name || t.serverName || "custom",
    };
  });

  const request = create(StreamUnifiedChatRequestWithToolsSchema, {
    request: {
      conversation,
      allowLongFileScan: true,
      explicitContext: { instruction: "" },
      canHandleFilenamesAfterLanguageIds: true,
      modelDetails: { modelName, maxMode },
      useWeb: "",
      shouldCache: true,
      currentFile: { path: "cursor\\aisettings", unknown8: true, unknown9: true },
      useNewCompressionScheme: true,
      isChat: !isAgentic,
      conversationId: crypto.randomUUID(),
      environmentInfo: {
        exthostPlatform: process.platform || "linux",
        exthostArch: process.arch || "x64",
        exthostRelease: process.version || "v20.0.0",
        exthostShell: process.env.SHELL || "/bin/bash",
        localTimestamp: new Date().toISOString(),
      },
      isAgentic,
      supportedTools: isAgentic ? [1] : [],
      fullConversationHeadersOnly: messageIds,
      useUnifiedChatPrompt: true,
      mcpTools,
      useFullInputsContext: false,
      unknown38: false,
      unifiedMode: isAgentic ? UNIFIED_MODE.AGENT : UNIFIED_MODE.CHAT,
      toolsRequiringAcceptedReturn: "",
      shouldDisableTools: !isAgentic,
      thinkingLevel,
      usesRules: false,
      modeUsesAutoApply: isAgentic,
      unifiedModeName: isAgentic ? "Agent" : "Ask",
    }
  });

  return { request, toolResultRequests };
}

/**
 * Build a tool result request for bidi streaming.
 */
function buildToolResultRequest(tr) {
  return create(StreamUnifiedChatRequestWithToolsSchema, {
    clientSideToolV2Result: create(ClientSideToolV2ResultSchema, {
      tool: CLIENT_SIDE_TOOL_V2_MCP,
      mcpResult: create(MCPResultSchema, {
        selectedTool: tr.name || tr.tool_name || "",
        result: tr.result || tr.content || "",
      }),
      toolCallId: tr.tool_call_id || "",
      modelCallId: tr.model_call_id || "",
    })
  });
}

// ==================== RESPONSE TYPES ====================

/**
 * @typedef {Object} CursorResponse
 * @property {string} text - Accumulated text
 * @property {Array} toolCalls - Tool calls [{id, name, rawArgs, mcpParams, modelCallId, isPartial, isLast}]
 * @property {string} thinkingText - Thinking/reasoning text
 * @property {Array} frames - Raw decoded response frames for SSE streaming
 * @property {string|null} error - Error message if any
 */

// ==================== MAIN REQUEST FUNCTION ====================

/**
 * Make a ConnectRPC request to Cursor API.
 * Returns an array of decoded response frames for streaming.
 *
 * @param {Object} config - Provider config {baseUrl, headers, clientVersion, ...}
 * @param {Array} messages - OpenAI-format messages
 * @param {string} modelName - Model name
 * @param {Array} tools - Tools
 * @param {Object} credentials - {accessToken, providerSpecificData: {machineId, ghostMode}}
 * @param {Object} opts - {reasoningEffort, maxMode, signal, skipToolResultFrames}
 * @returns {Promise<CursorResponse>}
 */
export async function makeConnectRequest(config, messages, modelName, tools, credentials, opts = {}) {
  const baseUrl = config.baseUrl || config;
  debugLog(`[CONNECT] makeConnectRequest: model=${modelName}, tools=${tools?.length || 0}, msgs=${messages?.length || 0}`);
  const transport = getTransport(baseUrl, credentials, config.headers);
  const client = createClient(ChatService, transport);
  const { request } = buildConnectRequest(
    messages, modelName, tools, opts.reasoningEffort, opts.maxMode
  );
  debugLog(`[CONNECT] Request built: mcpTools=${request.request?.mcpTools?.length || 0}, isAgentic=${request.request?.isAgentic}, unifiedMode=${request.request?.unifiedMode}`);

  const frames = [];
  let textTotal = "";
  let thinkingText = "";
  const toolCalls = [];
  let error = null;

  // Keep the request stream alive until we've received all response frames.
  // Closing too early (half-close) causes Cursor to truncate responses.
  let closeGenerator;
  async function* requestStream() {
    yield request;
    await new Promise(r => { closeGenerator = r; });
  }

  try {
    const stream = client.streamUnifiedChatWithTools(requestStream());
    let frameCount = 0;

    for await (const response of stream) {
      frameCount++;
      debugLog(`[CONNECT] Frame #${frameCount}: text=${(response.response?.text || '').length}chars, toolCall=${!!response.toolCall?.toolCallId}`);
      // Extract text
      const text = response.response?.text || "";
      const thinking = response.response?.thinking;
      const serverBubbleId = response.response?.serverBubbleId || "";

      // Build frame object for SSE transformation
      const frame = {
        text: text || null,
        thinking: thinking ? { text: thinking.text || "", signature: thinking.signature || "" } : null,
        serverBubbleId: serverBubbleId || null,
        usageUuid: response.response?.usageUuid || null,
        toolCall: null,
        error: null,
      };

      if (text) textTotal += text;
      if (thinking?.text) thinkingText += thinking.text;

      // Extract tool call
      if (response.toolCall?.toolCallId) {
        const tc = response.toolCall;
        const mcpToolName = tc.mcpParams?.tools?.[0]?.name || tc.name || "";
        const isLast = tc.isLastMessage || false;

        frame.toolCall = {
          id: tc.toolCallId,
          name: mcpToolName,
          rawArgs: tc.rawArgs || "",
          mcpParams: tc.mcpParams,
          modelCallId: tc.modelCallId || "",
          isPartial: !isLast && !tc.rawArgs,
          isLast,
          toolIndex: tc.toolIndex || 0,
        };

        toolCalls.push(frame.toolCall);

        // Got a complete tool call — close the request stream and stop reading
        if (isLast || tc.rawArgs) {
          debugLog(`[CONNECT] Got complete tool_call (isLast=${isLast}), closing stream`);
          frames.push(frame);
          break;
        }
      }

      frames.push(frame);
    }
  } catch (err) {
    error = `[${err.code || "unknown"}] ${err.message}`;
    console.error(`[CONNECT-DBG] Error: ${error}`);
  } finally {
    // Signal the generator to close (request stream half-close)
    closeGenerator?.();
  }

  // Abort error after break is expected when we got tool_calls
  if (error && toolCalls.length > 0 && frames.length > 0) {
    debugLog(`[CONNECT] Clearing abort error — got ${toolCalls.length} tool_calls with ${frames.length} frames`);
    error = null;
  }

  debugLog(`[CONNECT] Result: frames=${frames.length}, text=${textTotal.length}chars, toolCalls=${toolCalls.length}, error=${error || 'none'}`);
  return { text: textTotal, toolCalls, thinkingText, frames, error };
}

/**
 * Make a ConnectRPC bidi request that supports sending tool results back.
 * The onToolCall callback is called for each tool call, and its return value
 * is sent back as a tool result.
 *
 * @param {string} baseUrl
 * @param {Array} messages
 * @param {string} modelName
 * @param {Array} tools
 * @param {Object} credentials
 * @param {Function} onToolCall - async (toolCall) => { result: string }
 * @param {Object} opts
 * @returns {Promise<CursorResponse>}
 */
export async function makeConnectBidiRequest(baseUrl, messages, modelName, tools, credentials, onToolCall, opts = {}) {
  const transport = getTransport(baseUrl, credentials);
  const client = createClient(ChatService, transport);
  const { request } = buildConnectRequest(
    messages, modelName, tools, opts.reasoningEffort, opts.maxMode
  );

  const frames = [];
  let textTotal = "";
  let thinkingText = "";
  const toolCalls = [];
  let error = null;

  const pendingResults = [];
  let resolveWait = null;
  let streamDone = false;

  function pushResult(msg) {
    pendingResults.push(msg);
    if (resolveWait) { resolveWait(); resolveWait = null; }
  }

  function markDone() {
    streamDone = true;
    if (resolveWait) { resolveWait(); resolveWait = null; }
  }

  async function* requestStream() {
    yield request;
    while (!streamDone) {
      if (pendingResults.length > 0) {
        yield pendingResults.shift();
      } else {
        await new Promise(r => { resolveWait = r; });
      }
    }
  }

  try {
    const stream = client.streamUnifiedChatWithTools(requestStream());

    for await (const response of stream) {
      const text = response.response?.text || "";
      const thinking = response.response?.thinking;

      const frame = {
        text: text || null,
        thinking: thinking ? { text: thinking.text || "", signature: thinking.signature || "" } : null,
        serverBubbleId: response.response?.serverBubbleId || null,
        usageUuid: response.response?.usageUuid || null,
        toolCall: null,
        error: null,
      };

      if (text) textTotal += text;
      if (thinking?.text) thinkingText += thinking.text;

      if (response.toolCall?.toolCallId) {
        const tc = response.toolCall;
        const mcpToolName = tc.mcpParams?.tools?.[0]?.name || tc.name || "";

        frame.toolCall = {
          id: tc.toolCallId,
          name: mcpToolName,
          rawArgs: tc.rawArgs || "",
          mcpParams: tc.mcpParams,
          modelCallId: tc.modelCallId || "",
          isPartial: false,
          isLast: tc.isLastMessage || false,
          toolIndex: tc.toolIndex || 0,
        };

        toolCalls.push(frame.toolCall);

        // Call the tool handler and send result back
        if (onToolCall && tc.rawArgs) {
          try {
            const result = await onToolCall(frame.toolCall);
            const toolResultMsg = buildToolResultRequest({
              tool_call_id: tc.toolCallId,
              name: mcpToolName,
              result: result?.result || "",
              model_call_id: tc.modelCallId || "",
            });
            pushResult(toolResultMsg);
          } catch (toolErr) {
            console.error(`[CONNECT] Tool execution error: ${toolErr.message}`);
          }
        }
      }

      frames.push(frame);
    }
  } catch (err) {
    error = `[${err.code || "unknown"}] ${err.message}`;
    console.error(`[CONNECT] Bidi error: ${error}`);
  } finally {
    markDone();
  }

  return { text: textTotal, toolCalls, thinkingText, frames, error };
}

export { buildConnectRequest, buildToolResultRequest, getTransport };
