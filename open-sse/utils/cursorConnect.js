/**
 * cursorConnect.js — ConnectRPC client for Cursor API
 *
 * Replaces manual HTTP/2 + ConnectRPC framing with native @connectrpc/connect.
 * Uses generated protobuf stubs from cursor.proto.
 */

import { createClient } from "@connectrpc/connect";
import { createConnectTransport, createGrpcWebTransport } from "@connectrpc/connect-node";
import { create } from "@bufbuild/protobuf";
import crypto from "crypto";

import {
  ChatService,
  StreamUnifiedChatRequestWithToolsSchema,
  ErrorDetailsSchema,
} from "../gen/cursor_pb.js";

import { proxyAwareFetch } from "./proxyFetch.js";

// ==================== CONSTANTS ====================

const ROLE = { USER: 1, ASSISTANT: 2 };
const UNIFIED_MODE = { CHAT: 1, AGENT: 2 };
const THINKING_LEVEL = { UNSPECIFIED: 0, MEDIUM: 1, HIGH: 2 };
const CLIENT_SIDE_TOOL_V2_MCP = 19;

// Env-gated debug logging (set CURSOR_CONNECT_DEBUG=1 to enable)
const CURSOR_CONNECT_DEBUG = process.env.CURSOR_CONNECT_DEBUG === "1";
const debugLog = (...args) => CURSOR_CONNECT_DEBUG && console.log(...args);

/**
 * Extract ErrorDetails from a ConnectRPC error (if present).
 * Returns { message, code, errorDetails } where errorDetails
 * contains the Cursor-specific error enum and custom details.
 */
function extractErrorInfo(err) {
  const code = err.code || "unknown";
  const message = `[${code}] ${err.message}`;
  let errorDetails = null;

  // ConnectError has findDetails() to extract typed protobuf error details
  if (typeof err.findDetails === "function") {
    try {
      const details = err.findDetails(ErrorDetailsSchema);
      if (details.length > 0) {
        const d = details[0];
        errorDetails = {
          error: d.error,           // Cursor error enum value (e.g. 41 for RESOURCE_EXHAUSTED)
          title: d.details?.title || null,
          detail: d.details?.detail || null,
          isExpected: d.isExpected || false,
        };
        debugLog(`[CONNECT] ErrorDetails: error=${d.error}, title=${d.details?.title}, detail=${d.details?.detail}`);
      }
    } catch (e) {
      debugLog(`[CONNECT] Failed to parse ErrorDetails: ${e.message}`);
    }
  }

  return { message, code, errorDetails };
}

// ==================== TRANSPORT FACTORY ====================

/**
 * Create a ConnectRPC transport with auth interceptor.
 * Fresh transport per request — accepts pre-built headers from the executor
 * so the same headers are used for both sending and reporting.
 * Matches the stateless pattern used by all other providers.
 *
 * @param {string} baseUrl
 * @param {Object} headers - Pre-built headers from executor's buildHeaders()
 * @param {Object|null} proxyOptions
 */
function getTransport(baseUrl, headers = {}, proxyOptions = null) {
  const useProxy = proxyOptions?.connectionProxyEnabled === true || proxyOptions?.enabled === true;

  // Filter headers for ConnectRPC (skip pseudo-headers and te)
  const connectHeaders = {};
  for (const [key, value] of Object.entries(headers)) {
    if (key.startsWith(":") || key === "te") continue;
    connectHeaders[key] = String(value);
  }

  const interceptors = [
    (next) => async (req) => {
      for (const [key, value] of Object.entries(connectHeaders)) {
        req.header.set(key, value);
      }
      return next(req);
    },
  ];

  return useProxy
    ? createGrpcWebTransport({
        baseUrl,
        httpVersion: "1.1",
        fetch: (url, init) => proxyAwareFetch(url, init, proxyOptions),
        interceptors,
      })
    : createConnectTransport({
        baseUrl,
        httpVersion: "2",
        interceptors,
      });
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
 * @returns {{ request: object }}
 */
function buildConnectRequest(messages, modelName, tools = [], reasoningEffort = null, maxMode = false) {
  const hasTools = tools?.length > 0;
  const isAgentic = hasTools;

  // Build conversation messages
  const conversation = [];
  const messageIds = [];

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
      chatModeEnum: isAgentic ? 2 : 1,
      toolResults,
    });

    messageIds.push({ messageId: msgId, role });
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
      currentFile: {
        contentsStartAtLine: 1,
        cursorPosition: { line: 0, column: 0 },
        totalNumberOfLines: 1,
        selection: {
          startPosition: { line: 0, column: 0 },
          endPosition: { line: 0, column: 0 },
        },
      },
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
      unifiedModeName: isAgentic ? "agent" : "Ask",
    }
  });

  return { request };
}


// ==================== RESPONSE TYPES ====================

/**
 * @typedef {Object} CursorResponse
 * @property {Array} frames - Raw decoded response frames
 * @property {Object|null} error - Error info if any
 */

// ==================== FRAME BUILDER ====================

/**
 * Build a normalized frame object from a ConnectRPC response.
 * Shared between makeConnectRequest (buffered) and streamConnectRequest (streaming).
 *
 * @param {Object} response - ConnectRPC StreamUnifiedChatResponseWithToolsIdempotent
 * @returns {{ text: string|null, thinking: Object|null, serverBubbleId: string|null, usageUuid: string|null, toolCall: Object|null, error: null }}
 */
function buildFrame(response) {
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
  }

  return frame;
}

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
export async function makeConnectRequest(config, messages, modelName, tools, opts = {}) {
  const baseUrl = config.baseUrl || config;
  debugLog(`[CONNECT] makeConnectRequest: model=${modelName}, tools=${tools?.length || 0}, msgs=${messages?.length || 0}`);
  const transport = getTransport(baseUrl, opts.headers || {}, opts.proxyOptions);
  const client = createClient(ChatService, transport);
  const { request } = buildConnectRequest(
    messages, modelName, tools, opts.reasoningEffort, opts.maxMode
  );
  debugLog(`[CONNECT] Request built: mcpTools=${request.request?.mcpTools?.length || 0}, isAgentic=${request.request?.isAgentic}, unifiedMode=${request.request?.unifiedMode}`);

  const frames = [];
  let error = null;

  // Keep the request stream alive until we've received all response frames.
  // Closing too early (half-close) causes Cursor to truncate responses.
  let closeGenerator;
  async function* requestStream() {
    yield request;
    await new Promise(r => { closeGenerator = r; });
  }

  try {
    const stream = client.streamUnifiedChatWithTools(requestStream(), { signal: opts.signal });
    let frameCount = 0;

    for await (const response of stream) {
      frameCount++;
      debugLog(`[CONNECT] Frame #${frameCount}: text=${(response.response?.text || '').length}chars, toolCall=${!!response.toolCall?.toolCallId}`);
      const frame = buildFrame(response);

      if (frame.toolCall) {

        // Got a complete tool call — close the request stream and stop reading.
        // Cursor's bidi stream expects tool results back; since 9router is a proxy
        // (not a bidi client), we break here and let the caller re-send with results.
        if (frame.toolCall.isLast || frame.toolCall.rawArgs) {
          debugLog(`[CONNECT] Got complete tool_call (isLast=${frame.toolCall.isLast}), closing stream`);
          frames.push(frame);
          break;
        }
      }

      frames.push(frame);
    }
  } catch (err) {
    const errorInfo = extractErrorInfo(err);
    error = errorInfo;
    console.error(`[CONNECT] Error: ${errorInfo.message}`);
  } finally {
    // Signal the generator to close (request stream half-close)
    closeGenerator?.();
  }

  // Abort error after break is expected when we got tool_calls —
  // breaking the for-await abandons the ConnectRPC stream mid-read,
  // which causes a cancel/abort error. This is expected, not a real failure.
  if (error && frames.length > 0 && frames.some(f => f.toolCall)) {
    debugLog(`[CONNECT] Clearing expected abort error — got tool_calls with ${frames.length} frames`);
    error = null;
  }

  debugLog(`[CONNECT] Result: frames=${frames.length}, error=${error?.message || 'none'}`);
  return { frames, error };
}

/**
 * Streaming variant of makeConnectRequest.
 * Yields Frame objects as they arrive from the ConnectRPC stream.
 * Use for SSE streaming — the consumer can emit SSE chunks immediately.
 *
 * @param {Object} config
 * @param {Array} messages
 * @param {string} modelName
 * @param {Array} tools
 * @param {Object} credentials
 * @param {Object} opts - {reasoningEffort, maxMode, signal}
 * @yields {Object} Frame - { text, thinking, serverBubbleId, usageUuid, toolCall, error }
 */
export async function* streamConnectRequest(config, messages, modelName, tools, opts = {}) {
  const baseUrl = config.baseUrl || config;
  debugLog(`[CONNECT] streamConnectRequest: model=${modelName}, tools=${tools?.length || 0}, msgs=${messages?.length || 0}`);
  const transport = getTransport(baseUrl, opts.headers || {}, opts.proxyOptions);
  const client = createClient(ChatService, transport);
  const { request } = buildConnectRequest(
    messages, modelName, tools, opts.reasoningEffort, opts.maxMode
  );

  let closeGenerator;
  async function* requestStream() {
    yield request;
    await new Promise(r => { closeGenerator = r; });
  }

  try {
    const stream = client.streamUnifiedChatWithTools(requestStream(), { signal: opts.signal });
    let frameCount = 0;

    for await (const response of stream) {
      frameCount++;
      debugLog(`[CONNECT-STREAM] Frame #${frameCount}: text=${(response.response?.text || '').length}chars, toolCall=${!!response.toolCall?.toolCallId}`);
      const frame = buildFrame(response);

      if (frame.toolCall) {
        // Complete tool call — yield and stop.
        // Cursor's bidi stream expects tool results back; 9router proxies
        // the tool call to the caller who re-sends with results.
        if (frame.toolCall.isLast || frame.toolCall.rawArgs) {
          debugLog(`[CONNECT-STREAM] Got complete tool_call (isLast=${frame.toolCall.isLast}), ending stream`);
          yield frame;
          return;
        }
      }

      yield frame;
    }
  } catch (err) {
    const errorInfo = extractErrorInfo(err);
    debugLog(`[CONNECT-STREAM] Error: ${errorInfo.message}`);
    yield { text: null, thinking: null, serverBubbleId: null, usageUuid: null, toolCall: null, error: errorInfo };
  } finally {
    closeGenerator?.();
  }
}

export { buildConnectRequest, getTransport };
