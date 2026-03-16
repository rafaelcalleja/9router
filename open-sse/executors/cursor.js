import { BaseExecutor } from "./base.js";
import { PROVIDERS } from "../config/providers.js";
import { HTTP_STATUS } from "../config/runtimeConfig.js";
import { errorResponse } from "../utils/error.js";
import { buildCursorHeaders } from "../utils/cursorChecksum.js";
import { makeConnectRequest } from "../utils/cursorConnect.js";
import { estimateUsage } from "../utils/usageTracking.js";
import { FORMATS } from "../translator/formats.js";

// Env-gated debug logging (set CURSOR_DEBUG=1 to enable)
const CURSOR_DEBUG = process.env.CURSOR_DEBUG === "1";
const debugLog = (...args) => CURSOR_DEBUG && console.log(...args);

/**
 * Detect rate limit / resource exhaustion from error strings.
 * ConnectRPC errors arrive as "[resource_exhausted] ..."
 * Cursor's ErrorDetails may include "rate limit" / "quota" keywords.
 * Preserves 9router's accountFallback behavior (429 → exponential backoff).
 */
function isRateLimitError(errorStr) {
  if (!errorStr) return false;
  const lower = errorStr.toLowerCase();
  return lower.includes('resource_exhausted') || lower.includes('rate_limit') || lower.includes('rate limit');
}

export class CursorExecutor extends BaseExecutor {
  constructor() {
    super("cursor", PROVIDERS.cursor);
  }

  buildUrl() {
    return `${this.config.baseUrl}${this.config.chatPath}`;
  }

  buildHeaders(credentials) {
    const accessToken = credentials.accessToken;
    const machineId = credentials.providerSpecificData?.machineId;
    const ghostMode = credentials.providerSpecificData?.ghostMode !== false;

    if (!machineId) {
      throw new Error("Machine ID is required for Cursor API");
    }

    return buildCursorHeaders(accessToken, machineId, ghostMode, {
      clientVersion: this.config.clientVersion,
      headers: this.config.headers
    });
  }

  async execute({ model, body, stream, credentials, signal, log, proxyOptions = null }) {
    const url = this.buildUrl();
    const headers = this.buildHeaders(credentials);
    const messages = body.messages || [];
    const tools = body.tools || [];
    const reasoningEffort = body.reasoning_effort || null;
    const maxMode = true;

    log?.debug?.("CURSOR", `ConnectRPC request: model=${model}, tools=${tools.length}, msgs=${messages.length}`);

    try {
      const result = await makeConnectRequest(
        this.config, messages, model, tools, credentials,
        { reasoningEffort, maxMode, signal }
      );

      if (result.error) {
        const err = result.error; // { message, code, errorDetails }
        const isAuth = err.message.includes('[16]') || err.message.includes('unauthenticated');
        const isRateLimit = isRateLimitError(err.message);
        log?.warn?.("CURSOR", `ConnectRPC error: ${err.message}`);

        // Use ErrorDetails for richer error info when available (matches old createErrorResponse)
        const errorMsg = err.errorDetails?.title
          || err.errorDetails?.detail
          || err.message
          || "API Error";

        const status = isAuth ? 401 : isRateLimit ? HTTP_STATUS.RATE_LIMITED : HTTP_STATUS.BAD_REQUEST;
        const type = isAuth ? "authentication_error" : isRateLimit ? "rate_limit_error" : "api_error";
        const code = isAuth ? "unauthorized"
          : err.errorDetails?.error != null ? String(err.errorDetails.error)
          : isRateLimit ? "rate_limited" : "";

        return {
          response: new Response(JSON.stringify({
            error: { message: errorMsg, type, code }
          }), {
            status,
            headers: { "Content-Type": "application/json" }
          }),
          url, headers, transformedBody: body
        };
      }

      const transformedResponse = stream !== false
        ? this.transformFramesToSSE(result.frames, model, body)
        : this.transformFramesToJSON(result.frames, model, body);

      return { response: transformedResponse, url, headers, transformedBody: body };
    } catch (error) {
      log?.error?.("CURSOR", `ConnectRPC exception: ${error.message}`);
      const errorResponse = new Response(JSON.stringify({
        error: { message: error.message, type: "connection_error", code: "" }
      }), {
        status: HTTP_STATUS.SERVER_ERROR,
        headers: { "Content-Type": "application/json" }
      });
      return { response: errorResponse, url, headers, transformedBody: body };
    }
  }

  /**
   * Transform decoded ConnectRPC frames to JSON response.
   */
  transformFramesToJSON(frames, model, body) {
    const responseId = `chatcmpl-cursor-${Date.now()}`;
    const created = Math.floor(Date.now() / 1000);
    let totalContent = "";
    const toolCalls = [];
    const toolCallsMap = new Map();
    const finalizedIds = new Set();

    for (const frame of frames) {
      if (frame.error) {
        if (!totalContent && toolCallsMap.size === 0) {
          return new Response(JSON.stringify({
            error: {
              message: frame.error,
              type: "rate_limit_error",
              code: "rate_limited"
            }
          }), {
            status: HTTP_STATUS.RATE_LIMITED,
            headers: { "Content-Type": "application/json" }
          });
        }
        break;
      }

      if (frame.toolCall) {
        const tc = frame.toolCall;
        if (toolCallsMap.has(tc.id)) {
          const existing = toolCallsMap.get(tc.id);
          existing.function.arguments += tc.rawArgs || "";
          existing.isLast = tc.isLast;
        } else {
          toolCallsMap.set(tc.id, {
            id: tc.id,
            type: "function",
            function: { name: tc.name, arguments: tc.rawArgs || "" },
            isLast: tc.isLast,
          });
        }
        if (tc.isLast) {
          const final = toolCallsMap.get(tc.id);
          finalizedIds.add(tc.id);
          toolCalls.push({ id: final.id, type: final.type, function: final.function });
        }
      }

      if (frame.text) totalContent += frame.text;
    }

    // Finalize remaining tool calls (stream may end without isLast=true)
    for (const [id, tc] of toolCallsMap.entries()) {
      if (!finalizedIds.has(id)) {
        toolCalls.push({ id: tc.id, type: tc.type, function: tc.function });
      }
    }

    const message = { role: "assistant", content: totalContent || null };
    if (toolCalls.length > 0) message.tool_calls = toolCalls;

    const usage = estimateUsage(body, totalContent.length, FORMATS.OPENAI);
    const completion = {
      id: responseId, object: "chat.completion", created, model,
      choices: [{ index: 0, message, finish_reason: toolCalls.length > 0 ? "tool_calls" : "stop" }],
      usage
    };

    debugLog(`[CURSOR] JSON: finish_reason=${completion.choices[0].finish_reason}, text=${totalContent.length}chars, toolCalls=${toolCalls.length}`);
    return new Response(JSON.stringify(completion), {
      status: 200, headers: { "Content-Type": "application/json" }
    });
  }

  /**
   * Transform decoded ConnectRPC frames to SSE stream.
   */
  transformFramesToSSE(frames, model, body) {
    const responseId = `chatcmpl-cursor-${Date.now()}`;
    const created = Math.floor(Date.now() / 1000);
    const chunks = [];
    let totalContent = "";
    const toolCalls = [];
    const toolCallsMap = new Map();
    const finalizedIds = new Set();
    const emittedToolCallIds = new Set();

    for (const frame of frames) {
      if (frame.error) {
        if (chunks.length === 0 && totalContent === "" && toolCallsMap.size === 0) {
          return errorResponse(HTTP_STATUS.RATE_LIMITED, frame.error);
        }
        break;
      }

      if (frame.toolCall) {
        const tc = frame.toolCall;

        // Ensure role chunk exists
        if (chunks.length === 0) {
          chunks.push(`data: ${JSON.stringify({
            id: responseId, object: "chat.completion.chunk", created, model,
            choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }]
          })}\n\n`);
        }

        if (toolCallsMap.has(tc.id)) {
          const existing = toolCallsMap.get(tc.id);
          existing.function.arguments += tc.rawArgs || "";
          existing.isLast = tc.isLast;
          if (tc.rawArgs) {
            emittedToolCallIds.add(tc.id);
            chunks.push(`data: ${JSON.stringify({
              id: responseId, object: "chat.completion.chunk", created, model,
              choices: [{ index: 0, delta: { tool_calls: [{
                index: existing.index, id: tc.id, type: "function",
                function: { name: tc.name, arguments: tc.rawArgs }
              }] }, finish_reason: null }]
            })}\n\n`);
          }
        } else {
          const toolCallIndex = toolCalls.length;
          finalizedIds.add(tc.id);
          toolCalls.push({ ...tc, index: toolCallIndex, type: "function", function: { name: tc.name, arguments: tc.rawArgs || "" } });
          toolCallsMap.set(tc.id, { ...tc, index: toolCallIndex, type: "function", function: { name: tc.name, arguments: tc.rawArgs || "" } });
          emittedToolCallIds.add(tc.id);
          chunks.push(`data: ${JSON.stringify({
            id: responseId, object: "chat.completion.chunk", created, model,
            choices: [{ index: 0, delta: { tool_calls: [{
              index: toolCallIndex, id: tc.id, type: "function",
              function: { name: tc.name, arguments: tc.rawArgs || "" }
            }] }, finish_reason: null }]
          })}\n\n`);
        }
      }

      if (frame.text) {
        totalContent += frame.text;
        chunks.push(`data: ${JSON.stringify({
          id: responseId, object: "chat.completion.chunk", created, model,
          choices: [{ index: 0,
            delta: chunks.length === 0 && toolCalls.length === 0
              ? { role: "assistant", content: frame.text }
              : { content: frame.text },
            finish_reason: null }]
        })}\n\n`);
      }
    }

    // Finalize remaining tool calls
    for (const [id, tc] of toolCallsMap.entries()) {
      if (!finalizedIds.has(id)) {
        const idx = toolCalls.length;
        toolCalls.push({ id: tc.id, type: "function", index: idx, function: tc.function });
        if (!emittedToolCallIds.has(tc.id)) {
          chunks.push(`data: ${JSON.stringify({
            id: responseId, object: "chat.completion.chunk", created, model,
            choices: [{ index: 0, delta: { tool_calls: [{
              index: idx, id: tc.id, type: "function", function: tc.function
            }] }, finish_reason: null }]
          })}\n\n`);
        }
      }
    }

    if (chunks.length === 0 && toolCalls.length === 0) {
      chunks.push(`data: ${JSON.stringify({
        id: responseId, object: "chat.completion.chunk", created, model,
        choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }]
      })}\n\n`);
    }

    const usage = estimateUsage(body, totalContent.length, FORMATS.OPENAI);
    chunks.push(`data: ${JSON.stringify({
      id: responseId, object: "chat.completion.chunk", created, model,
      choices: [{ index: 0, delta: {}, finish_reason: toolCalls.length > 0 ? "tool_calls" : "stop" }],
      usage
    })}\n\n`);

    debugLog(`[CURSOR] SSE: ${chunks.length} chunks, finish_reason=${toolCalls.length > 0 ? "tool_calls" : "stop"}, text=${totalContent.length}chars, toolCalls=${toolCalls.length}`);

    return new Response(chunks.join(""), {
      status: 200,
      headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive" }
    });
  }

  async refreshCredentials() {
    return null;
  }
}

export default CursorExecutor;
