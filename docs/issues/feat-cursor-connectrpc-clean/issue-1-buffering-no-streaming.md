# Issue 1: Full Response Buffering — No Real Streaming

**Severity:** 🔴 Critical  
**Files:** `open-sse/utils/cursorConnect.js`, `open-sse/executors/cursor.js`  
**Branch:** `feat/cursor-connectrpc-clean`  
**Observed symptom:** TTFT (Time To First Token) of ~36 seconds for a response that should stream progressively.

---

## Problem

`makeConnectRequest()` in `cursorConnect.js` is an `async` function that accumulates **all** ConnectRPC response frames into an array before returning. The executor (`cursor.js`) awaits this function at line 64:

```javascript
const result = await makeConnectRequest(
  this.config, messages, cleanModel, tools, credentials,
  { reasoningEffort, maxMode, signal }
);
```

This means:
1. The request is sent to Cursor API
2. Cursor starts generating tokens (streaming response)
3. **9router buffers every single frame** in `cursorConnect.js:315` (`const frames = []`) inside the `for await` loop (line 333)
4. Only once Cursor finishes generating **all** tokens does `makeConnectRequest` return
5. Then `cursor.js` transforms the buffered frames into SSE chunks or JSON
6. The client receives the entire response at once

The client sees zero output for the entire model generation time (~36s for large responses), then all tokens arrive in a burst.

## How `master` worked

The `master` branch also buffered the HTTP/2 response, but it used a **unary** HTTP/2 POST (`makeHttp2Request` / `makeFetchRequest`). That approach had lower overhead because:
- It was a single request-response cycle, not a bidi stream
- No generator function (`requestStream()`) holding the connection open
- The HTTP/2 request naturally completed when all data was received

Both approaches buffer, but the ConnectRPC bidi stream has additional overhead from keeping the request stream open via the `await new Promise(r => { closeGenerator = r; })` pattern (line 326).

## Root cause in code

**`cursorConnect.js` lines 315–399:**
```javascript
const frames = [];
// ...
for await (const response of stream) {
  // ... builds frame objects
  frames.push(frame);
}
// ... only returns after loop completes
return { text: textTotal, toolCalls, thinkingText, frames, error };
```

**`cursor.js` lines 64–67:**
```javascript
const result = await makeConnectRequest(/* ... */);
// ↑ blocks until ALL frames are collected
```

## Proposed fix

Change `makeConnectRequest` to return an `AsyncGenerator` or `ReadableStream` that yields frames as they arrive:

```javascript
// Option A: AsyncGenerator
export async function* streamConnectRequest(config, messages, model, tools, credentials, opts) {
  // ... setup ...
  for await (const response of stream) {
    yield parseFrame(response);  // Emit each frame immediately
  }
}

// Option B: ReadableStream
export function makeConnectStream(config, messages, model, tools, credentials, opts) {
  return new ReadableStream({
    async start(controller) {
      for await (const response of stream) {
        controller.enqueue(parseFrame(response));
      }
      controller.close();
    }
  });
}
```

Then in `cursor.js`, `transformFramesToSSE` should become a streaming transform that pipes frames to SSE chunks on-the-fly, instead of accumulating all chunks into `chunks.join("")`.

## Impact if not fixed

- TTFT equals total generation time (unacceptable for any streaming use case)
- Users see a frozen UI for 10–60+ seconds depending on response length
- Timeout errors if the total generation time exceeds client timeouts
- Defeats the entire purpose of SSE streaming

---

## ✅ Resolution (2026-03-16)

**Status:** Fixed  
**Commit:** `95a3c4b` on `feat/cursor-connectrpc-clean`

### Implementation

Chose **Option A (AsyncGenerator)** from the proposed fix, combined with a **TransformStream** for SSE conversion:

1. **`cursorConnect.js`** — Added `streamConnectRequest()` async generator that yields `Frame` objects one-at-a-time as they arrive from the ConnectRPC bidi stream. Same parameters as `makeConnectRequest` but uses `async function*` + `yield`.

2. **`cursor.js`** — Added `transformFramesToSSEStream()` using the **TransformStream pattern** (same architecture as the Kiro executor). Modified `execute()` to dispatch `stream !== false` to the new streaming path while `stream === false` continues using the buffered `makeConnectRequest` + `transformFramesToJSON`.

### Architecture

```
streamConnectRequest() [async generator, yields Frame per ConnectRPC message]
  ↓ serialized as JSON lines
ReadableStream (pull: reads one frame per pull from generator)
  ↓ pipeThrough()
TransformStream (transform: JSON frame → SSE text chunk)
  ↓ returned as Response body
handleStreamingResponse → pipeWithDisconnect → passthrough → client
```

### Key lesson: `ReadableStream({ async start() })` does NOT stream

The initial implementation used `ReadableStream({ async start(controller) { for await ... enqueue() } })`. This **buffers all chunks** because `pipeThrough()` defers consumption until `start()` resolves, which doesn't happen until the entire generator is consumed. All `enqueue()` calls during `start()` accumulate in the internal queue and flush at once.

The fix was to use the **TransformStream** pattern (as proven by the Kiro executor): a source `ReadableStream` with `pull()` that yields serialized frame data, piped through a `TransformStream` whose `transform()` callback converts each chunk to SSE text and enqueues immediately.
