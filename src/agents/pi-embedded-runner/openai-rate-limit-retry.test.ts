import type { StreamFn } from "@mariozechner/pi-agent-core";
import { createAssistantMessageEventStream } from "@mariozechner/pi-ai";
import { describe, expect, it, vi } from "vitest";
import {
  computeBackoffDelayMs,
  isOpenAICompatApi,
  parseRetryAfterMs,
  wrapStreamFnWithRateLimitRetry,
} from "./openai-rate-limit-retry.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeErrorEvent(errorMessage: string) {
  return {
    type: "error" as const,
    reason: "error" as const,
    error: {
      stopReason: "error" as const,
      errorMessage,
      inputTokens: 0,
      outputTokens: 0,
      thinking: false,
      functionCalls: [],
      toolResults: [],
    },
  };
}

function makeDoneEvent() {
  return {
    type: "done" as const,
    reason: "done" as const,
    message: {
      stopReason: "done" as const,
      errorMessage: undefined,
      inputTokens: 0,
      outputTokens: 0,
      thinking: false,
      functionCalls: [],
      toolResults: [],
    },
  };
}

/** Build a StreamFn that emits the provided sequence of events. */
function makeStreamFn(events: object[]): StreamFn {
  return (_model, _context, _options) => {
    const stream = createAssistantMessageEventStream();
    (async () => {
      for (const ev of events) {
        stream.push(ev as Parameters<typeof stream.push>[0]);
      }
    })().catch(() => {});
    return stream;
  };
}

/** Collect all events from a stream into an array. */
async function collectEvents(stream: ReturnType<StreamFn>): Promise<object[]> {
  const events: object[] = [];
  for await (const ev of await stream) {
    events.push(ev);
  }
  return events;
}

// ---------------------------------------------------------------------------
// parseRetryAfterMs
// ---------------------------------------------------------------------------

describe("parseRetryAfterMs", () => {
  it("returns undefined for messages without retry-after", () => {
    expect(parseRetryAfterMs("Rate limit exceeded")).toBeUndefined();
    expect(parseRetryAfterMs("429 Too Many Requests")).toBeUndefined();
    expect(parseRetryAfterMs("")).toBeUndefined();
  });

  it("parses integer seconds", () => {
    expect(parseRetryAfterMs("retry-after: 30")).toBe(30_000);
    expect(parseRetryAfterMs("Retry-After: 60")).toBe(60_000);
  });

  it("parses fractional seconds (rounds up)", () => {
    expect(parseRetryAfterMs("retry-after: 1.5")).toBe(1_500);
    expect(parseRetryAfterMs("retry-after: 0.1")).toBe(100);
  });

  it("ignores zero and negative values", () => {
    expect(parseRetryAfterMs("retry-after: 0")).toBeUndefined();
    expect(parseRetryAfterMs("retry-after: -5")).toBeUndefined();
  });

  it("handles various spacing/casing variants", () => {
    expect(parseRetryAfterMs("retryafter:10")).toBe(10_000);
    expect(parseRetryAfterMs("RETRY-AFTER: 20")).toBe(20_000);
  });
});

// ---------------------------------------------------------------------------
// computeBackoffDelayMs
// ---------------------------------------------------------------------------

describe("computeBackoffDelayMs", () => {
  it("uses exponential backoff without retry-after", () => {
    // attempt 0: base = 1000 * 2^0 = 1000; jitter 0–249 → range [1000, 1249]
    const d0 = computeBackoffDelayMs(0);
    expect(d0).toBeGreaterThanOrEqual(1000);
    expect(d0).toBeLessThan(1250);

    // attempt 3: base = 1000 * 2^3 = 8000; range [8000, 8249]
    const d3 = computeBackoffDelayMs(3);
    expect(d3).toBeGreaterThanOrEqual(8000);
    expect(d3).toBeLessThan(8250);
  });

  it("caps at MAX_DELAY_MS (60 000 ms)", () => {
    // attempt 10: base = 1000 * 2^10 = 1 024 000 → capped to 60 000
    const d = computeBackoffDelayMs(10);
    expect(d).toBeGreaterThanOrEqual(60_000);
    expect(d).toBeLessThan(60_250);
  });

  it("uses retryAfterMs instead of exponential backoff", () => {
    const d = computeBackoffDelayMs(0, 45_000);
    expect(d).toBeGreaterThanOrEqual(45_000);
    expect(d).toBeLessThan(45_250);
  });

  it("caps retryAfterMs at MAX_DELAY_MS", () => {
    const d = computeBackoffDelayMs(0, 120_000);
    expect(d).toBeGreaterThanOrEqual(60_000);
    expect(d).toBeLessThan(60_250);
  });
});

// ---------------------------------------------------------------------------
// isOpenAICompatApi
// ---------------------------------------------------------------------------

describe("isOpenAICompatApi", () => {
  it("returns true for OpenAI-compatible api values", () => {
    expect(isOpenAICompatApi("openai-completions")).toBe(true);
    expect(isOpenAICompatApi("openai-responses")).toBe(true);
    expect(isOpenAICompatApi("azure-openai-responses")).toBe(true);
    expect(isOpenAICompatApi("openai-codex-responses")).toBe(true);
  });

  it("returns false for other api values", () => {
    expect(isOpenAICompatApi("anthropic")).toBe(false);
    expect(isOpenAICompatApi("ollama")).toBe(false);
    expect(isOpenAICompatApi(undefined)).toBe(false);
    expect(isOpenAICompatApi("gemini")).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// wrapStreamFnWithRateLimitRetry
// ---------------------------------------------------------------------------

describe("wrapStreamFnWithRateLimitRetry", () => {
  // Suppress log.warn output during tests.
  vi.mock("./logger.js", () => ({ log: { warn: vi.fn(), info: vi.fn(), debug: vi.fn() } }));
  vi.mock("../../utils.js", () => ({
    sleep: vi.fn().mockResolvedValue(undefined),
    resolveUserPath: vi.fn(),
  }));

  const fakeModel = {} as Parameters<StreamFn>[0];
  const fakeContext = {} as Parameters<StreamFn>[1];

  it("passes through a successful stream unchanged", async () => {
    const done = makeDoneEvent();
    const baseFn = makeStreamFn([done]);
    const wrapped = wrapStreamFnWithRateLimitRetry(baseFn, { maxRetries: 3 });

    const events = await collectEvents(wrapped(fakeModel, fakeContext));
    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({ type: "done" });
  });

  it("retries on 429 rate-limit error and eventually succeeds", async () => {
    const rateLimitErr = makeErrorEvent("429 rate limit exceeded");
    const done = makeDoneEvent();

    let calls = 0;
    const baseFn: StreamFn = (_model, _context, _options) => {
      calls++;
      return makeStreamFn(calls < 3 ? [rateLimitErr] : [done])(_model, _context, _options);
    };

    const wrapped = wrapStreamFnWithRateLimitRetry(baseFn, { maxRetries: 5 });
    const events = await collectEvents(wrapped(fakeModel, fakeContext));

    expect(calls).toBe(3);
    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({ type: "done" });
  });

  it("surfaces the rate-limit error after exhausting all retries", async () => {
    const rateLimitErr = makeErrorEvent("429 rate limit exceeded");

    const baseFn = makeStreamFn([rateLimitErr]);
    const wrapped = wrapStreamFnWithRateLimitRetry(baseFn, { maxRetries: 3 });

    const events = await collectEvents(wrapped(fakeModel, fakeContext));

    expect(events).toHaveLength(1);
    const [ev] = events as [{ type: string; error: { errorMessage: string } }];
    expect(ev.type).toBe("error");
    // Error message should be annotated with retry count.
    expect(ev.error.errorMessage).toMatch(/retried 3 times/);
  });

  it("does NOT retry on 401 errors", async () => {
    const authErr = makeErrorEvent("401 Unauthorized");

    let calls = 0;
    const baseFn: StreamFn = (_model, _context, _options) => {
      calls++;
      return makeStreamFn([authErr])(_model, _context, _options);
    };

    const wrapped = wrapStreamFnWithRateLimitRetry(baseFn, { maxRetries: 3 });
    const events = await collectEvents(wrapped(fakeModel, fakeContext));

    // Should only call once, passing through immediately.
    expect(calls).toBe(1);
    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({ type: "error" });
  });

  it("does NOT retry on 400 errors", async () => {
    const badReqErr = makeErrorEvent("400 Bad Request");

    let calls = 0;
    const baseFn: StreamFn = (_model, _context, _options) => {
      calls++;
      return makeStreamFn([badReqErr])(_model, _context, _options);
    };

    const wrapped = wrapStreamFnWithRateLimitRetry(baseFn, { maxRetries: 3 });
    await collectEvents(wrapped(fakeModel, fakeContext));

    expect(calls).toBe(1);
  });

  it("does NOT retry on 403 errors", async () => {
    const forbiddenErr = makeErrorEvent("403 Forbidden");

    let calls = 0;
    const baseFn: StreamFn = (_model, _context, _options) => {
      calls++;
      return makeStreamFn([forbiddenErr])(_model, _context, _options);
    };

    const wrapped = wrapStreamFnWithRateLimitRetry(baseFn, { maxRetries: 3 });
    await collectEvents(wrapped(fakeModel, fakeContext));

    expect(calls).toBe(1);
  });

  it("passes through non-rate-limit errors immediately without retry", async () => {
    const serverErr = makeErrorEvent("500 Internal Server Error");

    let calls = 0;
    const baseFn: StreamFn = (_model, _context, _options) => {
      calls++;
      return makeStreamFn([serverErr])(_model, _context, _options);
    };

    const wrapped = wrapStreamFnWithRateLimitRetry(baseFn, { maxRetries: 3 });
    const events = await collectEvents(wrapped(fakeModel, fakeContext));

    expect(calls).toBe(1);
    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({ type: "error" });
    // Not annotated.
    expect((events[0] as { error: { errorMessage: string } }).error.errorMessage).not.toMatch(
      /retried/,
    );
  });

  it("recognizes 'Too Many Requests' as a rate-limit error", async () => {
    const tooManyErr = makeErrorEvent("429 Too Many Requests - too many requests");

    let calls = 0;
    const baseFn: StreamFn = (_model, _context, _options) => {
      calls++;
      return makeStreamFn(calls < 2 ? [tooManyErr] : [makeDoneEvent()])(_model, _context, _options);
    };

    const wrapped = wrapStreamFnWithRateLimitRetry(baseFn, { maxRetries: 3 });
    const events = await collectEvents(wrapped(fakeModel, fakeContext));

    expect(calls).toBe(2);
    expect(events[0]).toMatchObject({ type: "done" });
  });
});
