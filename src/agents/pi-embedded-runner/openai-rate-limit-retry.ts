import type { StreamFn } from "@mariozechner/pi-agent-core";
import type { AssistantMessage, AssistantMessageEvent } from "@mariozechner/pi-ai";
import { createAssistantMessageEventStream } from "@mariozechner/pi-ai";
import { sleep } from "../../utils.js";
import { isRateLimitErrorMessage } from "../pi-embedded-helpers.js";
import { log } from "./logger.js";

const MAX_RETRIES = 5;
const BASE_DELAY_MS = 1000;
const MAX_DELAY_MS = 60_000;
const JITTER_MAX_MS = 250;

// These status codes should never trigger a retry.
const NO_RETRY_CODES_RE = /\b(400|401|403)\b/;

function isNonRetryableError(errorMessage: string): boolean {
  return NO_RETRY_CODES_RE.test(errorMessage);
}

/** Parse a Retry-After value (in seconds) from an error message, if present. */
export function parseRetryAfterMs(errorMessage: string): number | undefined {
  const match = errorMessage.match(/retry[- ]?after[:\s]+(\d+(?:\.\d+)?)/i);
  if (match?.[1]) {
    const seconds = Number.parseFloat(match[1]);
    if (Number.isFinite(seconds) && seconds > 0) {
      return Math.ceil(seconds * 1000);
    }
  }
  return undefined;
}

/**
 * Compute delay for a retry attempt.
 * Uses Retry-After if provided, otherwise exponential backoff, both with jitter.
 *
 * @param attempt - 0-indexed attempt number (0 = first retry after initial failure)
 * @param retryAfterMs - Optional Retry-After value from the server, in milliseconds
 */
export function computeBackoffDelayMs(attempt: number, retryAfterMs?: number): number {
  const base = retryAfterMs && retryAfterMs > 0 ? retryAfterMs : BASE_DELAY_MS * 2 ** attempt;
  const capped = Math.min(base, MAX_DELAY_MS);
  return capped + Math.floor(Math.random() * JITTER_MAX_MS);
}

/**
 * Returns true if the model's API is an OpenAI-compatible provider that benefits
 * from this retry layer. Restricts retry logic to OpenAI-based APIs only.
 */
export function isOpenAICompatApi(api?: string): boolean {
  return (
    api === "openai-completions" ||
    api === "openai-responses" ||
    api === "azure-openai-responses" ||
    api === "openai-codex-responses"
  );
}

type RateLimitRetryOpts = {
  /** Provider identifier used in log messages. */
  provider?: string;
  /** Run/correlation ID used in log messages. */
  runId?: string;
  /** Override the max number of attempts for testing. Defaults to MAX_RETRIES (5). */
  maxRetries?: number;
};

/**
 * Wraps a StreamFn with exponential-backoff retry logic for HTTP 429 rate-limit errors.
 *
 * Behaviour:
 * - Retries up to `maxRetries` (default 5) times on 429 / rate-limit errors.
 * - Backoff: min(1000 * 2^attempt, 60 000) ms + up to 250 ms jitter.
 * - If the error message contains a Retry-After value, that delay is used instead.
 * - Does NOT retry on 400 / 401 / 403 errors.
 * - Logs attempt number, wait duration, and correlation details on every retry.
 * - After exhausting retries the original rate-limit error is surfaced to the caller.
 */
export function wrapStreamFnWithRateLimitRetry(
  baseFn: StreamFn,
  opts?: RateLimitRetryOpts,
): StreamFn {
  const maxAttempts = opts?.maxRetries ?? MAX_RETRIES;
  const providerLabel = opts?.provider ?? "unknown";
  const runId = opts?.runId;

  return (model, context, options) => {
    const outerStream = createAssistantMessageEventStream();

    (async () => {
      // lastRateLimitError holds the most recent rate-limit error event so we can
      // surface it after exhausting retries without constructing a synthetic message.
      let lastRateLimitError: (AssistantMessageEvent & { type: "error" }) | null = null;

      for (let attempt = 0; attempt < maxAttempts; attempt++) {
        let hitRateLimit = false;

        try {
          const innerStream = await baseFn(model, context, options);
          for await (const event of innerStream) {
            if (event.type === "error") {
              const errMsg = event.error.errorMessage ?? "";

              // Only retry on rate-limit errors, and never on 400/401/403.
              if (isRateLimitErrorMessage(errMsg) && !isNonRetryableError(errMsg)) {
                hitRateLimit = true;
                lastRateLimitError = event;
                break; // Don't forward — will retry or surface after loop.
              }

              // Non-retryable error: pass through immediately.
              outerStream.push(event);
              return;
            }

            // Forward all non-error events (including "done") to the outer stream.
            outerStream.push(event);
            if (event.type === "done") {
              return;
            }
          }
        } catch (unexpectedErr) {
          // baseFn should never throw (it catches internally), but guard defensively.
          log.warn(
            `openai-rate-limit-retry: unexpected throw from baseFn: ${String(unexpectedErr)}`,
          );
          outerStream.end();
          return;
        }

        if (!hitRateLimit) {
          // Stream ended without a terminal event — shouldn't happen normally.
          outerStream.end();
          return;
        }

        // This was the last allowed attempt; don't wait — fall through to error surface.
        if (attempt >= maxAttempts - 1) {
          break;
        }

        // Parse Retry-After from the error message (best-effort; headers are not available
        // at this layer since the pi-ai library does not propagate HTTP response headers).
        const errMsg = lastRateLimitError?.error.errorMessage ?? "";
        const retryAfterMs = parseRetryAfterMs(errMsg);
        const delayMs = computeBackoffDelayMs(attempt, retryAfterMs);

        log.warn(
          [
            `openai-rate-limit-retry: 429 on attempt ${attempt + 1}/${maxAttempts},`,
            `waiting ${delayMs}ms before retry`,
            retryAfterMs !== undefined ? `(retry-after: ${retryAfterMs}ms)` : "",
            `provider=${providerLabel}`,
            runId ? `runId=${runId}` : "",
          ]
            .filter(Boolean)
            .join(" "),
        );

        await sleep(delayMs);
      }

      // All attempts exhausted — surface the last rate-limit error.
      log.warn(
        [
          `openai-rate-limit-retry: max retries (${maxAttempts}) exhausted,`,
          `surfacing rate-limit error`,
          `provider=${providerLabel}`,
          runId ? `runId=${runId}` : "",
        ]
          .filter(Boolean)
          .join(" "),
      );

      if (lastRateLimitError) {
        // Annotate the error message so downstream can see retry exhaustion context.
        const annotated: AssistantMessage = {
          ...lastRateLimitError.error,
          errorMessage: `${lastRateLimitError.error.errorMessage ?? "429 Rate limit exceeded"} (retried ${maxAttempts} times)`,
        };
        outerStream.push({ type: "error", reason: "error", error: annotated });
      } else {
        outerStream.end();
      }
    })().catch((asyncErr: unknown) => {
      log.warn(`openai-rate-limit-retry: async error in retry loop: ${String(asyncErr)}`);
      try {
        outerStream.end();
      } catch {
        // ignore secondary error from end()
      }
    });

    return outerStream;
  };
}
