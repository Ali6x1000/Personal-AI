export type TokenResponse = {
  token: string;
  livekit_url: string;
  room: string;
  identity: string;
};

export type FetchTokenOptions = {
  devTrace?: boolean;
  /**
   * Stable room slug for QA only. By default each call creates `alijr-<uuid>` so concurrent users don't
   * share audio. For shared demos, pass the same name (avoid in production concurrency).
   */
  roomName?: string;
  /** AbortSignal (e.g. user cancelled). */
  signal?: AbortSignal;
};

/** Backend API root (FastAPI). Local default mirrors uvicorn on port 8000. */
export function getBackendBaseUrl(): string {
  const raw =
    typeof process !== "undefined" && process.env.NEXT_PUBLIC_BACKEND_URL
      ? process.env.NEXT_PUBLIC_BACKEND_URL
      : "http://localhost:8000";
  return raw.replace(/\/$/, "");
}

function newSessionRoomSlug(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `alijr-${crypto.randomUUID()}`;
  }
  return `alijr-${Date.now()}-${Math.random().toString(36).slice(2, 12)}`;
}

function mergeAbortSignals(outer?: AbortSignal, inner?: AbortSignal): AbortSignal | undefined {
  if (!outer && !inner) return undefined;
  if (!outer) return inner;
  if (!inner) return outer;
  if (outer.aborted || inner.aborted) return outer.aborted ? outer : inner;

  const merged = new AbortController();
  const abort = () => merged.abort();
  outer.addEventListener("abort", abort, { once: true });
  inner.addEventListener("abort", abort, { once: true });
  return merged.signal;
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    let t: ReturnType<typeof setTimeout> | undefined;
    const onAbort = () => {
      clearTimeout(t);
      reject(new DOMException("Aborted", "AbortError"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
    t = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
  });
}

/** Health probe — distinguishes “backend down/CORS/wrong URL” vs LiveKit failures. */
export async function pingTokenBackend(signal?: AbortSignal, timeoutMs = 12000): Promise<boolean> {
  const ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
  const timer =
    ctrl != null ? setTimeout(() => ctrl.abort(), timeoutMs) : undefined;
  const sig = mergeAbortSignals(signal, ctrl?.signal);

  try {
    const res = await fetch(`${getBackendBaseUrl()}/`, {
      method: "GET",
      headers: { Accept: "application/json" },
      cache: "no-store",
      signal: sig,
    });
    return res.ok;
  } catch {
    return false;
  } finally {
    if (timer != null) clearTimeout(timer);
  }
}

async function mintTokenOnce(
  roomName: string,
  participantName: string,
  devTrace: boolean,
  signal?: AbortSignal,
): Promise<Response> {
  const BACKEND_URL = getBackendBaseUrl();
  const params = new URLSearchParams();
  params.set("name", participantName);
  params.set("room_name", roomName);
  if (devTrace) {
    params.set("dev_trace", "true");
  }

  return fetch(`${BACKEND_URL}/getToken?${params.toString()}`, {
    method: "GET",
    headers: { Accept: "application/json" },
    cache: "no-store",
    signal,
  });
}

async function mintWithTimeout(
  roomName: string,
  participantName: string,
  devTrace: boolean,
  outerSignal: AbortSignal | undefined,
  perTryMs: number,
): Promise<Response> {
  const timeoutCtrl = typeof AbortController !== "undefined" ? new AbortController() : null;
  const timer =
    timeoutCtrl != null ? setTimeout(() => timeoutCtrl.abort(), perTryMs) : undefined;
  const merged = mergeAbortSignals(outerSignal, timeoutCtrl?.signal);

  try {
    return await mintTokenOnce(roomName, participantName, devTrace, merged);
  } finally {
    if (timer != null) clearTimeout(timer);
  }
}

/**
 * Mint a LiveKit JWT using a UNIQUE room per call (default `alijr-<uuid>`) so users never hear each other.
 * Retries on transient failures; probes `/` first for clearer errors.
 */
export async function fetchToken(
  participantName: string,
  options: FetchTokenOptions = {},
): Promise<TokenResponse> {
  const roomName =
    typeof options.roomName === "string" && options.roomName.trim().length > 0
      ? options.roomName.trim()
      : newSessionRoomSlug();

  const devTrace = Boolean(options.devTrace);
  let healthy = await pingTokenBackend(options.signal);
  if (!healthy) {
    await sleep(400, options.signal).catch(() => undefined);
    healthy = await pingTokenBackend(options.signal);
  }
  if (!healthy) {
    throw new Error(
      `Cannot reach token API at ${getBackendBaseUrl()}. Check NEXT_PUBLIC_BACKEND_URL, HTTPS/mixed-content, ` +
        `FastAPI CORS (CORS_ALLOW_ORIGINS / CORS_ALLOW_ORIGIN_REGEX), and that the server is up.`,
    );
  }

  const maxAttempts = 4;
  const perTryMs = 20000;
  let lastError: unknown = null;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const response = await mintWithTimeout(
        roomName,
        participantName,
        devTrace,
        options.signal,
        perTryMs,
      );

      if (!response.ok) {
        const retryable = response.status === 429 || response.status >= 502;
        const err = new Error(`Token request failed: ${response.status} ${response.statusText}`);
        if (!retryable || attempt === maxAttempts) {
          throw err;
        }
        lastError = err;
      } else {
        const payload = (await response.json()) as Partial<TokenResponse>;
        const token = payload.token?.trim() ?? "";
        const livekitUrl =
          payload.livekit_url?.trim() ||
          (typeof process !== "undefined" ? process.env.NEXT_PUBLIC_LIVEKIT_URL?.trim() : "") ||
          "";
        if (!token) {
          throw new Error("Token endpoint returned an empty token.");
        }
        if (!livekitUrl) {
          throw new Error(
            "Token response missing livekit_url; set LIVEKIT_URL on the backend or NEXT_PUBLIC_LIVEKIT_URL on the frontend.",
          );
        }

        return {
          token,
          livekit_url: livekitUrl,
          room: payload.room ?? roomName,
          identity: payload.identity ?? participantName,
        };
      }
    } catch (e) {
      lastError = e;
      if (
        e instanceof Error &&
        (e.name === "AbortError" || e.message.includes("abort"))
      ) {
        throw e;
      }
      if (attempt === maxAttempts) {
        break;
      }
    }

    await sleep(Math.min(2000, 400 * Math.pow(2, attempt - 1)), options.signal).catch(() => undefined);
  }

  if (lastError instanceof Error) {
    throw lastError;
  }
  throw new Error("Token mint failed after retries.");
}
