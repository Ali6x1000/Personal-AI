export type TokenResponse = {
  token: string;
  livekit_url: string;
  room: string;
  identity: string;
};

export type FetchTokenOptions = {
  devTrace?: boolean;
};

/** Backend API root (FastAPI). Local default mirrors uvicorn on port 8000. */
export function getBackendBaseUrl(): string {
  const raw =
    typeof process !== "undefined" && process.env.NEXT_PUBLIC_BACKEND_URL
      ? process.env.NEXT_PUBLIC_BACKEND_URL
      : "http://localhost:8000";
  return raw.replace(/\/$/, "");
}

/**
 * Mint a LiveKit token via GET `/getToken` (aligned with production AWS/Vercel setup).
 * Override backend with NEXT_PUBLIC_BACKEND_URL (e.g. https://api-alijr.alinawaf.com).
 */
export async function fetchToken(
  participantName: string,
  options: FetchTokenOptions = {},
): Promise<TokenResponse> {
  const BACKEND_URL = getBackendBaseUrl();
  const params = new URLSearchParams();
  params.set("name", participantName);
  if (options.devTrace) {
    params.set("dev_trace", "true");
  }

  const response = await fetch(`${BACKEND_URL}/getToken?${params.toString()}`, {
    method: "GET",
    headers: { Accept: "application/json" },
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`Token request failed: ${response.status} ${response.statusText}`);
  }

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
    room: payload.room ?? "alijr-test",
    identity: payload.identity ?? participantName,
  };
}
