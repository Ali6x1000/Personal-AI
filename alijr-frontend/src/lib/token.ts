export type TokenResponse = {
  token: string;
  livekit_url: string;
  room: string;
  identity: string;
};

export type FetchTokenOptions = {
  devTrace?: boolean;
};

export async function fetchToken(
  participantName: string,
  options: FetchTokenOptions = {},
): Promise<TokenResponse> {
  const normalizedIdentity = `web-${participantName
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "")}-${Date.now()}`;

  const base =
    typeof process !== "undefined" && process.env.NEXT_PUBLIC_TOKEN_API_URL
      ? process.env.NEXT_PUBLIC_TOKEN_API_URL.replace(/\/$/, "")
      : "http://localhost:8000";

  const response = await fetch(`${base}/token`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({
      room_name: "alijr-test",
      participant_identity: normalizedIdentity,
      participant_name: participantName,
      dev_trace: Boolean(options.devTrace),
    }),
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`Token request failed: ${response.status} ${response.statusText}`);
  }

  const payload = (await response.json()) as Partial<TokenResponse>;
  const token = payload.token?.trim() ?? "";
  const livekitUrl = payload.livekit_url?.trim() ?? "";
  if (!token || !livekitUrl) {
    throw new Error("Token endpoint returned an empty token.");
  }

  return {
    token,
    livekit_url: livekitUrl,
    room: payload.room ?? "alijr-test",
    identity: payload.identity ?? participantName,
  };
}
