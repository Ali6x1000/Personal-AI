export type TokenResponse = {
  token: string;
  livekit_url: string;
  room: string;
  identity: string;
};

export async function fetchToken(participantName: string): Promise<TokenResponse> {
  const normalizedIdentity = `web-${participantName
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "")}-${Date.now()}`;

  const response = await fetch("http://localhost:8000/token", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({
      room_name: "alijr-test",
      participant_identity: normalizedIdentity,
      participant_name: participantName,
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
