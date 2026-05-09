"use client";

import {
  BarVisualizer,
  LiveKitRoom,
  RoomAudioRenderer,
  useTranscriptions,
  useVoiceAssistant,
} from "@livekit/components-react";
import { useMemo, useState } from "react";
import { fetchToken } from "@/lib/token";

type ConnectionState = "disconnected" | "connecting" | "connected";

function AgentView() {
  const { state, audioTrack } = useVoiceAssistant();
  const transcriptions = useTranscriptions();

  const transcriptRows = useMemo(
    () =>
      transcriptions.map((item, index) => ({
        id: `${item.participantInfo.identity}-${index}`,
        speaker: item.participantInfo.identity,
        text: item.text,
      })),
    [transcriptions],
  );

  return (
    <div className="mt-8 grid gap-6">
      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/80 p-5">
        <div className="mb-3 text-xs uppercase tracking-wider text-zinc-400">
          Agent state: <span className="font-medium text-zinc-200">{state}</span>
        </div>
        <BarVisualizer
          state={state}
          trackRef={audioTrack}
          barCount={12}
          options={{ minHeight: 8, maxHeight: 96 }}
          className="h-24"
        />
      </div>

      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/80 p-5">
        <h2 className="mb-3 text-sm font-medium text-zinc-200">Live Transcript</h2>
        <div className="max-h-80 space-y-3 overflow-y-auto pr-2">
          {transcriptRows.length === 0 ? (
            <p className="text-sm text-zinc-400">Transcript will appear here once speech starts.</p>
          ) : (
            transcriptRows.map((row) => (
              <div key={row.id} className="rounded-xl bg-zinc-950/70 p-3">
                <p className="mb-1 text-xs uppercase tracking-wide text-zinc-500">{row.speaker}</p>
                <p className="text-sm text-zinc-100">{row.text}</p>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  const [connectionState, setConnectionState] = useState<ConnectionState>("disconnected");
  const [token, setToken] = useState<string | null>(null);
  const [serverUrl, setServerUrl] = useState<string | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);

  const liveKitUrl = serverUrl ?? process.env.NEXT_PUBLIC_LIVEKIT_URL;

  const startCall = async () => {
    try {
      setError(null);
      setConnectionState("connecting");
      const tokenResp = await fetchToken("AliJR Web Client");
      setServerUrl(tokenResp.livekit_url);
      setToken(tokenResp.token);
    } catch (err) {
      setConnectionState("disconnected");
      setToken(null);
      setError(err instanceof Error ? err.message : "Unable to start call.");
    }
  };

  const endCall = () => {
    setToken(null);
    setServerUrl(undefined);
    setConnectionState("disconnected");
  };

  return (
    <main className="min-h-screen bg-zinc-950 px-6 py-10 text-zinc-100">
      <div className="mx-auto max-w-4xl">
        <div className="rounded-3xl border border-zinc-800 bg-zinc-900/70 p-8 shadow-2xl shadow-black/30">
          <h1 className="text-3xl font-semibold tracking-tight">AliJR Voice Console</h1>
          <p className="mt-2 text-sm text-zinc-400">
            LiveKit + Gemini + RAG assistant for coursework and research.
          </p>

          <div className="mt-6 flex flex-wrap items-center gap-3">
            {connectionState !== "connected" && (
              <button
                onClick={startCall}
                disabled={connectionState === "connecting" || !liveKitUrl}
                className="rounded-xl bg-emerald-500 px-4 py-2 text-sm font-medium text-black transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:bg-zinc-700 disabled:text-zinc-400"
              >
                {connectionState === "connecting" ? "Connecting..." : "Start Call"}
              </button>
            )}

            {connectionState === "connected" && (
              <button
                onClick={endCall}
                className="rounded-xl bg-rose-500 px-4 py-2 text-sm font-medium text-white transition hover:bg-rose-400"
              >
                End Call
              </button>
            )}

            <span className="rounded-full border border-zinc-700 px-3 py-1 text-xs uppercase tracking-wide text-zinc-300">
              {connectionState}
            </span>
          </div>

          {!liveKitUrl && (
            <p className="mt-4 text-sm text-amber-400">
              Missing NEXT_PUBLIC_LIVEKIT_URL in your frontend environment.
            </p>
          )}
          {error && <p className="mt-4 text-sm text-rose-400">{error}</p>}

          {token && (
            <LiveKitRoom
              serverUrl={liveKitUrl}
              token={token}
              audio={true}
              video={false}
              connect={true}
              onConnected={() => setConnectionState("connected")}
              onDisconnected={() => {
                setConnectionState("disconnected");
                setToken(null);
                setServerUrl(undefined);
                setError("Disconnected from LiveKit room.");
              }}
              onError={(err) => {
                setConnectionState("disconnected");
                setToken(null);
                setServerUrl(undefined);
                setError(err.message || "LiveKit connection failed.");
              }}
              className="mt-6"
            >
              <RoomAudioRenderer />
              <AgentView />
            </LiveKitRoom>
          )}
        </div>
      </div>
    </main>
  );
}
