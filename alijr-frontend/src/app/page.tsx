"use client";

import {
  BarVisualizer,
  LiveKitRoom,
  RoomAudioRenderer,
  useTranscriptions,
  useVoiceAssistant,
} from "@livekit/components-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { DevTraceFeed } from "@/components/DevTraceFeed";
import { fetchToken } from "@/lib/token";

type ConnectionState = "disconnected" | "connecting" | "connected";

function AgentView({ devTraceEnabled }: { devTraceEnabled: boolean }) {
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

      <DevTraceFeed enabled={devTraceEnabled} />

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
  const [devTraceEnabled, setDevTraceEnabled] = useState(false);
  const [livekitRoomId, setLivekitRoomId] = useState<string | null>(null);

  const callAbortRef = useRef<AbortController | null>(null);
  /** True when user clicked End / Cancel — avoid treating LiveKit disconnect as an error. */
  const intentionalLeaveRef = useRef(false);

  /** Prefer URL returned by the token API; optional NEXT_PUBLIC_LIVEKIT_URL override when set at build time. */
  const liveKitUrl =
    serverUrl ??
    (typeof process !== "undefined" ? process.env.NEXT_PUBLIC_LIVEKIT_URL?.trim() : undefined) ??
    "";

  const resetSession = () => {
    intentionalLeaveRef.current = true;
    callAbortRef.current?.abort();
    callAbortRef.current = null;
    setToken(null);
    setServerUrl(undefined);
    setLivekitRoomId(null);
    setConnectionState("disconnected");
    setError(null);
  };

  const endCall = () => {
    resetSession();
  };

  useEffect(() => {
    return () => {
      callAbortRef.current?.abort();
    };
  }, []);

  const startCall = async () => {
    callAbortRef.current?.abort();
    const ac = new AbortController();
    callAbortRef.current = ac;
    intentionalLeaveRef.current = false;

    try {
      setError(null);
      setConnectionState("connecting");
      setToken(null);
      setServerUrl(undefined);
      setLivekitRoomId(null);

      const tokenResp = await fetchToken("AliJR Web Client", {
        devTrace: devTraceEnabled,
        signal: ac.signal,
      });

      if (ac.signal.aborted) {
        return;
      }

      setServerUrl(tokenResp.livekit_url);
      setLivekitRoomId(tokenResp.room);
      setToken(tokenResp.token);
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        setConnectionState("disconnected");
        return;
      }
      setConnectionState("disconnected");
      setToken(null);
      setServerUrl(undefined);
      setLivekitRoomId(null);
      setError(err instanceof Error ? err.message : "Unable to start call.");
    } finally {
      if (callAbortRef.current === ac) {
        callAbortRef.current = null;
      }
    }
  };

  const inLiveSession = Boolean(token && livekitRoomId && liveKitUrl);

  return (
    <main className="min-h-screen bg-zinc-950 px-6 py-10 text-zinc-100">
      <div className="mx-auto max-w-4xl">
        <div className="rounded-3xl border border-zinc-800 bg-zinc-900/70 p-8 shadow-2xl shadow-black/30">
          <h1 className="text-3xl font-semibold tracking-tight">AliJR Assistant</h1>
          <p className="mt-2 text-sm text-zinc-400">Virtual Avatar of Ali Nawaf</p>

          <div className="mt-6 flex flex-wrap items-center gap-3">
            {connectionState !== "connected" && (
              <label className="flex cursor-pointer items-center gap-2 rounded-xl border border-zinc-700 bg-zinc-950/60 px-3 py-2 text-xs text-zinc-300 transition hover:border-zinc-600">
                <input
                  type="checkbox"
                  checked={devTraceEnabled}
                  onChange={(e) => setDevTraceEnabled(e.target.checked)}
                  disabled={connectionState === "connecting"}
                  className="accent-amber-500"
                />
                <span>
                  Developer trace{" "}
                  <span className="text-zinc-500">(RAG / filters → UI + worker logs)</span>
                </span>
              </label>
            )}

            {connectionState !== "connected" && (
              <button
                type="button"
                onClick={startCall}
                disabled={connectionState === "connecting"}
                className="rounded-xl bg-emerald-500 px-4 py-2 text-sm font-medium text-black transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:bg-zinc-700 disabled:text-zinc-400"
              >
                {connectionState === "connecting" ? "Connecting..." : "Start Call"}
              </button>
            )}

            {(connectionState === "connecting" || connectionState === "connected") && (
              <button
                type="button"
                onClick={endCall}
                className="rounded-xl bg-rose-500 px-4 py-2 text-sm font-medium text-white transition hover:bg-rose-400"
              >
                {connectionState === "connecting" ? "Cancel" : "End call"}
              </button>
            )}

            <span className="rounded-full border border-zinc-700 px-3 py-1 text-xs uppercase tracking-wide text-zinc-300">
              {connectionState}
            </span>
          </div>

          <p className="mt-4 text-xs leading-relaxed text-zinc-500">
            Backend:{" "}
            <code className="rounded bg-zinc-950/80 px-1.5 py-0.5 text-zinc-400">
              {typeof process !== "undefined"
                ? process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "") || "http://localhost:8000"
                : "http://localhost:8000"}
            </code>
            . Each session uses its own LiveKit room. Set{" "}
            <code className="rounded bg-zinc-950/80 px-1 py-0.5 text-zinc-400">NEXT_PUBLIC_BACKEND_URL</code> on
            Vercel. Optional{" "}
            <code className="rounded bg-zinc-950/80 px-1 py-0.5 text-zinc-400">NEXT_PUBLIC_LIVEKIT_URL</code>{" "}
            override.
          </p>
          {error && <p className="mt-4 text-sm text-rose-400">{error}</p>}

          {inLiveSession && (
            <LiveKitRoom
              key={`${livekitRoomId}-${token!.slice(0, 16)}`}
              serverUrl={liveKitUrl}
              token={token!}
              audio={true}
              video={false}
              connect={true}
              options={{
                disconnectOnPageLeave: true,
              }}
              onConnected={() => {
                intentionalLeaveRef.current = false;
                setConnectionState("connected");
              }}
              onDisconnected={() => {
                setConnectionState("disconnected");
                setToken(null);
                setServerUrl(undefined);
                setLivekitRoomId(null);
                if (!intentionalLeaveRef.current) {
                  setError("Disconnected from LiveKit room.");
                } else {
                  setError(null);
                }
                intentionalLeaveRef.current = false;
              }}
              onError={(err) => {
                setConnectionState("disconnected");
                setToken(null);
                setServerUrl(undefined);
                setLivekitRoomId(null);
                intentionalLeaveRef.current = false;
                setError(err.message || "LiveKit connection failed.");
              }}
              className="mt-6"
            >
              <RoomAudioRenderer />
              <AgentView devTraceEnabled={devTraceEnabled} />
            </LiveKitRoom>
          )}
        </div>
      </div>
    </main>
  );
}
