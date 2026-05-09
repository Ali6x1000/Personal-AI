"use client";

import { useRoomContext } from "@livekit/components-react";
import { ConnectionState, RoomEvent } from "livekit-client";
import { useCallback, useEffect, useState } from "react";

/** Must match ``dev_trace.ALIJR_DEV_TRACE_TOPIC`` in the Python agent. */
export const ALIJR_DEV_TRACE_TOPIC = "alijr-dev-trace";

/** Must match ``dev_trace.ALIJR_DEV_TRACE_CONTROL_TOPIC`` — tells the agent to enable tracing. */
export const ALIJR_DEV_TRACE_CONTROL_TOPIC = "alijr-dev-trace-control";

type TraceLine = { id: string; text: string };

export function DevTraceFeed({ enabled }: { enabled: boolean }) {
  const room = useRoomContext();
  const [open, setOpen] = useState(true);
  const [lines, setLines] = useState<TraceLine[]>([]);

  const appendLine = useCallback((text: string) => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    setLines((prev) => [...prev.slice(-150), { id, text }]);
  }, []);

  useEffect(() => {
    if (!enabled || !room) {
      return undefined;
    }

    const controlPayload = new TextEncoder().encode(
      JSON.stringify({ v: 1, alijr_dev_trace: true }),
    );

    const publishControl = () => {
      try {
        room.localParticipant.publishData(controlPayload, {
          reliable: true,
          topic: ALIJR_DEV_TRACE_CONTROL_TOPIC,
        });
      } catch {
        /* room may still be handshaking */
      }
    };

    if (room.state === ConnectionState.Connected) {
      publishControl();
      const t = window.setTimeout(publishControl, 400);
      const t2 = window.setTimeout(publishControl, 1200);
      return () => {
        window.clearTimeout(t);
        window.clearTimeout(t2);
      };
    }

    const onConnected = () => {
      publishControl();
      window.setTimeout(publishControl, 400);
      window.setTimeout(publishControl, 1200);
    };
    room.on(RoomEvent.Connected, onConnected);
    return () => {
      room.off(RoomEvent.Connected, onConnected);
    };
  }, [enabled, room]);

  useEffect(() => {
    if (!enabled || !room) {
      return undefined;
    }

    const onData = (
      payload: Uint8Array,
      _participant?: unknown,
      _kind?: unknown,
      topic?: string,
    ) => {
      if (topic !== ALIJR_DEV_TRACE_TOPIC) {
        return;
      }
      const raw = new TextDecoder().decode(payload);
      let pretty = raw;
      try {
        const j = JSON.parse(raw) as {
          type?: string;
          title?: string;
          rows?: unknown;
          ts?: number;
        };
        if (j.type === "panel" && j.title != null) {
          const head = j.ts != null ? `[${new Date(j.ts * 1000).toISOString()}] ` : "";
          pretty = `${head}[panel] ${j.title}\n${JSON.stringify(j.rows, null, 2)}`;
        } else {
          pretty = JSON.stringify(j, null, 2);
        }
      } catch {
        /* keep raw text */
      }
      appendLine(pretty);
    };

    room.on("dataReceived", onData);
    return () => {
      room.off("dataReceived", onData);
    };
  }, [appendLine, enabled, room]);

  if (!enabled) {
    return null;
  }

  return (
    <div className="mt-6 rounded-2xl border border-amber-800/40 bg-amber-950/25 p-4 shadow-inner shadow-black/20">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between text-left"
      >
        <span className="text-xs font-semibold uppercase tracking-wider text-amber-200/90">
          Developer trace
        </span>
        <span className="text-xs text-amber-400/80">{open ? "Hide" : "Show"}</span>
      </button>
      <p className="mt-1 text-[11px] leading-relaxed text-amber-100/50">
        Mirrors agent RAG / filter panels over LiveKit data (topic{" "}
        <code className="rounded bg-black/30 px-1 text-amber-200/90">{ALIJR_DEV_TRACE_TOPIC}</code>
        ). Optional worker env <code className="rounded bg-black/30 px-1">ALIJR_DEV_MODE=1</code> adds
        stderr logging for every session; leave this checkbox off for production.
      </p>
      {open && (
        <pre className="mt-3 max-h-96 overflow-auto rounded-xl border border-amber-900/30 bg-black/40 p-3 font-mono text-[11px] leading-snug text-amber-50/90">
          {lines.length === 0 ? (
            <span className="text-amber-200/40">
              Waiting for trace packets from the agent…
            </span>
          ) : (
            lines.map((line) => (
              <div key={line.id} className="mb-4 whitespace-pre-wrap border-b border-amber-900/20 pb-3 last:mb-0 last:border-b-0">
                {line.text}
              </div>
            ))
          )}
        </pre>
      )}
    </div>
  );
}
