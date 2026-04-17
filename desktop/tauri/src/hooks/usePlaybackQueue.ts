import { RefObject, useEffect, useRef } from "react";
import { listen } from "@tauri-apps/api/event";
import { AvatarStageHandle } from "../components/avatar/AvatarStage";

interface AudioPayload {
  audio?: string;
  text?: string;
  turn_id?: number | null;
}

export function usePlaybackQueue(avatarRef: RefObject<AvatarStageHandle | null>) {
  const queueRef = useRef<AudioPayload[]>([]);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);
  const busyRef = useRef(false);

  useEffect(() => {
    const playNext = () => {
      if (busyRef.current) {
        return;
      }

      const item = queueRef.current.shift();
      if (!item?.audio) {
        return;
      }

      busyRef.current = true;
      const audio = new Audio(`data:audio/wav;base64,${item.audio}`);
      currentAudioRef.current = audio;
      avatarRef.current?.connectAudioElement(audio);
      audio.play().catch((error) => {
        console.error("audio playback failed", error);
      });
      audio.addEventListener(
        "ended",
        () => {
          busyRef.current = false;
          currentAudioRef.current = null;
          avatarRef.current?.stopLipSync();
          playNext();
        },
        { once: true },
      );
    };

    let unlistenAudio: (() => void) | null = null;
    let unlistenStop: (() => void) | null = null;

    void listen<AudioPayload>("host://audio-ready", (event) => {
      queueRef.current.push(event.payload);
      playNext();
    }).then((dispose) => {
      unlistenAudio = dispose;
    });

    void listen("host://audio-stop", () => {
      queueRef.current = [];
      currentAudioRef.current?.pause();
      currentAudioRef.current = null;
      busyRef.current = false;
      avatarRef.current?.stopLipSync();
    }).then((dispose) => {
      unlistenStop = dispose;
    });

    return () => {
      unlistenAudio?.();
      unlistenStop?.();
      currentAudioRef.current?.pause();
    };
  }, [avatarRef]);
}
