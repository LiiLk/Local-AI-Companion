export {};

declare global {
  interface Window {
    Live2DManager?: {
      init(config: {
        canvasId: string;
        modelPath: string;
        modelName: string;
        defaultExpression?: string;
        scale: number;
        position: { x: number; y: number };
        debug?: boolean;
      }): Promise<void>;
      setExpression(name: string): void;
      setLipSync(value: number): void;
      setScale(scale: number): void;
      setPosition(x: number, y: number): void;
      playMotion(group: string, index?: number): void;
      connectAudioElement?(audio: HTMLAudioElement): void;
      stopLipSync?(): void;
      toggleDebug?(): boolean;
      isInitialized?(): boolean;
    };
  }
}
