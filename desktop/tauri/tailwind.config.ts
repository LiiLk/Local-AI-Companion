import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        shell: {
          void: "#02030A",
          panel: "rgba(8, 14, 24, 0.78)",
          stroke: "rgba(160, 222, 255, 0.16)",
          cyan: "#79E7FF",
          magenta: "#FF6FC9",
          muted: "#8AA7C2",
          text: "#F3F7FF",
          success: "#91F4B4",
          warning: "#FFC982",
          error: "#FF8AA5"
        }
      },
      boxShadow: {
        halo: "0 0 0 1px rgba(151,227,255,0.08), 0 30px 120px rgba(70,166,255,0.18)",
        glass: "0 24px 80px rgba(0, 8, 24, 0.45)"
      },
      backgroundImage: {
        "shell-mesh":
          "radial-gradient(circle at 82% 24%, rgba(81, 193, 255, 0.2), transparent 32%), radial-gradient(circle at 22% 18%, rgba(255, 112, 203, 0.12), transparent 30%), linear-gradient(180deg, rgba(3, 8, 18, 0.18), rgba(3, 8, 18, 0.04))",
      },
      backdropBlur: {
        xs: "2px"
      },
      keyframes: {
        pulseHalo: {
          "0%, 100%": { opacity: "0.7", transform: "scale(1)" },
          "50%": { opacity: "1", transform: "scale(1.03)" }
        }
      },
      animation: {
        "pulse-halo": "pulseHalo 3.6s ease-in-out infinite"
      }
    },
  },
  plugins: [],
} satisfies Config;
