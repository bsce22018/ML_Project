"use client";
import { useEffect, useRef, useState } from "react";
import axios from "axios";

type Scores = {
  player: number;
  ai: number;
  ties: number;
};

type PredictionResponse = {
  player_move: string;
  ai_move: string;
  result: string;
  scores: Scores;
};

type Landmark = {
  x: number;
  y: number;
  z: number;
};

export default function Game() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [scores, setScores] = useState<Scores>({ player: 0, ai: 0, ties: 0 });
  const [playerMove, setPlayerMove] = useState<string>("");
  const [aiMove, setAiMove] = useState<string>("");
  const [result, setResult] = useState<string>("");



  useEffect(() => {
    const startCamera = async () => {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) videoRef.current.srcObject = stream;
    };

    startCamera();
  }, []);

  const sendFrame = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const canvas = document.createElement("canvas");
    const width = videoRef.current.videoWidth;
    const height = videoRef.current.videoHeight;
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(videoRef.current, 0, 0, width, height);

    const imageData = canvas.toDataURL("image/jpeg");

    try {
      const res = await axios.post<
        PredictionResponse & { landmarks: Landmark[] }
      >("http://localhost:5000/predict", { frame: imageData });

      const data = res.data;
      setPlayerMove(data.player_move);
      setAiMove(data.ai_move);
      setResult(data.result);
      setScores(data.scores);

      const overlayCtx = canvasRef.current.getContext("2d");
      if (overlayCtx) {
        overlayCtx.clearRect(
          0,
          0,
          canvasRef.current.width,
          canvasRef.current.height
        );

        const moveEmojiMap: Record<string, string> = {
          Rock: "✊",
          Paper: "✋",
          Scissors: "✌",
          Unknown: "❓",
        };

        const aiMoveText = `${data.ai_move} ${
          moveEmojiMap[data.ai_move] || "❓"
        }`;

        overlayCtx.font = "bold 60px Arial";
        overlayCtx.textAlign = "center";
        overlayCtx.textBaseline = "middle";

        const textWidth = overlayCtx.measureText(aiMoveText).width;
        const padding = 40;
        const boxWidth = textWidth + padding;
        const boxHeight = 100;
        const boxX = 30;
        const boxY = 30;

        overlayCtx.fillStyle = "rgba(0, 0, 0, 0.6)";
        overlayCtx.strokeStyle = "#00ffcc";
        overlayCtx.lineWidth = 4;
        overlayCtx.fillRect(boxX, boxY, boxWidth, boxHeight);
        overlayCtx.strokeRect(boxX, boxY, boxWidth, boxHeight);

        overlayCtx.fillStyle = "#00ffcc";
        overlayCtx.shadowColor = "black";
        overlayCtx.shadowBlur = 8;
        overlayCtx.fillText(
          aiMoveText,
          boxX + boxWidth / 2,
          boxY + boxHeight / 2
        );
      }
    } catch (err) {
      console.error("Prediction error:", err);
    }
  };





  const resetScores = async () => {
    await axios.post("http://localhost:5000/reset");
    setScores({ player: 0, ai: 0, ties: 0 });
    setPlayerMove("");
    setAiMove("");
    setResult("");
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      if (ctx)
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  return (
    <div className="bg-slate-900 text-white min-h-screen py-10 px-5 flex flex-col items-center font-sans">
      <div className="flex flex-wrap gap-20 justify-center">
        <div
          className="relative rounded-2xl overflow-hidden"
          style={{
            width: 640,
            height: 480,
            border: "5px solid #14b8a6",
            boxShadow: "0 0 15px #14b8a6",
          }}
        >
          <video
            ref={videoRef}
            autoPlay
            playsInline
            width={640}
            height={480}
            className="absolute"
          />
          <canvas
            ref={canvasRef}
            width={640}
            height={480}
            className="absolute"
          />
        </div>

        <div className="text-left min-w-[300px] max-w-[400px] flex flex-col justify-start">
          <h1 className="text-5xl text-teal-400 mb-4 text-left animate-fadeIn">
            Rock Paper Scissors ✊✋✌
          </h1>
          <p className="text-slate-300 mb-6 text-base">
            Raise your hand in front of the camera to play. Rock = ✊ | Paper =
            ✋ | Scissors = ✌
          </p>

          <div className="text-2xl flex mb-5 justify-between w-full max-w-md">
            <p>
              <strong>👤 You:</strong>{" "}
              <span className="text-sky-400">{playerMove || "Waiting..."}</span>
            </p>
            <p>
              <strong>🤖 AI:</strong>{" "}
              <span className="text-pink-300">{aiMove || "Waiting..."}</span>
            </p>
          </div>

          <h2
            className={`text-3xl mb-5 transition-all duration-300 ${
              result.includes("win")
                ? "text-green-400"
                : result.includes("tie")
                ? "text-slate-200"
                : "text-red-400"
            }`}
          >
            {result || "Make a move!"}
          </h2>

          <div
            className="text-lg p-5 rounded-xl border-2 border-teal-400 mb-5"
            style={{
              background: "linear-gradient(to right, #1e293b, #0f172a)",
              boxShadow: "0 0 10px rgba(20, 184, 166, 0.5)",
            }}
          >
            <p>
              👤 Player: <strong>{scores.player}</strong>
            </p>
            <p>
              🤖 AI: <strong>{scores.ai}</strong>
            </p>
            <p>
              🤝 Ties: <strong>{scores.ties}</strong>
            </p>
          </div>

          <div className="space-x-2">
            <button
              onClick={sendFrame}
              className="bg-teal-400 text-slate-900 font-bold py-3 px-6 rounded-lg text-base transition-all hover:brightness-110 hover:scale-105"
            >
              Start Game
            </button>
            <button
              onClick={resetScores}
              className="bg-red-400 text-slate-900 font-bold py-3 px-6 rounded-lg text-base transition-all hover:brightness-110 hover:scale-105"
            >
              Reset Scores
            </button>
           

          </div>
        </div>
      </div>

      <style>{`
        @keyframes fadeIn {
          0% { opacity: 0; transform: translateY(-20px); }
          100% { opacity: 1; transform: translateY(0); }
        }
        .animate-fadeIn {
          animation: fadeIn 2s ease-in-out;
        }
      `}</style>
    </div>
  );
}
