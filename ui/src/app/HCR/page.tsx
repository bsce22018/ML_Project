"use client";
import { useEffect, useRef, useState } from "react";

export default function HillClimbRace() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [gesture, setGesture] = useState<string>("Unknown");

  useEffect(() => {
    let lastSent = Date.now();
    let animationFrameId: number;

    const captureFrame = async () => {
      if (!videoRef.current || !canvasRef.current) {
        animationFrameId = requestAnimationFrame(captureFrame);
        return;
      }

      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const width = videoRef.current.videoWidth;
      const height = videoRef.current.videoHeight;

      canvas.width = width;
      canvas.height = height;
      ctx.drawImage(videoRef.current, 0, 0, width, height);

      const now = Date.now();
      if (now - lastSent > 500) {
        lastSent = now;
        const imageData = canvas.toDataURL("image/jpeg");

        try {
          const res = await fetch("http://localhost:5000/gesture", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: imageData }),
          });

          const data = await res.json();
          if (data.gesture) {
            setGesture(data.gesture);
          }
        } catch (error) {
          console.error("Error sending gesture data:", error);
        }
      }

      animationFrameId = requestAnimationFrame(captureFrame);
    };

    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        animationFrameId = requestAnimationFrame(captureFrame);
      } catch (err) {
        console.error("Error accessing camera:", err);
      }
    };

    startCamera();

    return () => cancelAnimationFrame(animationFrameId);
  }, []);

  return (
    <div className="bg-slate-900 text-white min-h-screen py-10 px-5 font-sans">
      {/* Header Section */}
      <div className="w-full text-center mb-10 animate-fadeIn">
        <h1 className="text-5xl text-pink-400 font-bold mb-3">
          Hill Climb Racing 🚗✋✊
        </h1>
        <p className="text-slate-300 text-lg">
          Use your hand gestures to control the car!
        </p>
        <ul className="text-slate-200 list-inline flex justify-center gap-10 mt-4 text-md md:text-lg">
          <li>
            ✋ <b>Open hand</b> = <span className="text-yellow-300">GAS</span>
          </li>
          <li>
            ✊ <b>Closed fist</b> ={" "}
            <span className="text-yellow-300">BRAKE</span>
          </li>
          <li>
            🤷 <b>Other</b> = <span className="text-yellow-300">Unknown</span>
          </li>
        </ul>
      </div>

      {/* Main Content: Side by Side */}
      <div className="flex flex-col md:flex-row justify-center items-center gap-10">
        {/* Camera Feed */}
        <div
          className="relative rounded-2xl overflow-hidden shadow-lg"
          style={{
            width: 640,
            height: 480,
            border: "5px solid #e68bbe",
            boxShadow: "0 0 15px #e68bbe",
          }}
        >
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            width={640}
            height={480}
            className="absolute rounded-2xl"
          />
          <canvas
            ref={canvasRef}
            width={640}
            height={480}
            className="absolute"
          />

          {/* Gesture Overlay */}
          <div className="absolute bottom-3 left-1/2 transform -translate-x-1/2 bg-black/70 px-5 py-2 rounded-full text-2xl font-semibold text-pink-300 shadow-lg">
            {gesture}
          </div>
        </div>

        {/* Game Frame */}
        <div
          className="rounded-2xl overflow-hidden shadow-lg"
          style={{
            width: 640,
            height: 480,
            border: "5px solid #7fffd4",
            boxShadow: "0 0 15px #7fffd4",
          }}
        >
          <iframe
            src="https://hillclimbrace.io/"
            width="100%"
            height="100%"
            className="rounded-2xl"
            allow="autoplay; fullscreen"
            sandbox="allow-same-origin allow-scripts allow-forms allow-popups"
          ></iframe>
        </div>
      </div>

      <style>{`
        @keyframes fadeIn {
          0% { opacity: 0; transform: translateY(-20px); }
          100% { opacity: 1; transform: translateY(0); }
        }
        .animate-fadeIn {
          animation: fadeIn 1s ease-in-out;
        }
        .list-inline > li {
          display: inline-block;
        }
      `}</style>
    </div>
  );
}
