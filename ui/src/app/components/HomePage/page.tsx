"use client";
import { useRouter } from "next/navigation";

export default function Home() {
  const router = useRouter();

  return (
    <div className="min-h-screen bg-slate-900 text-white px-6 py-12 font-sans flex flex-col items-center">
      <h1 className="text-5xl font-bold text-cyan-400 mb-4 animate-fadeIn">
        AI Game Zone 🎮
      </h1>
      <p className="text-lg text-slate-300 mb-12 max-w-xl text-center">
        Welcome to the AI-powered gaming experience. Choose a game below to get
        started!
      </p>

      <div className="grid gap-10 sm:grid-cols-1 md:grid-cols-2">
        <div
          onClick={() => router.push("/RPS")}
          className="group bg-gradient-to-br from-slate-800 to-slate-900 border border-teal-400 rounded-2xl px-6 py-10 cursor-pointer transition-all duration-300 hover:scale-105 shadow-[0_0_20px_rgba(20,184,166,0.4)] w-full md:w-80 min-h-[200px]"
        >
          <h2 className="text-3xl flex justify-center font-semibold text-teal-400 mb-2 group-hover:text-teal-300 transition">
            ✊✋✌
          </h2>
          <h2 className="text-3xl text-center font-semibold text-teal-400 mb-2 group-hover:text-teal-300 transition">
            Rock Paper Scissors
          </h2>
          <p className="text-slate-400 text-center group-hover:text-white transition">
            Use your hand gestures to play against an AI in real-time!
          </p>
        </div>

        <div
          onClick={() => router.push("/HCR")}
          className="group bg-gradient-to-br from-slate-800 to-slate-900 border border-pink-500 rounded-2xl px-6 py-10 cursor-pointer transition-all duration-300 hover:scale-105 shadow-[0_0_20px_rgba(236,72,153,0.3)] w-full md:w-80 min-h-[200px]"
        >
          <h2 className="text-4xl flex justify-center font-semibold text-pink-300 mb-2 group-hover:text-pink-400 transition">
            🚗
          </h2>
          <h2 className="text-3xl text-center font-semibold text-pink-300 mb-2 group-hover:text-pink-400 transition">
            Hill Climbing Game
          </h2>
          <p className="text-slate-400 text-center group-hover:text-white transition">
            Control a car on a bumpy hill using your hand gestures.
          </p>
        </div>
      </div>

      <style jsx>{`
        .animate-fadeIn {
          animation: fadeIn 1.5s ease-in-out;
        }

        @keyframes fadeIn {
          0% {
            opacity: 0;
            transform: translateY(-20px);
          }
          100% {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  );
}
