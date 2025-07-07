import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import WoodJudgePose from "../assets/woodjudge-pose.png";

export default function QuestionInputPage() {
  const [query, setQuery] = useState("");
  const navigate = useNavigate();

  const handleSubmit = () => {
    if (!query.trim()) return;

    // 채팅 페이지로 이동하면서 초기 질문 전달
    navigate("/chat", { state: { initialQuestion: query } });
  };

  return (
    <div className="flex flex-col items-center justify-center h-full space-y-6">
      <img
        src={WoodJudgePose}
        alt="우드저지"
        className="w-40 h-40 object-contain"
      />
      <h2 className="text-2xl font-bold text-center text-gray-800">
        당신의 작고 소중한 법률 요정 우드저지에요.
        <br />
        어떤 사건을 도와드릴까요~?
      </h2>
      <div className="w-full max-w-xl bg-gray-100 rounded-2xl shadow px-4 py-3 flex items-center">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          placeholder="무엇이든 물어보세요"
          className="flex-grow bg-transparent text-gray-800 placeholder-gray-500 focus:outline-none"
        />
        <button
          onClick={handleSubmit}
          className="text-green-600 hover:text-green-800 px-2"
        >
          📨
        </button>
      </div>
    </div>
  );
}
