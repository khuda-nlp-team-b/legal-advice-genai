import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import WoodJudgePose from "../assets/woodjudge-pose.png";

export default function QuestionInputPage() {
  const [query, setQuery] = useState("");
  const navigate = useNavigate();

  const handleSubmit = () => {
    if (!query.trim()) return;

    // Mock 결과
    const result = {
      summary: "택배기사의 과실로 인한 물건 파손 사건과 유사",
      verdict_estimate: "손해배상 30만 원 인정 가능성",
      recommendations: [
        "파손 물건 사진, 영상 확보",
        "택배사에 손해배상 청구",
        "필요 시 소비자원 또는 법원에 청구",
      ],
      referenced_cases: [
        {
          case_id: "2021나123456",
          title: "손해배상청구소송",
          summary: "택배기사의 부주의로 물건이 파손되어 손해배상 책임 인정",
          similarity: 0.89,
          court: "서울고등법원",
          date: "2021-03-15",
          reference_rules: ["민법 제750조", "민법 제756조"],
          precedent: "대법원 2010다54085",
          recommendations: [
            "물건 파손 시 손해배상 책임 가능",
            "과실 입증 시 피해자 중심 판결 경향",
          ],
        },
      ],
    };

    const today = new Date().toISOString().split("T")[0];
    const historyEntry = {
      id: Date.now().toString(),
      date: today,
      title: query,
      result: result.verdict_estimate,
      recommendations: result.recommendations,
      cases: result.referenced_cases,
    };

    const prev = localStorage.getItem("lawgpt_history");
    const history = prev ? JSON.parse(prev) : [];
    localStorage.setItem("lawgpt_history", JSON.stringify([historyEntry, ...history]));

    navigate("/result", { state: { query, result } });
  };

  return (
    <div className="flex flex-col items-center justify-center h-full space-y-6">
      <img
        src={WoodJudgePose}
        alt="우드저지"
        className="w-40 h-40 object-contain"
      />
      <h2 className="text-2xl font-bold text-center text-gray-800">
        당신의 작고 소중한 법률 요정 우드저지에요.<br />어떤 사건을 도와드릴까요~?
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
