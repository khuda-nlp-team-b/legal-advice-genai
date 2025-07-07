import React, { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import woodjudge from "../assets/woodjudge.png";

export default function AnswerResultPage() {
  const { state } = useLocation();
  const navigate = useNavigate();

  const [query, setQuery] = useState<string>("");
  const [result, setResult] = useState<any>(null);
  const [activeTab, setActiveTab] = useState<"strategy" | "cases">("strategy");
  const [isBookmarked, setIsBookmarked] = useState(false);
  const [feedback, setFeedback] = useState<"up" | "down" | null>(null);

  useEffect(() => {
    if (state?.query && state?.result) {
      setQuery(state.query);
      setResult(state.result);
    } else {
      const stored = localStorage.getItem("lawgpt_history");
      if (stored) {
        const latest = JSON.parse(stored)[0];
        if (latest) {
          setQuery(latest.title);
          setResult({
            verdict_estimate: latest.result,
            recommendations: latest.recommendations || [],
            referenced_cases: latest.cases || [],
          });
        }
      }
    }
  }, [state]);

  const handleBookmark = () => {
    const saved = JSON.parse(localStorage.getItem("lawgpt_bookmarks") || "[]");
    if (isBookmarked) {
      const updated = saved.filter((item: any) => item.title !== query);
      localStorage.setItem("lawgpt_bookmarks", JSON.stringify(updated));
    } else {
      saved.unshift({ id: Date.now().toString(), title: query, ...result });
      localStorage.setItem("lawgpt_bookmarks", JSON.stringify(saved));
    }
    setIsBookmarked(!isBookmarked);
  };

  const copySummary = () => {
    navigator.clipboard.writeText(`${query}\n\n판단 요약: ${result?.verdict_estimate}`);
  };

  if (!result) return <p className="text-center">결과를 불러오는 중입니다...</p>;

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      {/* 요약 박스 */}
      <div className="bg-white border-l-4 border-blue-500 shadow p-4 rounded-xl flex gap-4 items-start animate-fade-in">
        <img src={woodjudge} alt="우드저지" className="w-12 h-12" />
        <div>
          <p className="text-sm text-gray-500 font-medium">📌 사건 요약</p>
          <p className="text-lg font-semibold text-blue-900 mt-1">
            {result?.verdict_estimate}
          </p>
          <p className="text-xs text-gray-400 mt-1">
            ※ 아래 대응 전략 및 유사 판례를 참고하세요
          </p>
          <div className="flex gap-2 mt-2 text-sm">
            <button onClick={handleBookmark} className="text-yellow-600 hover:underline">
              {isBookmarked ? "★ 저장됨" : "☆ 저장하기"}
            </button>
            <button onClick={copySummary} className="text-blue-600 hover:underline">
              📋 복사하기
            </button>
          </div>
        </div>
      </div>

      {/* 질문/답변 말풍선 */}
      <div className="flex flex-col space-y-3 animate-slide-up">
        <div className="self-start bg-gray-100 px-4 py-3 rounded-xl shadow text-sm max-w-[80%]">
          🙋‍♂️ <b>질문:</b> {query}
        </div>
        <div className="self-end bg-green-100 text-green-800 px-4 py-3 rounded-xl shadow text-sm max-w-[80%]">
          ✅ <b>판단:</b> {result?.verdict_estimate}
        </div>
        <div className="self-start flex items-start gap-2">
          <img src={woodjudge} alt="woodjudge" className="w-10 h-10" />
          <div className="bg-yellow-50 border border-yellow-200 px-4 py-3 rounded-xl shadow text-sm max-w-[80%]">
            안녕하세요~ 우드저지에요! 아래에 법률 조언을 정리해봤어요 🍀
          </div>
        </div>
      </div>

      {/* 탭 선택 */}
      <div className="flex gap-4 border-b mt-6">
        <button
          className={`pb-2 text-sm font-medium ${activeTab === "strategy" ? "text-blue-600 border-b-2 border-blue-600" : "text-gray-400"}`}
          onClick={() => setActiveTab("strategy")}
        >
          🧠 대응 전략
        </button>
        <button
          className={`pb-2 text-sm font-medium ${activeTab === "cases" ? "text-blue-600 border-b-2 border-blue-600" : "text-gray-400"}`}
          onClick={() => setActiveTab("cases")}
        >
          📚 유사 판례
        </button>
      </div>

      {/* 전략 / 판례 탭 콘텐츠 */}
      {activeTab === "strategy" && (
        <div className="bg-pink-50 border border-pink-200 p-4 rounded-xl shadow animate-fade-in">
          <ul className="list-disc pl-5 text-sm text-gray-800 space-y-1">
            {result?.recommendations?.map((r: string, i: number) => (
              <li key={i}>{r}</li>
            ))}
          </ul>

          {/* 피드백 */}
          <div className="flex gap-3 mt-4 text-sm items-center">
            <span className="text-gray-600">이 조언이 도움이 되었나요?</span>
            <button
              className={`text-xl ${feedback === "up" ? "text-green-600" : "text-gray-400"}`}
              onClick={() => setFeedback("up")}
            >👍</button>
            <button
              className={`text-xl ${feedback === "down" ? "text-red-600" : "text-gray-400"}`}
              onClick={() => setFeedback("down")}
            >👎</button>
          </div>
        </div>
      )}

      {activeTab === "cases" && (
        <div className="space-y-3 animate-fade-in">
          {result?.referenced_cases?.map((c: any) => (
            <div
              key={c.case_id}
              className="bg-amber-50 border border-amber-200 p-4 rounded-xl hover:bg-amber-100 transition"
            >
              <p className="font-semibold text-amber-900">📚 [{c.case_id}] {c.court}</p>
              <p className="text-sm text-gray-700 mt-1">↳ {c.summary}</p>
              <button
                onClick={() => navigate(`/case/${c.case_id}`)}
                className="text-blue-600 text-xs mt-2 underline"
              >
                🔍 판례 상세 보기
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
