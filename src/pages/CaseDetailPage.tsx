import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";

export default function CaseDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState<any | null>(null);

  useEffect(() => {
    const history = JSON.parse(localStorage.getItem("lawgpt_history") || "[]");
    for (const entry of history) {
      const matched = entry.cases?.find((c: any) => c.case_id === id);
      if (matched) {
        setData(matched);
        return;
      }
    }

    const bookmarks = JSON.parse(localStorage.getItem("lawgpt_bookmarks") || "[]");
    for (const entry of bookmarks) {
      if (entry.case_id === id) {
        setData(entry);
        return;
      }

      const matched = entry.referenced_cases?.find((c: any) => c.case_id === id);
      if (matched) {
        setData(matched);
        return;
      }
    }
  }, [id]);

  if (!data) return <p className="text-center text-gray-500">📄 판례를 불러오는 중입니다...</p>;

  return (
    <div className="p-6 max-w-2xl mx-auto space-y-6 animate-fade-in">
      <button
        onClick={() => navigate(-1)}
        className="text-blue-600 text-sm hover:underline"
      >
        🔙 돌아가기
      </button>

      <div className="bg-white rounded-xl shadow p-4 space-y-2">
        <h2 className="text-xl font-bold">📄 {data.case_id} 판결문</h2>
        <p><b>🧾 사건명:</b> {data.title || "제목 없음"}</p>
        <p><b>🏛 재판부:</b> {data.court || "정보 없음"}</p>
        <p><b>📅 판결일:</b> {data.date || "날짜 미상"}</p>
      </div>

      <div className="bg-yellow-50 border border-yellow-100 rounded-xl shadow p-4 space-y-2">
        <h3 className="text-lg font-semibold">📑 판결 요지</h3>
        {data.summary ? (
          <ul className="list-disc pl-5">
            {Array.isArray(data.summary)
              ? data.summary.map((s: string, i: number) => <li key={i}>{s}</li>)
              : <li>{data.summary}</li>}
          </ul>
        ) : (
          <p className="text-sm text-gray-500">요약 정보가 없습니다.</p>
        )}
        <p className="pt-2">📎 관련 법조항: {data.reference_rules?.join(", ") || "없음"}</p>
        <p>📎 관련 판례: {data.precedent || "없음"}</p>
      </div>
    </div>
  );
}
