import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

interface HistoryItem {
  id: string;
  date: string;
  title: string;
  result: string;
  recommendations?: string[];
  cases?: any[];
}

export default function HistoryPage() {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [filterDate, setFilterDate] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    const stored = localStorage.getItem("lawgpt_history");
    if (stored) {
      const parsed = JSON.parse(stored) as HistoryItem[];
      const sorted = parsed.sort((a, b) => b.date.localeCompare(a.date));
      setHistory(sorted);
    }
  }, []);

  const handleDelete = (id: string) => {
    const updated = history.filter((item) => item.id !== id);
    setHistory(updated);
    localStorage.setItem("lawgpt_history", JSON.stringify(updated));
  };

  const filtered = filterDate
    ? history.filter((item) => item.date === filterDate)
    : history;

  return (
    <div className="max-w-3xl mx-auto p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold flex items-center gap-2">🧾 질문 기록</h2>
        <input
          type="date"
          value={filterDate}
          onChange={(e) => setFilterDate(e.target.value)}
          className="text-sm border px-2 py-1 rounded"
        />
      </div>

      <div className="space-y-4">
        {filtered.map((item) => (
          <div
            key={item.id}
            className="bg-white rounded-xl shadow p-4 animate-fade-in cursor-pointer hover:bg-gray-50"
            onClick={() =>
              navigate("/result", {
                state: {
                  query: item.title,
                  result: {
                    verdict_estimate: item.result,
                    recommendations: item.recommendations || [],
                    referenced_cases: item.cases || [],
                  },
                },
              })
            }
          >
            <div className="flex justify-between items-center">
              <div className="space-y-1">
                <p className="text-sm text-gray-500">📅 {item.date}</p>
                <div className="bg-gray-100 px-3 py-2 rounded-xl text-sm">
                  🙋‍♂️ {item.title}
                </div>
                <div className="bg-green-100 text-green-800 px-3 py-2 rounded-xl text-sm">
                  ✅ {item.result}
                </div>
              </div>
              <button
                className="text-red-500 text-sm hover:underline"
                onClick={(e) => {
                  e.stopPropagation();
                  handleDelete(item.id);
                }}
              >
                삭제
              </button>
            </div>
          </div>
        ))}
        {filtered.length === 0 && (
          <p className="text-center text-gray-500 text-sm">기록이 없습니다.</p>
        )}
      </div>
    </div>
  );
}
