import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import woodjudge from "../assets/woodjudge.png";

export default function BookmarkPage() {
  const [bookmarks, setBookmarks] = useState<any[]>([]);
  const navigate = useNavigate();

  useEffect(() => {
    const stored = localStorage.getItem("lawgpt_bookmarks");
    if (stored) {
      setBookmarks(JSON.parse(stored));
    }
  }, []);

  const handleDelete = (id: string) => {
    const updated = bookmarks.filter((item) => item.id !== id);
    setBookmarks(updated);
    localStorage.setItem("lawgpt_bookmarks", JSON.stringify(updated));
  };

  return (
    <div className="max-w-3xl mx-auto p-6 space-y-6">
      <h2 className="text-2xl font-bold flex items-center gap-2">
        ⭐ 저장한 기록
      </h2>

      {bookmarks.length === 0 ? (
        <p className="text-sm text-gray-500">아직 저장한 질문이 없습니다.</p>
      ) : (
        bookmarks.map((item) => (
          <div
            key={item.id}
            className="bg-white border-l-4 border-yellow-400 p-4 rounded-xl shadow hover:shadow-md transition cursor-pointer animate-fade-in"
            onClick={() =>
              navigate("/result", {
                state: {
                  query: item.title,
                  result: {
                    verdict_estimate: item.verdict_estimate,
                    recommendations: item.recommendations || [],
                    referenced_cases: item.referenced_cases || [],
                  },
                },
              })
            }
          >
            <div className="flex justify-between items-start">
              <div>
                <p className="text-sm text-gray-500">📅 저장됨</p>
                <p className="text-md font-semibold mt-1">🙋 {item.title}</p>
                <p className="text-green-700 text-sm mt-1">
                  ✅ {item.verdict_estimate}
                </p>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleDelete(item.id);
                }}
                className="text-sm text-red-500 hover:underline"
              >
                삭제
              </button>
            </div>
          </div>
        ))
      )}
    </div>
  );
}
