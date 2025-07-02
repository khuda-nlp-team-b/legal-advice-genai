import React, { useState } from "react";

export default function SettingsModal({ onClose }: { onClose: () => void }) {
  const [responseLength, setResponseLength] = useState("중간");
  const [language, setLanguage] = useState("한국어");
  const [courtFilter, setCourtFilter] = useState("전체");

  const handleSave = () => {
    // TODO: 실제 저장 로직 (전역 상태 또는 localStorage)
    console.log({
      responseLength,
      language,
      courtFilter,
    });
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-40 backdrop-blur-sm flex items-center justify-center z-50 animate-fade-in">
      <div className="bg-white p-6 rounded-xl w-full max-w-md space-y-5 shadow-lg">
        <div className="flex justify-between items-center">
          <h2 className="text-lg font-bold">⚙ 설정</h2>
          <button onClick={onClose} className="text-gray-500 hover:text-black">✖</button>
        </div>

        <div className="space-y-4 text-sm">
          <div>
            <label className="block mb-1 font-medium">응답 길이</label>
            <select
              value={responseLength}
              onChange={(e) => setResponseLength(e.target.value)}
              className="w-full border px-3 py-2 rounded"
            >
              <option>짧게</option>
              <option>중간</option>
              <option>길게</option>
            </select>
          </div>

          <div>
            <label className="block mb-1 font-medium">언어</label>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="w-full border px-3 py-2 rounded"
            >
              <option>한국어</option>
              <option>영어</option>
            </select>
          </div>

          <div>
            <label className="block mb-1 font-medium">법원 필터</label>
            <select
              value={courtFilter}
              onChange={(e) => setCourtFilter(e.target.value)}
              className="w-full border px-3 py-2 rounded"
            >
              <option>전체</option>
              <option>대법원</option>
              <option>고등법원</option>
              <option>지방법원</option>
            </select>
          </div>
        </div>

        <button
          onClick={handleSave}
          className="w-full bg-green-600 text-white py-2 rounded hover:bg-green-700 transition"
        >
          저장
        </button>
      </div>
    </div>
  );
}
