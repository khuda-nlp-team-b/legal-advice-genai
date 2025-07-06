import React, { useState } from "react";
import { Link } from "react-router-dom";

export default function Sidebar({
  onSettingsClick,
  open,
  setOpen,
}: {
  onSettingsClick: () => void;
  open: boolean;
  setOpen: (open: boolean) => void;
}) {
  return (
    <div
      className={`bg-green-50 border-r p-4 transition-all fixed left-0 top-0 h-full z-10 ${
        open ? "w-56" : "w-14"
      }`}
    >
      <button className="mb-6 text-green-600" onClick={() => setOpen(!open)}>
        ☰
      </button>
      {open && (
        <>
          <div className="text-green-800 font-bold text-sm border-b border-green-300 pb-2 mb-4">
            📁 메뉴
          </div>
          <nav className="space-y-3 text-sm">
            <Link
              to="/"
              className="block text-green-700 hover:text-green-900 font-medium"
            >
              🏠 홈
            </Link>
            <Link
              to="/chat"
              className="block text-green-700 hover:text-green-900 font-medium"
            >
              💬 채팅 상담
            </Link>
            <Link
              to="/history"
              className="block text-green-700 hover:text-green-900 font-medium"
            >
              🕘 기록 보기
            </Link>
            <Link
              to="/bookmarks"
              className="block text-green-700 hover:text-green-900 font-medium"
            >
              ⭐ 저장한 기록
            </Link>
            <button
              className="text-green-700 hover:text-green-900 font-medium"
              onClick={onSettingsClick}
            >
              ⚙ 설정
            </button>
          </nav>
        </>
      )}
    </div>
  );
}
