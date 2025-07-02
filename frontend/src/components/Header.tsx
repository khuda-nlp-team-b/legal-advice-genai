import React from "react";
import { Link } from "react-router-dom";
import woodjudge from "../assets/woodjudge.png";

export default function Header() {
  return (
    <header className="p-4 border-b shadow-sm flex justify-between items-center bg-white sticky top-0 z-10">
      <Link to="/" className="flex items-center space-x-2">
        <img src={woodjudge} alt="logo" className="w-8 h-8 rounded-full" />
        <span className="text-lg font-bold text-green-600">법률GPT</span>
      </Link>
      <Link to="/history" className="text-sm text-blue-600 underline">
        기록 보기
      </Link>
    </header>
  );
}
