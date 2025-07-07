import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import ChatPage from "./pages/ChatPage";
import QuestionInputPage from "./pages/QuestionInputPage";
import AnswerResultPage from "./pages/AnswerResultPage";
import CaseDetailPage from "./pages/CaseDetailPage";
import HistoryPage from "./pages/HistoryPage";
import BookmarkPage from "./pages/BookmarkPage";

import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import SettingsModal from "./components/SettingsModal";

export default function App() {
  const [isSettingsOpen, setSettingsOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <Router>
      <div className="min-h-screen flex bg-white text-gray-900">
        <Sidebar
          onSettingsClick={() => setSettingsOpen(true)}
          open={sidebarOpen}
          setOpen={setSidebarOpen}
        />
        <div
          className={`flex-1 flex flex-col transition-all ${
            sidebarOpen ? "ml-56" : "ml-14"
          }`}
        >
          <Header />
          <main className="flex-grow p-6 overflow-y-auto">
            <Routes>
              <Route path="/" element={<QuestionInputPage />} />
              <Route path="/chat" element={<ChatPage />} />
              <Route path="/question" element={<QuestionInputPage />} />
              <Route path="/result" element={<AnswerResultPage />} />
              <Route path="/case/:id" element={<CaseDetailPage />} />
              <Route path="/history" element={<HistoryPage />} />
              <Route path="/bookmarks" element={<BookmarkPage />} />
            </Routes>
          </main>
        </div>
        {isSettingsOpen && (
          <SettingsModal onClose={() => setSettingsOpen(false)} />
        )}
      </div>
    </Router>
  );
}
