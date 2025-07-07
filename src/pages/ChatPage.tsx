import React, { useState, useRef, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import ChatMessage from "../components/ChatMessage";

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
  isLoading?: boolean;
  feedback?: boolean | null;
}

const ChatPage: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const initialQuestion = location.state?.initialQuestion;

  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      text: "안녕하세요! ⚖️ 판례 기반 법률 상담을 도와드리는 AI 어시스턴트입니다.\n궁금하신 법적 문제를 말씀해 주시면 관련 판례를 기반으로 설명드리겠습니다.",
      isUser: false,
      timestamp: new Date(),
    },
  ]);
  const [inputText, setInputText] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const initialHandled = useRef(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // 초기 질문 처리
  useEffect(() => {
    if (initialQuestion && !initialHandled.current) {
      initialHandled.current = true;
      sendMessage(initialQuestion);
    }
    // eslint-disable-next-line
  }, [initialQuestion]);

  // 질문 전송 및 답변 수신
  const sendMessage = async (text: string) => {
    if (!text.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text,
      isUser: true,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);

    const loadingMessage: Message = {
      id: `loading-${Date.now()}`,
      text: "답변 생성 중입니다...",
      isUser: false,
      timestamp: new Date(),
      isLoading: true,
    };
    setMessages((prev) => [...prev, loadingMessage]);

    try {
      const response = await fetch("http://localhost:8000/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text }),
      });
      const data = await response.json();

      setMessages((prev) =>
        prev
          .filter((msg) => !msg.isLoading)
          .concat([
            {
              id: Date.now().toString(),
              text: data.answer,
              isUser: false,
              timestamp: new Date(),
            },
          ])
      );
    } catch (error) {
      console.error("Error fetching answer:", error);
      setMessages((prev) =>
        prev
          .filter((msg) => !msg.isLoading)
          .concat([
            {
              id: Date.now().toString(),
              text: "⚠️ 답변 생성 중 오류가 발생했습니다. 다시 시도해 주세요.",
              isUser: false,
              timestamp: new Date(),
            },
          ])
      );
    }
  };

  // 판례 상세 호출
  const handleCaseClick = async (caseNumber: string) => {
    const loadingMessage: Message = {
      id: `loading-case-${caseNumber}`,
      text: `📂 판례번호 ${caseNumber} 정보를 불러오는 중입니다...`,
      isUser: false,
      timestamp: new Date(),
      isLoading: true,
    };
    setMessages((prev) => [...prev, loadingMessage]);

    try {
      const response = await fetch("http://localhost:8000/api/case", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ caseNumber }),
      });
      const data = await response.json();

      if (response.ok && data.caseInfo) {
        setMessages((prev) =>
          prev
            .filter((msg) => !msg.isLoading)
            .concat([
              {
                id: Date.now().toString(),
                text: `📋 판례번호: ${caseNumber}\n\n${data.caseInfo}`,
                isUser: false,
                timestamp: new Date(),
              },
            ])
        );
      } else {
        setMessages((prev) =>
          prev
            .filter((msg) => !msg.isLoading)
            .concat([
              {
                id: Date.now().toString(),
                text: `⚠️ 판례번호 ${caseNumber}의 정보를 찾을 수 없습니다.`,
                isUser: false,
                timestamp: new Date(),
              },
            ])
        );
      }
    } catch (error) {
      console.error("Error fetching case info:", error);
      setMessages((prev) =>
        prev
          .filter((msg) => !msg.isLoading)
          .concat([
            {
              id: Date.now().toString(),
              text: "⚠️ 판례 정보 조회 중 오류가 발생했습니다.",
              isUser: false,
              timestamp: new Date(),
            },
          ])
      );
    }
  };

  // 피드백 처리
  const handleFeedback = (messageId: string, isPositive: boolean) => {
    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === messageId ? { ...msg, feedback: isPositive } : msg
      )
    );
    console.log(
      `피드백: ${isPositive ? "좋아요" : "싫어요"} - 메시지 ID: ${messageId}`
    );
  };

  // 메시지 전송 핸들러
  const handleSendMessage = () => {
    if (!inputText.trim()) return;
    sendMessage(inputText);
    setInputText("");
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto">
      {/* 헤더 */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <h1 className="text-xl font-semibold text-gray-900">법률 상담 AI</h1>
        <p className="text-sm text-gray-600">
          판례 기반 법률 상담으로 보다 정확한 가이드를 제공합니다.
        </p>
      </div>

      {/* 메시지 영역 */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {messages.map((message) => (
          <ChatMessage
            key={message.id}
            message={message.text}
            isUser={message.isUser}
            timestamp={message.timestamp}
            isLoading={message.isLoading}
            onFeedback={
              !message.isUser
                ? (isPositive) => handleFeedback(message.id, isPositive)
                : undefined
            }
            onCaseClick={!message.isUser ? handleCaseClick : undefined}
          />
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* 입력 영역 */}
      <div className="border-t border-gray-200 bg-white px-6 py-4">
        <div className="flex space-x-4">
          <div className="flex-1">
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="법적 질문을 입력해보세요..."
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              rows={3}
            />
          </div>
          <button
            onClick={handleSendMessage}
            disabled={!inputText.trim()}
            className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            전송
          </button>
        </div>
        <p className="text-xs text-gray-500 mt-2">
          Enter로 전송, Shift+Enter로 줄바꿈
        </p>
      </div>
    </div>
  );
};

export default ChatPage;
