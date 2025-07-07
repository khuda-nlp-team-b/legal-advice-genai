import React, { useState } from "react";

interface ChatMessageProps {
  message: string;
  isUser: boolean;
  timestamp: Date;
  isLoading?: boolean;
  onFeedback?: (isPositive: boolean) => void;
  onCaseClick?: (caseNumber: string) => void;
}

const ChatMessage: React.FC<ChatMessageProps> = ({
  message,
  isUser,
  timestamp,
  isLoading = false,
  onFeedback,
  onCaseClick,
}) => {
  const [feedbackGiven, setFeedbackGiven] = useState<boolean | null>(null);

  const handleFeedback = (isPositive: boolean) => {
    if (feedbackGiven !== null) return; // 이미 피드백을 준 경우
    setFeedbackGiven(isPositive);
    onFeedback?.(isPositive);
  };
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div className={`max-w-3xl ${isUser ? "order-2" : "order-1"}`}>
        <div
          className={`rounded-2xl px-4 py-3 ${
            isUser ? "bg-blue-500 text-white" : "bg-gray-100 text-gray-900"
          }`}
        >
          {isLoading ? (
            <div className="flex items-center space-x-2">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                <div
                  className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                  style={{ animationDelay: "0.1s" }}
                ></div>
                <div
                  className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                  style={{ animationDelay: "0.2s" }}
                ></div>
              </div>
              <span className="text-sm">답변을 생성하고 있습니다...</span>
            </div>
          ) : (
            <div className="whitespace-pre-wrap">
              {message.split(/(\[판례번호:\d+\])/).map((part, index) => {
                const match = part.match(/\[판례번호:(\d+)\]/);
                if (match) {
                  return (
                    <button
                      key={index}
                      onClick={() => onCaseClick?.(match[1])}
                      className="inline-block px-2 py-1 mx-1 bg-blue-100 text-blue-700 rounded-md hover:bg-blue-200 transition-colors text-sm font-medium"
                    >
                      {part}
                    </button>
                  );
                }
                return part;
              })}
            </div>
          )}
        </div>
        <div
          className={`text-xs text-gray-500 mt-1 ${
            isUser ? "text-right" : "text-left"
          }`}
        >
          {timestamp.toLocaleTimeString("ko-KR", {
            hour: "2-digit",
            minute: "2-digit",
          })}
        </div>

        {/* 피드백 버튼 (AI 답변에만 표시) */}
        {!isUser && !isLoading && onFeedback && (
          <div className="flex items-center space-x-2 mt-2">
            <span className="text-xs text-gray-500">
              이 답변이 도움이 되었나요?
            </span>
            <div className="flex space-x-1">
              <button
                onClick={() => handleFeedback(true)}
                disabled={feedbackGiven !== null}
                className={`px-2 py-1 text-xs rounded-md transition-colors ${
                  feedbackGiven === true
                    ? "bg-green-500 text-white"
                    : feedbackGiven === false
                    ? "bg-gray-200 text-gray-400"
                    : "bg-gray-100 hover:bg-green-100 text-gray-600 hover:text-green-600"
                }`}
              >
                👍 좋아요
              </button>
              <button
                onClick={() => handleFeedback(false)}
                disabled={feedbackGiven !== null}
                className={`px-2 py-1 text-xs rounded-md transition-colors ${
                  feedbackGiven === false
                    ? "bg-red-500 text-white"
                    : feedbackGiven === true
                    ? "bg-gray-200 text-gray-400"
                    : "bg-gray-100 hover:bg-red-100 text-gray-600 hover:text-red-600"
                }`}
              >
                👎 싫어요
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;
