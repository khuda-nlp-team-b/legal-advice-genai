import React, { useState } from "react";
import LegalTerm from "../components/LegalTerm";

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
  const [expanded, setExpanded] = useState(false);

  const safeMessage = typeof message === "string" ? message : "";
  const shouldTruncate = safeMessage.length > 300;
  const displayedText =
    expanded || !shouldTruncate ? safeMessage : safeMessage.slice(0, 300) + "...";

  const legalTerms = ["폭행", "과실", "상해", "손해배상", "불법행위", "책임"];

  const parseMessageWithComponents = (text: string) => {
    if (!text) return null;

    // 판례번호 정규표현식
    const caseRegex = /(판례번호[:：]?\s?)(\d{5,})/g;

    // 법률 용어 정규표현식 (단어 단위 매칭)
    const termRegex = new RegExp(`(${legalTerms.join("|")})`, "g");

    return text.split("\n").map((line, idx) => {
      const caseMatches = [...line.matchAll(caseRegex)];
      let processedLine: (string | React.ReactNode)[] = [];

      if (caseMatches.length > 0) {
        let lastIndex = 0;
        caseMatches.forEach((match, i) => {
          const [fullMatch, label, number] = match;
          const matchIndex = match.index ?? 0;

          processedLine.push(line.substring(lastIndex, matchIndex));
          processedLine.push(
            label,
            <button
              key={`case-${idx}-${i}`}
              onClick={() => onCaseClick && onCaseClick(number.trim())}
              className="text-blue-600 underline hover:text-blue-800 mx-1"
            >
              {number.trim()}
            </button>
          );
          lastIndex = matchIndex + fullMatch.length;
        });
        processedLine.push(line.substring(lastIndex));
      } else {
        processedLine = [line];
      }

      // 두 번째 단계: 법률 용어 하이라이팅
      const highlightedLine = processedLine.flatMap((segment, i) => {
        if (typeof segment !== "string") return [segment];
        const parts = segment.split(termRegex).map((part, j) => {
          if (legalTerms.includes(part)) {
            return (
              <LegalTerm term={part} key={`term-${idx}-${i}-${j}`}>
                {part}
              </LegalTerm>
            );
          }
          return part;
        });
        return parts;
      });

      return (
        <p key={`line-${idx}`} className="whitespace-pre-wrap">
          {highlightedLine}
        </p>
      );
    });
  };

  return (
    <div className={`flex flex-col ${isUser ? "items-end" : "items-start"} w-full`}>
      <div
        className={`rounded-lg p-3 max-w-[80%] break-words ${
          isUser ? "bg-blue-100 text-right" : "bg-gray-100 text-left"
        }`}
      >
        {isLoading ? (
          <div className="flex items-center space-x-2">
            <svg
              className="animate-spin h-5 w-5 text-gray-500"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              ></circle>
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8v8H4z"
              ></path>
            </svg>
            <span>답변 생성 중입니다...</span>
          </div>
        ) : (
          <>
            {parseMessageWithComponents(displayedText)}
            {shouldTruncate && !expanded && (
              <button
                onClick={() => setExpanded(true)}
                className="text-sm text-blue-500 mt-2 hover:underline"
              >
                더 보기
              </button>
            )}
          </>
        )}
      </div>

      <div className="text-xs text-gray-500 mt-1">
        {timestamp
          ? timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
          : ""}
      </div>

      {!isUser && onFeedback && !isLoading && (
        <div className="flex space-x-2 mt-1">
          <button
            onClick={() => onFeedback(true)}
            className="text-green-500 text-sm hover:underline"
          >
            👍 도움이 됐어요
          </button>
          <button
            onClick={() => onFeedback(false)}
            className="text-red-500 text-sm hover:underline"
          >
            👎 도움이 안 됐어요
          </button>
        </div>
      )}
    </div>
  );
};

export default ChatMessage;
