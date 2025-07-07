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
      text: "안녕하세요! 법률 상담을 도와드리는 AI 어시스턴트입니다. 어떤 법적 문제로 도움이 필요하신가요?",
      isUser: false,
      timestamp: new Date(),
    },
  ]);
  const [inputText, setInputText] = useState("");
  const [isFirstQuestion, setIsFirstQuestion] = useState(true); // 최초 질문 여부
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const initialHandled = useRef(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // 초기 질문 자동 처리
  useEffect(() => {
    if (initialQuestion && !initialHandled.current) {
      initialHandled.current = true;
      setInputText("");
      handleSendMessage(initialQuestion);
    }
    // eslint-disable-next-line
  }, [initialQuestion]);

  const handleFeedback = (messageId: string, isPositive: boolean) => {
    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === messageId ? { ...msg, feedback: isPositive } : msg
      )
    );
    // TODO: 피드백 서버 전송
  };

  const handleCaseClick = async (caseNumber: string) => {
    setMessages((prev) => [
      ...prev,
      {
        id: `loading-case-${caseNumber}`,
        text: "판례 정보를 조회 중입니다...",
        isUser: false,
        timestamp: new Date(),
        isLoading: true,
      },
    ]);

    try {
      const response = await fetch("http://localhost:8000/api/case", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ caseNumber }),
      });

      if (response.ok) {
        const data = await response.json();
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
    } catch (err) {
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

  const handleSendMessage = async (forcedText?: string) => {
    const textToSend = forcedText ?? inputText.trim();
    if (!textToSend) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: textToSend,
      isUser: true,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setInputText("");

    const streamingMessageId = "ai-" + Date.now().toString();
    setMessages((prev) => [
      ...prev,
      {
        id: streamingMessageId,
        text: "",
        isUser: false,
        timestamp: new Date(),
      },
    ]);

    try {
      const endpoint = isFirstQuestion
        ? "http://localhost:8000/api/ask"
        : "http://localhost:8000/api/conversation";

      const body = isFirstQuestion
        ? JSON.stringify({ question: textToSend })
        : JSON.stringify({
            question: textToSend,
            history: messages
              .filter((msg) => msg.text.trim() !== "")
              .map((msg) => ({
                role: msg.isUser ? "user" : "assistant",
                content: msg.text,
              })),
          });

      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split("\n");

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              try {
                const data = JSON.parse(line.slice(6));
                if (data.chunk) {
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === streamingMessageId
                        ? { ...msg, text: msg.text + data.chunk }
                        : msg
                    )
                  );
                }
              } catch (e) {
                console.error("JSON parse error:", e);
              }
            }
          }
        }
      }

      if (isFirstQuestion) {
        setIsFirstQuestion(false); // 최초 질문 후 이후부터 conversation 모드로 전환
      }
    } catch (err) {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === streamingMessageId
            ? {
                ...msg,
                text: "⚠️ 답변 생성 중 오류가 발생했습니다. 다시 시도해 주세요.",
              }
            : msg
        )
      );
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto">
      {/* 채팅 헤더 */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <h1 className="text-xl font-semibold text-gray-900">법률 상담 AI</h1>
        <p className="text-sm text-gray-600">
          판례 기반 법률 상담을 도와드립니다
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
              onKeyPress={handleKeyPress}
              placeholder="법적 문제에 대해 질문해주세요..."
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              rows={3}
              disabled={false}
            />
          </div>
          <button
            onClick={() => handleSendMessage()}
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
