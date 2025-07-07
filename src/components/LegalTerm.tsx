// src/components/LegalTerm.tsx

import React, { useState } from "react";
import { fetchLegalDefinition } from "../utils/fetchLegalDefinition";
import { useLegalDefinitionStore } from "../utils/legalDefinitionStore"; // ✅ zustand store import

interface LegalTermProps {
  term: string;
  children: React.ReactNode;
}

const LegalTerm: React.FC<LegalTermProps> = ({ term, children }) => {
  const [loading, setLoading] = useState(false);
  const setDefinition = useLegalDefinitionStore((state) => state.setDefinition); // ✅ zustand setter 구독

  const handleClick = async () => {
    console.log(`[LegalTerm] Clicked term: ${term}`); 
    setLoading(true);
    try {
      const def = await fetchLegalDefinition(term);
      setDefinition(term, def);
    } catch (error) {
      console.error(`[LegalTerm] 정의 fetch 실패:`, error);
      setDefinition(term, "정의 정보를 불러오는 중 오류가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <span
      className="cursor-pointer text-blue-600 hover:underline"
      onClick={handleClick}
    >
      {children}
      {loading && (
        <span className="ml-1 text-xs text-gray-400">(불러오는 중...)</span>
      )}
    </span>
  );
};

export default LegalTerm;
