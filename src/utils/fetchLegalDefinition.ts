// src/utils/fetchLegalDefinition.ts

export const fetchLegalDefinition = async (term: string): Promise<string> => {
    try {
        // 실제 API 연동 전 테스트용 더미 데이터
        const mockDefinitions: Record<string, string> = {
            "폭행": "상대방의 신체에 대해 물리적 힘을 가하여 침해하는 행위.",
            "과실": "주의의무를 다하지 않아 타인에게 손해를 입히는 것.",
            "상해": "신체의 완전성을 훼손하거나 건강을 해치는 것.",
            "손해배상": "타인의 불법행위로 입은 손해를 금전으로 보상받는 것.",
            "불법행위": "고의 또는 과실로 타인에게 손해를 가하는 행위.",
            "책임": "법적 의무를 이행할 의무."
        };

        return mockDefinitions[term] || `${term}의 정의를 찾을 수 없습니다.`;
    } catch (error) {
        console.error("용어 정의 불러오기 오류:", error);
        return "정의 정보를 불러오는 중 오류가 발생했습니다.";
    }
};
