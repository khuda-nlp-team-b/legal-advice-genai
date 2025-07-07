# WOODJUDGE
![image](https://github.com/user-attachments/assets/0bd29f86-f1e1-4372-8992-0beb5f446ede)
자연어로 상황을 입력하면 유사 판례 기반의 법률 조언을 제공하는 AI 서비스, "WOODJUDGE"입니다.

# 주제 선정 이유
  
- 법률 지식이 부족한 일반인에게 로펌 방문은 비용과 심리적 부담으로 장벽이 높음
- 인터넷을 통한 정보 탐색은 신뢰성 있는 자료를 찾기 어렵고, 과정 또한 복잡하고 번거로움
- 자연어로 상황을 입력하면 유사 판례 기반의 법률 조언을 제공하는 AI 서비스

# 기존 서비스의 한계

### GPT 기반 법률 상담
- 웹 문서 기반으로 상담 → 법률적 근거 부족

### 기존 판례 검색 서비스(Ex: https://bigcase.ai/)
- 키워드 검색만 가능, 자연어 처리 불가 → 구체적인 행동 추천 기능 없음

# 우리 서비스의 차별점
![image](https://github.com/user-attachments/assets/b0604bc1-031f-4990-bca4-476e674e5deb)

# ⚙️ 서비스 아키텍처
![image](https://github.com/user-attachments/assets/072be938-c6f0-45f8-b5be-98d6f987689c)

# 🔎 세부 기술

## 1️⃣ RAG<br>

### Text split<br>
 - 임베딩될 벡터의 단위 결정<br>
 - 사용 모델 : Langchain - RecursiveCharacterTextSplitter

### Embedding<br>
 - 벡터의 임베딩 방법 결정<br>
 - 사용 모델 : KURE(Korea University Retrieval Embedding model) - sentence_transformer

## 2️⃣ 프롬프트 엔지니어링<br>

### 프롬프트 조건에 따라 아래의 5개 항목 출력

1. 사실관계: 유사 판례의 배경, 쟁점, 적용법리, 판결을 서술
2. 예상결과: 유사 판례를 근거로 해당 사건의 예상 판결 서술 
3. 대응전략: 사용자에게 권장되는 실행 가능 대처 방안 설명
4. 우려사항: 발생 가능한 법적 우려사항 서술
5. 관련판례: 인용한 모든 판례에 대해 판례번호와 유사점 명시

## 3️⃣ 백엔드 / 프론트엔드<br>

### 백엔드<br>
 - Python<br>
 - FastAPI<br>

### 프론트엔드<br>
 - React<br>
 - TailwindCSS<br>
 - TypeScript<br>

# 🎥 시연 결과
![image](https://github.com/user-attachments/assets/e4e14869-5d61-4068-a0b2-125d3eb0e4a9)

# ✔️ 결론

## 개선점
#### - 사건 유형별 맞춤 로직 고도화
#### - 변호사 연계 통한 통합 상담 플랫폼화

## 기대효과
#### - RAG, 프롬포트 -> 정확도 향상
#### - 법률 용어를 몰라도 일상어로 질문 가능
#### - 출처 기반 신뢰 확보
