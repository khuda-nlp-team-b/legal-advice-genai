from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from jinja2 import Template
import pymysql
import logging

# ------------------ 환경 및 로깅 설정 ------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)

# ------------------ FastAPI 앱 초기화 ------------------
app = FastAPI()

# CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Jinja 템플릿 로드 ------------------
with open("./prompts/answer_synth.j2", "r", encoding="utf-8") as f:
    answer_template = Template(f.read())

# ------------------ DB 및 벡터스토어 설정 ------------------
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT"))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sbert-nli",
    model_kwargs={'device': 'cpu'}
)

vectorstore = Chroma(
    persist_directory="./db",
    embedding_function=embeddings,
    collection_name='LAW_RAG'
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ------------------ MySQL 연결 ------------------
def get_mysql_connection():
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        db=DB_NAME,
        charset='utf8',
        cursorclass=pymysql.cursors.DictCursor
    )

# ------------------ Request Models ------------------
class QueryRequest(BaseModel):
    query: str

class AdviseRequest(BaseModel):
    user_query: str
    structured_issue: Dict[str, Any]
    retrieved_cases: List[Dict[str, Any]]

# ------------------ /retrieve ------------------
@app.post("/retrieve")
async def retrieve_case(request: QueryRequest):
    query = request.query
    docs = retriever.invoke(query)
    retrieved = []
    seen_case_ids = set()

    conn = get_mysql_connection()
    cursor = conn.cursor()

    for doc in docs:
        case_id = doc.metadata.get("source")
        if case_id in seen_case_ids:
            continue
        seen_case_ids.add(case_id)

        cursor.execute("SELECT * FROM 판례 WHERE 판례일련번호 = %s", (case_id,))
        result = cursor.fetchone()
        if not result:
            continue

        retrieved.append({
            "info": {
                "id": case_id,
                "caseNm": result.get("사건명", ""),
                "judmnAdjuDe": result.get("선고일자", ""),
                "caseNo": result.get("사건번호", ""),
                "courtNm": result.get("법원명", "")
            },
            "jdgmn": result.get("판결요지", ""),
            "summary": result.get("판시사항", ""),
            "content": result.get("판례내용", ""),
            "similarity": doc.metadata.get("similarity", 1.0)
        })

    conn.close()
    return {"retrieved_cases": retrieved}

# ------------------ /advise ------------------
@app.post("/advise")
async def advise(request: AdviseRequest):
    user_query = request.user_query

    retrieved_context_list = []
    for item in request.retrieved_cases:
        content = item.get("content", "")
        case_id = item.get("info", {}).get("id", "unknown")
        retrieved_context_list.append(f"{content} [ref:{case_id}]")
    retrieved_context = "\n\n".join(retrieved_context_list) or "관련 판례를 찾지 못했습니다."

    rendered_prompt = answer_template.render(context=retrieved_context, user_query=user_query)

    logging.info(f"LLM 호출 준비, prompt 길이: {len(rendered_prompt)}")
    try:
        response = llm.invoke(rendered_prompt)
        logging.info("LLM 호출 성공")
    except Exception as e:
        logging.error(f"LLM 호출 실패: {e}")
        return {"advice": "답변 생성 중 오류가 발생했습니다. 관리자에게 문의하세요."}

    return {"advice": response.content}

# ------------------ /legal-advice ------------------
@app.post("/legal-advice")
async def legal_advice(request: QueryRequest):
    query = request.query
    docs = retriever.invoke(query)
    retrieved = []
    seen_case_ids = set()

    conn = get_mysql_connection()
    cursor = conn.cursor()

    for doc in docs:
        case_id = doc.metadata.get("source")
        if case_id in seen_case_ids:
            continue
        seen_case_ids.add(case_id)

        cursor.execute("SELECT * FROM 판례 WHERE 판례일련번호 = %s", (case_id,))
        result = cursor.fetchone()
        if not result:
            continue

        retrieved.append({
            "info": {
                "id": case_id,
                "caseNm": result.get("사건명", ""),
                "judmnAdjuDe": result.get("선고일자", ""),
                "caseNo": result.get("사건번호", ""),
                "courtNm": result.get("법원명", "")
            },
            "jdgmn": result.get("판결요지", ""),
            "summary": result.get("판시사항", ""),
            "content": result.get("판례내용", ""),
            "similarity": doc.metadata.get("similarity", 1.0)
        })

    conn.close()

    retrieved_context_list = []
    for item in retrieved:
        content = item.get("content", "")
        case_id = item.get("info", {}).get("id", "unknown")
        retrieved_context_list.append(f"{content} [ref:{case_id}]")
    retrieved_context = "\n\n".join(retrieved_context_list) or "관련 판례를 찾지 못했습니다."

    rendered_prompt = answer_template.render(context=retrieved_context, user_query=query)

    logging.info(f"LLM 호출 준비, prompt 길이: {len(rendered_prompt)}")
    try:
        response = llm.invoke(rendered_prompt)
        logging.info("LLM 호출 성공")
    except Exception as e:
        logging.error(f"LLM 호출 실패: {e}")
        return {
            "summary": "답변 생성 중 오류가 발생했습니다.",
            "verdict_estimate": "",
            "recommendations": ["답변 생성 중 오류가 발생했습니다."],
            "referenced_cases": retrieved,
            "sources": []
        }

    summary_text = f"{retrieved[0]['info']['caseNm']} 사건과 유사" if retrieved else "유사 사례 없음"

    return {
        "summary": summary_text,
        "verdict_estimate": "판례 기반 예상 결과 포함",
        "recommendations": response.content.splitlines(),
        "referenced_cases": retrieved,
        "sources": [
            {
                "case_id": item["info"]["id"],
                "url": f"https://www.law.go.kr/precInfoP.do?precSeq={item['info']['id']}"
            } for item in retrieved
        ]
    }

# ------------------ 서버 실행 ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
