import os
import argparse
import jinja2
import time
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from dotenv import load_dotenv
import sys
from utils import util as u
import torch
from langchain_community.embeddings import SentenceTransformerEmbeddings


load_dotenv()

# ────────────────── 1) 경로·환경 ──────────────────
BASE_PATH  = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(BASE_PATH, "prompts")

# ↓↓↓  db_test/<여기에 db 입력>  부분만 바꿔주세요 ↓↓↓
DB_SUBDIR = "LAW_RAG_500_75"
#DB_DIR    = os.path.join(BASE_PATH, "db_test", DB_SUBDIR)
# ↑↑↑  db_test/<여기에 db 입력>  부분만 바꿔주세요 ↑↑↑

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# ────────────────── 2) DB 접속 정보 ──────────────────
# (create_db.retrieve_db 호출을 위해 환경변수로 설정)
HOST        = os.getenv("DB_HOST")
PORT        = int(os.getenv("DB_PORT", 3306))
USER        = os.getenv("DB_USER")
PASSWORD    = os.getenv("DB_PASSWORD")
DB_NAME     = os.getenv("DB_NAME")
#BASE_DB_DIR = os.path.join(BASE_PATH, "db_test")   # create_db.py의 base_db_dir 인자

# ────────────────── 3) Jinja2 템플릿 로드 ──────────────────
env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(PROMPT_DIR),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)
#query_tpl  = env.get_template("query_rewrite.j2")  # 템플릿 파일명이 정확한지 확인하세요 :contentReference[oaicite:0]{index=0}
answer_tpl = env.get_template("answer_synth.j2")

def get_llm():
    return ChatOpenAI(api_key=OPENAI_KEY, model="gpt-4o-mini", temperature=1)

def rewrite_query(user_query: str) -> str:
    prompt = query_tpl.render(user_query=user_query)
    llm = get_llm()
    print("🔄 질의 재작성(LLM) …", end=" ")
    start = time.perf_counter()
    resp = llm.invoke(prompt)
    print(f"✔ ({time.perf_counter()-start:.1f}s)")
    return resp.content.strip() if hasattr(resp, "content") else resp.strip()

def run_rag(user_query: str, vectorstore, k: int = 5) -> str:
    # 1) 검색어 재작성
    #search_query = rewrite_query(user_query)
    #print("   ↪ 검색어:", search_query)

    # 2) MySQL+Chroma 통합 전문 조회 (create_db.retrieve_db 호출)
    print("DB 검색 중...")
    results = u.retrieve_db(
        user_query,
        HOST, PORT, USER, PASSWORD, DB_NAME,vectorstore,k=k
    )

    # 3) 검색 결과 처리 및 템플릿 적용
    if not results or len(results) == 0:
        return "유사 판례를 찾지 못했습니다."

    # 상위 k개 결과 모두 병합
    contexts = []
    full_documents = []

    for i, item in enumerate(results):
        # 각 결과에서 필요한 정보 추출
        contexts.append(f"{i+1}. {item['유사문단']} [판례번호:{item['판례일련번호']}]")
        full_documents.append(f"--- 문서 {i+1} ---\n{item['전문']}")

    answer = answer_tpl.render(
        context1=results[0]['유사문단'] + f" [판례번호:{results[0]['판례일련번호']}]",
        full1=results[0]['전문'],
        context2=results[1]['유사문단'] + f" [판례번호:{results[1]['판례일련번호']}]",
        full2=results[1]['전문'],
        context3=results[2]['유사문단'] + f" [판례번호:{results[2]['판례일련번호']}]",
        full3=results[2]['전문'],
        context4=results[3]['유사문단'] + f" [판례번호:{results[3]['판례일련번호']}]",
        full4=results[3]['전문'],
        context5=results[4]['유사문단'] + f" [판례번호:{results[4]['판례일련번호']}]",
        full5=results[4]['전문'],
        user_query=user_query
    )

    
    llm = get_llm()
    print("🔄 답변 생성(LLM) …", end=" ")
    start = time.perf_counter()
    resp = llm.invoke(answer)
    print(f"✔ ({time.perf_counter()-start:.1f}s)")
    
    return resp.content.strip() if hasattr(resp, "content") else resp.strip()

def setup_db(base_db_dir='./db'):
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✅ CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
        print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("❌ CUDA 사용 불가능 - CPU 모드로 실행됩니다")
    device = "cuda" if cuda_available else "cpu"
    
    vectorstore = Chroma(
        persist_directory=base_db_dir,
        embedding_function=SentenceTransformerEmbeddings(model_name='nlpai-lab/KURE-v1', model_kwargs={"device": device}),
        collection_name='LAW_RAG_500_75'
    )
    print('벡터스토어 로드 완료')
    return vectorstore

def main():
    vectorstore = setup_db()
    while True:
        user_query = input("💬 처한 법적 상황과 걱정하는 점을 알려주세요: ").strip()
        if user_query == 'exit':
            break
        answer = run_rag(user_query,vectorstore)
        print(user_query)
        print("📌 최종 요약\n", answer)
        
    
    return answer

if __name__ == "__main__":
    main()