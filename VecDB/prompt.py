import os
import argparse
import jinja2
import time
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from dotenv import load_dotenv
load_dotenv()  

# ────────────────── 1) 경로·환경 ──────────────────
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(BASE_PATH, "prompts")

# ↓↓↓  db_test/<여기에 db 입력>  부분만 바꿔주세요 ↓↓↓
DB_SUBDIR = "LAW_RAG_TEST_250_25_openai"
#DB_DIR = os.path.join(BASE_PATH, "db_test", DB_SUBDIR)
# ↑↑↑  db_test/<여기에 db 입력>  부분만 바꿔주세요 ↑↑↑

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# ────────────────── 2) Jinja2 템플릿 로드 ──────────────────
env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(PROMPT_DIR),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)
query_tpl = env.get_template("query_rewrite.j2")
answer_tpl = env.get_template("answer_synth.j2")

# ────────────────── 3) LLM 래퍼 ──────────────────

def get_llm():
    return ChatOpenAI(api_key=OPENAI_KEY, model_name="gpt-3.5-turbo", temperature=0)

# ────────────────── 4) 파이프라인 함수 ──────────────────

def rewrite_query(user_query: str) -> str:
    prompt = query_tpl.render(user_query=user_query)
    llm = get_llm()
    start = time.perf_counter()
    print("🔄 질의 재작성(LLM) …", end=" ")
    resp = llm.invoke(prompt)
    elapsed = time.perf_counter() - start
    print(f"✔ ({elapsed:.1f}s)")
    return resp.content.strip() if hasattr(resp, "content") else resp.strip()


def run_rag(user_query: str, k: int = 3) -> str:
    # 1) 검색어 재작성
    search_query = rewrite_query(user_query)
    print("   ↪ 검색어:", search_query)

    # 2) 벡터스토어 로드
    '''    if not os.path.isdir(DB_DIR):
            raise FileNotFoundError(f"벡터 DB 경로가 존재하지 않습니다: {DB_DIR}")'''
    
    print("📂 벡터 DB 로드 중 …", end=" ")
    start = time.perf_counter()
    embeddings = OpenAIEmbeddings(api_key=OPENAI_KEY)
    vect = Chroma(
        persist_directory='./db_test',
        embedding_function=embeddings,
        collection_name=DB_SUBDIR,
    )
    print(f"✔ ({time.perf_counter()-start:.1f}s)")

    # 3) 유사 문단 검색
    print("🔍 검색 …", end=" ")
    start = time.perf_counter()
    docs = vect.as_retriever(search_kwargs={"k": k}).invoke(search_query)
    print(f"✔ ({time.perf_counter()-start:.1f}s, {len(docs)}개)")

    # 4) 컨텍스트 조립
    context = "\n\n".join(f"[ref:{d.metadata['source']}] " + d.page_content for d in docs)

    # 5) 답변 생성
    print("🖋️ 답변 생성 …", end=" ")
    start = time.perf_counter()
    prompt = answer_tpl.render(context=context, user_query=user_query)
    llm = get_llm()
    result = llm.invoke(prompt)
    print(f"✔ ({time.perf_counter()-start:.1f}s)")
    return result.content if hasattr(result, "content") else result

# ────────────────── 5) CLI ──────────────────

def main():
    parser = argparse.ArgumentParser(
        description="판례 RAG 검색기 (OpenAI 전용, DB 재생성 없음)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # query 인수를 *선택* 으로 두고, 없으면 stdin 으로 입력받음
    parser.add_argument(
        "query",
        nargs="*",
        help="검색할 질문",
    )
    parser.add_argument("--k", type=int, default=3, help="가져올 상위 문서 수 (default=3)")

    args = parser.parse_args()

    # positional 인수가 없으면 인터랙티브 입력
    if not args.query:
        user_query = input("💬 질문을 입력하세요: ").strip()
    else:
        user_query = " ".join(args.query)

    if not user_query:
        parser.error("질문이 비어 있습니다.")
 
    answer = run_rag(user_query, args.k)
    print("\n📌 최종 요약\n", answer)


if __name__ == "__main__":
    main()
