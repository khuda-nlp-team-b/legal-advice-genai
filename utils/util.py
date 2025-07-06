from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings   
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pandas as pd
import pymysql
import os
import warnings
import chromadb
from langchain.embeddings import SentenceTransformerEmbeddings
import torch
import time
from datetime import datetime, timedelta
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

warnings.filterwarnings('ignore')


def get_mysql_connection(host,port,username,password,db_name):
    try:
        conn = pymysql.connect(host=host, port=port, user=username, password=password, db=db_name,charset='utf8',cursorclass=pymysql.cursors.DictCursor)
        print(f"MySQL 연결 성공")
        return conn
    except Exception as e:
        print(f"MySQL 연결 오류: {e}")
        return None

def get_document(conn,source):
    try: 
        with conn.cursor() as cursor:
            sql = f'SELECT 판례일련번호, 판례내용 FROM 판례 WHERE 판례일련번호 = {source}'
            cursor.execute(sql)
            result = cursor.fetchone()
            return result
    except Exception as e:
        print(f"MySQL 조회 오류: {e}")
        return None


def multiquery_retrieve_db(query,host,port,username,password,db_name,base_db_dir='./db',k=1):
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
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.5}
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)
    prompt = PromptTemplate.from_template(
        """
        By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of distance-based similarity search.  
        Your response must be a list of values separated solely by new-line characters, e.g.  
        foo\nbar\nbaz\n

        #ORIGINAL QUESTION:  
        {question}

        #Answer in Korean:
        ### Role
        You are a “Legal-document RAG” multi-query generator.  
        Based on the single incident scenario provided below, create **5** search queries that will surface a broad range of relevant case-law materials.

        ### Input format  
        [Incident scenario]

        ### Output rules  
        1. Output **one query per line**, max 120 characters (including spaces).  
        2. Provide only the raw query strings—no numbering, quotes, comments, or explanations.  
        3. Avoid duplicate or near-duplicate queries; use distinct wording and perspectives.  
        4. Separate lines with `\n` only, following the pattern `foo\nbar\nbaz\n`.

        ### Query-writing guidelines  
        - Restate the **key facts/acts** and **main legal issues** (tort, vicarious liability, bailment/custody, insurer subrogation, etc.) in varied ways.  
        - Restate the situation in a legal way so that it can be used as a search query.

        """
    )
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
        prompt = prompt,
        include_original=True,
    )
    results = multiquery_retriever.invoke(query)
    conn = get_mysql_connection(host,port,username,password,db_name)
    # 결과 출력
    output = []
    for i, doc in enumerate(results):
        meta = doc.metadata
        result = get_document(conn, meta['source'])
        output.append({
            "rank": i+1,
            "판례일련번호": meta['source'],
            "사건명": meta.get('case_type'),
            "유사문단": doc.page_content.strip(),
            "전문": result['판례내용'] if result else None
        })
    return output

    
def retrieve_db(query,conn,vectorstore,k=1):
    
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.85},
        return_metadata=True
    )
    
    '''cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    compressor = CrossEncoderReranker(model=cross_encoder,top_n=k)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )
    
    results = compression_retriever.invoke(query)'''
    results = retriever.invoke(query)
    
    print('벡터스토어 검색 중...')

    # 결과 출력
    output = []
    for i, doc in enumerate(results):
        meta = doc.metadata
        result = get_document(conn, meta['source'])
        output.append({
        "rank": i+1,
        "판례일련번호": meta['source'],
        "사건명": meta.get('case_type'),
        "유사문단": doc.page_content.strip(),
        "전문": result['판례내용'] if result else None
        })
        #print(meta['source'])
        #print(doc.page_content)
    return output


def check_db(base_db_dir='./db'):
    print(f'🔍 DB 경로: {base_db_dir}')
    try:
        client = chromadb.PersistentClient(path=base_db_dir)
        collections = client.list_collections()
        if not collections:
            print('❌ 컬렉션이 존재하지 않습니다.')
            return
        print(f'✅ {len(collections)}개의 컬렉션이 존재합니다:')
        for col in collections:
            print(f'\n📁 컬렉션 이름: {col.name}')
            print(f'  - id: {col.id}')
            print(f'  - 메타데이터: {col.metadata}')
            print(f'  - document count: {col.count()}')
    except Exception as e:
        print(f'❌ DB 확인 중 오류 발생: {e}')
    
def delete_collection(collection_name, base_db_dir='./db'):
    print(f'🗑️ 컬렉션 삭제 시도: {collection_name} (DB 경로: {base_db_dir})')
    try:
        client = chromadb.PersistentClient(path=base_db_dir)
        client.delete_collection(name=collection_name)
        print(f'✅ 컬렉션 "{collection_name}" 삭제 완료!')
    except Exception as e:
        print(f'❌ 컬렉션 삭제 중 오류 발생: {e}')

def get_llm(openai_key):
    return ChatOpenAI(api_key=openai_key, model="gpt-4o-mini", temperature=0.5)

def run_rag(user_query: str, vectorstore, k: int = 5, conn = None,answer_tpl = None,openai_key = None) -> str:
    # 1) 검색어 재작성
    #search_query = rewrite_query(user_query)
    #print("   ↪ 검색어:", search_query)

    # 2) MySQL+Chroma 통합 전문 조회 (create_db.retrieve_db 호출)
    print("DB 검색 중...")
    results = retrieve_db(
        user_query,
        conn,
        vectorstore,
        k=k
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

    llm = get_llm(openai_key)
    print("🔄 답변 생성(LLM) …", end=" ")
    start = time.perf_counter()
    resp = llm.invoke(answer)
    print(f"✔ ({time.perf_counter()-start:.1f}s)")

    # ''' '''로 감싸진 답변이면 제거
    content = resp.content.strip() if hasattr(resp, "content") else resp.strip()
    if content.startswith("'''") and content.endswith("'''"):
        content = content[3:-3].strip()
    return content

def setup_db(base_db_dir='./db'):
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✅ CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
        print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("❌ CUDA 사용 불가능 - CPU 모드로 실행됩니다")
    device = "cuda" if cuda_available else "cpu"
    #check_db()
    vectorstore = Chroma(
        persist_directory=base_db_dir,
        embedding_function=SentenceTransformerEmbeddings(model_name='nlpai-lab/KURE-v1', model_kwargs={"device": device}),
        collection_name='LAW_RAG_500_75'
    )
    print('벡터스토어 로드 완료')
    return vectorstore

if __name__ == "__main__":
    check_db()