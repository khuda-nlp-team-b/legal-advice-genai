from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings   
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pandas as pd
import pymysql
import os
import warnings
import chromadb
<<<<<<< HEAD
from langchain.embeddings import SentenceTransformerEmbeddings
=======
from langchain_community.embeddings import SentenceTransformerEmbeddings
>>>>>>> 37c4389 (배포 전 로컬 테스트 완료)
import torch
import time
from datetime import datetime, timedelta
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
<<<<<<< HEAD
=======
import asyncio
from langchain.chains import ConversationChain  
from langchain.memory import ConversationBufferMemory

# LangChain 텔레메트리 비활성화
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
>>>>>>> 37c4389 (배포 전 로컬 테스트 완료)

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


<<<<<<< HEAD

def save_df(host,port,username,password,db_name):
    conn = pymysql.connect(host=host, port=port, user=username, password=password, db=db_name)
    df = pd.read_sql('SELECT * FROM 판례', conn)
    df.to_csv('판례.csv', index=False)
    conn.close()
    

def create_db(base_db_dir='C:\\db'):
    # CUDA 사용 가능 여부 확인
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✅ CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
        print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("❌ CUDA 사용 불가능 - CPU 모드로 실행됩니다")
    
    # 시작 시간 기록
    start_time = time.time()
    
    # 현재 스크립트 디렉토리 기준으로 파일 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, '판례.csv')
    
    # 파일 존재 여부 확인
    if not os.path.exists(csv_file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {csv_file_path}")
        print(f"현재 디렉토리: {os.getcwd()}")
        print(f"스크립트 디렉토리: {script_dir}")
        print("사용 가능한 파일들:")
        for file in os.listdir(script_dir):
            if file.endswith('.csv'):
                print(f"  - {file}")
        return
    
    # 판례.csv 파일 읽기
    print(f"📁 파일 경로: {csv_file_path}")
    df_판례 = pd.read_csv(csv_file_path)
    print(f"📊 총 {len(df_판례)}개의 판례 데이터를 로드했습니다")

    # 기존 벡터 DB 디렉토리가 존재하면 완전 삭제
    if os.path.exists(base_db_dir):
        try:
            import shutil
            shutil.rmtree(base_db_dir)
            print(f"[삭제] 기존 벡터 DB 디렉토리가 완전히 삭제되었습니다: {base_db_dir}")
        except Exception as e:
            print(f"[경고] 기존 DB 디렉토리 삭제 중 오류 발생: {e}")
    
    # DB 디렉토리 생성
    os.makedirs(base_db_dir, exist_ok=True)
    
    # Document 객체 리스트 생성
    print('Document 객체 리스트 생성 중...')
    docs = []
    for i, row in df_판례.iterrows():
        metadata = {
            "source": row['판례일련번호'],
            'case_type': row['사건명']
        }
        doc = Document(page_content=str(row['판례내용']), metadata=metadata)
        docs.append(doc)
    print('Document 객체 리스트 생성 완료')
    # model_name = 'intfloat/multilingual-e5-large-instruct'
    embeddings = SentenceTransformerEmbeddings(model_name='nlpai-lab/KURE-v1', model_kwargs={"device": "cpu"})
    
    print('텍스트 분할 중...')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250, ##### 1500
        chunk_overlap=25 ##### 300
    )
    split_docs = text_splitter.split_documents(docs)
    print(f'텍스트 분할 완료: {len(docs)}개 문서 → {len(split_docs)}개 청크')
    
    # 임베딩 모델 설정 (CUDA 사용 여부에 따라)
    device = "cuda" if cuda_available else "cpu"
    embeddings = SentenceTransformerEmbeddings(model_name='nlpai-lab/KURE-v1', model_kwargs={"device": device})
    print(f"🤖 임베딩 모델 로드 완료 (장치: {device})")
    
    # 첫 번째 벡터스토어 생성 (chunk_size=250, chunk_overlap=50)
    print('\n=== 첫 번째 벡터스토어 생성 (chunk_size=250, chunk_overlap=50) ===')
    chunk_start_time = time.time()
    
    print('텍스트 분할 중...')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=50
    )
    
    split_docs = text_splitter.split_documents(docs)
    print(f'텍스트 분할 완료: {len(docs)}개 문서 → {len(split_docs)}개 청크')
    
    # 자식 청크를 저장할 벡터스토어 생성
    print('벡터스토어 생성 중...')
    total_docs = len(split_docs)
    print(f'총 {len(docs)}개의 문서를 {total_docs}개의 청크로 처리합니다...')
    
    # 배치 크기 설정 (메모리 관리를 위해)
    batch_size = 1000
    processed = 0
    processing_times = []  # 각 배치의 처리 시간을 저장할 리스트
    
    for i in range(0, total_docs, batch_size):
        batch_start_time = time.time()
        batch_docs = split_docs[i:i + batch_size]
        processed += len(batch_docs)
        
        if i == 0:
            # 첫 번째 배치로 벡터스토어 생성
            vectorstore = Chroma.from_documents(
                documents=batch_docs,
                embedding=embeddings,
                collection_name='LAW_RAG_250_50',
                persist_directory=base_db_dir
            )
        else:
            # 나머지 배치는 추가
            vectorstore.add_documents(batch_docs)
        
        # 배치 처리 시간 계산
        batch_time = time.time() - batch_start_time
        processing_times.append(batch_time)
        
        # 이전 배치들의 평균 처리 시간으로 예상 완료시간 계산
        if len(processing_times) > 1:
            avg_time_per_batch = sum(processing_times) / len(processing_times)
            remaining_batches = (total_docs - processed) // batch_size + (1 if (total_docs - processed) % batch_size > 0 else 0)
            estimated_remaining_time = remaining_batches * avg_time_per_batch
            estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining_time)
            
            print(f'진행률: {processed}/{total_docs} ({processed/total_docs*100:.1f}%) - 예상 완료: {estimated_completion.strftime("%H:%M:%S")} (평균 배치시간: {avg_time_per_batch:.1f}초)')
        else:
            # 첫 번째 배치 후에는 아직 평균을 계산할 수 없으므로 현재 배치 시간으로 추정
            remaining_batches = (total_docs - processed) // batch_size + (1 if (total_docs - processed) % batch_size > 0 else 0)
            estimated_remaining_time = remaining_batches * batch_time
            estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining_time)
            
            print(f'진행률: {processed}/{total_docs} ({processed/total_docs*100:.1f}%) - 예상 완료: {estimated_completion.strftime("%H:%M:%S")} (현재 배치시간: {batch_time:.1f}초)')
    
    vectorstore.persist()
    chunk_time = time.time() - chunk_start_time
    print(f'✅ 첫 번째 벡터스토어 생성 완료! (소요시간: {chunk_time/60:.1f}분)')

    # 두 번째 벡터스토어 생성 (chunk_size=500, chunk_overlap=75)
    print('\n=== 두 번째 벡터스토어 생성 (chunk_size=500, chunk_overlap=75) ===')
    chunk_start_time = time.time()
    
    print('두번째 텍스트 분할 중...')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=75
    )
    
    split_docs = text_splitter.split_documents(docs)
    print(f'텍스트 분할 완료: {len(docs)}개 문서 → {len(split_docs)}개 청크')
    
    # 자식 청크를 저장할 벡터스토어 생성
    print('벡터스토어 생성 중...')
    total_docs = len(split_docs)
    print(f'총 {len(docs)}개의 문서를 {total_docs}개의 청크로 처리합니다...')
    
    # 배치 크기 설정
    batch_size = 1000
    processed = 0
    processing_times = []  # 각 배치의 처리 시간을 저장할 리스트
    
    for i in range(0, total_docs, batch_size):
        batch_start_time = time.time()
        batch_docs = split_docs[i:i + batch_size]
        processed += len(batch_docs)
        
        if i == 0:
            # 첫 번째 배치로 벡터스토어 생성
            vectorstore = Chroma.from_documents(
                documents=batch_docs,
                embedding=embeddings,
                collection_name='LAW_RAG_500_75',
                persist_directory=base_db_dir
            )
            return
        else:
            # 나머지 배치는 추가
            vectorstore.add_documents(batch_docs)
        
        # 배치 처리 시간 계산
        batch_time = time.time() - batch_start_time
        processing_times.append(batch_time)
        
        # 이전 배치들의 평균 처리 시간으로 예상 완료시간 계산
        if len(processing_times) > 1:
            avg_time_per_batch = sum(processing_times) / len(processing_times)
            remaining_batches = (total_docs - processed) // batch_size + (1 if (total_docs - processed) % batch_size > 0 else 0)
            estimated_remaining_time = remaining_batches * avg_time_per_batch
            estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining_time)
            
            print(f'진행률: {processed}/{total_docs} ({processed/total_docs*100:.1f}%) - 예상 완료: {estimated_completion.strftime("%H:%M:%S")} (평균 배치시간: {avg_time_per_batch:.1f}초)')
        else:
            # 첫 번째 배치 후에는 아직 평균을 계산할 수 없으므로 현재 배치 시간으로 추정
            remaining_batches = (total_docs - processed) // batch_size + (1 if (total_docs - processed) % batch_size > 0 else 0)
            estimated_remaining_time = remaining_batches * batch_time
            estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining_time)
            
            print(f'진행률: {processed}/{total_docs} ({processed/total_docs*100:.1f}%) - 예상 완료: {estimated_completion.strftime("%H:%M:%S")} (현재 배치시간: {batch_time:.1f}초)')
    
    vectorstore.persist()
    chunk_time = time.time() - chunk_start_time
    print(f'✅ 두 번째 벡터스토어 생성 완료! (소요시간: {chunk_time/60:.1f}분)')
    
    # 전체 완료 시간 계산
    total_time = time.time() - start_time
    print(f"\n🎉 전체 작업 완료!")
    print(f"📊 총 소요시간: {total_time/60:.1f}분")
    print(f"📁 저장 위치: {base_db_dir}")
    print(f"[완료] 판례 {len(docs)}건이 원본과 청크로 분리되어 저장되었습니다.")

=======
>>>>>>> 37c4389 (배포 전 로컬 테스트 완료)
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
<<<<<<< HEAD
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
=======
        Your response must be a list of values separated solely by new-line characters, e.g.  
        foo\nbar\nbaz\n

        #ORIGINAL QUESTION:  
        {question}

        #Answer in Korean:
        ### Role
        You are a "Legal-document RAG" multi-query generator.  
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
>>>>>>> 37c4389 (배포 전 로컬 테스트 완료)

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

    
<<<<<<< HEAD
def retrieve_db(query,host,port,username,password,db_name,vectorstore,k=1):
    
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.5},
=======
def retrieve_db(query,conn,vectorstore,k=1,threshold=0.0):
    
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.85, "score_threshold": threshold},
>>>>>>> 37c4389 (배포 전 로컬 테스트 완료)
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

<<<<<<< HEAD
    if not conn:
        conn = get_mysql_connection(host,port,username,password,db_name)
=======
>>>>>>> 37c4389 (배포 전 로컬 테스트 완료)
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
<<<<<<< HEAD
    conn.close()
=======
>>>>>>> 37c4389 (배포 전 로컬 테스트 완료)
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

<<<<<<< HEAD
=======
def get_llm(openai_key):
    return ChatOpenAI(api_key=openai_key, model="gpt-4o-mini", temperature=0.5)

def docs2tpl(results,answer_tpl,user_query,k=5):
    contexts = []
    full_documents = []
    for i, item in enumerate(results[:k]):
        contexts.append(f"{i+1}. {item['유사문단']} [판례번호:{item['판례일련번호']}]")
        full_documents.append(f"--- 문서 {i+1} ---\n{item['전문']}")

    # k개에 맞게 동적으로 렌더링
    render_data = {'user_query': user_query}
    
    for i in range(k):
        if i < len(results):
            render_data[f'context{i+1}'] = results[i]['유사문단'] + f" [판례번호:{results[i]['판례일련번호']}]"
            render_data[f'full{i+1}'] = results[i]['전문']
        else:
            # k개보다 적은 결과가 있는 경우 빈 문자열로 채움
            render_data[f'context{i+1}'] = ""
            render_data[f'full{i+1}'] = ""
    
    answer = answer_tpl.render(**render_data)
    return answer

async def run_rag(user_query: str, vectorstore, k: int = 5, conn = None,answer_tpl = None,openai_key = None) -> str:
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
    answer = docs2tpl(results,answer_tpl,user_query)

    llm = get_llm(openai_key)
    print("🔄 답변 생성(LLM) …", end=" ")
    start = time.perf_counter()
    
    # 스트리밍하면서 내용을 모아서 리턴
    full_response = ""
    async for chunk in llm.astream(answer):
        content = chunk.content
        if content:
            print(content, end="", flush=True)
            full_response += str(content)
    
    print(f"✔ ({time.perf_counter()-start:.1f}s)")

    # ''' '''로 감싸진 답변이면 제거
    content = full_response.strip()
    if content.startswith("'''") and content.endswith("'''"):
        content = content[3:-3].strip()
    return content

async def run_rag_stream(user_query: str, vectorstore, k: int = 5, conn = None, answer_tpl = None, openai_key = None):
    """스트리밍 방식으로 답변을 생성하는 함수 - 각 청크를 yield"""
    # 1) 검색어 재작성
    print("DB 검색 중...")
    results = retrieve_db(
        user_query,
        conn,
        vectorstore,
        k=k
    )

    # 3) 검색 결과 처리 및 템플릿 적용
    if not results or len(results) == 0:
        yield "유사 판례를 찾지 못했습니다."
        return

    answer = docs2tpl(results,answer_tpl,user_query)

    llm = get_llm(openai_key)
    
    # 스트리밍하면서 각 청크를 yield
    async for chunk in llm.astream(answer):
        if not hasattr(run_rag_stream, '_first_chunk_printed'):
            print("📌 답변 \n", end="", flush=True)
            run_rag_stream._first_chunk_printed = True
        content = chunk.content
        if content:
            #print(content, end="", flush=True)
            yield str(content)

    # content 변수가 정의되지 않았으므로 제거
    # async generator에서는 return 값을 가질 수 없음

def set_conversation(query,answer,model):
    memory = ConversationBufferMemory()
    conversation = ConversationChain( 
        llm=model,
        memory=memory,
        verbose=True
    )
    memory.save_context({'input':query},{'output':answer})
    return conversation

async def run_conversation(conversation,user_query,vectorstore,conn,k=5,answer_tpl=None,openai_key=None):
    print("DB 검색 중...")
    results = retrieve_db(
        user_query,
        conn,
        vectorstore,
        k=k,
        threshold=0.7
    )
    
    answer = docs2tpl(results,answer_tpl,user_query,k=3)
    
    async for chunk in conversation.astream(answer):
        # ConversationChain.astream() returns dictionaries with 'response' key
        if isinstance(chunk, dict) and 'response' in chunk:
            content = chunk['response']
            if content:
                yield str(content)

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

>>>>>>> 37c4389 (배포 전 로컬 테스트 완료)
if __name__ == "__main__":
    check_db()