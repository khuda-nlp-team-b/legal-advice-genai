from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings   
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.schema import Document
import pandas as pd
import pymysql
from langchain_community.storage import SQLStore
from langchain_huggingface import HuggingFaceEmbeddings
import os
import warnings
import chromadb
from langchain.embeddings import SentenceTransformerEmbeddings
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



def save_df(host,port,username,password,db_name):
    conn = pymysql.connect(host=host, port=port, user=username, password=password, db=db_name)
    df = pd.read_sql('SELECT * FROM 판례', conn)
    df.to_csv('판례.csv', index=False)
    conn.close()
    

def create_db(api_key,base_db_dir='./db'):
    # 판례.csv 파일 읽기
    df_판례 = pd.read_csv('판례.csv')

    
    # 기존 벡터 DB가 존재하면 삭제
    try:
        client = chromadb.PersistentClient(path=base_db_dir)
        client.delete_collection(name='LAW_RAG')
        print(f"[삭제] 기존 벡터 DB가 삭제되었습니다: {base_db_dir}")
    except Exception as e:
        print(f"[경고] 기존 DB 삭제 중 오류 발생: {e}")
    
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
    embeddings = SentenceTransformerEmbeddings(model_name='nlpai-lab/KURE-v1',kwargs={"device": "cuda"})
    
    print('텍스트 분할 중...')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250, ##### 1500
        chunk_overlap=25 ##### 300
    )
    
    split_docs = text_splitter.split_documents(docs)
    print('텍스트 분할 완료')
    
     # 자식 청크를 저장할 벡터스토어 생성
    print('벡터스토어 생성 중...')
    total_docs = len(split_docs)
    print(f'총 {len(docs)}개의 문서를 {total_docs}개의 청크로 처리합니다...')
    
    # 배치 크기 설정
    batch_size = 1000
    processed = 0
    
    for i in range(0, total_docs, batch_size):
        batch_docs = split_docs[i:i + batch_size]
        processed += len(batch_docs)
        
        if i == 0:
            # 첫 번째 배치로 벡터스토어 생성
            vectorstore = Chroma.from_documents(
                documents=batch_docs,
                embedding=embeddings,
                collection_name='LAW_RAG',
                persist_directory=base_db_dir
            )
            return
        else:
            # 나머지 배치는 추가
            vectorstore.add_documents(batch_docs)
        
        print(f'진행률: {processed}/{total_docs} ({processed/total_docs*100:.1f}%)')
    
    print('벡터스토어 생성 완료!')

    vectorstore.persist()
    
    print(f"[완료] 판례 {len(docs)}건이 원본과 청크로 분리되어 저장되었습니다.")
    print(f"원본 문서: SQL 테이블 '판례_원본'")
    print(f"청크 벡터: {base_db_dir}/LAW_RAG")
    
def retrieve_db(query,host,port,username,password,db_name,api_key,base_db_dir='./db'):
    print('벡터스토어 생성 중...')
    vectorstore = Chroma(
        persist_directory=base_db_dir,
        embedding_function=SentenceTransformerEmbeddings(model_name='nlpai-lab/KURE-v1',kwargs={"device": "cuda"}),
        collection_name='LAW_RAG'
    )
    print('벡터스토어 생성 완료')
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    print('벡터스토어 검색 중...')
    results = retriever.invoke(query)
    
    conn = get_mysql_connection(host,port,username,password,db_name)
    # 결과 출력
    for i, doc in enumerate(results):
        meta = doc.metadata
        print(f"\n🔍 [결과 {i+1}]")
        print(f"▶ 판례일련번호 : {meta['source']}")
        print(f"▶ 사건명 : {meta['case_type']}")
        print("▶ 유사 문단:", doc.page_content.strip())
        result = get_document(conn,meta['source'])
        print('▶ 전체 판례:',result['판례내용'])
        print("\n" + "="*50)