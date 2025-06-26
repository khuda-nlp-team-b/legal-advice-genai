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
warnings.filterwarnings('ignore')



def save_df(host,port,username,password,db_name):
    conn = pymysql.connect(host=host, port=port, user=username, password=password, db=db_name)
    df = pd.read_sql('SELECT * FROM 판례', conn)
    df.to_csv('판례.csv', index=False)
    conn.close()
    

def create_db(base_db_dir='./db'):
    # 판례.csv 파일 읽기
    df_판례 = pd.read_csv('판례.csv',nrows=10)

    
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
    model_name = 'intfloat/multilingual-e5-large-instruct'
    hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    print('텍스트 분할 중...')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )
    
    split_docs = text_splitter.split_documents(docs)
    print('텍스트 분할 완료')
    
    # 자식 청크를 저장할 벡터스토어 생성
    print('벡터스토어 생성 중...')
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=hf_embeddings,
        collection_name='LAW_RAG',
        persist_directory=base_db_dir
    )

    vectorstore.persist()
    
    print(f"[완료] 판례 {len(docs)}건이 원본과 청크로 분리되어 저장되었습니다.")
    print(f"원본 문서: SQL 테이블 '판례_원본'")
    print(f"청크 벡터: {base_db_dir}/LAW_RAG")
    
def retrieve_db(query,base_db_dir='./db'):
    df=pd.read_csv('판례.csv',nrows=10)
    vectorstore = Chroma(
        persist_directory=base_db_dir,
        embedding_function=HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large-instruct'),
        collection_name='LAW_RAG'
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    print('벡터스토어 검색 중...')
    results = retriever.invoke(query)
    
    # 결과 출력
    for i, doc in enumerate(results):
        meta = doc.metadata
        print(f"\n🔍 [결과 {i+1}]")
        print(f"▶ 판례일련번호 : {meta['source']}")
        print(f"▶ 사건명 : {meta['case_type']}")
        print("▶ 유사 문단:", doc.page_content.strip())
        print('원본 판례:',df.loc[df['판례일련번호'] == meta['source']]['판례내용'].values[0])
        print("\n" + "="*50)