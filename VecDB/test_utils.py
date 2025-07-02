import pandas as pd
import chromadb
import os
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import shutil
from dotenv import load_dotenv
from langchain.embeddings import SentenceTransformerEmbeddings
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')


def create_test_db(api_key, base_db_dir='./db_test'):
    # 판례.csv 파일 읽기
    df_판례 = pd.read_csv('판례.csv')
    df_sampled = df_판례.sample(n=1000,random_state=42)

    
    try:
        client = chromadb.PersistentClient(path=base_db_dir)
        #디렉토리 전체를 삭제
        shutil.rmtree(base_db_dir)
        print(f"[삭제] 기존 테스트 벡터 DB들이 모두 삭제되었습니다: {base_db_dir}")
    except Exception as e:
        print(f"[경고] 기존 DB 삭제 중 오류 발생: {e}")
    
    # DB 디렉토리 생성
    os.makedirs(base_db_dir,exist_ok=True)
    
    # Document 객체 리스트 생성
    print('테스트용 Document 객체 리스트 생성 중...')
    docs = []
    for i, row in df_sampled.iterrows():
        metadata = {
            "source": row['판례일련번호'],
            'case_type': row['사건명']
        }
        doc = Document(page_content=str(row['판례내용']), metadata=metadata)
        docs.append(doc)
    print('테스트용 Document 객체 리스트 생성 완료')
    openai_embeddings = OpenAIEmbeddings(api_key=api_key,model='text-embedding-3-small')
    sentence_transformer_embeddings = SentenceTransformerEmbeddings(model_name='nlpai-lab/KURE-v1',model_kwargs={'device':'cuda'})
    # 여러 chunk_size, chunk_overlap 조합에 대해 두 임베딩 모델로 벡터스토어 생성
    chunk_configs = [
        (250, 25),
        (500, 50),
        (1000, 100)
    ]
    batch_size = 250

    for chunk_size, chunk_overlap in chunk_configs:
        print(f"\n=== [설정] chunk_size={chunk_size}, chunk_overlap={chunk_overlap} ===")
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        split_docs = recursive_splitter.split_documents(docs)
        print('텍스트 분할 완료')

        collection_name = f'LAW_RAG_TEST_{chunk_size}_{chunk_overlap}'
        total_docs = len(split_docs)

        # OpenAI Embeddings 벡터스토어 생성
        print('테스트용 openai 벡터스토어 생성 중...')
        print(f'총 {len(docs)}개의 판례를 {len(split_docs)}개의 청크로 처리합니다...')
        processed = 0
        for i in range(0, total_docs, batch_size):
            batch_docs = split_docs[i:i + batch_size]
            if i == 0:
                vectorstore = Chroma.from_documents(
                    documents=batch_docs,
                    embedding=openai_embeddings,
                    collection_name=f'{collection_name}_openai',
                    persist_directory=base_db_dir
                )
            else:
                vectorstore.add_documents(batch_docs)
            processed += len(batch_docs)
            print(f'[openai] 진행률: {processed}/{total_docs} ({processed/total_docs*100:.1f}%)')
        print(f'테스트용 벡터스토어 생성 완료 : {collection_name}_openai')
        vectorstore.persist()

        # SentenceTransformer 벡터스토어 생성
        print('테스트용 sentence_transformer 벡터스토어 생성 중...')
        processed = 0
        for i in range(0, total_docs, batch_size):
            batch_docs = split_docs[i:i + batch_size]
            if i == 0:
                vectorstore_sentence_transformer = Chroma.from_documents(
                    documents=batch_docs,
                    embedding=sentence_transformer_embeddings,
                    collection_name=f'{collection_name}_sentence_transformer',
                    persist_directory=base_db_dir
                )
            else:
                vectorstore_sentence_transformer.add_documents(batch_docs)
            processed += len(batch_docs)
            print(f'[sentence_transformer] 진행률: {processed}/{total_docs} ({processed/total_docs*100:.1f}%)')
        print(f'테스트용 sentence_transformer 벡터스토어 생성 완료 : {collection_name}_sentence_transformer')
        vectorstore_sentence_transformer.persist()

    








