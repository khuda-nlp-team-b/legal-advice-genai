from langchain_community.vectorstores import Chroma
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document


def migrate_chroma_to_qdrant(
    chroma_dir,
    qdrant_path,
    collection_name,
    embedding_model='nlpai-lab/KURE-v1',
    device='cpu'
):
    """
    ChromaDB의 컬렉션을 Qdrant로 이전하는 함수.
    chroma_dir: ChromaDB persist_directory 경로
    qdrant_path: Qdrant 저장 경로 (로컬 파일)
    collection_name: 마이그레이션할 컬렉션 이름
    embedding_model: 임베딩 모델명
    device: 'cpu' 또는 'cuda'
    """
    # 1. Chroma에서 Document와 Embedding 불러오기
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model, model_kwargs={"device": device})
    chroma = Chroma(
        persist_directory=chroma_dir,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    # 모든 문서와 메타데이터 추출
    chroma_data = chroma.get()
    docs = []
    for content, meta in zip(chroma_data['documents'], chroma_data['metadatas']):
        docs.append(Document(page_content=content, metadata=meta))

    # 2. Qdrant 초기화 및 컬렉션 생성
    client = QdrantClient(path=qdrant_path)
    # 벡터 사이즈는 임베딩 모델에 따라 다름 (예: KURE-v1은 768)
    vector_size = len(embeddings.embed_documents(["테스트"])[0])
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    except Exception as e:
        print(f"[경고] 컬렉션 생성 중 오류 (이미 존재할 수 있음): {e}")

    # 3. Qdrant에 문서 적재
    qdrant = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    print(f'벡터 저장소 생성 완료, 문서 배치 적재 중...')
    batch_size = 1000
    total_docs = len(docs)
    for i in range(0, total_docs, batch_size):
        batch_docs = docs[i:i+batch_size]
        qdrant.add_documents(batch_docs)
        print(f"  - {i+1} ~ {min(i+len(batch_docs), total_docs)} / {total_docs} 문서 적재 완료")
    print(f"✅ {total_docs}개 문서가 Qdrant로 배치 이전되었습니다.")


if __name__ == "__main__":
    # 사용 예시: 경로/컬렉션명/임베딩모델/디바이스는 필요에 따라 수정
    migrate_chroma_to_qdrant(
        chroma_dir="./db",           # ChromaDB 경로
        qdrant_path="C:/qdrant_db",   # Qdrant 저장 경로
        collection_name="LAW_RAG_500_75",
        embedding_model='nlpai-lab/KURE-v1',
        device='cpu'
    ) 