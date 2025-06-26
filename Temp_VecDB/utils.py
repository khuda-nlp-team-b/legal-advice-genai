from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def create_db(data, api_key):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250,
        chunk_overlap=50,
        encoding_name='cl100k_base'
    )

    all_texts = []
    all_label = []

    for i, row in data.iterrows():
        texts = text_splitter.split_text(row['판례내용'])
        for i in range(len(texts)):
            all_label.append(row['사건명'])
        all_texts.extend(texts)

    embeddings_model = OpenAIEmbeddings(api_key=api_key)

    # all_label이 all_texts와 길이가 같다고 가정
    metadatas = [{'label': label} for label in all_label]

    db = Chroma.from_texts(
        all_texts,
        embedding=embeddings_model,
        collection_name='history',
        persist_directory='./db/chromadb',
        collection_metadata={'hnsw:space': 'cosine'},
        metadatas=metadatas
    )

def retrieve_db(query, api_key, data):
    db = Chroma(
        persist_directory=f"./db/chromadb",
        collection_name=f'history',
        embedding_function=OpenAIEmbeddings(api_key=api_key)
    )

    docs = db.similarity_search(query, k=1)
    print('[관련문장]:')
    print(docs[0].page_content)
    print()

    print('[전체문장]: ')
    target_label = list(docs[0].metadata.values())[0]
    matching_content = data[data['사건명'] == target_label]['판례내용']
    print(matching_content.values[0])