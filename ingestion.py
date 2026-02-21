import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()


def main():
    print("Hello from langchain-learning!")
    loader = TextLoader("mediumblog1.txt", encoding="utf-8")
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Splitted into {len(texts)} chunks.")

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    print("ingesting...")
    index_name = os.environ.get("INDEX_NAME")
    if not index_name:
        raise ValueError("INDEX_NAME environment variable is not set. Please add it to your .env file.")

    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=index_name
    )
    print("finish")


if __name__ == "__main__":
    main()
