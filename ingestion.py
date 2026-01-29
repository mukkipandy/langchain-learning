import asyncio
import os
import ssl

import certifi
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import log_header, log_info, Colors, log_success

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"  # Note the full path
)
vectorstore = PineconeVectorStore(
    index_name="langchain-doc-index", embedding=embeddings
)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site",
        Colors.PURPLE,
    )
    # Crawl the documentation site

    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 2,
            "extract_depth": "advanced",
        }
    )
    all_docs = [Document(page_content=result['raw_content'], metadata={"source": result['url']}) for result in
                res["results"]]
    log_success(f"TavilyCrawl: Successfully crawled {len(all_docs)} URLs from documentation site")


if __name__ == "__main__":
    asyncio.run(main())
