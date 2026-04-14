import os
import logging
from dotenv import load_dotenv

# Import our custom modules
from loaddoc import load_and_chunk_documents
from vectorize import save_chunks_to_qdrant
from retrieving import get_reranking_retriever
from llm import build_rag_pipeline

logging.basicConfig(
    level=logging.INFO,

    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # 1. Load Environment Variables (OPENAI_API_KEY, COHERE_API_KEY)
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("COHERE_API_KEY"):
        logger.error("Missing required API keys in .env file.")
        return

    # Path to sample document (Update this path to a real PDF you have)
    doc_path = "sample_document.pdf"
    if not os.path.exists(doc_path):
        logger.error(f"Please provide a valid document at {doc_path}")
        return

    try:
        # Step 1: Load and Semantically Chunk the Document
        chunks = load_and_chunk_documents(doc_path)
        
        # Step 2: Embed and Save to Qdrant
        # (For repeated runs, you'd check if the index exists to skip ingestion)
        vector_store = save_chunks_to_qdrant(chunks)
        
        # Step 3: Setup Retriever with Reranking
        retriever = get_reranking_retriever(vector_store)
        
        # Step 4: Build the LLM Chain
        rag_chain = build_rag_pipeline(retriever)
        
        # Step 5: Execute Query
        query = "What is the main topic discussed in the document?"
        logger.info(f"Executing query: '{query}'")
        
        response = rag_chain.invoke({"input": query})
        
        print("\n" + "="*50)
        print("QUERY:", query)
        print("-" * 50)
        print("ANSWER:", response["answer"])
        print("="*50 + "\n")
        
        # Production debugging: View which chunks the reranker selected
        print("Top Reranked Context Used:")
        for i, doc in enumerate(response["context"]):
            print(f"\n[Source {i+1}] {doc.page_content[:200]}...")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()