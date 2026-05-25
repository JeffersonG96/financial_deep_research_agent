# # Dense, Sparce, Hybrid and Reranking
# - Implementar busqueda hibrida con filtros
# - Aplicar reranking para mejores resultados
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain.messages import HumanMessage, SystemMessage
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.tools import tool
from langchain_core.documents import Document

#metadata extraction from llm
from agent.schema import ChunkMetadata
from qdrant_client.models import Filter, FieldCondition, MatchValue
import subprocess
import sys
import os

#configuration
COLLECTION_NAME = "financial_docs"
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"
RERANKER_MODEL = "BAAI/bge-reranker-base"
url = os.getenv("QDRANT_URL")


#initialize llm
llm = ChatGoogleGenerativeAI(model=LLM_MODEL)
#gemini embedding
embedding = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
#sparse embedding
sparse_embedding = FastEmbedSparse(model_name="Qdrant/bm25")

#connect to existing collection
vector_store = QdrantVectorStore.from_existing_collection(
    collection_name=COLLECTION_NAME,
    embedding=embedding,
    sparse_embedding=sparse_embedding,
    url=url,
    retrieval_mode=RetrievalMode.HYBRID
)

# ### Filter Extraction with LLM

def extract_filters(user_query: str):
    system_prompt = f"""
        Extract metadata filters from the query. Return None for fields not mentioned.

            #### EXAMPLES
            COMPANY MAPPINGS:
            - Amazon/AMZN -> amazon
            - Google/Alphabet/GOOGL/GOOG -> google
            - Apple/AAPL -> apple
            - Microsoft/MSFT -> microsoft
            - Tesla/TSLA -> tesla
            - Nvidia/NVDA -> nvidia
            - Meta/Facebook/FB -> meta

            DOC TYPE:
            - Annual report -> 10-k
            - Quarterly report -> 10-q
            - Current report -> 8-k

            EXAMPLES:
            "Amazon Q3 2024 revenue" -> {{"company_name": "amazon", "doc_type": "10-q", "fiscal_year": 2024, "fiscal_quarter": "q3"}}
            "Apple 2023 annual report" -> {{"company_name": "apple", "doc_type": "10-k", "fiscal_year": 2023}}
            "Tesla profitability" -> {{"company_name": "tesla"}}

            Extract metadata based on the user query only:
        """
    structured_llm = llm.with_structured_output(ChunkMetadata)
    metadata = structured_llm.invoke([SystemMessage(system_prompt), HumanMessage(f"user query: {user_query}")])
    if metadata:
        filters = metadata.model_dump(exclude_none=True)
    else: 
        metadata = {}
    return filters

#metadata filtering
@tool
def hybrid_search(query: str, k: int = 10) -> list[Document]:
    """Hybrid search (dense + sparse vectors)
    
    Args: 
        query: search query
        k: number of results
    
    Return:
        List of documents objects"""
    filters = extract_filters(query)
    qdrant_filter = None
    
    if filters:
        condition = [FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value)) for key, value in filters.items()]
        qdrant_filter = Filter(must=condition)
    
    result = vector_store.similarity_search(query=query, k=k, filter=qdrant_filter)
    return result

#reranking for better result
 
def rerank_results(query: str, documents: list, top_k:int = 5):
    """Rerank documents using cross-encoder
    Return:
        List of (score, Documents) tuples sorted by relevance"""
    
    reranker = HuggingFaceCrossEncoder(
        model_name=RERANKER_MODEL,
        model_kwargs={'device':'cpu'}
    )

    query_doc_pairs = [(query, doc.page_content) for doc in documents]

    scores = reranker.score(query_doc_pairs)
    reranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    reranked = reranked[:top_k]
    return [rank[1] for rank in reranked]


@tool
def live_finance_researcher(query: str) -> str:
    """Research stocks usinf yahoo finance MCP async function.
    Call this tool to get:
    - current stock prices and real-time maket data
    - lastest dinancial news
    - stock recimmendationand analyst rating
    - recents stock actions(splits, dividends)
    
    args:
        query: the financial research question about current market data
    Return: 
         Research results from yahoo finance"""

    code=f"""
import asyncio
from scripts_py.yahoo_mcp import finance_research
asyncio.run(finance_research('{query}'))"""

    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, encoding="utf-8", cwd="D:/Cursos/Agentes/financial_deep_research_agent")
    return result.stdout if result.stdout else "No se obtuvo respuesta"