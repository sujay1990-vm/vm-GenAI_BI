from langchain_core.documents import Document
import pickle
import os
from typing import TypedDict, List, Optional
from langchain_community.vectorstores import FAISS
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llm import get_embedding_model
from typing import Annotated, TypedDict
import operator
from langchain_core.documents import Document


class RAGWorkerState(TypedDict):
    query: str
    rag_outputs: List[Document] 


DB_PATH = r"vector_db"
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
embeddings = get_embedding_model()


# Load FAISS vector store (child chunks)
vectorstore = FAISS.load_local(
    DB_PATH,
    embeddings=embeddings,  # Your OpenAI or other embedding model
    allow_dangerous_deserialization=True
)

# Load parent docstore
with open(os.path.join(DB_PATH, "parent_docstore.pkl"), "rb") as f:
    parent_data = pickle.load(f)

docstore = InMemoryStore()
docstore.mset(list(parent_data.items()))  # pass as list of (id, doc) tuples

# Initialize retriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)


from langchain_core.tools import tool
from functools import partial
from typing import List

def make_rag_worker_tool(retriever):
    @tool
    def rag_worker_tool(query: str) -> str:
        """
        Retrieves relevant documents for the given query using RAG.
        Returns top 3 results as a combined string summary.
        """
        print(f"📥 RAG Worker received query: {query}")

        # Retrieve top-k child chunks and trace back to parent docs
        results = retriever.invoke(query)

        if not results:
            return "No relevant documents found."

        # Return a summarized version of top 3 parent documents
        top_results = results[:3]
        combined_text = "\n\n---\n\n".join([doc.page_content for doc in top_results])

        return combined_text

    return rag_worker_tool

# def rag_worker(state: RAGWorkerState):
#     query = state["query"]
#     print(f"📥 RAG Worker received query: {query}")

#     # Step 1: Retrieve top-k child chunks based on similarity
#     child_chunks = vectorstore.similarity_search(query, k=5)

#     # Step 2: Extract unique parent document IDs from child chunk metadata
#     parent_ids = list({chunk.metadata.get("doc_id") for chunk in child_chunks if "doc_id" in chunk.metadata})

#     # Step 3: Fetch parent documents from the docstore using mget
#     parent_docs_map = docstore.mget(parent_ids)  # returns list in same order, may include None
#     parent_docs = [doc for doc in parent_docs_map if doc is not None]

#     if not parent_docs:
#         print("⚠️ No parent documents found. Falling back to child chunks.")
#         return {"rag_outputs": child_chunks[:3]}

#     # Step 4: Return top N parent documents
#     return {"rag_outputs": parent_docs[:3]}