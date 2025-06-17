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
from pydantic import BaseModel
from langchain.tools import Tool
from langchain_core.tools import tool
from functools import partial
from typing import List

# class RAGWorkerState(TypedDict):
#     query: str
#     rag_outputs: List[Document] 


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

class RAGInputSchema(BaseModel):
    query: str

def make_rag_worker_tool(retriever):
    @tool
    def rag_worker_tool(query: str) -> str:
        """
        Retrieves relevant documents for the given query using RAG.
        Returns top 3 results with file name and document content to improve trust.
        """
        print(f"📥 RAG Worker received query: {query}")

        # Retrieve top-k documents
        results = retriever.invoke(query)

        if not results:
            return "No relevant documents found."

        # Take top 3, show filename + content
        top_results = results[:3]
        formatted_chunks = []

        for i, doc in enumerate(top_results, 1):
            filename = doc.metadata.get("filename", "Unknown File")
            content = doc.page_content.strip()
            formatted_chunks.append(f"📄 **Source {i}: {filename}**\n\n{content}")

        return "\n\n---\n\n".join(formatted_chunks)

    return rag_worker_tool


# def make_rag_worker_tool(retriever):
#     def _rag_worker(query: str) -> str:
#         """
#         Retrieves relevant documents for the given query using semantic search.
#         Returns top 3 results with filename and document content to improve trust and traceability.
#         """
#         print(f"📥 RAG Worker received query: {query}")
#         results = retriever.invoke(query)

#         if not results:
#             return "No relevant documents found."

#         top_results = results[:3]
#         formatted_chunks = []

#         for i, doc in enumerate(top_results, 1):
#             filename = doc.metadata.get("filename", "Unknown File")
#             content = doc.page_content.strip()
#             formatted_chunks.append(f"📄 **Source {i}: {filename}**\n\n{content}")

#         return "\n\n---\n\n".join(formatted_chunks)

#     # Full function spec using Tool.from_function
#     return Tool.from_function(
#         func=_rag_worker,
#         name="rag_worker_tool",
#         description=(
#             "Retrieves relevant policy, coverage, or guideline excerpts from unstructured PDFs or Word documents. "
#             "Use this tool when the query relates to terms like 'soft threshold', 'eligibility criteria', "
#             "'complex BI escalation', or other concepts not found in structured SQL tables. "
#             "Returns the top 3 most relevant document snippets, including filenames, for auditability."
#         ),
#         args_schema=RAGInputSchema,
#         return_direct=True  # optional: set True if you want to return the output as-is to user
#     )

