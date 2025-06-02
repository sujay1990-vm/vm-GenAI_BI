from langchain_core.documents import Document
import pickle
from langchain_community.vectorstores import FAISS
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever

DB_PATH = r"vector_db"
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
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
docstore.store = parent_data

# Initialize retriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)


def rag_worker(state: RAGWorkerState):
    query = state["query"]
    print(f"📥 RAG Worker received query: {query}")

    # Assume retriever returns List[Document]
    results = retriever.invoke(query)

    # Return top 3 results (as-is, not just content)
    return {"rag_outputs": results[:3]}