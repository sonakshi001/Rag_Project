import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load environment variables
project_dir = Path(__file__).resolve().parent
load_dotenv(project_dir / ".env")
load_dotenv(project_dir / ".dist" / ".env")
api_key = (os.getenv("NVIDIA_API_KEY") or os.getenv("API_KEY") or "").strip()
if not api_key:
    raise RuntimeError(
        "NVIDIA API key is missing. Set NVIDIA_API_KEY in the project .env file "
        "or environment."
    )

# 2. Document Loading
pdf_candidates = [
    project_dir / "document_loader" / "STATS.pdf",
    *project_dir.glob("*.pdf"),
]
pdf_path = next(
    (candidate for candidate in pdf_candidates if candidate.is_file()),
    None,
)
if pdf_path is None or not pdf_path.is_file():
    raise FileNotFoundError(f"PDF not found. Place a PDF in {project_dir}.")

print(f"Loading document from: {pdf_path}")
loader = PyPDFLoader(pdf_path)
docs = loader.load()
print(f"Loaded {len(docs)} document pages.")

# 3. Document Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80,
)
chunks = text_splitter.split_documents(docs)
print(f"Created {len(chunks)} text chunks.")

# 4. Embeddings & In-Memory Vector Store (Qdrant + FastEmbed)
print("Creating embeddings and indexing into Qdrant vector store...")
embeddings = FastEmbedEmbeddings()

client = QdrantClient(location=":memory:")
client.create_collection(
    collection_name="rag_collection",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="rag_collection",
    embedding=embeddings,
)
vectorstore.add_documents(chunks)
print("Indexing complete!")

# 5. Initialize NVIDIA LLM
# Note: nemotron uses reasoning tokens, so max_completion_tokens should be >= 1024
llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    nvidia_api_key=api_key,
    temperature=0.2,
    max_completion_tokens=2500,
)

# 6. Retriever & RAG Prompt
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert tutor for the subject covered in the PDF. "
        "Answer the question accurately based ONLY on the provided context. "
        "If the answer cannot be found in the context, say that you don't know.\n\n"
        "Context:\n{context}"
    ),
    ("human", "{question}"),
])

rag_chain = prompt | llm | StrOutputParser()

# 7. Question generation
question_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an exam-question generator. Use only the supplied PDF context. "
        "Generate exactly one clear, original question based on a topic in the context. "
        "Do not copy questions verbatim from the context. "
        "The question can be short-answer or descriptive. "
        "Return only the question. Do not include numbering or an answer."
    ),
    ("human", "Generate one exam question.\n\nContext:\n{context}")
])

question_chain = question_prompt | llm | StrOutputParser()

def generate_question():
    print("\n" + "="*60)
    print(f"Generating 1 question from: {pdf_path.name}")
    print("="*60)

    retrieved_docs = retriever.invoke(
        "important topics, concepts, units, short questions, and descriptive questions"
    )
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    question = question_chain.invoke({"context": context})
    print("\nGenerated question:")
    print(question)
    print("="*60)
    return question

# 8. RAG question answering
def query_rag(question: str):
    print("\n" + "="*60)
    print(f"Query: {question}")
    print("="*60)
    
    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)
    print(f"\nRetrieved {len(retrieved_docs)} relevant context chunk(s):")
    for i, doc in enumerate(retrieved_docs, start=1):
        page_num = doc.metadata.get("page", "Unknown")
        preview = doc.page_content.strip().replace("\n", " ")[:150]
        print(f" [{i}] (Page {page_num}): {preview}...")

    # Build context string
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Generate response
    print("\nGenerating response from NVIDIA LLM...")
    response = rag_chain.invoke({"context": context, "question": question})
    print("\nAnswer:")
    print(response)
    print("="*60)
    return response

if __name__ == "__main__":
    my_question = "What is a subroutine? How do subroutines help in program writing? "
    print(f"\nYour Question: {my_question}")
    query_rag(my_question)