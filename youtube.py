# ============================
# RAG-Based YouTube Summarizer
# ============================

from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ---------------- CONFIG ----------------
TOP_K = 3

# ---------------- USER INPUT ----------------
youtube_url = input("Enter YouTube video URL: ")
question = input("Enter your question about the video: ")

# ---------------- 1. LOAD YOUTUBE VIDEO ----------------
loader = YoutubeLoader.from_youtube_url(
    youtube_url,
    add_video_info=False
)
documents = loader.load()

# ---------------- 2. SPLIT INTO CHUNKS ----------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)

# ---------------- 3. CREATE EMBEDDINGS ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# ---------------- 4. VECTOR STORE ----------------
vectorstore = FAISS.from_documents(chunks, embeddings)

# ---------------- 5. RETRIEVER ----------------
retriever = vectorstore.as_retriever(
    search_kwargs={"k": TOP_K}
)

# ---------------- 6. LOAD LLM ----------------
llm_pipeline = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.3,
        "max_new_tokens": 120,
        "return_full_text": False
    }
)


llm = ChatHuggingFace(llm=llm_pipeline)

# ---------------- 7. PROMPT ----------------
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Summarize the YouTube video transcript strictly using the provided context
and answer ONLY what is asked in the question.

If the context is insufficient, reply exactly:
"Sorry I cannot provide a summary at this time."

Context:
{context}

Question:
{question}

Answer:
"""
)

# ---------------- 8. RAG CHAIN ----------------
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

# ---------------- 9. RUN ----------------
response = rag_chain.invoke(question)

print("\n--- RESPONSE ---\n")
print(response.content)
