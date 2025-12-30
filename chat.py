from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document


# ---------------- CONFIG ----------------
TEXT_FILE_PATH = "notice.txt"  
TOP_K = 3


with open(TEXT_FILE_PATH, "r", encoding="utf-8") as f:
    text = f.read().strip()

if not text:
    raise ValueError("notice.txt is empty")

documents = [Document(page_content=text)]


# ---------------- 2. SPLIT INTO CHUNKS ----------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)


# ---------------- 3. EMBEDDINGS ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------- 4. VECTOR STORE ----------------
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})


# ---------------- 5. LOAD LLM ----------------
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.3,
        "max_new_tokens": 60
    }
)

model = ChatHuggingFace(llm=llm)


# ---------------- 6. STRICT PROMPT ----------------
prompt = ChatPromptTemplate.from_template(
    """Answer ONLY with the final answer.
Do NOT explain.
Do NOT repeat the context.
If the answer is not present, say exactly: NOT FOUND IN NOTICE.

Context:
{context}

Question:
{question}

Answer:"""
)


# ---------------- 7. RAG CHAIN ----------------
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
)


# ---------------- 8. ASK FUNCTION ----------------
def answer_question(question: str) -> str:
    response = rag_chain.invoke(question)
    answer = response.content.strip()

    # Safety cleanup
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()

    return answer


# ---------------- 9. CLI ----------------
if __name__ == "__main__":
    print("\nText-based RAG ready. Type 'exit' to quit.\n")

    while True:
        q = input("Ask a question: ").strip()
        if q.lower() == "exit":
            break

        print("\nAnswer:")
        print(answer_question(q))
        print("-" * 40)
