import os
from pdf2image import convert_from_path
import pytesseract
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import ChatHuggingFace
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from flask import Flask, render_template, request, jsonify

# ---------------- FILE PATHS ----------------
PDF_FILE_PATH = "Notice.pdf"
TESSERACT_PATH = r"C:\ocr\tesseract\tesseract.exe"
POPPLER_BIN_PATH = r"C:\poppler\poppler-25.12.0\Library\bin"

# ---------------- FIX WINDOWS EXECUTION ----------------
os.environ["PATH"] += os.pathsep + POPPLER_BIN_PATH
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ---------------- OCR FUNCTION ----------------
def extract_text_from_scanned_pdf(pdf_path: str) -> str:
    pages = convert_from_path(pdf_path, dpi=300)
    full_text = ""
    for page in pages:
        full_text += pytesseract.image_to_string(page, lang="eng") + "\n"
    return full_text.strip()

# ---------------- 1. OCR LOAD ----------------
text = extract_text_from_scanned_pdf(PDF_FILE_PATH)
if not text:
    raise ValueError("OCR failed: no text extracted from PDF")
documents = [Document(page_content=text)]

# ---------------- 2. SPLIT INTO CHUNKS ----------------
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# ---------------- 3. EMBEDDINGS ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- 4. VECTOR STORE ----------------
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------------- 5. LOAD LLM ----------------
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={"temperature": 0.3, "max_new_tokens": 60}
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
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
)

# ---------------- 8. ASK FUNCTION ----------------
def answer_question(question: str) -> str:
    response = rag_chain.invoke(question)
    answer = response.content.strip()
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    return answer

# ---------------- FLASK APP ----------------
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '').strip()
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    answer = answer_question(question)
    return jsonify({'answer': answer})

# ---------------- 9. RUN ----------------
if __name__ == "__main__":
    print("\nOCR-based RAG ready. Type 'exit' to quit.\n")
    while True:
        q = input("Ask a question: ").strip()
        if q.lower() == "exit":
            break
        print("\nAnswer:")
        print(answer_question(q))
        print("-" * 40)
