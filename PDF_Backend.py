from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import numpy as np
from dotenv import load_dotenv
from astrapy import DataAPIClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings

# =========================
# 1️⃣ Load environment variables
# =========================
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN_RAG")
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_KEY_RAG")

ASTRA_API_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_ENDPOINT = os.getenv("ASTRA_VECTOR_DB_ENDPOINT")
ASTRA_TABLE = "qa_sih_demo"

# =========================
# 2️⃣ FastAPI setup
# =========================
app = FastAPI(title="⛑️ Mine Survival Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 3️⃣ Astra DB connection (lazy fail-safe)
# =========================
collection = None
try:
    if not ASTRA_API_TOKEN or not ASTRA_ENDPOINT:
        raise ValueError("Missing Astra DB credentials in .env file")

    client = DataAPIClient(ASTRA_API_TOKEN)
    db = client.get_database_by_api_endpoint(ASTRA_ENDPOINT)
    collection = db.get_collection(ASTRA_TABLE)
    print(f"✅ Connected to Astra DB collection: {ASTRA_TABLE}")
except Exception as e:
    print(f"❌ Failed to connect to Astra DB: {e}")
    collection = None

# =========================
# 4️⃣ Lazy-loaded embeddings & LLM
# =========================
embedding_model = None
llm = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print("🔄 Loading HuggingFace Embeddings...")
        embedding_model = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
    return embedding_model

def get_llm():
    global llm
    if llm is None:
        print("🔄 Loading Gemini LLM...")
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    return llm

chat_histories = {}  # session_id → ChatMessageHistory


# =========================
# 5️⃣ API Models
# =========================
class AskRequest(BaseModel):
    session_id: str
    question: str


# =========================
# 6️⃣ Cosine similarity helper
# =========================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# =========================
# 7️⃣ Safe top-k vector search (lowered memory usage)
# =========================
def get_top_docs(query, k=3, limit=100):
    if collection is None:
        return []

    query_vector = np.array(get_embedding_model().embed_query(query))
    try:
        # ✅ Lower limit reduces memory usage
        all_docs = collection.find({}, limit=limit)
    except Exception as e:
        print(f"❌ Astra query failed: {e}")
        return []

    scores = []
    for doc in all_docs:
        vector = doc.get("vector")
        if vector is None or len(vector) == 0:
            continue
        vector = np.array(vector)
        sim = cosine_similarity(query_vector, vector)
        scores.append((sim, doc))

    scores.sort(reverse=True, key=lambda x: x[0])
    return [doc for sim, doc in scores[:k]]


# =========================
# 8️⃣ Health check
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


# =========================
# 9️⃣ Ask endpoint
# =========================
@app.post("/ask")
async def ask_question(req: AskRequest):
    try:
        session_id = req.session_id
        question = req.question

        # Initialize chat history for new session
        if session_id not in chat_histories:
            chat_histories[session_id] = ChatMessageHistory()
        history = chat_histories[session_id]

        # Get top documents
        top_docs = get_top_docs(question, k=3)
        if not top_docs:
            return JSONResponse(
                {"answer": "⚠️ No relevant documents found for this query."},
                status_code=200
            )

        context_text = "\n\n".join([doc.get("body_blob", "") for doc in top_docs])

        # Prepare system prompt
        system_prompt = (
            "⚠️ You are a Mine Disaster Survival Assistant. "
            "Always respond with clear, step-by-step survival guidance for trapped miners. "
            "Use the provided documents if available. "
            "Format answers as a numbered list of survival steps.\n\n{context}"
        ).format(context=context_text)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        question_answer_chain = qa_prompt | get_llm()
        conversational_chain = RunnableWithMessageHistory(
            question_answer_chain,
            lambda _: history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        resp = conversational_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )

        answer = resp.content if hasattr(resp, "content") else str(resp)
        return {"answer": answer}

    except Exception as e:
        return JSONResponse(
            {"error": f"Server Error: {str(e)}"},
            status_code=500
        )


# =========================
# 10️⃣ Run server
# =========================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
