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

# Load .env variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN_RAG")
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_KEY_RAG")

ASTRA_API_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_ENDPOINT = os.getenv("ASTRA_VECTOR_DB_ENDPOINT")
ASTRA_TABLE = "qa_sih_demo"

app = FastAPI(title="⛑️ Mine Survival Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-load globals
embedding_model = None
llm = None
collection = None
chat_histories = {}

# Safe Astra connection
try:
    if ASTRA_API_TOKEN and ASTRA_ENDPOINT:
        client = DataAPIClient(ASTRA_API_TOKEN)
        db = client.get_database_by_api_endpoint(ASTRA_ENDPOINT)
        collection = db.get_collection(ASTRA_TABLE)
        print(f"✅ Connected to Astra DB collection: {ASTRA_TABLE}")
    else:
        print("⚠️ Astra DB credentials missing, skipping DB connection.")
except Exception as e:
    print(f"❌ Failed to connect to Astra DB: {e}")

# Lazy model getters
def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print("🔄 Loading HuggingFace Embeddings...")
        embedding_model = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
        print("✅ Embeddings loaded")
    return embedding_model

def get_llm():
    global llm
    if llm is None:
        print("🔄 Loading Gemini LLM...")
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        print("✅ LLM loaded")
    return llm

# Models
class AskRequest(BaseModel):
    session_id: str
    question: str

# Helpers
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_top_docs(query, k=3, limit=200):
    if collection is None:
        return []
    try:
        query_vector = np.array(get_embedding_model().embed_query(query))
        all_docs = collection.find({}, limit=limit)
    except Exception as e:
        print(f"❌ Astra query failed: {e}")
        return []
    scores = []
    for doc in all_docs:
        vector = doc.get("vector")
        if not vector:
            continue
        sim = cosine_similarity(query_vector, np.array(vector))
        scores.append((sim, doc))
    scores.sort(reverse=True, key=lambda x: x[0])
    return [doc for sim, doc in scores[:k]]

# Routes
@app.get("/")
def root():
    return {"status": "running", "message": "⛑️ Mine Survival Assistant API is up and healthy!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
async def ask_question(req: AskRequest):
    try:
        print(f"📝 Question: {req.question}")
        session_id = req.session_id
        if session_id not in chat_histories:
            chat_histories[session_id] = ChatMessageHistory()
        history = chat_histories[session_id]

        top_docs = get_top_docs(req.question)
        if not top_docs:
            return {"answer": "⚠️ No relevant documents found for this query."}

        context_text = "\n\n".join([doc.get("body_blob", "") for doc in top_docs])
        system_prompt = (
            "⚠️ You are a Mine Disaster Survival Assistant. "
            "Always respond with clear, step-by-step survival guidance for trapped miners. "
            "Use the provided documents if available. "
            "Format answers as a numbered list of survival steps.\n\n"
            f"{context_text}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        chain = qa_prompt | get_llm()
        conversational_chain = RunnableWithMessageHistory(
            chain,
            lambda _: history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        resp = conversational_chain.invoke(
            {"input": req.question},
            config={"configurable": {"session_id": session_id}}
        )

        answer = resp.content if hasattr(resp, "content") else str(resp)
        print(f"✅ Answer: {answer[:120]}...")
        return {"answer": answer}

    except Exception as e:
        print(f"❌ Error in /ask: {e}")
        return JSONResponse({"error": f"Server Error: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
