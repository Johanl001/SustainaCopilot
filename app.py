import os
import uvicorn
import json
import sqlite3
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

# --- Database setup ---
db_path = "database.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS reports (id INTEGER PRIMARY KEY, title TEXT, summary TEXT, actions TEXT)")
conn.commit()

# --- LLM and Embedding setup ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0.3)
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 12})

# --- Pydantic models ---
class QueryModel(BaseModel):
    query: str

class ReportResponseModel(BaseModel):
    title: str
    summary: str
    actions: str
    
class ReportSummaryModel(BaseModel):
    title: str
    summary: str
    memo: str

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("shutdown")
def shutdown_event():
    try:
        conn.close()
    except Exception:
        pass

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/analyze-report")
async def analyze_report_endpoint(file: UploadFile = File(...)):
    try:
        # Validate and persist upload
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported at this time.")
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
        original_filename = os.path.basename(file.filename) or "uploaded.pdf"
        file_path = os.path.join("uploads", original_filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        # Load and split PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load_and_split()

        # Ingest into existing persistent vector store
        vector_db.add_documents(documents)

        # Use LLM to generate structured summary and actions
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", 'You are a Sustainability Officer. Using the provided context, write a 500-word summary and 3-5 actionable bullet points for employees. Return STRICT JSON with keys "summary" (string) and "actions" (array of strings). Context: {context}'),
            ("user", "{input}"),
        ])
        summary_chain = create_stuff_documents_chain(llm, summary_prompt)
        context_text = "\n\n".join([d.page_content for d in documents[:8]])
        raw_response = summary_chain.invoke({"input": original_filename, "context": context_text})

        if isinstance(raw_response, str):
            response_text = raw_response
        else:
            response_text = getattr(raw_response, "content", str(raw_response))

        try:
            data = json.loads(response_text)
            summary_text = data.get("summary", "")
            actions_text = json.dumps(data.get("actions", []))
        except Exception:
            # Fallback: store raw text as summary when JSON parsing fails
            summary_text = response_text
            actions_text = json.dumps([])

        # Save to database
        cursor.execute(
            "INSERT INTO reports (title, summary, actions) VALUES (?, ?, ?)",
            (original_filename, summary_text, actions_text),
        )
        conn.commit()

        return {"message": "Report analyzed successfully."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-report-summary")
async def get_report_summary(query: QueryModel):
    try:
        # Retrieve relevant report chunks based on query
        docs = retriever.invoke(query.query) or []
        context_text = "\n\n".join([d.page_content for d in docs]) if docs else ""
        
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant for a Sustainability Copilot. Summarize the following document sections and provide a 500-word overview. Context: {context}"),
            ("user", "{input}"),
        ])
        
        summary_chain = create_stuff_documents_chain(llm, summary_prompt)
        raw_response = summary_chain.invoke({"context": context_text, "input": "Generate a report summary."})

        if isinstance(raw_response, str):
            summary_text = raw_response
        else:
            summary_text = getattr(raw_response, "content", str(raw_response))

        return ReportSummaryModel(title="Report Summary", summary=summary_text, memo="Actionable Memo for Employees")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)

