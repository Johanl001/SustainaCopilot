import os
import uvicorn
import json
import sqlite3
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
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
retriever = vector_db.as_retriever()

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

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/analyze-report")
async def analyze_report_endpoint(file: UploadFile = File(...)):
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    loader = PyPDFLoader(file_path)
    documents = loader.load_and_split()
    
    db = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")
    
    # Use LLM to generate summary and actions
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Sustainability Officer. Summarize the following report in 500 words and generate 3-5 actionable bullet points for employees. Use a professional but easy-to-read tone. Report: {report}"),
        ("user", "{input}"),
    ])
    
    summary_chain = create_stuff_documents_chain(llm, summary_prompt)
    response = summary_chain.invoke({"input": file.filename, "report": documents})

    # Save to database
    cursor.execute("INSERT INTO reports (title, summary, actions) VALUES (?, ?, ?)", (file.filename, response['summary'], response['actions']))
    conn.commit()

    return {"message": "Report analyzed successfully."}

@app.post("/get-report-summary")
async def get_report_summary(query: QueryModel):
    # Retrieve relevant report chunks based on query
    docs = retriever.get_relevant_documents(query.query)
    
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant for a Sustainability Copilot. Summarize the following document sections and provide a 500-word overview. Context: {context}"),
        ("user", "{input}"),
    ])
    
    summary_chain = create_stuff_documents_chain(llm, summary_prompt)
    response = summary_chain.invoke({"context": docs, "input": "Generate a report summary."})

    # You would need to add logic here to extract title and memo from the response
    # For a simple solution, we'll return the full response as the summary
    return ReportSummaryModel(title="Report Summary", summary=response, memo="Actionable Memo for Employees")