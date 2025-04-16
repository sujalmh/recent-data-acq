from fastapi import FastAPI, UploadFile, Query, File, Form, Depends, HTTPException
from pydantic import Field, BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
import hashlib, sqlite3, sys, os, requests, logging, asyncio
from crawl4ai import *
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Milvus
import logging
import httpx

from dotenv import load_dotenv
load_dotenv()

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

logging.basicConfig(filename="logs.txt", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from fastapi import FastAPI
from contextlib import asynccontextmanager
from pymilvus import connections

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: connect to Milvus
    connections.connect("default", host="localhost", port="19530")
    print("Connected to Milvus.")

    yield  # Application runs during this time

    # Shutdown: disconnect from Milvus
    connections.disconnect("default")
    print("Disconnected from Milvus.")


app = FastAPI(lifespan=lifespan)


API_KEY = os.getenv("RD_API_KEY")
API_KEY_NAME = "access_token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

PDF_URLS_DB = "pdfs.db"
PDF_SAVE_DIR = "pdfs"
URLS = [
    "https://www.rbi.org.in/Scripts/NotificationUser.aspx",
    "https://www.rbi.org.in/Scripts/HalfYearlyPublications.aspx?head=Monetary%20Policy%20Report",
    "https://www.rbi.org.in/Scripts/BimonthlyPublications.aspx?head=Survey%20of%20Professional%20Forecasters%20-%20Bi-monthly",
    "https://www.rbi.org.in/Scripts/BimonthlyPublications.aspx?head=Inflation%20Expectations%20Survey%20of%20Households%20-%20Bi-monthly",
    "https://www.rbi.org.in/Scripts/BimonthlyPublications.aspx?head=Consumer%20Confidence%20Survey%20-%20Bi-monthly",
    "https://www.rbi.org.in/Scripts/BimonthlyPublications.aspx?head=Rural%20Consumer%20Confidence%20Survey%20-%20Bi-monthly",
    "https://www.rbi.org.in/Scripts/AnnualPublications.aspx?head=Handbook%20of%20Statistics%20on%20Indian%20Economy",
    "https://www.rbi.org.in/Scripts/AnnualPublications.aspx?head=Handbook%20of%20Statistics%20on%20Indian%20States"
]

class PDFLinkOutput(BaseModel):
    pdf_links: List[str] = Field(description="List of PDF URLs updated or posted yesterday")

def get_relevant_links_from_markdown(markdown_text, date):
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    chunks = splitter.split_text(markdown_text)

    prompt = PromptTemplate(
        input_variables=["text", "date"],
        template="""
You are a helpful assistant. From the following markdown of links and descriptions, extract only the PDF URLs that were updated or posted on {date}. Analyse from the text surrounding the pdf link
Strictly return pdf links from {date} only. Do not return any other links or text.
Markdown:
{text}

Return only the matching links as a list of URLs.
"""
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(PDFLinkOutput)

    chain = prompt | structured_llm

    relevant_links = []
    
    date_str = (date - timedelta(days=1)).strftime("%B %d %Y")
    
    for chunk in chunks:
        result = chain.invoke({"text": chunk, "date": date_str})
        relevant_links.extend(result.pdf_links)

    return relevant_links

def init_db():
    conn = sqlite3.connect(PDF_URLS_DB)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pdfs (
        url TEXT PRIMARY KEY,
        hash TEXT,
        indexed BOOLEAN DEFAULT 0,
        path TEXT
    )""")
    conn.commit()
    return conn 

def has_been_downloaded(conn, url):
    return conn.execute("SELECT 1 FROM pdfs WHERE url = ?", (url,)).fetchone()

def save_pdf_metadata(conn, url, content, path):
    hash_digest = hashlib.sha256(content).hexdigest()
    conn.execute(
        "INSERT OR IGNORE INTO pdfs (url, hash, indexed, path) VALUES (?, ?, ?, ?)",
        (url, hash_digest, False, path)
    )
    conn.commit()

async def download_pdf(conn, url):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=15)
        if response.status_code == 200 and not has_been_downloaded(conn, url):
            os.makedirs(PDF_SAVE_DIR, exist_ok=True)
            filename = os.path.join(PDF_SAVE_DIR, os.path.basename(url))
            with open(filename, "wb") as f:
                f.write(response.content)
            save_pdf_metadata(conn, url, response.content, filename)
            logging.info(f"✅ Downloaded: {filename}")
    except Exception as e:
        logging.error(f"❌ Failed to download {url}: {e}")

async def extract_pdfs_from_url(url, conn, crawler, date):
    await crawler.start()
    result = await crawler.arun(url=url)
    await crawler.close()
    markdown_text = result.markdown

    relevant_links = get_relevant_links_from_markdown(markdown_text, date)

    for link in relevant_links:
        if link and not link.startswith("http"):
            from urllib.parse import urljoin
            link = urljoin(url, link)
        if link:
            await download_pdf(conn, link)

async def scrape_and_download(date):
    conn = init_db()

    async with AsyncWebCrawler() as crawler:
        for url in URLS:
            try:
                
                await extract_pdfs_from_url(url, conn, crawler, date)
                
            except Exception as e:
                logging.warning(f"⚠️ Error crawling {url}: {e}")
    conn.close()

def sync_scrape_and_download(date):
    result = asyncio.run(scrape_and_download(date))
    return result

NEW_COLLECTION_NAME = "recent_pdf_embeddings"


def drop_collection(collection_name: str):
    connections.connect(host="localhost", port="19530")
    
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"Dropped existing collection: {collection_name}")


def setup_collection(collection_name: str):
    if utility.has_collection(collection_name):
        return

    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    }

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, index_params=index_params, dim=1536),
        FieldSchema(name="page_content", dtype=DataType.VARCHAR, max_length=65535),
    ]

    schema = CollectionSchema(fields=fields, description="PDF Embedding Collection")
    Collection(name=collection_name, schema=schema)


@app.post("/index", dependencies=[Depends(verify_api_key)])
def reindex_new_pdfs():
    setup_collection(NEW_COLLECTION_NAME)

    conn = sqlite3.connect(PDF_URLS_DB)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT url, hash, path FROM pdfs WHERE indexed = 0")
    except sqlite3.Error as e:
        conn.close()
        print(f"SQLite error: {e}")
        return {"error": "Database query failed."}
    
    pdf_entries = cursor.fetchall()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = []

    for url, pdf_hash, path in pdf_entries:
        pdf_filename = path
        if not os.path.exists(pdf_filename):
            continue
        try:
            loader = PyPDFLoader(pdf_filename)
            docs = loader.load()
            splits = text_splitter.split_documents(docs)

            for doc in splits:
                doc.metadata["page_content"] = doc.page_content

            all_splits.extend(splits)
            cursor.execute("UPDATE pdfs SET indexed = 1 WHERE hash = ?", (pdf_hash,))
        except Exception as e:
            print(f"Failed to process {url}: {e}")
            continue

    conn.commit()
    conn.close()

    if all_splits:
        batch_size = 500
        for i in range(0, len(all_splits), batch_size):
            batch = all_splits[i:i + batch_size]
            try:
                Milvus.from_documents(
                    documents=batch,
                    embedding=OpenAIEmbeddings(),
                    collection_name=NEW_COLLECTION_NAME,
                    connection_args=None,
                    vector_field="embedding",
                    text_field="page_content",
                )
            except Exception as e:
                conn.rollback()
                return {"message": f"Failed to insert batch {i}: {e}"}
        return {"message": f"Indexed {len(all_splits)} chunks into {NEW_COLLECTION_NAME}."}
    else:
        return {"message": "No new PDFs to index."}


@app.post("/ask", dependencies=[Depends(verify_api_key)])
async def ask_question_new(
    question: str = Form(...)
):
    if not utility.has_collection(NEW_COLLECTION_NAME):
        return JSONResponse(status_code=400, content={"error": "No documents uploaded yet."})

    setup_collection(NEW_COLLECTION_NAME)
    
    vectorstore = Milvus(
        embedding_function=OpenAIEmbeddings(),
        collection_name=NEW_COLLECTION_NAME,
        connection_args=None,
        vector_field="embedding",
        text_field="page_content",
    )

    logger.info(f"[ASK_NEW] Query: {question}")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})  # Fetch more to de-duplicate
    all_docs = retriever.get_relevant_documents(question)
    logger.debug(f"[ASK_NEW] Retrieved {len(all_docs)} documents")

    # Filter for unique page content
    seen = set()
    unique_docs = []
    for doc in all_docs:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            unique_docs.append(doc)
        if len(unique_docs) == 5:
            break

    if not unique_docs:
        return {
            "query": question,
            "top_5_chunks": [],
            "message": "No relevant documents found."
        }

    return {
        "query": question,
        "top_5_chunks": [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in unique_docs
        ]
    }

@app.post("/run", dependencies=[Depends(verify_api_key)])
def run_scraper(date: Optional[str] = Query(None)):
    if date:
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return {"error": "Invalid date format. Use YYYY-MM-DD."}
    else:
        date_obj = datetime.now()

    formatted_date = date_obj.strftime("%B %d %Y")

    sync_scrape_and_download(date_obj)
    return {"status": "Scraping and PDF download task completed", "date": formatted_date}
