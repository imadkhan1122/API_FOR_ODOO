from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from fastapi import Form
from pydantic import BaseModel
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
import os
import re
import logging

# Initialize FastAPI app
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.kmizeolite.com"],  # Allow only your frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)
# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Define a request model for the API key
class OpenAIKeyRequest(BaseModel):
    openai_api_key: str

class Document:
    """Class to represent a PDF document with metadata and page content."""
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

def extract_pdf_metadata(pdf_file):
    """Extract metadata from the PDF document."""
    metadata = {}
    try:
        with open(pdf_file, 'rb') as f:
            parser = PDFParser(f)
            document = PDFDocument(parser)
            if document.info:
                metadata = document.info[0]  # Extract metadata from the PDF
    except Exception as e:
        logging.error(f"Error extracting metadata: {e}")
    return metadata

def clean_text(text):
    """Remove URLs, references, tables, and other undesired elements from the text."""
    try:
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        
        # Remove references (e.g., [1], (1))
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\(\d+\)', '', text)
        
        # Remove extra spaces or tabs
        text = re.sub(r'(\s{2,}|\t)', ' ', text)
        
        # Remove figure/table captions (e.g., "Figure 1:", "Table 2:")
        text = re.sub(r'(Figure \d+:|Table \d+:)', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    except Exception as e:
        logging.error(f"Error cleaning text: {e}")
        return text

def extract_pdf_text_by_page(pdf_file):
    """Extract text from each page of a PDF file using high-level API."""
    try:
        text = extract_text(pdf_file)  # Extracts the entire text of the PDF
        pages = text.split('\f')  # Split text by form feed character (page breaks)
        return [clean_text(page) for page in pages[:4] if page.strip()]  # Clean each page's text
    except Exception as e:
        logging.error(f"Error extracting text: {e}")
        return [] 

def process_pdf(pdf_file):
    """Extract metadata and page-by-page text from a PDF document."""
    metadata = extract_pdf_metadata(pdf_file)
    text_pages = extract_pdf_text_by_page(pdf_file)
          
    # Create a Document object for each page
    documents = [Document(page_content=page, metadata=metadata) for page in text_pages]
    return documents

def get_summary(doc_objs, openai_api_key: str):
    # Define prompt
    prompt_template = """Please analyze the following text and provide the following details:
                        1.Clearly identify and state the title of the article or chapter.
                        2. Write a detailed, comprehensive, and coherent summary that thoroughly captures the main points, key insights, supporting details, and any critical arguments presented in the text. The summary should ensure no significant information is missed and should be proportionate to the text length.
                        3. Explain the relevance and significance of the article, including its importance to its field or topic, how it contributes to ongoing discussions, and the benefits it offers to readers.
                        4. Extract keywords if keywords of the text.
                        Analyze the following text \n {text}:
                        CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    # Define LLM chain
    llm = ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=openai_api_key)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
                                                                                             
    return stuff_chain.run(doc_objs)

@app.post("/upload-pdf/")
async def upload_pdf(
    file: UploadFile = File(..., max_size=10_000_000),  # 10 MB limit
    openai_api_key: str = Form(...)  # Accept the API key in the request body
):
    try:
        logging.debug(f"Received file: {file.filename}")
        
        # Save the uploaded file temporarily
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        logging.debug(f"File saved to: {file_path}")
        
        # Process the PDF
        documents = process_pdf(file_path)
        
        # Generate summary using the OpenAI API key from the request body
        summary = get_summary(documents, openai_api_key)
        
        # Clean up the temporary file
        os.remove(file_path)
        
        return JSONResponse(content={"summary": summary})
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
