import os
import json
import pickle
import asyncio
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
import openai
from openai import OpenAI
import glob
import re
import uuid
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("rag_system.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger("rag_system")

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define constants
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072  # Dimensions for text-embedding-3-large
VECTOR_STORE_PATH = "data/vector_store.pkl"
DOCS_DIR = "data/documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

class Document:
    """Class to represent a document with metadata and content."""
    def __init__(self, 
                 content: str, 
                 metadata: Dict[str, Any] = None, 
                 doc_id: str = None):
        self.content = content
        self.metadata = metadata or {}
        self.doc_id = doc_id or str(uuid.uuid4())
        self.embedding = None

class VectorStore:
    """A simple vector store implementation to save and retrieve document embeddings."""
    def __init__(self, embedding_dimensions: int = EMBEDDING_DIMENSIONS):
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None
        self.embedding_dimensions = embedding_dimensions
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store."""
        self.documents.extend(documents)
        
    def set_embeddings(self, embeddings: List[List[float]]):
        """Set embeddings for the documents."""
        if not self.embeddings:
            self.embeddings = np.array(embeddings)
        else:
            self.embeddings = np.vstack([self.embeddings, np.array(embeddings)])
    
    def save(self, path: str = VECTOR_STORE_PATH):
        """Save the vector store to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Vector store saved to {path} with {len(self.documents)} documents")
    
    @staticmethod
    def load(path: str = VECTOR_STORE_PATH) -> 'VectorStore':
        """Load the vector store from disk."""
        try:
            with open(path, 'rb') as f:
                vector_store = pickle.load(f)
            logger.info(f"Vector store loaded from {path} with {len(vector_store.documents)} documents")
            return vector_store
        except (FileNotFoundError, EOFError):
            logger.warning(f"Vector store not found at {path}, creating new one")
            return VectorStore()
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Document, float]]:
        """Search for documents similar to the query embedding."""
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        # Convert query embedding to numpy array
        query_embedding_np = np.array(query_embedding)
        
        # Calculate cosine similarity
        dot_product = np.dot(self.embeddings, query_embedding_np)
        query_norm = np.linalg.norm(query_embedding_np)
        doc_norm = np.linalg.norm(self.embeddings, axis=1)
        cosine_similarities = dot_product / (query_norm * doc_norm)
        
        # Get indices of top k most similar documents
        top_indices = np.argsort(cosine_similarities)[::-1][:top_k]
        
        # Return documents and their similarity scores
        return [(self.documents[i], float(cosine_similarities[i])) for i in top_indices]

def get_embedding(text: str) -> List[float]:
    """Get embedding for text using OpenAI's embedding API."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        dimensions=EMBEDDING_DIMENSIONS
    )
    return response.data[0].embedding

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end > len(text):
            end = len(text)
        
        # Find a good breaking point (e.g., end of sentence or paragraph)
        if end < len(text):
            # Try to find the end of a sentence or paragraph
            for punct in ['\n\n', '\n', '.', '!', '?', ';']:
                last_punct = text.rfind(punct, start, end)
                if last_punct != -1:
                    end = last_punct + 1
                    break
        
        chunks.append(text[start:end])
        start = end - chunk_overlap
    
    return chunks

async def process_csv_file(file_path: str) -> List[Document]:
    """Process a CSV file into Documents."""
    documents = []
    try:
        df = pd.read_csv(file_path)
        
        # Process each row as a document
        for idx, row in df.iterrows():
            # Convert row to string representation
            content = "\n".join([f"{col}: {val}" for col, val in row.items()])
            
            # Create document with metadata
            metadata = {
                "source": file_path,
                "type": "csv",
                "row": idx,
                "columns": list(df.columns)
            }
            
            documents.append(Document(content=content, metadata=metadata))
            
        # Also create a document for the CSV schema
        schema_content = "CSV Schema:\n" + "\n".join([f"Column: {col}, Type: {df[col].dtype}" for col in df.columns])
        schema_metadata = {
            "source": file_path,
            "type": "csv_schema",
            "num_rows": len(df),
            "columns": list(df.columns)
        }
        documents.append(Document(content=schema_content, metadata=schema_metadata))
        
        # Create a summary document with basic stats
        summary_content = f"CSV Summary for {os.path.basename(file_path)}:\n"
        summary_content += f"Total rows: {len(df)}\n"
        summary_content += f"Columns: {', '.join(df.columns)}\n"
        
        # Add basic statistics for numeric columns
        for col in df.select_dtypes(include=['number']).columns:
            summary_content += f"\nColumn {col} statistics:\n"
            summary_content += f"Min: {df[col].min()}\n"
            summary_content += f"Max: {df[col].max()}\n"
            summary_content += f"Mean: {df[col].mean()}\n"
            summary_content += f"Median: {df[col].median()}\n"
        
        summary_metadata = {
            "source": file_path,
            "type": "csv_summary",
            "num_rows": len(df),
            "columns": list(df.columns)
        }
        documents.append(Document(content=summary_content, metadata=summary_metadata))
        
    except Exception as e:
        logger.error(f"Error processing CSV file {file_path}: {e}")
    
    return documents

async def process_pdf_file(file_path: str) -> List[Document]:
    """Process a PDF file into Documents."""
    documents = []
    try:
        # This is a placeholder for PDF processing logic
        # In a real implementation, you would use a library like PyPDF2 or pdfplumber
        # For this example, we'll just use a placeholder
        
        # Import PyPDF2 conditionally to not require it for this example
        try:
            import PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Process each page
                full_text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    full_text += page_text + "\n\n"
                    
                    # Create a document for each page
                    metadata = {
                        "source": file_path,
                        "type": "pdf",
                        "page": page_num + 1,
                        "total_pages": len(pdf_reader.pages)
                    }
                    documents.append(Document(content=page_text, metadata=metadata))
                
                # Chunk the full document text
                chunks = chunk_text(full_text)
                for i, chunk in enumerate(chunks):
                    metadata = {
                        "source": file_path,
                        "type": "pdf_chunk",
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                    documents.append(Document(content=chunk, metadata=metadata))
                
        except ImportError:
            # If PyPDF2 is not available, create a placeholder document
            metadata = {
                "source": file_path,
                "type": "pdf",
                "note": "PDF processing not available - PyPDF2 not installed"
            }
            documents.append(Document(
                content=f"This is a placeholder for the content of {file_path}. "
                        f"Install PyPDF2 to process PDF files.",
                metadata=metadata
            ))
        
    except Exception as e:
        logger.error(f"Error processing PDF file {file_path}: {e}")
    
    return documents

async def index_documents(docs_dir: str = DOCS_DIR, force_reindex: bool = False) -> VectorStore:
    """Index documents from the docs directory."""
    # Check if vector store already exists
    if os.path.exists(VECTOR_STORE_PATH) and not force_reindex:
        return VectorStore.load()
    
    # Create vector store
    vector_store = VectorStore()
    
    # Create docs directory if it doesn't exist
    os.makedirs(docs_dir, exist_ok=True)
    
    # Find all CSV and PDF files
    csv_files = glob.glob(os.path.join(docs_dir, "**", "*.csv"), recursive=True)
    pdf_files = glob.glob(os.path.join(docs_dir, "**", "*.pdf"), recursive=True)
    
    logger.info(f"Found {len(csv_files)} CSV files and {len(pdf_files)} PDF files")
    
    # Process CSV files
    all_documents = []
    for file_path in csv_files:
        documents = await process_csv_file(file_path)
        all_documents.extend(documents)
    
    # Process PDF files
    for file_path in pdf_files:
        documents = await process_pdf_file(file_path)
        all_documents.extend(documents)
    
    # Add documents to vector store
    vector_store.add_documents(all_documents)
    
    # Calculate embeddings for all documents
    logger.info(f"Calculating embeddings for {len(all_documents)} documents")
    embeddings = []
    for doc in all_documents:
        embedding = get_embedding(doc.content)
        doc.embedding = embedding
        embeddings.append(embedding)
    
    # Set embeddings in vector store
    vector_store.set_embeddings(embeddings)
    
    # Save vector store
    vector_store.save()
    
    return vector_store

class RAGSystem:
    """
    RAG system for retrieving and answering questions about datasets and documentation.
    """
    def __init__(self, domain_description: str):
        """
        Initialize the RAG system.
        
        Args:
            domain_description: Description of the domain
        """
        self.domain_description = domain_description
        self.vector_store = None
    
    async def ensure_indexed(self):
        """Ensure documents are indexed in the vector store."""
        if self.vector_store is None:
            self.vector_store = await index_documents()
    
    async def retrieve_relevant_documents(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Retrieve documents relevant to the query.
        
        Args:
            query: The user's query
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        await self.ensure_indexed()
        
        # Get query embedding
        query_embedding = get_embedding(query)
        
        # Search for similar documents
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        return [doc for doc, score in results]
    
    async def answer_query(self, query: str) -> Tuple[str, str]:
        """
        Answer a query using the RAG system.
        
        Args:
            query: The user's query
            
        Returns:
            Tuple of (reasoning, response)
        """
        # Retrieve relevant documents
        relevant_docs = await self.retrieve_relevant_documents(query)
        
        if not relevant_docs:
            return "No relevant documents found", "<p>I don't have enough information to answer that question. Could you provide more details?</p>"
        
        # Prepare context from relevant documents
        context = "\n\n".join([f"Document {i+1}:\n{doc.content}" for i, doc in enumerate(relevant_docs)])
        
        # Get metadata for reasoning
        metadata = [doc.metadata for doc in relevant_docs]
        
        # Prepare the prompt for OpenAI
        prompt = f"""
        You are an AI assistant helping with datasets and documentation for an XAI (Explainable AI) system.
        
        User Query: {query}
        
        Based on the following documents, please provide a clear and accurate response to the user's query.
        
        {context}
        
        Domain Description: {self.domain_description}
        
        Provide a concise, HTML-formatted response that directly answers the user's question.
        """
        
        # Get completion from OpenAI
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant that helps with questions about datasets and documentation."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract and format the response
        response_text = response.choices[0].message.content
        
        # Ensure the response is HTML formatted
        if not response_text.strip().startswith("<"):
            response_text = f"<p>{response_text}</p>"
        
        # Generate reasoning
        reasoning = f"Retrieved {len(relevant_docs)} relevant documents: {json.dumps(metadata, indent=2)}"
        
        return reasoning, response_text

# Test function
async def test_rag_system():
    """Test the RAG system."""
    # Create a test CSV file if it doesn't exist
    test_csv_path = os.path.join(DOCS_DIR, "test.csv")
    os.makedirs(os.path.dirname(test_csv_path), exist_ok=True)
    
    if not os.path.exists(test_csv_path):
        df = pd.DataFrame({
            "age": [25, 30, 35, 40, 45],
            "income": [30000, 50000, 70000, 90000, 110000],
            "education": ["High School", "Bachelors", "Masters", "PhD", "Bachelors"],
            "occupation": ["Clerk", "Manager", "Director", "Executive", "Engineer"]
        })
        df.to_csv(test_csv_path, index=False)
        print(f"Created test CSV file at {test_csv_path}")
    
    # Initialize RAG system
    rag_system = RAGSystem(domain_description="Income prediction model using census data")
    
    # Test queries
    queries = [
        "What is the average income in the dataset?",
        "How many people have a Bachelor's degree?",
        "What is the relationship between education and income?",
        "What columns are available in the dataset?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        reasoning, response = await rag_system.answer_query(query)
        print(f"Reasoning: {reasoning}")
        print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(test_rag_system()) 