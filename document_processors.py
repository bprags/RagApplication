from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import chromadb
from chromadb.config import Settings
import hashlib
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np
import os

from dotenv import load_dotenv
load_dotenv()

@dataclass
class DocumentMetadata:
    """Metadata for processed documents"""
    source_type: str  # 'html', 'pdf'
    source_path: str
    title: str
    created_at: datetime
    modified_at: datetime
    content_hash: str
    chunk_count: int
    file_size: int
    additional_metadata: Dict[str, Any] = None

@dataclass 
class ProcessedChunk:
    """Represents a processed document chunk"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    document_id: str
    chunk_index: int
    embedding: Optional[List[float]] = None

class EmbeddingGenerator(ABC):
    """Abstract base class for embedding generators."""

    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        pass

class SentenceTransformerEmbeddingGenerator(EmbeddingGenerator):
    """Embedding generator using SentenceTransformer."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        self._embedding_dimension = self._model.get_sentence_embedding_dimension()
        print(f"Loaded SentenceTransformer model: {model_name} (dimension: {self._embedding_dimension})")

    def generate_embedding(self, text: str) -> List[float]:
        try:
            embedding = self._model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return [0.0] * self._embedding_dimension

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self._model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            return [[0.0] * self._embedding_dimension] * len(texts)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

class OpenAIEmbeddingGenerator(EmbeddingGenerator):
    """Embedding generator using OpenAI API."""

    def __init__(self, model_name: str = "text-embedding-3-small"):
        import openai
        import os
        self._openai = openai
        self._api_key = os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self._openai.api_key = self._api_key
        self._model_name = model_name
        self._embedding_dimension = 1536 if "small" in model_name else 3072

    def generate_embedding(self, text: str) -> List[float]:
        try:
            response = self._openai.embeddings.create(
                input=text,
                model=self._model_name
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return [0.0] * self._embedding_dimension

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            response = self._openai.embeddings.create(
                input=texts,
                model=self._model_name
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            return [[0.0] * self._embedding_dimension] * len(texts)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

class GeminiEmbeddingGenerator(EmbeddingGenerator):
    """Embedding generator using Gemini API."""

    def __init__(self, model_name: str = "models/embedding-001"):
        import google.generativeai as genai
        import os
        self._genai = genai
        self._api_key = os.getenv("GOOGLE_API_KEY")
        if not self._api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        self._genai.configure(api_key=self._api_key)
        self._model_name = model_name
        self._embedding_dimension = 768

    def generate_embedding(self, text: str) -> List[float]:
        try:
            response = self._genai.embed_content(
                model=self._model_name,
                content=text,
                task_type="retrieval_document"
            )
            return response["embedding"]
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return [0.0] * self._embedding_dimension

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        # Gemini API does not support batch embedding as of now; process sequentially
        return [self.generate_embedding(text) for text in texts]

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

class BaseDocumentProcessor(ABC):
    """Abstract base class for document processors"""
    
    def __init__(self, chroma_client: chromadb.PersistentClient, embedding_generator: EmbeddingGenerator):
        self.chroma_client = chroma_client
        self.embedding_generator = embedding_generator
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
    @abstractmethod
    def extract_content(self, file_path: Union[str, Path]) -> str:
        """Extract raw content from the document"""
        pass
    
    @abstractmethod
    def clean_content(self, raw_content: str) -> str:
        """Clean and preprocess the extracted content"""
        pass
    
    @abstractmethod
    def extract_metadata(self, file_path: Union[str, Path]) -> DocumentMetadata:
        """Extract metadata from the document"""
        pass
    
    @abstractmethod
    def get_collection_name(self) -> str:
        """Return the ChromaDB collection name for this processor"""
        pass
    
    def generate_chunks(self, content: str, document_id: str) -> List[ProcessedChunk]:
        """Split content into chunks for vector storage"""
        chunks = []
        words = content.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_content = " ".join(chunk_words)
            
            chunk_id = hashlib.md5(f"{document_id}_{i}".encode()).hexdigest()
            
            chunk = ProcessedChunk(
                content=chunk_content,
                metadata={
                    "document_id": document_id,
                    "chunk_index": len(chunks),
                    "start_word": i,
                    "end_word": min(i + self.chunk_size, len(words))
                },
                chunk_id=chunk_id,
                document_id=document_id,
                chunk_index=len(chunks),
                embedding=None  # Will be generated later
            )
            chunks.append(chunk)
            
        return chunks
    
    def generate_embeddings_for_chunks(self, chunks: List[ProcessedChunk]) -> None:
        """Generate embeddings for all chunks in batch"""
        print(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Extract text content for batch processing
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings in batch (more efficient)
        embeddings = self.embedding_generator.generate_embeddings_batch(texts)
        
        # Assign embeddings back to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        print(f"Generated {len(embeddings)} embeddings successfully")
    
    def process_document(self, file_path: Union[str, Path]) -> List[ProcessedChunk]:
        """Main processing pipeline for a document"""
        # Extract content
        raw_content = self.extract_content(file_path)
        
        # Clean content
        cleaned_content = self.clean_content(raw_content)
        
        # Generate document ID
        document_id = hashlib.md5(str(file_path).encode()).hexdigest()
        
        # Generate chunks
        chunks = self.generate_chunks(cleaned_content, document_id)
        
        # Generate embeddings for chunks
        self.generate_embeddings_for_chunks(chunks)
        
        # Extract and attach metadata
        metadata = self.extract_metadata(file_path)
        for chunk in chunks:
            chunk.metadata.update({
                "source_type": metadata.source_type,
                "source_path": metadata.source_path,
                "title": metadata.title,
                "created_at": metadata.created_at.isoformat(),
                "content_hash": metadata.content_hash,
                "embedding_model": self.embedding_generator.model_name
            })
            
        return chunks
    
    def index_chunks(self, chunks: List[ProcessedChunk]) -> bool:
        """Index chunks into ChromaDB collection with embeddings"""
        
        try:
             self.chroma_client.delete_collection(name=self.get_collection_name())
             print(f"Removed existing collection {self.get_collection_name()}")
        except Exception as e:
            print(f"Error removing collection {self.get_collection_name()}: {e}")

        try:
            collection = self.chroma_client.get_or_create_collection(
                name=self.get_collection_name()
            )
            
            # Prepare data for batch insertion
            documents = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            ids = [chunk.chunk_id for chunk in chunks]
            embeddings = [chunk.embedding for chunk in chunks]
            
            # Validate embeddings
            if any(emb is None for emb in embeddings):
                print("Warning: Some chunks missing embeddings")
                return False
            
            # Add to collection with embeddings
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            
            print(f"Successfully indexed {len(chunks)} chunks with embeddings to {self.get_collection_name()}")
            return True
            
        except Exception as e:
            print(f"Error indexing chunks: {e}")
            return False

class HTMLDocumentProcessor(BaseDocumentProcessor):
    """Processor for HTML documents (Confluence/Wiki)"""
    
    def __init__(self, chroma_client: chromadb.PersistentClient, embedding_generator: EmbeddingGenerator):
        super().__init__(chroma_client, embedding_generator)
        try:
            from bs4 import BeautifulSoup
            import requests
            self.BeautifulSoup = BeautifulSoup
            self.requests = requests
        except ImportError:
            raise ImportError("Required packages: pip install beautifulsoup4 requests")
    
    def extract_content(self, file_path: Union[str, Path]) -> str:
        """Extract content from HTML file or URL"""
        if str(file_path).startswith(('http://', 'https://')):
            # Handle URL
            response = self.requests.get(str(file_path))
            response.raise_for_status()
            return response.text
        else:
            # Handle local file
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def clean_content(self, raw_content: str) -> str:
        """Clean HTML tags and extract meaningful text"""
        soup = self.BeautifulSoup(raw_content, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Extract text content
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def extract_metadata(self, file_path: Union[str, Path]) -> DocumentMetadata:
        """Extract metadata from HTML document"""
        raw_content = self.extract_content(file_path)
        soup = self.BeautifulSoup(raw_content, 'html.parser')
        
        # Extract title
        title_tag = soup.find('title')
        title = title_tag.text.strip() if title_tag else str(file_path)
        
        # Generate content hash
        content_hash = hashlib.md5(raw_content.encode()).hexdigest()
        
        # File stats
        if not str(file_path).startswith(('http://', 'https://')):
            file_stats = Path(file_path).stat()
            file_size = file_stats.st_size
            modified_at = datetime.fromtimestamp(file_stats.st_mtime)
        else:
            file_size = len(raw_content)
            modified_at = datetime.now()
        
        return DocumentMetadata(
            source_type="html",
            source_path=str(file_path),
            title=title,
            created_at=datetime.now(),
            modified_at=modified_at,
            content_hash=content_hash,
            chunk_count=0,  # Will be updated after chunking
            file_size=file_size,
            additional_metadata={
                "urls": [link.get('href') for link in soup.find_all('a', href=True)],
                "images": [img.get('src') for img in soup.find_all('img', src=True)]
            }
        )
    
    def get_collection_name(self) -> str:
        """Return collection name for HTML documents"""
        return "html_documents"

class PDFDocumentProcessor(BaseDocumentProcessor):
    """Processor for PDF documents"""
    
    def __init__(self, chroma_client: chromadb.PersistentClient, embedding_generator: EmbeddingGenerator):
        super().__init__(chroma_client, embedding_generator)
        try:
            import pdfplumber
            self.pdfplumber = pdfplumber
        except ImportError:
            raise ImportError("Required package: pip install pdfplumber")
    
    def extract_content(self, file_path: Union[str, Path]) -> str:
        """Extract text content from PDF"""
        text_content = []
        
        with self.pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_content.append(text)
        
        return '\n'.join(text_content)
    
    def clean_content(self, raw_content: str) -> str:
        """Clean PDF extracted text"""
        # Remove extra whitespace and normalize
        lines = raw_content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 2:  # Skip very short lines
                cleaned_lines.append(line)
        
        # Join and clean up spacing
        text = ' '.join(cleaned_lines)
        text = ' '.join(text.split())  # Normalize whitespace
        
        return text
    
    def extract_metadata(self, file_path: Union[str, Path]) -> DocumentMetadata:
        """Extract metadata from PDF document"""
        file_path = Path(file_path)
        
        with self.pdfplumber.open(file_path) as pdf:
            # Extract PDF metadata
            pdf_metadata = pdf.metadata or {}
            title = pdf_metadata.get('Title', file_path.stem)
            
            # Generate content hash from first page
            first_page_text = pdf.pages[0].extract_text() if pdf.pages else ""
            content_hash = hashlib.md5(first_page_text.encode()).hexdigest()
        
        file_stats = file_path.stat()
        
        return DocumentMetadata(
            source_type="pdf",
            source_path=str(file_path),
            title=title,
            created_at=datetime.fromtimestamp(file_stats.st_ctime),
            modified_at=datetime.fromtimestamp(file_stats.st_mtime),
            content_hash=content_hash,
            chunk_count=0,
            file_size=file_stats.st_size,
            additional_metadata={
                "page_count": len(pdf.pages) if 'pdf' in locals() else 0,
                "pdf_metadata": pdf_metadata
            }
        )
    
    def get_collection_name(self) -> str:
        """Return collection name for PDF documents"""
        return "pdf_documents"

class DocumentIndexer:
    """Main indexer class that orchestrates document processing with embeddings"""
    
    def __init__(
        self,
        chroma_db_path: str = "./chroma_db",
        embedding_provider: str = "sentence-transformer",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize document indexer with embedding support

        Args:
            chroma_db_path: Path to ChromaDB storage
            embedding_provider: 'sentence-transformer', 'openai', or 'gemini'
            embedding_model: Model name for the provider
        """
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # Select embedding generator
        if embedding_provider == "openai":
            self.embedding_generator = OpenAIEmbeddingGenerator(model_name="text-embedding-3-small")
        elif embedding_provider == "gemini":
            self.embedding_generator = GeminiEmbeddingGenerator(embedding_model)
        else:
            self.embedding_generator = SentenceTransformerEmbeddingGenerator(embedding_model)

        # Initialize processors with embedding support
        self.processors = {
            'html': HTMLDocumentProcessor(self.chroma_client, self.embedding_generator),
            'pdf': PDFDocumentProcessor(self.chroma_client, self.embedding_generator)
        }

        print(f"DocumentIndexer initialized with embedding provider: {embedding_provider}, model: {embedding_model}")
    
    def process_file(self, file_path: Union[str, Path], file_type: str) -> bool:
        """Process a single file with embedding generation"""
        if file_type not in self.processors:
            print(f"Unsupported file type: {file_type}")
            return False
        
        try:
            print(f"Processing {file_path} as {file_type}...")
            processor = self.processors[file_type]
            chunks = processor.process_document(file_path)
            
            if chunks:
                success = processor.index_chunks(chunks)
                if success:
                    print(f"✅ Successfully indexed {len(chunks)} chunks from {file_path}")
                    return True
                else:
                    print(f"❌ Failed to index chunks from {file_path}")
                    return False
            else:
                print(f"⚠️ No content extracted from {file_path}")
                return False
                
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            return False
    
    def process_directory(self, directory_path: Union[str, Path]) -> Dict[str, int]:
        """Process all supported files in a directory"""
        directory_path = Path(directory_path)
        results = {"html": 0, "pdf": 0, "errors": 0}
        
        print(f"Processing directory: {directory_path}")
        
        # Process HTML files
        html_files = list(directory_path.rglob("*.html"))
        print(f"Found {len(html_files)} HTML files")
        for html_file in html_files:
            if self.process_file(html_file, 'html'):
                results["html"] += 1
            else:
                results["errors"] += 1
        
        # Process PDF files  
        pdf_files = list(directory_path.rglob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files")
        for pdf_file in pdf_files:
            if self.process_file(pdf_file, 'pdf'):
                results["pdf"] += 1
            else:
                results["errors"] += 1
        
        return results
    
    def search_similar(self, query: str, collection_name: str, n_results: int = 5) -> List[Dict]:
        """Search for similar documents using semantic similarity"""
        try:
            collection = self.chroma_client.get_collection(collection_name)
            
            # Generate embedding for the query
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            # Search using the query embedding
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            print(f"Found {len(results.get('documents', [[]])[0])} similar documents")
            return results
            
        except Exception as e:
            print(f"Error searching collection {collection_name}: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics about indexed collections"""
        stats = {}
        try:
            collections = self.chroma_client.list_collections()
            for collection in collections:
                count = collection.count()
                stats[collection.name] = count
        except Exception as e:
            print(f"Error getting collection stats: {e}")
        
        return stats

# Example usage
if __name__ == "__main__":
    # Initialize the indexer with embedding support
    indexer = DocumentIndexer(
        chroma_db_path="./knowledge_base_db",
        embedding_provider="openai",  # Fast and efficient
        embedding_model="text-embedding-3-small"  # Higher quality alternative
    )
    
    # Process individual files
    indexer.process_file("https://www.crash.net/bsb/results/1074861/1/2025-british-superbikes-snetterton-race-results-1", "html")
    #indexer.process_file("technical_manual.pdf", "pdf")
    
    # # Process entire directory
    # results = indexer.process_directory("./documents")
    # print(f"Processing complete: {results}")
    
    # Get collection statistics
    stats = indexer.get_collection_stats()
    print(f"Collection stats: {stats}")
    
    # Search example with semantic similarity
    search_results = indexer.search_similar(
        "What were the race results", 
        "html_documents",
        n_results=3
    )
    
    if search_results and 'documents' in search_results:
        print(f"\nTop search results:")
        documents = search_results['documents'][0]
        distances = search_results.get('distances', [[]])[0]
        metadatas = search_results.get('metadatas', [[]])[0]
        
        for i, (doc, distance, metadata) in enumerate(zip(documents, distances, metadatas)):
            print(f"\n{i+1}. Similarity: {1-distance:.3f}")
            print(f"   Source: {metadata.get('title', 'Unknown')}")
            print(f"   Content: {doc[:200]}...")
    
    # Example: Batch processing with progress tracking
    print("\n" + "="*50)
    print("BATCH PROCESSING EXAMPLE")
    print("="*50)
    
    # You can also customize chunk size and overlap
    indexer.processors['html'].chunk_size = 800
    indexer.processors['html'].chunk_overlap = 150
    
    print("Indexer ready for batch processing!")
    print(f"Embedding model: {indexer.embedding_generator.model_name}")
    print(f"Embedding dimension: {indexer.embedding_generator.embedding_dimension}")