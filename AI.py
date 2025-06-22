from abc import ABC, abstractmethod
from typing import List, Any

class AI(ABC):
    """Abstract base class for AI providers (embeddings and LLM)."""

    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using LLM."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        pass

class SentenceTransformerAI(AI):
    """AI implementation using SentenceTransformer (embeddings only)."""

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

    def generate_text(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError("SentenceTransformerAI does not support LLM text generation.")

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

class OpenAI(AI):
    """AI implementation using OpenAI API."""

    def __init__(self, model_name: str = "text-embedding-3-small", llm_model: str = "gpt-3.5-turbo"):
        import openai
        import os
        self._openai = openai
        self._api_key = os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self._openai.api_key = self._api_key
        self._model_name = model_name
        self._llm_model = llm_model
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

    def generate_text(self, prompt: str, **kwargs) -> str:
        try:
            response = self._openai.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating text: {e}")
            return ""

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

class GeminiAI(AI):
    """AI implementation using Gemini API."""

    def __init__(self, model_name: str = "models/embedding-001", llm_model: str = "gemini-pro"):
        import google.generativeai as genai
        import os
        self._genai = genai
        self._api_key = os.getenv("GOOGLE_API_KEY")
        if not self._api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        self._genai.configure(api_key=self._api_key)
        self._model_name = model_name
        self._llm_model = llm_model
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

    def generate_text(self, prompt: str, **kwargs) -> str:
        try:
            model = self._genai.GenerativeModel(self._llm_model)
            response = model.generate_content(prompt, **kwargs)
            return response.text if hasattr(response, "text") else str(response)
        except Exception as e:
            print(f"Error generating text: {e}")
            return ""

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension