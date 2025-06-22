from typing import List, Dict, Any
from dotenv import load_dotenv
from AI import AI  # Assuming AI is defined in AI.py
load_dotenv()  # Load environment variables from .env file

class AIAssistant:
    """
    Handles semantic search and LLM interaction for answering user queries.
    """

    def __init__(self, chroma_client, ai: 'AI', collection_name: str):
        """
        Args:
            chroma_client: ChromaDB PersistentClient instance
            ai: An instance of AI (OpenAI or GeminiAI)
            collection_name: Name of the ChromaDB collection to search
        """
        self.chroma_client = chroma_client
        self.ai = ai
        self.collection_name = collection_name

    def fetch_relevant_context(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Fetch relevant documents from ChromaDB for the query."""
        collection = self.chroma_client.get_collection(self.collection_name)
        query_embedding = self.ai.generate_embedding(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        docs = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]
        context_chunks = []
        for doc, meta, dist in zip(docs, metadatas, distances):
            context_chunks.append({
                "content": doc,
                "metadata": meta,
                "distance": dist
            })
        return context_chunks

    def build_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Build a prompt for the LLM using the query and context."""
        context_text = "\n\n".join(
            f"Context {i+1}:\n{chunk['content'][:500]}" for i, chunk in enumerate(context_chunks)
        )
        prompt = (
            f"Answer the following question using the provided context, stick to the information provided in the context ONLY, If you do not know the answer, just say that you don't know.\n\n"
            f"{context_text}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        return prompt

    def answer_query(self, query: str, n_results: int = 3, **llm_kwargs) -> str:
        """Fetch context, build prompt, send to LLM, and return/display the answer."""
        print(f"Fetching relevant context for query: {query}")
        context_chunks = self.fetch_relevant_context(query, n_results=n_results)
        if not context_chunks:
            print("No relevant context found. Sending query directly to LLM.")
            prompt = query
        else:
            prompt = self.build_prompt(query, context_chunks)
        print("\n--- Prompt sent to LLM ---\n")
        print(prompt[:1000] + ("..." if len(prompt) > 1000 else ""))
        print("\n--- LLM Response ---\n")
        answer = self.ai.generate_text(prompt, **llm_kwargs)
        print(answer)
        return answer

if __name__ == "__main__":
    from chromadb import PersistentClient
    from AI import OpenAI  # Assuming OpenAI is defined in AI.py

    # Example usage
    chroma_client = PersistentClient(path="./knowledge_base_db")
    print(f"ChromaDB client initialized. {chroma_client.get_collection('html_documents')} collections found.")
    ai = OpenAI(model_name="text-embedding-3-small", llm_model="gpt-3.5-turbo")
    assistant = AIAssistant(chroma_client, ai, collection_name="html_documents")

    while True:
        query = input("Enter your question (or 'q' to quit): ")
        if query.lower() == 'q':
            break
        answer = assistant.answer_query(query)
        print("\n--- Answer ---\n")
        print(answer)
 