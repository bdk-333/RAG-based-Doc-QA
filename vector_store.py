import chromadb
from config import COLLECTION_NAME

def setup_vector_store(chunks, metadata, embedding_model, client):
    """
    Creates or recreates a ChromaDB collection and stores document embeddings.
    """
    try:
        if client.get_collection(name=COLLECTION_NAME):
            client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadata,
        ids=ids
    )
    
    return collection

def query_vector_store(query, collection, embedding_model, top_n=5):
    """
    Queries the vector store to find the most similar document chunks.
    """
    if collection is None:
        return None, None, None
    
    query_embedding = embedding_model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_n,
        include=['documents', 'metadatas', 'distances']
    )
    
    return results['documents'][0], results['metadatas'][0], results['distances'][0]