from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss

# get resume text
RESUME_TEXT = """
Name: Kirtan Parekh
Contact links:
- Email: kirtansp333@gmail.com
- LinkedIn: www.linkedin.com/in/kirtan-s-parekh
Skills:
- HTML5
- CSS3
- JavaScript
- PHP
- Linux Operating System
- SQL
- MySQL database
- Data Structures and Algorithms
- Adaptability
- Work responsibility
- Self-confidence

Experience:
- Hungry Point food for the and items to A sample ordering website (using HTML, CSS, JavaScript)
- The online Tic-Tac-Toe Game (using JavaScript)

Education:
- Bachelor Engineering in Computer Technology (B.E.) at Patel Institute of Pursuing ( completed)
  - Education: University of Michigan
  - Field of Study: Web Development

Projects:
- Hungry Point food for the and items to A sample ordering website, build using HTML, CSS, Javascript
- The online Tic-Tac-Toe Game, an website game of JavaScript
- UX Design (Issued on : 8th November 2022)
- Get started with UX (Issued on 16th December 2022)

Certifications:
- SAP CSR Program - Code Unnati
- Edunet Foundation's Industry implemented technologies

Achievements:
- Completed HSC from R.P.T.P Science College with passing percentage of 75.23
- Completed from with passing percentage of SSC I. B. Patel English School (82.33)

Certificates and Badges:
- None mentioned in the details
"""

def generate_chunks(text, chunk_size=500, chunk_overlap=100):
    """Generates chunks from the given text

    Args:
        resume (str): a string containing text of a document.

    Returns:
        list[str]: chunks. Each chunk is a string of certain length of characters from the given text
    """
    
    # text splitter obj
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len
    )
    
    # generating chunks
    chunks = text_splitter.split_text(text)
    
    return chunks


# function to generate embedding of chunks
def generate_embedding(chunks):
    
    # create a model
    model = SentenceTransformer("all-MiniLM-L6-V2")
    
    # embed the chunks
    chunk_embeddings = model.encode(chunks)
    
    # return embedding obj
    return chunk_embeddings


if __name__ == "__main__":
    
    # get chunks for resume text:
    chunks = generate_chunks(RESUME_TEXT, 300, 50)
    
    for i, chunk in enumerate(chunks):
        print(f"--------------CHUNK - {i+1} ------------------")
        print(chunk)
        print("\n")
        
    # get chunk embeddings
    chunk_embedding = generate_embedding(chunks)
    
    print("Shape of embeddings:", chunk_embedding.shape)
    print(f"First chunk embedding:\nChunk: {chunks[0]}\nNow it's embedding: {chunk_embedding[0]}")
    
    # create and store embeddings into FAISS database
    embedding_dim = chunk_embedding.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)    # create index
    
    index.add(chunk_embedding)    # add embeddings to the index
    
    print(f"Index created with {index.ntotal} vectors.")
    
    # Searching part
    user_query = "What is the name in the resume?"
    query_embedding = generate_embedding([user_query])    # embed user query
    
    k = 3
    distances, indices = index.search(query_embedding, k)
    
    print(f"Found {k} most similar chunk for query: '{user_query}'")
    
    relevant_chunks = [chunks[i] for i in indices[0]]
    for i, chunk in enumerate(relevant_chunks):
        print(f"Relevant chunk {i+1} (Distance: {distances[0][i]:.4f})")
        print(chunk)
        print("\n")