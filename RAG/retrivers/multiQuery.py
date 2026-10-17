from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

docs = [
    Document(
        page_content="""
        Retrieval-Augmented Generation (RAG) is a technique that
        combines information retrieval with large language models.
        First, relevant documents are retrieved from a vector database.
        Then, the retrieved context is provided to an LLM to generate
        an accurate and context-aware answer. RAG helps reduce
        hallucinations and allows LLMs to use external knowledge.
        """,
        metadata={"topic": "RAG", "source": "rag_notes.txt"}
    ),

    Document(
        page_content="""
        Vector databases store numerical representations of text
        called embeddings. These embeddings capture semantic meaning.
        When a user asks a question, the question is converted into
        an embedding, and the database retrieves documents with
        similar vectors. Examples of vector databases include Chroma,
        FAISS, and Pinecone.
        """,
        metadata={"topic": "Vector Database", "source": "vector_db.txt"}
    ),

    Document(
        page_content="""
        Embeddings convert words, sentences, or documents into
        numerical vectors. Semantically similar texts generally have
        embeddings that are close together in vector space.
        Embeddings are used in semantic search, document clustering,
        recommendation systems, and retrieval-augmented generation.
        """,
        metadata={"topic": "Embeddings", "source": "embeddings.txt"}
    ),

    Document(
        page_content="""
        A retriever is a component that searches a collection of
        documents and returns the most relevant results for a query.
        Vector store retrievers commonly use similarity search or
        maximum marginal relevance. Retrievers provide context to
        language models in RAG applications.
        """,
        metadata={"topic": "Retrievers", "source": "retrievers.txt"}
    ),

    Document(
        page_content="""
        Multi-Query Retrieval uses a language model to generate
        multiple alternative versions of a user's original question.
        Each query searches the document collection independently.
        The resulting documents are combined to improve recall and
        retrieve relevant information that a single query might miss.
        Duplicate documents may be removed from the combined results.
        """,
        metadata={"topic": "Multi-Query Retrieval", "source": "multi_query.txt"}
    ),

    Document(
        page_content="""
        Query expansion improves information retrieval by generating
        alternative search terms, synonyms, or related expressions.
        For example, the question 'How does RAG find information?'
        can be expanded into 'How does semantic document retrieval
        work?' and 'How does a vector database search for relevant
        context?' These queries can retrieve different relevant
        documents.
        """,
        metadata={"topic": "Query Expansion", "source": "query_expansion.txt"}
    ),

    Document(
        page_content="""
        Chunking divides long documents into smaller text segments.
        Each segment can be embedded and stored separately in a vector
        database. Smaller chunks can improve retrieval precision,
        while larger chunks may preserve more context. Chunk size
        and overlap should be selected according to the document type
        and the needs of the retrieval task.
        """,
        metadata={"topic": "Chunking", "source": "chunking.txt"}
    ),

    Document(
        page_content="""
        Hybrid search combines lexical search methods, such as BM25,
        with semantic vector search. Keyword search is useful when
        exact terms, identifiers, or technical names matter. Semantic
        search can find relevant text even when the query uses
        different words. Combining both methods can improve retrieval
        quality for a RAG application.
        """,
        metadata={"topic": "Hybrid Search", "source": "hybrid_search.txt"}
    ),
]

embeddings = GoogleGenerativeAIEmbeddings(
    model = "gemini-embedding-001"
)

vectorStore = Chroma.from_documents(
    docs,
    embeddings
)

llm = GoogleGenerativeAI(
    model = "gemini-3.5-flash-lite"
)

retriever = vectorStore.as_retriever()

multiQueryRetriever = MultiQueryRetriever.from_llm(
    retriever = retriever,
    llm = llm
)

query = "What is Query Decient"

docs = multiQueryRetriever.invoke("query")

for doc in docs:
    print(doc.page_content)


# = -  _  ""  ''  