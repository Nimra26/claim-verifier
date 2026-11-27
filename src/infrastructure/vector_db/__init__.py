# Chroma is optional (the "local-ml" extra): import it only when asked for, so pgvector installs never need it.
def __getattr__(name):
    if name == "ChromaVectorStore":
        from src.infrastructure.vector_db.chroma_vectorstore import ChromaVectorStore

        return ChromaVectorStore
    raise AttributeError(name)


__all__ = ["ChromaVectorStore"]
