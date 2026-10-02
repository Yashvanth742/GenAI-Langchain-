from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
)

texts = [
    "Hello this is yash",
    "and you all are beautiful"
]

vector = embedding.embed_documents(texts)

print(vector)