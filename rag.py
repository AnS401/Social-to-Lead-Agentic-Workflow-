import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

def create_vector_store():
    if not os.path.exists("data/knowledge_base.txt"):
        os.makedirs("data", exist_ok=True)
        content = (
            "AutoStream Pricing & Features:\n"
            "- Basic Plan: $29/month, 10 videos/month, 720p resolution.\n"
            "- Pro Plan: $79/month, Unlimited videos, 4K resolution, AI captions.\n"
            "Company Policies:\n"
            "- No refunds after 7 days.\n"
            "- 24/7 support available only on Pro plan."
        )
        with open("data/knowledge_base.txt", "w") as f:
            f.write(content)

    with open("data/knowledge_base.txt", "r") as f:
        raw_text = f.read()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1"
    )

    vectorstore = FAISS.from_texts(texts, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

def get_retriever():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1"
    )
    
    if not os.path.exists("faiss_index"):
        vectorstore = create_vector_store()
    else:
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    return vectorstore.as_retriever(search_kwargs={"k": 2})