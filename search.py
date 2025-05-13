import os
from googlesearch import search
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys

from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_ollama import OllamaLLM

from transformers import pipeline


ollama_model = "llama3.2:3b"
embedded_model = "mxbai-embed-large:latest"

question = "What it's a relu in genAI"

llm = OllamaLLM(
    base_url="http://192.168.0.100:11434",
    model=ollama_model,
    num_thread=12,
)


def google_search(query: str, num: int = 10) -> list:
    urls = []
    for i in search(query, num_results=num):
        urls.append(i)
    return urls


urls = google_search(query=question, num=10)
print(urls)

loader = WebBaseLoader(urls[1:])
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

all_splits = text_splitter.split_documents(docs)

oembed = OllamaEmbeddings(base_url="http://192.168.0.100:11434", model=embedded_model)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)


qachain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever(k=100))


query = qachain.invoke({"query": question})
response = query["result"]
print(response)
