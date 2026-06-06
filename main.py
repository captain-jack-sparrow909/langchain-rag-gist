import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
import langchain_openai
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


# the process is going to look like:
# TextLoader -> for loading the medium text, for loading of different kind of data, we've different loaders.
# TextSplitter -> Splitting the blog into smaller chunks
# OpenAIEmbeddings -> Embed the chunks and get vectors
# Pinecone -> store the embeddings in pinecone vector store

def main():
    print("Hello from rag-gist!")
    # ingesting
    print("\ningesting start...")
    loader = TextLoader("/Users/jabirkhan/Desktop/langchain-projects/rag-gist/mediumblog.txt",
    encoding='UTF-8')  #if UTF-8 didn't work then autodetect_encoding = True flag should be used 
    document = loader.load()

    # splitting text
    print("\nsplitting text...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents=document)
    print("number of created chunks: ", len(texts))

    # embedding the splitted text
    print("\nembedding ...")
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        model="text-embedding-3-small",
        dimensions=512,  # must match Pinecone index dimension
    )

    # storing embeddings into Pinecone DB
    PineconeVectorStore.from_documents(documents=texts, embedding=embeddings, index_name=os.environ.get("INDEX_NAME"))

    print("..ingestion finish..")


if __name__ == "__main__":
    main()
