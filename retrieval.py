from dotenv import load_dotenv

load_dotenv()
import os
from langchain_core.prompts import ChatPromptTemplate
# PromptTemplate creates a single string prompt.
# ChatPromptTemplate creates a list of chat messages with roles such as: system, human (user), assistant, tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

print("initializing components...")
embeddings = OpenAIEmbeddings(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        model="text-embedding-3-small",
        dimensions=512)
llm = ChatOpenAI(model='gpt-5-nano')
vector_store = PineconeVectorStore(index_name=os.environ.get('INDEX'), embedding=embeddings)

# retriever from pinecone to get relevant chunks:
retriever = vector_store.as_retriever(search_kwargs={"k": 3})   #return the top 3 most similar documents

# prompt: the context is going to be the augmentation part, and the question is going to be the original prompt
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the following context:
    {context}
    Question: {question}
    Provide a detailed answer
    """
)


#some auxillary functions:
def format_docs(docs):
    """Formate retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)
    #this one string is eventually going to be sent as context to the prompt_template

def retrieval_chain_without_lcel(query):
    """
    Simple retrieval chain without langchain expression language
    manually retrieve documents, format them, and generate response

    Limitations of this approach:
        - Manual step by step execution
        - no built-in streaming support
        - no async support without additional code
        - harder to compose with other chains
        - more verbose and error prone
    """

    # step 1: retrieve relevant documents
    docs = retriever.invoke(query)

    # step 2: format documents into string
    context = format_docs(docs=docs)

    # step 3: format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # step 4: invoke the LLM with the messages
    response = llm.invoke(messages)

    # step 5: return the response content
    return response.content




if __name__ == "__main__":
    print("--retrieving--")
    query = "what is pincone in machine learning?"

    ##########################################
    # Option 0: raw invocation without RAG
    ##########################################
    # print("\nRaw invocation without RAG")
    # result_raw = llm.invoke([HumanMessage(content=query)])
    # print("\nAnswer of raw invocaiton: ", result_raw.content)

    ##########################################
    # Option 1: invocation with RAG
    ##########################################
    retrieval_result_without_lcel = retrieval_chain_without_lcel(query)
    print(f"\nRetrieval without LCEL:\n{retrieval_result_without_lcel}")


