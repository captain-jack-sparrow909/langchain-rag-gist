from operator import itemgetter
from dotenv import load_dotenv

load_dotenv()
import os
from langchain_core.prompts import ChatPromptTemplate
# PromptTemplate creates a single string prompt.
# ChatPromptTemplate creates a list of chat messages with roles such as: system, human (user), assistant, tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
# Many LangChain chat models return structured message objects (such as AIMessage). If you only want the text content, StrOutputParser extracts it.
from langchain_core.runnables import RunnablePassthrough
# RunnablePassthrough simply passes its input through unchanged. Useful in RAG where you want to preserve original input while generating additional values.

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

def retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL
    return a chain that can be invoked with {"question": ...}

    Advantages over non-LCEL approach:
    - Declarative and composable - easy to chain operations using pipe operator |
    - Built-in streaming - chain.stream() works out of the box
    - Built-in async - chain.ainvoke() and chain.astream() available
    - Batch processing - chain.batch() for multiple inputs
    - Type safety - better integration with Langchain's type system
    - Less code
    - Reusable
    - Better debugging and observability
    """


    # retrieval_chain = (
    #     retriever | format_docs | # these are the 1st and 2nd step of retrieval_chain_without_lcel
    #     prompt_template | #this chain has combined the 3rd, 4th and 5th step of retrieval_chain_without_lcel
    #     llm |
    #     StrOutputParser()
    # ) 

    # possible issue: but the solution is there:
    # however the format_docs is just a python function and not Langchain's and you don't have invoke method on it 
    # but whenever we use python functions in LCEL, langchain will convert them into runnable lambdas.
    # eg: retriever | format_docs | prompt_template => converted into retriever | RunnableLambda(format_docs) | prompt_template

    # another issue:
    # the prompt_template need to receive context, and question, the output of -> retriever | format_docs isn't going to do this for us,
    # we need some way: we need to take the output of those steps and attribute it to the key "context", that's why the step shown below
    # is needed, and the above is commented
    retrieval_chain = (
        RunnablePassthrough.assign(
            context = itemgetter('question') | retriever | format_docs
        ) 
        | prompt_template 
        | llm
        | StrOutputParser()
    )

    # the input 'question' with which the retrieval_chain is invoked it'll let it pass through, but with assign method it'll add additional key 'context'
    # the value of 'context' is calculated by these steps: itemgetter("question") | retriever | format_docs

    return retrieval_chain
    






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
    # Option 1: invocation with RAG without using LCEL, main difficulty is that on LangSmith some of the steps won't be traceable
    ##########################################
    # retrieval_result_without_lcel = retrieval_chain_without_lcel(query)
    # print(f"\nRetrieval without LCEL:\n{retrieval_result_without_lcel}")

    ##########################################
    # Option 2: invocation with RAG using LCEL - BETTER APPROACH
    ##########################################
    chain_with_lcel = retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer of retrieval chain with LCEL\n")
    print(result_with_lcel)


