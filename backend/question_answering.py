from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

def answer_question(query, chunks):
    llm = OpenAI()
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    with get_openai_callback() as cb:
        docs = chunks.similarity_search(query=query, k=3)
        response = chain.run(input_documents=docs, question=query)
    return response
