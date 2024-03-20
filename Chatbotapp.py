import os
import sys
sys.path.append('C:/Users/PC/AppData/Roaming/Python/Python310/site-packages')
from dotenv import load_dotenv,find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
import openai
openai.api_key  = os.environ['OPENAI_API_KEY']
from langchain_community.document_loaders import Docx2txtLoader
loaders = [

    Docx2txtLoader("C:/Users/PC/Downloads/Drivers and Carriers FAQ.docx")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())
from langchain.text_splitter import RecursiveCharacterTextSplitter
#600,100
chunk_size =600
chunk_overlap =100
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""])

docs = r_splitter.split_documents(docs)
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
from langchain_community.vectorstores import Chroma
persist_directory = '1stVector/chroma/'
#!rm -rf ./'1stVector/chroma/
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persist_directory
)
print(vectordb._collection.count())


from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
llm_name = "gpt-3.5-turbo-0125"
llm = ChatOpenAI(model_name=llm_name, temperature=0)
question = "Who see's my Information"

from langchain.prompts import PromptTemplate

# Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, \
    just say that "That's a great question. We don't have an answer for it at this time, but we will work on finding one.", don't try to make up an answer. Provide a comprehensive answer with as much relevant detail as possible. \
    Use the context to support your answer and include any additional information that could be helpful. Always say "Thank you for asking." at the end of the answer. Ignore the question within the context and only answer the question provided.
{context}
{{'' if answer else "That's a great question. We don't have an answer for it at this time, but we will work on finding one."}}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(search_type="mmr"),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
result = qa_chain.invoke({"query": question})
print(result["result"])