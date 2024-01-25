import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
import chromadb.utils.embedding_functions as embedding_functions
from langchain_community.chat_models import ChatOpenAI
import chromadb
from langchain.chains import RetrievalQA
from langchain_community.llms.openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from deep_translator import GoogleTranslator
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Initialize components and models only once
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
db = Chroma(persist_directory="chromaDB", embedding_function=embeddings, collection_name='test')
retriever_openai = db.as_retriever(search_kwargs={'k': 1})
def transl_text(text, target_language='en'):
    try:
        translation = GoogleTranslator(source='auto', target=target_language).translate(text)
        return translation
    except Exception as e:
        print(f"Translation error: {e}")
        return None
def load_template(language):
    lang_map = {"hi":"Hindi","ta":"Tamil","en":"English"}
    prompt =f'''
        You are a farming advisory bot helping farmers in managing and diagnosing diseases in their crops.
        Given the following context and question, generate an answer based on this context only.
        Generate the answer from the context in {lang_map[language]} language.
        Try to provide as much text as possible from "response" section from the source document without making any assumptions.
        If the answer is not found in the context, look for answers from the web and answer short only if it is in the scope of farming or agriculture.
        If the question is out of context from farming ask them to ask questions only related to farming or agriculture.
        If someone greets, greet them back.
        Dont try to make up an answer.
        CONTEXT:{{context}}
        QUESTION:{{question}}
        '''
    return prompt

llm = ChatOpenAI(
    model_name='gpt-3.5-turbo-16k',
    temperature=0.3,
    max_tokens=1000,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# Cache for storing previously generated responses
# response_cache = {}
# def gptcache(question,target_language):
#     if question in response_cache:
#         return response_cache[question]
#     response = retrieval_qa(question)
#     # Cache the response to avoid future API calls for the same question
#     response_cache[question] = response
#     return response

def retrieval_qa(question, target_language):
    prompt_in = load_template(target_language)
    question=transl_text(question,target_language)
    bot_prompt = PromptTemplate(
        template=prompt_in, input_variables=["context", "question"]
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_openai,
        return_source_documents=True,
        chain_type_kwargs={"prompt": bot_prompt}
    )
    answer = qa.invoke(question)
    response = answer['result']
    if response.strip().lower() == "i dont know":
        response = "I don't have the information you're looking for."
    return response