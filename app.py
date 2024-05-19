import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
# from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
# from langchain.llms import HuggingFaceHub
from langchain.document_loaders import DirectoryLoader

def get_pdf_text(pdf_docs):
    
     all_texts = ""

     for pdf_doc in pdf_docs:
        # Read the PDF file
        pdf_reader = PdfReader(pdf_doc)
        
        # Extract text from each page and concatenate
        for page in pdf_reader.pages:
            all_texts += page.extract_text()
    
     return all_texts


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        # separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        
    )
    chunks = text_splitter.split_text(text)
    print(type(chunks))
    return chunks


def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    texts = []

    # Extract text from each chunk and append to the texts list
    for chunk in text_chunks:
        texts.append(chunk)

    # Create a Chroma vector store from the list of texts
    persist_directory = 'db'
    vectorstore = Chroma.from_texts(texts=texts, embedding=embeddings,persist_directory=persist_directory)

    return vectorstore


def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", torch_dtype=torch.float32)
    pipe = pipeline(
    "text2text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_length=512,
    temperature=0.5,
    top_p=0.95,
    repetition_penalty=1.15
    )
    llm = HuggingFacePipeline(pipeline=pipe)


    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Reverse the order of messages in the chat history
    chat_history = reversed(st.session_state.chat_history)

# Iterate over the reversed chat history
    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            st.write(bot_template.replace(
            "{{MSG}}", message.content), unsafe_allow_html=True)
            
        else:
            st.write(user_template.replace(
            "{{MSG}}", message.content), unsafe_allow_html=True)



def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector storemye
                
                vectorstore = get_vectorstore(text_chunks)
                st.subheader("Your documents are processed you can chat now!")
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()