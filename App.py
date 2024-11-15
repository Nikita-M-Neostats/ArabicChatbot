##Preliminary and dependencies
from dotenv import load_dotenv
import os
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
import streamlit as st
from langchain_community.vectorstores import FAISS
from operator import itemgetter
from langchain.schema.runnable import RunnableMap
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

#Loading secrets
os.environ["AZURE_OPENAI_ENDPOINT"] = st.secrets.OPENAI_ENDPOINT
os.environ["AZURE_OPENAI_API_KEY"] = st.secrets.OPENAI_API_KEY
doc_intelligence_endpoint = st.secrets.DOCUMENT_INTELLIGENCE_ENDPOINT
doc_intelligence_key = st.secrets.DOCUMENT_INTELLIGENCE_API_KEY
vector_store_address: str = st.secrets.AI_SEARCH_URL
vector_store_password: str = st.secrets.AI_SEARCH_PASSWORD
azure_deployment = st.secrets.EMBEDDING_DEPLOYMENT
api_version = st.secrets.OPENAI_API_VERSION
openai_deployment = st.secrets.OPENAI_DEPLOYMENT
#Inititalizing Embeddings model
embeddings = AzureOpenAIEmbeddings(azure_deployment=azure_deployment,
                                    openai_api_version=api_version)


##Function for formating loaded documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


##Function for reading files and generating knowlege index in local vector store
def file_loader(document):
    #Reading file text onto 'text' using PyPDF2 - PdfReader
    text=""
    
    pdf_reader=PdfReader(document)
    
    for page in pdf_reader.pages:
        text+=page.extract_text()

    #Initilizing langchain CharacterTextSplitter
    text_splitter=CharacterTextSplitter(separator="\n",
                                        chunk_size=1000,
                                        chunk_overlap=200,
                                        length_function=len)   
    chunks=text_splitter.split_text(text)

    #Storing into vector store
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("temp-index")


##Function for generating answer based on similarity search of the knowlege index
def chatbot_short(query: str):
    
    #Getting a retriever of the vector store
    folder = os.getcwd()+"/temp-index"
    knowledge_index = FAISS.load_local(folder_path=folder, index_name="index", embeddings=embeddings, allow_dangerous_deserialization=True)
    
    retriever = knowledge_index.as_retriever(search_type='similarity', search_kwargs={'k':3})

    #Building a RAG prompt
    prompt = ChatPromptTemplate(input_variables=['context', 'question'], 
                                metadata={'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt',
                                        'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'},
                                messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], 
                                                                                        template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer then answer normally while informing that you can answer from the read PDF, Use two-three sentences maximum, and keep the answer concise. Make sure arabic text should be align to right.\nQuestion: {question} \nContext: {context} \nAnswer:"))])

    #Azure OpenAI model - Using Indorama resource for now
    llm = AzureChatOpenAI(openai_api_version=api_version,
                        azure_deployment=openai_deployment,
                        temperature=0.2)

    #Building and invoking the RAG chain 
    rag_chain_from_docs = ({'context': lambda input: format_docs(input['documents']),
                            'question': itemgetter('question')}
                            | prompt | llm | StrOutputParser())

    rag_chain_with_source = RunnableMap({'documents': retriever,
                                        'question': RunnablePassthrough()}) | {'documents': lambda input: [doc.metadata for doc in input['documents']],
                                                                                'answer': rag_chain_from_docs}
    invoked_dict = rag_chain_with_source.invoke(query)

    return invoked_dict

def chatbot_long(query: str):
    
    #Getting a retriever of the vector store
    folder = os.getcwd()+"/temp-index"
    knowledge_index = FAISS.load_local(folder_path=folder, index_name="index", embeddings=embeddings, allow_dangerous_deserialization=True)
    
    retriever = knowledge_index.as_retriever(search_type='similarity', search_kwargs={'k':3})

    #Building a RAG prompt
    prompt = ChatPromptTemplate(input_variables=['context', 'question'], 
                                metadata={'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt',
                                        'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'},
                                messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], 
                                                                                        template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer then answer normally while informing that you can answer from the read PDF. Use five-six sentences maximum. Make sure to align arabic text to right. \nQuestion: {question} \nContext: {context} \nAnswer:"))])

    #Azure OpenAI model - Using Indorama resource for now
    llm = AzureChatOpenAI(openai_api_version=api_version,
                        azure_deployment=openai_deployment,
                        temperature=0.2)

    #Building and invoking the RAG chain 
    rag_chain_from_docs = ({'context': lambda input: format_docs(input['documents']),
                            'question': itemgetter('question')}
                            | prompt | llm | StrOutputParser())

    rag_chain_with_source = RunnableMap({'documents': retriever,
                                        'question': RunnablePassthrough()}) | {'documents': lambda input: [doc.metadata for doc in input['documents']],
                                                                                'answer': rag_chain_from_docs}
    invoked_dict = rag_chain_with_source.invoke(query)

    return invoked_dict

def handle_message():
    if st.session_state.input_text:
        # Add the user message to the chat history
        st.session_state.messages.append({"role": "user", "content": st.session_state.input_text})

        # Process the response (replace with your chatbot's logic)
        if st.session_state.answer_type == "Concise":
            response_dict = chatbot_short(st.session_state.input_text)
        else:
            response_dict = chatbot_long(st.session_state.input_text)

        response = response_dict.get('answer', "Sorry, I couldn't understand your question.")

        # Add the assistant's response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Clear the input field after sending the message
        st.session_state.input_text = ""

def main():
    rtl_css = """
    <style>
        .rtl-message {
            direction: rtl;
            text-align: right;
            margin-right: 0.5rem;
        }
        .rtl-text {
            direction: rtl;
            text-align: right;
            margin-right: 0.5rem;
            margin-bottom: 1rem;
        }
        .ltr-text {
            direction: ltr;
            text-align: left;
            margin-right: 0.5rem;
            margin-top: 0.1rem;
        }
        .stTextInput {
            text-align: right;
            position: fixed;
            bottom: 1rem; /* Adjust as needed */
            right: 10;
            width: 64%;
            direction: rtl;
            padding-top:0;
            padding-right: 40px; /* Adjust as needed for scroll bar */
            z-index: 9999; /* Ensures it stays above other elements */
        }
    }
    </style>
    """
    
    st.set_page_config(page_title="LegalAdvisorAI", page_icon="NeoStats_Logo_N.png", layout='wide')
    # Custom CSS for inline layout
    inline_css = """
    <style>
    .inline-container {
        display: flex;
        align-items: center;
        position: fixed;
        margin-top: 0;
        padding-top:0;
    }

    .inline-container h2 {
        margin: 0;
        margin-top: 0;
        padding-top:0;
    }
  
    </style>
    """
    st.markdown('<div class="content">', unsafe_allow_html=True)
    with st.container(border=True,height=500):
            st.markdown(rtl_css, unsafe_allow_html=True)
            # Main content area with top padding
            if "input_text" not in st.session_state:
                st.session_state.input_text = ""

            if "answer_type" not in st.session_state:
                st.session_state.answer_type = "Concise"
                    
            if 'messages' not in st.session_state:
                st.session_state.messages = []

                    # Display chat history
            for message in st.session_state.messages:
                if message['role'] == 'assistant':
                    with st.chat_message(message['role'], avatar='NeoStats_Logo_N.png'):
                        content = message['content']
                        if "(" in content and ")" in content:  # Check if both parentheses exist
                            arabic_part, english_part = content.split("(", 1)
                            english_part = english_part.rstrip(")")
                            st.markdown(f"<div class='rtl-text'>{arabic_part}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='ltr-text'>({english_part})</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='rtl-text'>{content}</div>", unsafe_allow_html=True)
                else:
                    with st.chat_message(message['role']):
                        st.markdown(f"<div class='rtl-text'>{message['content']}</div>", unsafe_allow_html=True)
            
    with st.sidebar:
        st.image("image.png", use_column_width=True) 
        st.subheader("تحميل ملف PDF (Upload PDF File)")
        doc = st.file_uploader("قم بتحميل الملف هنا، ثم انقر فوق الزر 'Load File' (Upload file here, and click on the 'Load File' button)", accept_multiple_files=False)
        if st.button("Load File"):
            with st.spinner("Loading file..."):
                if doc is not None:
                    file_loader(doc)

        st.session_state.answer_type = st.radio("نوع الإجابة (Answer Type): ", ['مقتضب (Concise)', 'مفصل (Detailed)'], captions=['إجابات أقصر وأكثر تلخيصا (Shorter, more summarised answers)', 'إجابات أطول وأكثر تفصيلا (Longer, more detailed answers)'])

    st.text_input("Enter your question", value=st.session_state.input_text, key="input_text", placeholder="سؤالك هنا...", on_change=handle_message, label_visibility="collapsed")

if __name__ == "__main__":
    main()
