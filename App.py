'''
AUTHOR: uday.marwah
TIME: 16-05-2024 10:01:27

NEO DOC DIVE: PDF RAG CHATBOT APP FOR NEOSTATS INTERNAL USE
'''

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
import pymupdf
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
    text_splitter=CharacterTextSplitter(separator="\n",
                                        chunk_size=1000,
                                        chunk_overlap=200,
                                        length_function=len)   
    chunks=text_splitter.split_text(text)

    #Storing into vector store
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("temp-index")
def chatbot_long(query: str):
    
    #Getting a retriever of the vector store
    folder = os.getcwd() + "/temp-index/"
    knowledge_index = FAISS.load_local(folder_path=folder, index_name="index", embeddings=embeddings, allow_dangerous_deserialization=True)
    
    retriever = knowledge_index.as_retriever(search_type='similarity', search_kwargs={'k':3})
    template_caseSummary="""
    You are an AML Assistant, tasked with helping users based on the cases provided as context.
        - If the user greets you, reply with a greeting and ask how you can assist.
        - If the user requests an email, generate a professional one.
        - If the user asks for a case summary, provide a properly formatted summary based on the case document.
        - If the user asks for generate summary for transaction made for particular case Id, provide a properly formatted summary based on the case document.
        - If the user's question is unrelated to the case, politely inform them that you can only respond to questions relevant to the case.
        - If the user requests suggestions on what steps to take next regarding the case, assist them by providing relevant and contextual recommendations.
    Make sure all responses are well-formatted, with appropriate line breaks.
    Case Note: {context}
    User’s Question: {question}
    """
    prompt = PromptTemplate(
    template=template_caseSummary,
    input_variables=["question", "context"]
    )
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

##Main method containing streamlit application UI 
def main():
    
    #setting up page configuration
    st.set_page_config(page_title="AML Assitant", page_icon="NeoStats_Logo_N.png", layout='wide')
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.chat_message("assistant", avatar='NeoStats_Logo_N.png'):
        st.markdown("<div>Hi! How can I help you today?</div>",unsafe_allow_html=True)

    #Setting up chat elements for assistant and user
    for message in st.session_state.messages:
        if message['role'] == 'assistant':
            with st.chat_message(message['role'], avatar='NeoStats_Logo_N.png'):
                st.markdown(message['content'])
        else:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

    #Sidebar for providing document that is to be read - passing to file_loader() function
    with st.sidebar:
        st.image('Neostats_Logo.png', output_format="PNG", width=300)
        st.title("AML Compliance Co-pilot\n")
        st.subheader("Upload Document")
        doc = st.file_uploader("Upload file here, and click on  the 'Load File' button", accept_multiple_files=False)
        if st.button("Load File"):
            with st.spinner("Loading file..."):
                file_loader(doc)

    #Taking user prompt - passing to chatbot() function
    if prompt := st.chat_input("Enter your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        response_dict = chatbot_long(prompt)
        response = response_dict['answer']
        response = response.replace("Response: ", "")
        response = response.replace("$", "Dollors")
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar='NeoStats_Logo_N.png'):

            st.markdown(response)
            print(response)

            #Appending messages so user can see chat history
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
