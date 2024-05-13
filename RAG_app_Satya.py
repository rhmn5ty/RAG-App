# Required installations

import streamlit as st
from dotenv import load_dotenv
import os
import shelve

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool, create_conversational_retrieval_agent
from langchain_openai.chat_models import ChatOpenAI

load_dotenv()

st.title("Streamlit RAG Chatbot Interface")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# PDF document loader setup
pdf_loader = PyPDFLoader("/Users/rahmansatya/Downloads/II2221_TB05_K01_T03_Sprint 1_Planning.pdf")
my_documents = pdf_loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(my_documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever()

# Tool and Agent setup
tool = create_retriever_tool(retriever, "Document_Name", "Searches and returns documents regarding the specific topic")
tools = [tool]
llm = ChatOpenAI(temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)

# Shelve for chat history
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])

def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

with st.sidebar:
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])

# Display chat messages
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Main chat interface
if prompt := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        # full_response = ""
        response = agent_executor.invoke([{"role": "user", "content": prompt}])['output']
        # for response in responses:
        #     full_response += response['content']
        #     message_placeholder.markdown(full_response)
        message_placeholder.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Save chat history after each interaction
save_chat_history(st.session_state.messages)
