import os
import glob
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.messages.chat import ChatMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_teddynote.prompts import load_prompt
from langchain_teddynote import logging


# API KEY 정보로드
load_dotenv()
logging.langsmith("[SOUNDMIND] RAG")


# UI 구현부-1 (채팅창)
st.title("[SOUNDMIND] RAG System")
user_input = st.chat_input("궁금한 내용을 물어보세요")
warning_msg = st.empty()


# UI 구현부-2 (사이드바) 
with st.sidebar:
    clear_btn = st.button("대화 초기화")
    uploade_file = st.file_uploader("파일 업로드", type=["PDF"])
    prompt_files = glob.glob("prompts_rag/*.yaml")
    selected_prompt = st.selectbox("원하는 프롬프트를 선택해주세요", prompt_files, index=0)
    selected_rag = st.selectbox("RAG 선택", ["Naive RAG", "Advanced RAG", "Moduler RAG"])
    selected_parser = st.selectbox("OutputParser 선택", ["StrOutputParser", "PydanticOutputParser", "JsonOutputParser"])
    prompt_files = glob.glob("prompts/*.yaml")
    selected_model = st.selectbox("LLM 선택", ["gpt-4.1", "gpt-4o", "gpt-4o-mini", "gpt-5", "gpt-5-mini","gpt-5-nano"])


# 초기화-1 (업로드 파일)
if not os.path.exists(".cache"):
    os.mkdir(".cache")

if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

# 초기화-2 (사용자 대화 메모리)
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 초기화-3 (Chain 유지)
if "chain" not in st.session_state:
    st.session_state["chain"] = None

# 초기화-4 (대화 초기화)
if clear_btn:
    st.session_state["messages"] = []

# 업로드 파일 처리
@st.cache_resource(show_spinner="업로드한 파일을 처리 중 입니다")


# RAG 1~5단계
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # RAG 로직-1 (문서 로드)
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    # RAG 로직-2 (문서 분할)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # RAG 로직-3 (임베딩)
    embeddings = OpenAIEmbeddings()

    # RAG 로직-4 (DB 생성)
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # RAG 로직-5 (검색기 생성)
    retriever = vectorstore.as_retriever()

    return retriever


# RAG 6~8단계
def create_chain(selected_prompt, retriever, model=selected_model):
    # RAG 로직-6 (프롬프트 정의)
    prompt = load_prompt(selected_prompt, encoding="utf-8")

    # RAG 로직-7 (언어 모델 생성)
    llm = ChatOpenAI(model=model, temperature=0.0)

    # RAG 로직-8 (LCEL 생성)
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
            | prompt
            | llm
            | StrOutputParser()
    )

    return chain


# 기능-1 (이전 대화 출력)
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 기능-2 (새로운 메시지 추가)
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 실행부
if uploade_file:
    retriever = embed_file(uploade_file)        # RAG 1~5단계
    chain = create_chain(selected_prompt, retriever)             # RAG 6~8단계
    st.session_state["chain"] = chain

print_messages()

if user_input:
    chain = st.session_state["chain"]

    if chain is not None:
        st.chat_message("user").write(user_input)
        ai_answer = ""
        
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            container = st.empty()

            for token in response:
                ai_answer += token
                container.markdown(ai_answer)
    else:
        warning_msg.error("파일을 업로드해 주세요")
    
    add_message("user", user_input)
    add_message("assistant", ai_answer)
    