import glob
import streamlit as st
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging


# API KEY 정보로드
load_dotenv()
logging.langsmith("[Soundmind] AI Chatbot")

# UI 구현부-1 : 채팅 창
st.title("[Soundmind] AI Chatbot")   # 타이틀
user_input = st.chat_input("궁금한 내용을 물어보세요")

# UI 구현부-2 : 사이드 바
with st.sidebar:
    clear_btn = st.button("대화 초기화")
    prompt_files = glob.glob("prompts_chatbot/*.yaml")
    selected_model = st.selectbox("LLM 선택", ["gpt-4.1", "gpt-4o", "gpt-4o-mini", "gpt-5", "gpt-5-mini","gpt-5-nano"])
    selected_prompt = st.selectbox("원하는 프롬프트를 선택해주세요", prompt_files, index=0)
    task_input = st.text_input("역할 설정 (선택사항)", "", placeholder="예: SNS 마케팅 담당자")

# =============================================================================================================

# 초기화 : 대화 초기화
if clear_btn:
    st.session_state["messages"] = []

    if "session_id" in st.session_state:
        session_id = st.session_state["session_id"]
        if session_id in st.session_state["store"]:
            st.session_state["store"][session_id].clear()

# 초기화 : Chain 유지
if "chain" not in st.session_state:
    st.session_state["chain"] = None

# 초기화 : 사용자 대화 메모리
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 초기화 : 세션 ID 정의 및 저장
if "store" not in st.session_state:
    st.session_state["store"] = {}

# 초기화 : 세션 ID를 설정
if "session_id" not in st.session_state:
    st.session_state["session_id"] = "single_user_chat_session"

SESSION_ID = st.session_state["session_id"]

# =============================================================================================================

# 기능-1 (이전 대화 출력)
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

# 기능-2 (새로운 메시지 추가)
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

# 기능-3 (세션 아이디 반환)
def get_session_history(session_id):
    if session_id not in st.session_state["store"]:
        st.session_state["store"][session_id] = ChatMessageHistory()

    return st.session_state["store"][session_id]

# 기능-4 (LCEL)
def create_chain(selected_prompt, task=""):
    """LangChain기반의 AI 챗봇을 생성합니다"""

    # 메모리 정의
    # memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

    # Runnable 설정
    # runnable = RunnablePassthrough.assign(
    #     chat_history=RunnableLambda(memory.load_memory_variables)
    #     | itemgetter("chat_history")
    # )
    
    # Prompt 정의
    if task and task.strip():
        system_message = f"{selected_prompt}\n\n system role: {task}\n위 역할에 맞게 전문적으로 대답해주세요."
    else:
        system_message = selected_prompt

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ]
    )

    # 언어 모델 생성
    llm = ChatOpenAI(model=selected_model, temperature=0.0)

    # 파서 생성
    output_parser = StrOutputParser()

    # 체인 생성
    # chain = runnable | prompt | llm | output_parser
    chain = prompt | llm | output_parser

    # LECL 구성
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,                    # 세션 기록을 가져오는 함수
        input_messages_key="question",          # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",    # 기록 메시지의 키
    )
    return chain_with_history

# =============================================================================================================

# 구현부-1 : 이전 대화 기록 출력
print_messages()

# 구현부-2 : 새로운 사용자 입력 처리
if user_input:
    # 사용자 메시지 저장
    add_message("user", user_input)

    # 사용자 메시지 화면에 출력
    st.chat_message("user").write(user_input)

    # AI 메시지 생성
    chain = create_chain(selected_prompt, task=task_input)
    config = {"configurable": {"session_id": SESSION_ID}}
    response = chain.stream({"question": user_input}, config=config)

    # AI 메시지 출력
    with st.chat_message("assistant"):
        container = st.empty()
        ai_answer = ""

        for token in response:
            ai_answer += token
            # container.markdown(ai_answer)
            container.write(ai_answer)

        # AI 메시지 저장
        add_message("assistant", ai_answer)
