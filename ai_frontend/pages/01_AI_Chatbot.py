# ============================================
# AI Chatbot - Backend API 통신 구조
# ============================================

import glob
import sys
sys.path.append('..')

import streamlit as st
from langchain_core.messages.chat import ChatMessage
from api_client import (
    chat_stream,
    get_chat_prompts,
    clear_chat_session,
    health_check
)


# 백엔드 서버 연결 확인
try:
    health = health_check()

    if health.get("status") != "healthy":
        st.error("⚠️ 백엔드 서버 연결 실패")
        st.info("백엔드 서버를 실행해주세요: `cd ai_backend && poetry run python -m app.main`")
        st.stop()

except Exception as e:
    st.error(f"⚠️ 백엔드 서버에 연결할 수 없습니다: {str(e)}")
    st.info("**Backend 서버 실행 방법:**\n```bash\ncd ai_backend\npoetry run python -m app.main\n```")
    st.stop()

#======================================================================================================================

# UI 구현부-1 : 로고
st.logo(
    "images/soundmind_CI_3.png",
    link="https://soundmind.life",
    icon_image="images/soundmind_CI_3.png",
    size="large"
)

# UI 구현부-2 : 채팅 창
st.title("[Soundmind] AI Chatbot")
user_input = st.chat_input("궁금한 내용을 물어보세요")


# UI 구현부-3 : 사이드바
with st.sidebar:
    ## 초기 경로 설정 값 (Prompt)
    prompt_files = glob.glob("../ai_backend/prompts/chatbot/*.yaml")

    ## 실제 UI 아이콘
    clear_btn = st.button("대화 초기화")
    selected_model = st.selectbox("LLM 선택", ["gpt-4.1", "gpt-4o", "gpt-4o-mini", "gpt-5", "gpt-5-mini", "gpt-5-nano"])
    selected_prompt = st.selectbox("Prompt 선택", prompt_files, index=0)
    task_input = st.text_input("역할 설정 (선택사항)", "", placeholder="예: SNS 마케팅 담당자")

#======================================================================================================================

# 세션 초기화
SESSION_ID = st.session_state["chatbot_session_id"]

# 세션 생성 유무 확인
if "chatbot_session_id" not in st.session_state:
    st.session_state["chatbot_session_id"] = "single_user_chat_session"

# 대화 기록 유무 확인
if "chatbot_messages" not in st.session_state:
    st.session_state["chatbot_messages"] = []

# 대화 기록 초기화
if clear_btn:
    st.session_state["chatbot_messages"] = []

    try:
        clear_chat_session(SESSION_ID)
        st.success("대화 기록이 초기화 되었습니다")
    except Exception as e:
        st.error(f"대화 기록 초기화가 실패하였습니다: {str(e)}")

#======================================================================================================================

# 유틸리티 함수
def print_messages():
    """이전 대화 출력"""
    for chat_message in st.session_state["chatbot_messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

def add_message(role, message):
    """새로운 메시지 추가"""
    st.session_state["chatbot_messages"].append(
        ChatMessage(role=role, content=message)
    )

#======================================================================================================================

# 구현부-1 : 이전 대화 기록 출력
print_messages()


# 구현부-2 : 새로운 사용자 입력 처리
if user_input:
    # 사용자 메시지 저장 및 표시
    add_message("user", user_input)
    st.chat_message("user").write(user_input)

    # AI 응답 생성 (Backend API 호출)
    try:
        response = chat_stream(
            session_id=SESSION_ID,
            message=user_input,
            model=selected_model,
            prompt_file=selected_prompt,
            task=task_input,
            temperature=0.0
        )

        # AI 메시지 스트리밍 출력
        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""

            for token in response:
                ai_answer += token
                container.write(ai_answer)

            # AI 메시지 저장
            add_message("assistant", ai_answer)

    except Exception as e:
        st.error(f"❌ AI 응답 생성 실패: {str(e)}")
        st.info("잠시 후 다시 시도해주세요.")
