# ============================================
# RAG System - Backend API 통신 구조
# ============================================

import os
import glob
import sys
sys.path.append('..')

import time
import streamlit as st
from langchain_core.messages.chat import ChatMessage
from api_client import (
    upload_pdf,
    rag_query_stream,
    get_rag_prompts,
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

# UI 구현부-2 : 채팅창
st.title("[Soundmind] RAG System")
user_input = st.chat_input("궁금한 내용을 물어보세요")
warning_msg = st.empty()


# UI 구현부-3 : Sidebar
with st.sidebar:
    ## 파일 업로더
    uploade_file = st.file_uploader("", type=["PDF"])

    ## 초기 경로 설정 값 (Prompt)
    prompt_files = glob.glob("prompts/rag/*.yaml")

    st.markdown("## [RAG Custom]")
    selected_model = st.selectbox("LLM 선택", ["gpt-4.1", "gpt-4o", "gpt-4o-mini", "gpt-5", "gpt-5-mini", "gpt-5-nano"])
    selected_api = st.selectbox("Documents Loader 선택", ["PDFPlumberLoader", "UpstageDocumentParseLoader"])
    selected_prompt = st.selectbox("Prompt 선택", prompt_files, index=0)
    selected_rag = st.selectbox("RAG 기술 선택", ["Naive RAG", "Advanced RAG", "Moduler RAG"])
    selected_parser = st.selectbox("OutputParser 선택", ["StrOutputParser"])
    clear_btn = st.button("대화 초기화")

#======================================================================================================================

# 세션 생성 유무 확인 (타임스탬프 기반)
if "rag_session_id" not in st.session_state:
    st.session_state["rag_session_id"] = "single_user_rag_session"

# 질의 기록 유무 확인
if "rag_messages" not in st.session_state:
    st.session_state["rag_messages"] = []

# 파일 업로드 유무 확인
if "rag_uploaded" not in st.session_state:
    st.session_state["rag_uploaded"] = False

# 세션 초기화
SESSION_ID = st.session_state["rag_session_id"]

# 내용 초기화
if clear_btn:
    st.session_state["rag_session_id"] = ""
    st.session_state["rag_messages"] = []
    st.session_state["rag_uploaded"] = False
    st.success("✓ 대화 기록과 업로드 문서가 초기화되었습니다")

#====================================================================================================================== 

# 유틸리티 함수
def print_messages():
    """이전 대화 출력"""
    for chat_message in st.session_state["rag_messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

def add_message(role, message):
    """새로운 메시지 추가"""
    st.session_state["rag_messages"].append(
        ChatMessage(role=role, content=message)
    )

#======================================================================================================================       

# 구현부-1 : 이전 대화 기록 출력
print_messages()

# 구현부-2 : 업로드 파일 AI Backend API 처리
if uploade_file and not st.session_state["rag_uploaded"]:
    with st.spinner(f"📄[{uploade_file.name}] 파일을 업로드 중입니다..."):

        try:
            result = upload_pdf(session_id=SESSION_ID, file=uploade_file)
            st.session_state["rag_uploaded"] = True
            st.success(f"✅[{result['filename']}] 업로드 완료")

        except Exception as e:
            st.session_state["rag_uploaded"] = False
            st.error(f"파일 업로드 실패: {str(e)}")


# 구현부-3 : 새로운 사용자 입력 처리
if user_input:
    if st.session_state.get("rag_uploaded"):

        # 사용자 메시지 저장 및 표시
        add_message("user", user_input)
        st.chat_message("user").write(user_input)

        # RAG 질의 (Backend API 호출)
        try:
            response = rag_query_stream(
                session_id=SESSION_ID,
                question=user_input,
                model=selected_model,
                prompt_file=selected_prompt,
                temperature=0.0
            )

            # AI 메시지 스트리밍 출력
            ai_answer = ""
            with st.chat_message("assistant"):
                container = st.empty()

                for token in response:
                    ai_answer += token
                    container.markdown(ai_answer)

            # AI 메시지 저장
            add_message("assistant", ai_answer)

        except Exception as e:
            st.error(f"❌ RAG 질의 실패: {str(e)}")
            st.info("잠시 후 다시 시도해주세요.")

    else:
        warning_msg.error("⚠️ 참고 자료를 업로드해주세요")
