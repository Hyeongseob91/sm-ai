import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="SOUNDMIND AI System",
    page_icon="🧠",
    layout="wide"
)

# 페이지 로고
st.logo(
    "soundmind_CI_3.png",
    link="https://soundmind.life",
    icon_image="soundmind_CI_3.png",
    size="large")

# 메인 타이틀
st.title("🧠 Soundmind AI System")
st.markdown("## LangChain 기반 AI 솔루션 플랫폼")
st.markdown("---")

# 프로젝트 소개
st.markdown("""
SOUNDMIND AI System은 LangChain과 OpenAI를 활용한 AI 솔루션 플랫폼입니다.
좌측 사이드바에서 원하는 기능을 선택하여 사용할 수 있습니다.
""")

st.markdown("##")

# 기능 소개 섹션
st.header("📌 주요 기능")
st.markdown("##")

# 2개의 컬럼으로 기능 카드 배치
col1, col2 = st.columns(2)

with col1:
    st.container()
    st.markdown("""
    <div style="padding: 20px; border-radius: 10px; border: 2px solid #4CAF50;">
        <h3>💬 AI Chatbot</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 기능 설명")
    st.markdown("""
    LangChain 기반의 대화형 AI 챗봇입니다. 사용자와 자연스러운 대화를 나누며 다양한 질문에 답변합니다.
    """)

    st.markdown("#### 주요 특징")
    st.markdown("""
    - ✅ **대화 히스토리 유지**: 이전 대화 내용을 기억하여 맥락있는 대답
    - ✅ **프롬프트 커스터마이징**: YAML 파일로 AI의 성격과 역할 설정
    - ✅ **역할 설정**: 특정 분야 전문가로 AI 역할 지정 가능
    - ✅ **다양한 LLM 모델 선택**: GPT-4, GPT-5 등 여러 모델 지원
    """)

    st.page_link("pages/01_AI_Chatbot.py", label="🚀 AI Chatbot 사용하기", icon="💬")

with col2:
    st.container()
    st.markdown("""
    <div style="padding: 20px; border-radius: 10px; border: 2px solid #2196F3;">
        <h3>📚 RAG System</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 기능 설명")
    st.markdown("""
    PDF 문서를 기반으로 질의응답하는 RAG(Retrieval-Augmented Generation) 시스템입니다.
    """)

    st.markdown("#### 주요 특징")
    st.markdown("""
    - ✅ **PDF 문서 업로드**: 원하는 문서를 업로드하여 지식 베이스 구축
    - ✅ **벡터 검색**: FAISS를 활용한 효율적인 문서 검색
    - ✅ **컨텍스트 기반 답변**: 업로드한 문서 내용을 바탕으로 정확한 답변
    - ✅ **다양한 RAG 방식**: Naive, Advanced, Modular RAG 선택 가능
    """)

    st.page_link("pages/02_RAG_System.py", label="🚀 RAG System 사용하기", icon="📚")

st.markdown("##")
st.markdown("---")

# 사용 방법 섹션
with st.expander("📖 사용 방법", expanded=False):
    st.markdown("""
    ### 1️⃣ 좌측 사이드바에서 기능 선택
    - **AI Chatbot**: 일반적인 대화형 AI 챗봇
    - **RAG System**: 문서 기반 질의응답 시스템

    ### 2️⃣ AI Chatbot 사용하기
    1. 좌측 사이드바에서 **AI Chatbot** 페이지로 이동
    2. 원하는 **LLM 모델**과 **프롬프트** 선택
    3. 필요시 **역할 설정**을 입력 (예: "SNS 마케팅 담당자")
    4. 채팅창에서 질문 입력
    5. **대화 초기화** 버튼으로 대화 기록 삭제 가능

    ### 3️⃣ RAG System 사용하기
    1. 좌측 사이드바에서 **RAG System** 페이지로 이동
    2. **PDF 파일 업로드**
    3. 원하는 **RAG 방식**, **프롬프트**, **LLM 모델** 선택
    4. 업로드한 문서에 대해 질문 입력
    5. **대화 초기화** 버튼으로 대화 기록 삭제 가능
    """)

# 기술 스택 섹션
with st.expander("🛠️ 기술 스택", expanded=False):
    st.markdown("""
    ### Frontend
    - **Streamlit**: 웹 애플리케이션 프레임워크

    ### AI/ML
    - **LangChain**: LLM 애플리케이션 개발 프레임워크
    - **OpenAI**: GPT 모델 API (ChatGPT)
    - **LangChain TeddyNote**: 한국어 프롬프트 및 유틸리티

    ### Vector Store & Embeddings
    - **FAISS**: 벡터 유사도 검색 라이브러리
    - **OpenAI Embeddings**: 텍스트 임베딩 생성

    ### Document Processing
    - **PDFPlumber**: PDF 문서 로딩
    - **RecursiveCharacterTextSplitter**: 문서 분할

    ### Utilities
    - **python-dotenv**: 환경 변수 관리
    - **LangSmith**: LangChain 모니터링 및 디버깅
    """)

# 푸터
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>🧠 SOUNDMIND AI System | Powered by LangChain & OpenAI</p>
</div>
""", unsafe_allow_html=True)
