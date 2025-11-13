# ============================================
# Chain Factory - LangChain 체인 생성
# ============================================

import glob
import os
import yaml
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from app.session_manager import get_session_history
from app.config import (
    CHATBOT_PROMPTS_DIR,
    RAG_PROMPTS_DIR,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP,
)

# ============================================
# 프롬프트 관리
# ============================================


def get_available_prompts(prompt_type: str = "chatbot") -> list:
    """
    사용 가능한 프롬프트 파일 목록 조회

    Args:
        prompt_type: 프롬프트 타입 ("chatbot" 또는 "rag")

    Returns:
        list: 프롬프트 파일 경로 리스트
    """
    if prompt_type == "chatbot":
        prompt_dir = CHATBOT_PROMPTS_DIR
    elif prompt_type == "rag":
        prompt_dir = RAG_PROMPTS_DIR
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")

    prompt_files = glob.glob(os.path.join(prompt_dir, "*.yaml"))
    # 경로를 상대 경로로 변환 (prompts/chatbot/01-general.yaml 형식)
    return [file.replace("\\", "/") for file in prompt_files]


# ============================================
# Chatbot Chain 생성
# ============================================


def create_chatbot_chain(
    prompt_file: str,
    model: str = DEFAULT_MODEL,
    task: str = "",
    temperature: float = DEFAULT_TEMPERATURE,
):
    """
    Chatbot Chain 생성

    Args:
        prompt_file: 프롬프트 파일 경로 (예: "prompts/chatbot/01-general.yaml")
        model: LLM 모델명
        task: 역할 설정 (선택사항)
        temperature: 생성 온도

    Returns:
        RunnableWithMessageHistory: 대화 히스토리를 포함한 체인
    """
    # YAML 파일에서 프롬프트 직접 로드
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_data = yaml.safe_load(f)

    # 템플릿 텍스트 추출
    template_text = prompt_data.get("template", "")
    print(f"[DEBUG] template_text type: {type(template_text)}")
    print(f"[DEBUG] template_text: {repr(template_text)}")

    # #Question: 이전 부분만 추출 (시스템 메시지용)
    if "#Question:" in template_text:
        base_prompt = template_text.split("#Question:")[0].strip()
        print(f"[DEBUG] Split succeeded, base_prompt: {repr(base_prompt)}")
    else:
        base_prompt = template_text.replace("{question}", "").strip()
        print(f"[DEBUG] No #Question: found, base_prompt: {repr(base_prompt)}")

    # 역할 설정이 있으면 추가
    if task and task.strip():
        system_message = (
            f"{base_prompt}\n\n"
            f"system role: {task}\n"
            f"위 역할에 맞게 전문적으로 대답해주세요."
        )
    else:
        system_message = base_prompt

    print(f"[DEBUG] system_message: {repr(system_message)}")

    # 프롬프트 템플릿 구성
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    # LLM 생성
    llm = ChatOpenAI(model=model, temperature=temperature)

    # 파서 생성
    output_parser = StrOutputParser()

    # Chain 구성
    chain = prompt | llm | output_parser

    # 히스토리 포함 Chain
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return chain_with_history


# ============================================
# RAG Pipeline
# ============================================


def create_rag_retriever(
    file_path: str,
    chunk_size: int = RAG_CHUNK_SIZE,
    chunk_overlap: int = RAG_CHUNK_OVERLAP,
):
    """
    PDF 파일을 로드하고 벡터 검색기 생성 (RAG 1~5단계)

    Args:
        file_path: PDF 파일 경로
        chunk_size: 문서 분할 청크 크기
        chunk_overlap: 청크 간 중복 크기

    Returns:
        retriever: FAISS 기반 벡터 검색기
    """
    # RAG 로직-1: 문서 로드
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    # RAG 로직-2: 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_documents = text_splitter.split_documents(docs)

    # RAG 로직-3: 임베딩 생성
    embeddings = OpenAIEmbeddings()

    # RAG 로직-4: 벡터 DB 생성
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # RAG 로직-5: 검색기 생성
    retriever = vectorstore.as_retriever()

    return retriever


def create_rag_chain(
    prompt_file: str,
    retriever,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
):
    """
    RAG Chain 생성 (RAG 6~8단계)

    Args:
        prompt_file: 프롬프트 파일 경로
        retriever: 벡터 검색기
        model: LLM 모델명
        temperature: 생성 온도

    Returns:
        chain: RAG 체인
    """
    # RAG 로직-6: 프롬프트 정의
    # YAML 파일에서 프롬프트 직접 로드
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_data = yaml.safe_load(f)

    # PromptTemplate 객체 생성
    template_text = prompt_data.get("template", "")
    input_vars = prompt_data.get("input_variables", ["context", "question"])
    prompt = PromptTemplate(template=template_text, input_variables=input_vars)

    # RAG 로직-7: LLM 생성
    llm = ChatOpenAI(model=model, temperature=temperature)

    # RAG 로직-8: LCEL 체인 구성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
