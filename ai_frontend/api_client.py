# ============================================
# API Client - Backend API 통신
# ============================================

import requests
import json
from typing import Iterator, Optional

# ============================================
# 설정
# ============================================

BACKEND_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"

# ============================================
# Chatbot API
# ============================================

def chat_stream(
        session_id: str,
        message: str,
        model: str,
        prompt_file: str,
        task: str = "",
        temperature: float = 0.0
    ) -> Iterator[str]:
    """
    스트리밍 채팅

    Args:
        session_id: 세션 ID
        message: 사용자 메시지
        model: LLM 모델명
        prompt_file: 프롬프트 파일 경로
        task: 역할 설정
        temperature: 생성 온도

    Yields:
        str: 응답 토큰
    """
    url = f"{BACKEND_URL}{API_PREFIX}/chat/stream"
    payload = {
        "session_id": session_id,
        "message": message,
        "model": model,
        "prompt_file": prompt_file,
        "task": task,
        "temperature": temperature
    }

    try:
        with requests.post(url, json=payload, stream=True, timeout=60) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith('data: '):
                        data_str = decoded[6:]  # 'data: ' 제거
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if 'token' in data:
                                yield data['token']
                            elif 'error' in data:
                                raise Exception(data['error'])
                        except json.JSONDecodeError:
                            continue
    except requests.exceptions.RequestException as e:
        raise Exception(f"API 요청 실패: {str(e)}")


def get_chat_prompts() -> list:
    """
    프롬프트 목록 조회

    Returns:
        list: 프롬프트 파일 경로 리스트
    """
    url = f"{BACKEND_URL}{API_PREFIX}/chat/prompts"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()["prompts"]
    except requests.exceptions.RequestException as e:
        raise Exception(f"프롬프트 목록 조회 실패: {str(e)}")


def clear_chat_session(session_id: str) -> dict:
    """
    채팅 세션 초기화

    Args:
        session_id: 세션 ID

    Returns:
        dict: 응답 메시지
    """
    url = f"{BACKEND_URL}{API_PREFIX}/chat/session/{session_id}"
    try:
        response = requests.delete(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"세션 초기화 실패: {str(e)}")


# ============================================
# RAG API
# ============================================

def upload_pdf(session_id: str, file) -> dict:
    """
    PDF 파일 업로드

    Args:
        session_id: 세션 ID
        file: 업로드할 파일 객체

    Returns:
        dict: 업로드 응답
    """
    url = f"{BACKEND_URL}{API_PREFIX}/rag/upload"
    files = {"file": file}
    data = {"session_id": session_id}

    try:
        response = requests.post(url, files=files, data=data, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"파일 업로드 실패: {str(e)}")


def rag_query_stream(
    session_id: str,
    question: str,
    model: str,
    prompt_file: str,
    temperature: float = 0.0
) -> Iterator[str]:
    """
    RAG 질의 (스트리밍)

    Args:
        session_id: 세션 ID
        question: 사용자 질문
        model: LLM 모델명
        prompt_file: 프롬프트 파일 경로
        temperature: 생성 온도

    Yields:
        str: 응답 토큰
    """
    url = f"{BACKEND_URL}{API_PREFIX}/rag/query"
    payload = {
        "session_id": session_id,
        "question": question,
        "model": model,
        "prompt_file": prompt_file,
        "temperature": temperature
    }

    try:
        with requests.post(url, json=payload, stream=True, timeout=60) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith('data: '):
                        data_str = decoded[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if 'token' in data:
                                yield data['token']
                            elif 'error' in data:
                                raise Exception(data['error'])
                        except json.JSONDecodeError:
                            continue
    except requests.exceptions.RequestException as e:
        raise Exception(f"RAG 질의 실패: {str(e)}")


def get_rag_prompts() -> list:
    """
    RAG 프롬프트 목록 조회

    Returns:
        list: 프롬프트 파일 경로 리스트
    """
    url = f"{BACKEND_URL}{API_PREFIX}/rag/prompts"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()["prompts"]
    except requests.exceptions.RequestException as e:
        raise Exception(f"RAG 프롬프트 목록 조회 실패: {str(e)}")


def check_rag_session(session_id: str) -> dict:
    """
    RAG 세션에 업로드된 문서 확인

    Args:
        session_id: 세션 ID

    Returns:
        dict: 문서 정보
    """
    url = f"{BACKEND_URL}{API_PREFIX}/rag/session/{session_id}/document"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"세션 확인 실패: {str(e)}")


# ============================================
# 헬스 체크
# ============================================

def health_check() -> dict:
    """
    Backend 서버 헬스 체크

    Returns:
        dict: 서버 상태
    """
    url = f"{BACKEND_URL}/health"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "unhealthy", "error": str(e)}
