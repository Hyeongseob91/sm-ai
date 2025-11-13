# ============================================
# Chat API - Chatbot REST API
# ============================================

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json
import asyncio
from app.chain_factory import create_chatbot_chain, get_available_prompts
from app.session_manager import clear_session, session_exists

router = APIRouter()

# ============================================
# 데이터 모델
# ============================================

class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    session_id: str
    message: str
    model: str = "gpt-4o"
    prompt_file: str
    task: Optional[str] = ""
    temperature: Optional[float] = 0.0


class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    session_id: str
    message: str
    role: str = "assistant"


# ============================================
# API 엔드포인트
# ============================================

@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    스트리밍 채팅 응답 (Server-Sent Events)

    사용자의 메시지에 대해 실시간으로 토큰 단위로 응답을 스트리밍합니다.
    """
    try:
        # Chain 생성
        chain = create_chatbot_chain(
            prompt_file=request.prompt_file,
            model=request.model,
            task=request.task,
            temperature=request.temperature
        )

        # 스트리밍 응답 생성
        async def generate():
            try:
                config = {"configurable": {"session_id": request.session_id}}

                # astream을 사용한 비동기 스트리밍
                async for chunk in chain.astream(
                    {"question": request.message},
                    config=config
                ):
                    yield f"data: {json.dumps({'token': chunk})}\n\n"

                # 스트리밍 종료 신호
                yield "data: [DONE]\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"ERROR in chat_stream: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message", response_model=ChatResponse)
async def chat_message(request: ChatRequest):
    """
    일반 채팅 응답 (non-streaming)

    전체 응답이 완성된 후 한 번에 반환합니다.
    """
    try:
        # Chain 생성
        chain = create_chatbot_chain(
            prompt_file=request.prompt_file,
            model=request.model,
            task=request.task,
            temperature=request.temperature
        )

        # 응답 생성
        config = {"configurable": {"session_id": request.session_id}}
        response = await chain.ainvoke(
            {"question": request.message},
            config=config
        )

        return ChatResponse(
            session_id=request.session_id,
            message=response,
            role="assistant"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prompts")
async def list_prompts():
    """
    사용 가능한 프롬프트 파일 목록 조회
    """
    try:
        prompts = get_available_prompts("chatbot")
        return {"prompts": prompts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    세션 초기화

    특정 세션의 대화 히스토리를 삭제합니다.
    """
    success = clear_session(session_id)

    if success:
        return {"message": "Session cleared successfully", "session_id": session_id}
    else:
        return {"message": "Session not found", "session_id": session_id}


@router.get("/session/{session_id}/exists")
async def check_session(session_id: str):
    """
    세션 존재 여부 확인
    """
    exists = session_exists(session_id)
    return {"session_id": session_id, "exists": exists}
