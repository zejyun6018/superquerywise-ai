import asyncio
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from contextlib import asynccontextmanager
from src.agent.types_defs.message_type import ChatRequest, Message
from src.agent.types_defs.state_type import OverallState
from src.agent.core import Pipeline


core_pipeline: Optional[Pipeline] = None


async def init_services():
    global core_pipeline
    core_pipeline =  await Pipeline.create()
    print("✅ All Initialized.") 


async def shutdown_services():
    print("✅ Shutting down...")
    await core_pipeline.close()
    import gc
    gc.collect()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_services()
    try:
        yield
    finally:
        await shutdown_services()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://10.184.16.27:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "gemma-3-27b-it",
            "object": "model",
            "created": 0,
            "owned_by": "local"
        }]
    }

def extract_final_answer(data):
    """
    從巢狀 dict 中遞迴尋找 final_answer 字串。
    若找不到則回傳 None。
    """
    if not isinstance(data, dict):
        return None

    # 如果本層有 final_answer 就回傳
    if "final_answer" in data:
        return data["final_answer"]

    # 否則遞迴搜尋所有子值（value）
    for value in data.values():
        if isinstance(value, dict):
            result = extract_final_answer(value)
            if result is not None:
                return result

    # 找不到的話回傳 None
    return None



@app.post("/v1/chat/completions")
async def proxy_to_graph(request: ChatRequest):
    messages = [m.dict() for m in request.messages]
    input_state = OverallState(messages=messages)

    if request.stream:
        async def event_stream():

            previous = ""
            
            try:
                async for chunk in core_pipeline.run_stream(input_state):      

                    data = chunk.get("merge_subanswers", {})
                    full_answer = data.get("final_answer", "")

                    if hasattr(full_answer, "item"):
                        full_answer = full_answer.item()

                    delta = full_answer[len(previous):]
                    previous = full_answer

                    for token in delta:
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': token}, 'index': 0}]})}\n\n"

                    await asyncio.sleep(0)


            except Exception as e:
                yield f"data: {json.dumps({'choices': [{'delta': {'content': f'Error: {str(e)}'}, 'index': 0}]})}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    else:
        final_state = await core_pipeline.run(input_state)
        answer = final_state.get("final_answer") or "Sorry, no answer found."
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": answer
                    },
                    "finish_reason": "stop",
                    "index": 0
                }
            ],
            "usage": {},
            "model": core_pipeline.llm.model_name
        }
        return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        port=7775  ,
        reload=True
    )
