import asyncio
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from contextlib import asynccontextmanager
from types_defs.message_type import ChatRequest, Message
from types_defs.state_type import OverallState
from core import Pipeline


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


@app.post("/v1/chat/completions")
async def proxy_to_graph(request: ChatRequest):
    messages = [m.dict() for m in request.messages]
    input_state = OverallState(messages=messages)
    done_event = asyncio.Event()
    result_container = {"answer": None}

    async def run_graph():
        final_state = await core_pipeline.run(input_state)

        print(final_state)

        result_container["answer"] = final_state.get("final_answer") or "Sorry, no answer found."
        done_event.set()

    asyncio.create_task(run_graph())

    if request.stream:
        async def event_stream():
            await asyncio.sleep(0.3)
            try:
                await asyncio.wait_for(done_event.wait(), timeout=30)
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'choices': [{'delta': {'content': 'Timeout waiting for answer.'}, 'index': 0}]})}\n\n"
                yield "data: [DONE]\n\n"
                return

            answer = result_container["answer"]
            for token in answer:
                yield f"data: {json.dumps({'choices': [{'delta': {'content': token}, 'index': 0}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        port=7777,
        reload=True
    )
