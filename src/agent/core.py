import operator
import json
import logging
from functools import reduce
from typing import List, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain.llms.base import LLM
import httpx
from types_defs.state_type import SlotInfo, SubQueryInfo, OverallState, SubQueryWrapper
from utils import helper
from utils import prompts_mng
from search_tools import GraphSearcher, VectorSearcher
import copy
import time
import redis.asyncio as redis
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer
import numpy as np
from qdrant_client.http import models
import hashlib
import uuid


logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
 


class VLLM_API(LLM):
    endpoint: str = "http://localhost:9999/v1/chat/completions"
    model_name: str = "/google/gemma-3-27b-it-GPTQ-4b-128g/"
    temperature: float = 0.0
    max_tokens: int = 10000

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        if stop:
            payload["stop"] = stop

        response = requests.post(self.endpoint, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def _call_async(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        if stop:
            payload["stop"] = stop

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def stream_chat(self, messages: List[dict]):
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": True
            }

            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", self.endpoint, json=payload, headers=headers) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line.removeprefix("data: ").strip()
                            if data_str == "[DONE]":
                                yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}
                            else:
                                yield json.loads(data_str)

    @property
    def _llm_type(self) -> str:
        return "vllm_api"

class Pipeline:
    def __init__(self, 
                 redis_client, 
                 qdrant_client, 
                 embedding_model):
        self.redis_client = redis_client
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.llm = VLLM_API()
        self.vector_searcher = VectorSearcher()
        self.graph_searcher = GraphSearcher()
        self.graph = self._build_graph()
 
    @classmethod
    async def create(cls, 
                     redis_url="redis://localhost:6379", 
                     qdrant_url="http://localhost:6333",
                     model_path="/meta/all-MiniLM-L12-v2/"):

        redis_client = redis.from_url(redis_url, decode_responses=True)
        qdrant_client = AsyncQdrantClient(url=qdrant_url)


        await helper.qdrant_collection_exists(
            qdrant_client,
            "semantic_decompose",
            384  
        )

        embedding_model = SentenceTransformer(model_path)

        return cls(redis_client, qdrant_client, embedding_model)

    def build_cache_key(self, text: str, prefix: str = "cache:semantic_decompose_") -> str:
            return prefix + hashlib.sha256(text.encode("utf-8")).hexdigest()

    def build_qdrant_point_id(self) -> str:
            return str(uuid.uuid4())

    async def semantic_and_decompose(self, state: OverallState) -> OverallState:
        helper.print_fun_name()
        question = state.messages[-1]["content"]
        cache_key = self.build_cache_key(question)

        # 1. Redis cache
        cached = await self.redis_client.get(cache_key)
        if cached:
            logger.info("Redis Cache hit")
            payload = json.loads(cached)
            state.slots = SlotInfo(**payload["slots"]) if payload.get("slots") else None
            state.intents = payload.get("intents")
            sub_queries = payload.get("sub_queries")
            if sub_queries:
                state.sub_queries = [SubQueryInfo(**sq) for sq in sub_queries]
            return state

        # 2. Qdrant async search
        emb = self.embedding_model.encode(question).astype(np.float32).tolist()
        try:
            search_result = await self.qdrant_client.query_points(
                collection_name="semantic_decompose",
                query=emb,
                limit=1
            )
        except UnexpectedResponse as e:
            logger.error(f"Qdrant query_points failed: {e}")
            search_result = None
 
        SIMILARITY_THRESHOLD = 0.85

        if search_result and search_result.points:
            top_point = search_result.points[0]
            similarity = top_point.score  
            logger.info(f"Qdrant top point similarity: {similarity}")

            if similarity is not None and similarity > SIMILARITY_THRESHOLD:
                payload_json = top_point.payload.get("state_payload")
                if payload_json:
                    payload = json.loads(payload_json)
                    await self.redis_client.set(cache_key, json.dumps(payload), ex=3600)

                    state.slots = SlotInfo(**payload["slots"]) if payload.get("slots") else None
                    state.intents = payload.get("intents")
                    sub_queries = payload.get("sub_queries")
                    if sub_queries:
                        state.sub_queries = [SubQueryInfo(**sq) for sq in sub_queries]

                    logger.info("Qdrant vector search hit with valid similarity")
                    return state

            logger.info("Qdrant result similarity below threshold, fallback to LLM")

        # 3. LLM fallback
        prompt = prompts_mng.build_semantic_decompose_prompt(question)
        response = await self.llm._call_async(prompt)

        try:
            parsed = json.loads(helper.extract_json(response))
        except json.JSONDecodeError:
            logger.error("[semantic_and_decompose] JSON parse error")
            raise

        slots_data = parsed.get("slots", {})
        state.slots = SlotInfo(**slots_data) if isinstance(slots_data, dict) else None
        state.intents = parsed.get("intents")

        sub_queries = []
        raw_subs = parsed.get("sub_questions", [])
        for q in raw_subs:
            try:
                slot_dict = q.pop("slots", {})
                slot_obj = SlotInfo(**slot_dict)
                subq = SubQueryInfo(**q, slots=[slot_obj])
                subq.step_log.append("decomposed")
                sub_queries.append(subq)
            except Exception as e:
                logger.error(f"[semantic_and_decompose] Error parsing sub-query: {e}")
                raise
        if sub_queries:
            state.sub_queries = sub_queries
            state.slots = reduce(operator.add, [sq.slots[0] for sq in sub_queries], SlotInfo())

        # write back to Redis & Qdrant cache （async）
        payload = {
            "slots": state.slots.model_dump() if state.slots else None,
            "intents": state.intents,
            "sub_queries": [sq.model_dump() for sq in state.sub_queries] if state.sub_queries else None,
        }

        payload_str = json.dumps(payload)
        await self.redis_client.set(cache_key, payload_str, ex=3600)
        await self.qdrant_client.upsert(
            collection_name="semantic_decompose",
            points=[
                models.PointStruct(
                    id=self.build_qdrant_point_id(),
                    vector=emb,
                    payload={"state_payload": payload_str}
                )
            ]
        )

        return state


    async def query_spec(self, wrapper: SubQueryWrapper) -> SubQueryInfo:

        print(wrapper.subquery.query)
        await self.vector_searcher.run_spec(wrapper.subquery)
        wrapper.subquery.step_log.append("queried_spec")
        return wrapper.subquery

    async def query_faq(self, wrapper: SubQueryWrapper) -> SubQueryInfo:

        print(wrapper.subquery.query)
        await self.vector_searcher.run_faq(wrapper.subquery)
        wrapper.subquery.step_log.append("queried_faq")
        return wrapper.subquery


    async def query_verified_comp(self, wrapper: SubQueryWrapper) -> SubQueryInfo:

        print(wrapper.subquery.query)
        #await self.vector_searcher.run_faq(wrapper.subquery)
        wrapper.subquery.step_log.append("queried_faq")
        return wrapper.subquery


    async def query_graph(self, wrapper: SubQueryWrapper) -> SubQueryInfo:
        print(wrapper.subquery.query)
        return wrapper.subquery
        await self.graph_searcher.run(wrapper.subquery)
        wrapper.subquery.step_log.append("queried_graph")
    

        

    async def merge_subanswers(self, state: OverallState) -> OverallState:


        question = state.messages[-1]["content"]
        state.sub_answers = [sq.sub_answer or "" for sq in state.sub_queries]

        prompt = prompts_mng.build_merge_prompt(question, state.sub_answers)
        final = await self.llm._call_async(prompt)

        if helper.extract_fail_flag(final):
            logger.warning("[merge_subanswers] LLM returned FAIL")
            metadata = state.sub_answers[0][0]["metadata"]
            filtered_metadata = {k: v for k, v in metadata.items() if k != "tags"}
            graph_query = SubQueryInfo(query=question, metadata=filtered_metadata)
            await self.query_graph(SubQueryWrapper(subquery=graph_query))
        else:
            state.final_answer = final.strip()

        return state

    def _build_graph(self):
        builder = StateGraph(OverallState)

        builder.add_node("semantic_and_decompose", self.semantic_and_decompose)
        builder.add_node("merge_subanswers", self.merge_subanswers)
        builder.add_node("query_spec", self.query_spec)
        builder.add_node("query_faq", self.query_faq)
        builder.add_node("query_graph", self.query_graph)
        builder.add_node("query_verified_comp", self.query_verified_comp)

        
        builder.add_edge(START, "semantic_and_decompose")

        def fan_out_and_route(state: OverallState) -> List[Send]:
            sends = []
            for subq in state.sub_queries:
                for source in subq.query_sources:
                    node = {
                        "vector_spec": "query_spec",
                        "vector_faq": "query_faq",
                        "graph_spec": "query_graph",
                        "sql_verified_component":"query_verified_comp"
                    }.get(source, "query_faq")  
                    sends.append(Send(node, SubQueryWrapper(subquery=subq)))
            return sends

        builder.add_conditional_edges("semantic_and_decompose", fan_out_and_route, ["query_spec", "query_faq", "query_graph","query_verified_comp"])
        builder.add_edge("query_spec", "merge_subanswers")
        builder.add_edge("query_faq", "merge_subanswers")
        builder.add_edge("query_graph", "merge_subanswers")
        builder.add_edge("query_verified_comp", "merge_subanswers")
        builder.add_edge("merge_subanswers", END)

        return builder.compile()

    async def run(self, state: OverallState):
        return await self.graph.ainvoke(state)


    async def close(self):
 
        if self.redis_client:
            await self.redis_client.close()

        if self.qdrant_client:
            await self.qdrant_client.close()




if __name__ == "__main__":
    import asyncio
    from types_defs.message_type import ChatRequest, Message

    async def main():
        pipeline = await Pipeline.create()
        input_state = OverallState(
            messages=[
                {"role": "user", "content": "I have trouble finding the heatsink for this MB X13DAI-T. I don’t see any references to a heatsink for this MB on your website."}
            ]
        )
        result = await pipeline.run(input_state)
        print("\nFinal Answer:\n", result.get("final_answer"))

    asyncio.run(main())