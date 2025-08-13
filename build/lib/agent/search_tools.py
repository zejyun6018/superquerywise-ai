from src.agent.types_defs.state_type import SubQueryInfo
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from uuid import uuid4
import json
import asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer, CrossEncoder
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import time
from collections import defaultdict
import numpy as np

class ToolInput(BaseModel):
    query_id: str = Field(..., description="Task tracking ID returned by the LLM")
    query: str = Field(..., description="User input question")
    context: Optional[Dict[str, Any]] = Field(None, description="User input context")


class GraphSearcher:
    def __init__(self, url: str = "http://localhost:8987/mcp/", timeout: int = 30):
        self.client = MultiServerMCPClient({
            "graph_db": {
                "url": url,
                "transport": "streamable_http",
                "timeout": timeout
            }
        })
        self.tool_name = "get_product_info_data"
    
    async def run(self, sub_query: SubQueryInfo) -> SubQueryInfo:
        query_text = sub_query.query
        query_context = sub_query.metadata
        print(f"[🔍] search_graph_with_slot called ----------- {query_text}")

        async with self.client.session("graph_db") as session:
            tools = await load_mcp_tools(session)

            tool = tools[0] if tools else None
            if tool is None:
                print(f"Tool '{self.tool_name}' not found.")
                sub_query.sub_answer = "None"
                return sub_query

            tool_input = ToolInput(
                query_id=str(uuid4()),
                query=query_text,
                context=query_context
            )
            
            response_raw = await tool.arun({"input": tool_input.dict()})
            print("Tool result:", response_raw)

            try:
                response = json.loads(response_raw)
            except json.JSONDecodeError as e:
                print("❌ JSON decode error:", e)
                response = {}

            if response.get("success"):
                print("------tool_response----OKOK\n")
                sub_query.sub_answer = response.get("result")
            else:
                sub_query.sub_answer = "None"

            return sub_query



# async def main():
#     searcher = GraphSearcher()
#     sub_query = SubQueryInfo(query="your query here", metadata={})
#     result = await searcher.search_graph(sub_query)
#     print(result.sub_answer)
#
# asyncio.run(main())




class VectorSearcher:
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        spec_collection: str = "supermicro_product_spec",
        faq_collection: str = "supermicro_faq",
        semantic_model_path: str = "/meta/paraphrase-multilingual-mpnet-base-v2",
        cross_encoder_path: str = "/meta/ms-marco-MiniLM-L12-v2",
        max_workers: int = 100,
    ):
        self.qdrant = AsyncQdrantClient(host=qdrant_host, port=qdrant_port)
        self.spec_collection = spec_collection
        self.faq_collection = faq_collection
        self.semantic_model = SentenceTransformer(semantic_model_path)
        self.cross_encoder = CrossEncoder(cross_encoder_path)
        self.rerank_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.BATCH_SIZE = 8
        self.MIN_FUSED_SCORE_THRESHOLD = 0.1
        self.VECTOR_SCORE_THRESHOLD = 0.1

    async def async_encode(self, text: str) -> List[float]:
        loop = asyncio.get_running_loop()
        try:
            vec = await loop.run_in_executor(None, partial(self.semantic_model.encode, text))
            return vec
        except Exception as e:
            print(f"[❌ ERROR] async_encode failed: {e}")
            return []

    async def async_encode_batch(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_running_loop()
        try:
            # 使用 batch encode, 不顯示進度條
            func = partial(self.semantic_model.encode, texts, show_progress_bar=False)
            vecs = await loop.run_in_executor(None, func)
            return vecs
        except Exception as e:
            print(f"[❌ ERROR] async_encode_batch failed: {e}")
            return []



    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    async def rerank_with_cross_encoder(self, pairs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.rerank_executor, self.cross_encoder.predict, pairs)





    async def merge_contents(self, items):
        merged = []
        for item in items:
            content = item.get("content", "")
            timestamp = item.get("timestamp", "")
            merged.append({
                "content": content,
                "timestamp": timestamp
            })
        return merged

    async def merge_tags(self, items):
        tag_set = set()
        for item in items:
            tags = item.get("metadata", {}).get("tags", [])
            for t in tags:
                tag_set.add(t)
        return list(tag_set)

    async def group_and_slim(self, items):
        start_time = time.perf_counter()

        grouped = defaultdict(list)
        for item in items:
            meta = item.get("metadata", {})
            key = (meta.get("product_name"), meta.get("doc_type"))
            if not key[0] or not key[1]:
                continue
            grouped[key].append(item)

        results = []
        for (product_name, doc_type), group_items in grouped.items():
            base_meta = group_items[0].get("metadata", {})
            merged_tags = await self.merge_tags(group_items)
            slim_meta = {
                "product_brand": base_meta.get("product_brand"),
                "product_generation": base_meta.get("product_generation"),
                "product_family": base_meta.get("product_family"),
                "product_name": product_name,
                "product_type": base_meta.get("product_type"),
                "doc_type": doc_type,
                "tags": merged_tags,
            }
            merged_content = await self.merge_contents(group_items)
            results.append({
                "qa_list": merged_content,
                "metadata": slim_meta
            })

        elapsed = time.perf_counter() - start_time
        print(f"--group_and_slim--elapsed: {elapsed:.3f} seconds")
        return results

    async def dedup_and_slim_spec_results(self, sub_query: SubQueryInfo) -> SubQueryInfo:
        new_sub_answer = await self.group_and_slim(sub_query.sub_answer)
        sub_query.sub_answer = new_sub_answer
        return sub_query


    async def rerank_spec_results(self, sub_query: SubQueryInfo, top_k: Optional[int] = None) -> SubQueryInfo:
        start_time = time.perf_counter()
        question = sub_query.query
        items = sub_query.sub_answer

        if not items:
            return sub_query

        try:

            filtered_items = [item for item in items if item.get("score", 0.0) >= self.VECTOR_SCORE_THRESHOLD]
            if not filtered_items:
                print(f"[Info] No items passed vector similarity threshold {self.VECTOR_SCORE_THRESHOLD}")
                sub_query.sub_answer = []
                return sub_query


            pairs = [(question, f"{item.get('content', '')}\n{json.dumps(item.get('metadata', {}))}") for item in filtered_items]


            scores = []
            for i in range(0, len(pairs), self.BATCH_SIZE):
                batch = pairs[i:i + self.BATCH_SIZE]
                batch_scores = await self.rerank_with_cross_encoder(batch)
                scores.extend(batch_scores)

            reranker_scores = [self.sigmoid(s) for s in scores]

            for item, score in zip(filtered_items, reranker_scores):
                item["reranker_score"] = float(score)

            vector_scores = [item.get("score", 0.0) for item in filtered_items]

            def normalize(arr):
                arr = np.array(arr)
                min_v = arr.min()
                max_v = arr.max()
                if max_v == min_v:
                    return np.zeros_like(arr)
                return (arr - min_v) / (max_v - min_v)


            norm_vector = normalize(vector_scores)
            norm_reranker = normalize(reranker_scores)

            weight_vector = 0.6
            weight_reranker = 0.4

            for i, item in enumerate(filtered_items):
                fused = weight_vector * norm_vector[i] + weight_reranker * norm_reranker[i]
                item["fused_score"] = fused


            items_sorted = sorted(filtered_items, key=lambda x: x["fused_score"], reverse=True)


            filtered_top_items = [item for item in items_sorted if item["fused_score"] >= self.MIN_FUSED_SCORE_THRESHOLD]


            if top_k is not None:
                filtered_top_items = filtered_top_items[:top_k]

            sub_query.sub_answer = filtered_top_items


            for rank, item in enumerate(filtered_top_items):
                print(f"{rank+1:02d}. fused_score={item['fused_score']:.4f}, reranker_score={item['reranker_score']:.4f}, vector_sim_score={item['score']:.4f}")
                print(f"Content: {item.get('content', '')[:80]}")

        except Exception as e:
            print(f"[❌ ERROR] rerank_spec_results failed: {e}")

        elapsed = time.perf_counter() - start_time
        print(f"[⏱️ rerank_spec_results] Took {elapsed:.3f} seconds")
        return sub_query


    async def run_spec_batch(self, sub_queries: List[SubQueryInfo]) -> List[SubQueryInfo]:
        # 將多條 query 一次 batch encode

        print("sub_queries",sub_queries)

        queries = [sq.query for sq in sub_queries]
        vectors = await self.async_encode_batch(queries)

        results = []
        for sq, vec in zip(sub_queries, vectors):
            product_names = sq.slots[0].product_name or []

            base_filter = FieldCondition(
                key="metadata.doc_type",
                match=MatchValue(value="specification")
            )

            if product_names:
                model_conditions = [
                    FieldCondition(
                        key="metadata.product_name",
                        match=MatchValue(value=model.upper())
                    ) for model in product_names
                ]
                product_filter = Filter(should=model_conditions)
                combined_filter = Filter(must=[base_filter, product_filter])
            else:
                combined_filter = Filter(must=[base_filter])

            hits = await self.qdrant.query_points(
                collection_name=self.spec_collection,
                query=vec,
                query_filter=combined_filter,
                limit=100
            )

            threshold = 0.1
            sq.sub_answer = [
                {**point.payload, "score": point.score}
                for point in hits.points
                if point.score is not None and point.score >= threshold
            ]

            await self.rerank_spec_results(sq, top_k=100)
            await self.dedup_and_slim_spec_results(sq)
            results.append(sq)
        return results


    async def run_spec(self, sub_query: SubQueryInfo, top_k: Optional[int] = None) -> SubQueryInfo:
        query_text = sub_query.query
        print(f"[🔍] search_spec_vector called ----------- {query_text}")

        vector = await self.async_encode(query_text)

        product_names = sub_query.slots[0].product_name or []
        print("[slots.product_name] →", product_names)


        base_filter = FieldCondition(
            key="metadata.doc_type",
            match=MatchValue(value="specification")
        )

        if product_names:
            model_conditions = [
                FieldCondition(
                    key="metadata.product_name",
                    match=MatchValue(value=model.upper())
                ) for model in product_names
            ]
            product_filter = Filter(should=model_conditions)
            combined_filter = Filter(must=[base_filter, product_filter])
        else:
            combined_filter = Filter(must=[base_filter])


            print("No motherboard_name, fallback only using doc_type filter")

        hits = await self.qdrant.query_points(
            collection_name=self.spec_collection,
            query=vector,
            query_filter=combined_filter,
            limit=100
        )

        threshold = 0.1
        sub_query.sub_answer = [
            {**point.payload, "score": point.score}
            for point in hits.points
            if point.score is not None and point.score >= threshold
        ]

        await self.rerank_spec_results(sub_query, top_k=50)
        await self.dedup_and_slim_spec_results(sub_query)

        return sub_query




    async def rerank_faq_results(self, sub_query: SubQueryInfo) -> SubQueryInfo:
        print("Check candidate scores before rerank:")
        for idx, item in enumerate(sub_query.sub_answer):
            print(f"[{idx}] score = {item.get('score', 'N/A')}, content preview = {item.get('answer', '')[:50]}")

        start_time = time.perf_counter()
        question = sub_query.query
        items = sub_query.sub_answer

        if not items:
            return sub_query

        try:
            pairs = [
                (question, f"{item.get('symptom', '')} {item.get('solution', '')}")
                for item in items
            ]

            scores = []
            for i in range(0, len(pairs), self.BATCH_SIZE):
                batch = pairs[i:i + self.BATCH_SIZE]
                batch_scores = await self.rerank_with_cross_encoder(batch)
                scores.extend(batch_scores)

            for item, score in zip(items, scores):
                item["reranker_score"] = float(score)

            vector_scores = [item.get("score", 0.0) for item in items]

            print("Vector similarity scores:")
            for idx, vs in enumerate(vector_scores):
                print(f"[{idx}] vector_score = {vs:.4f}")

            reranker_scores = [item["reranker_score"] for item in items]

            def normalize(arr):
                arr = np.array(arr)
                min_v = arr.min()
                max_v = arr.max()
                if max_v == min_v:
                    return np.zeros_like(arr)
                return (arr - min_v) / (max_v - min_v)

            norm_vector = normalize(vector_scores)
            norm_reranker = normalize(reranker_scores)

            weight_vector = 0.3
            weight_reranker = 0.7
            fused_raw = weight_vector * norm_vector + weight_reranker * norm_reranker
            fused_score_arr = normalize(fused_raw)

            for i, item in enumerate(items):
                item["fused_score"] = fused_score_arr[i]

            items_sorted = sorted(items, key=lambda x: x["fused_score"], reverse=True)

            top1_score = items_sorted[0]["fused_score"]
            top2_score = items_sorted[1]["fused_score"] if len(items_sorted) > 1 else 0.0
            confidence_level = "low"

            if top1_score >= 0.9:
                confidence_level = "high"
            elif top1_score >= 0.6:
                confidence_level = "medium"

            print(f"[🔎 Confidence] top1={top1_score:.4f}, top2={top2_score:.4f}, confidence={confidence_level}")

            if confidence_level == "high":
                sub_query.sub_answer = [items_sorted[0]]
            else:
                sub_query.sub_answer = []

            print(f"------- rerank_faq_results (Top results by fused_score) -------{question}")
            for rank, item in enumerate(items_sorted[:5]):
                print(f"{rank+1:02d}. fused_score={item['fused_score']:.4f}, reranker_score={item['reranker_score']:.4f}, vector_score={item.get('score', 0.0):.4f}")

        except Exception as e:
            print(f"[❌ ERROR] rerank_faq_results failed: {e}")

        elapsed = time.perf_counter() - start_time
        print(f"[⏱️ rerank_faq_results] Took {elapsed:.3f} seconds")

        return sub_query

    async def run_faq(self, sub_query: SubQueryInfo) -> SubQueryInfo:
        query_text = sub_query.query
        vector = await self.async_encode(query_text)

        product_names = sub_query.slots[0].product_name or []

        filter_conditions = [
            FieldCondition(key="doc_type", match=MatchValue(value="faq"))
        ]

        if product_names:
            tag_conditions = [
                FieldCondition(key="tags", match=MatchValue(value=tag))
                for tag in product_names
            ]
            filter_conditions.append(
                Filter(should=tag_conditions)
            )

        final_filter = Filter(must=filter_conditions)

        fallback_hits = await self.qdrant.query_points(
            collection_name=self.faq_collection,
            query=vector,
            query_filter=final_filter,
            limit=10
        )

        threshold = 0.1
        sub_query.sub_answer = [
            {**point.payload, "score": point.score} 
            for point in fallback_hits.points if point.score is not None and point.score >= threshold
        ]

        await self.rerank_faq_results(sub_query)

        return sub_query
