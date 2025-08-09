import streamlit as st
from typing import List
import json
import requests
import re
import datetime
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid

 
COLLECTION_NAME = "supermicro_faq"
EMBEDDING_MODEL_PATH = "/meta/all-mpnet-base-v2"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
 
VLLM_ENDPOINT = "http://localhost:9999/v1/chat/completions"
LLM_MODEL_NAME = "/google/gemma-3-27b-it-GPTQ-4b-128g/"

 
model = SentenceTransformer(EMBEDDING_MODEL_PATH)
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
 
existing_collections = [c.name for c in qdrant.get_collections().collections]
if COLLECTION_NAME not in existing_collections:
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

class VLLM_Chat_API:
    def __init__(self, endpoint=VLLM_ENDPOINT, model_name=LLM_MODEL_NAME, temperature=0.3, max_tokens=1024):
        self.endpoint = endpoint
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def stream_chat(self, messages: List[dict]):
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        response = requests.post(self.endpoint, json=payload, headers=headers, stream=True, timeout=60)
        full_text = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data: "):
                    content = decoded_line[len("data: "):]
                    if content == "[DONE]":
                        break
                    data = json.loads(content)
                    delta = data.get("choices", [])[0].get("delta", {})
                    if "content" in delta:
                        full_text += delta["content"]
                        yield delta["content"]

def extract_json_from_text(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group()
    return None

def extract_fields_from_qa(question: str, answer: str):
    system_prompt = (
        "You are an AI assistant specialized in Supermicro technical support.\n"
        "Based on the user's question and answer, strictly return a JSON with the following fields:\n"
        "- symptom (string): Describe the issue observed by the user.\n"
        "- solution (string): The suggested fix or troubleshooting approach.\n"
        "- tags (array of strings): Keywords related to the issue, such as model, error type, component, etc.\n"
        "- category (string): Must be exactly one of the following:\n"
        "    - Hardware\n"
        "    - BIOS\n"
        "    - Firmware\n"
        "    - Software\n"
        "    - Network\n"
        "    - Storage\n"
        "    - Boot\n"
        "    - Thermal\n"
        "    - Power\n"
        "    - Performance\n"
        "    - Compatibility\n"
        "    - General\n"
        "    - Management\n"
        "⚠️ Do NOT make up your own category. Use only one from the list above.\n"
        "⚠️ Strictly return JSON only, with no explanation or title.\n"
        "Example:\n"
        "{\"symptom\": \"Unable to enter BIOS screen, fans spinning at full speed.\", \"solution\": \"Reseat the CPU and clear CMOS.\", \"tags\": [\"X13DEG-R\", \"POST Fail\", \"Fan\", \"CMOS Reset\"], \"category\": \"Hardware\"}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}\nAnswer: {answer}"}
    ]

    llm = VLLM_Chat_API()
    full_response = ""
    for chunk in llm.stream_chat(messages):
        full_response += chunk

    json_text = extract_json_from_text(full_response)
    if not json_text:
        return {"error": "⚠️ No valid JSON detected, please try again.", "raw": full_response}

    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        return {"error": "⚠️ Failed to parse JSON, please try again.", "raw": full_response}
 
st.set_page_config(page_title="FAQ Auto Classification & Vector Storage", layout="centered")
st.header("🧠 FAQ Auto Classification & Vector DB Tool")

question = st.text_area("Question")
answer = st.text_area("Answer")

if st.button("Analyze and Auto-fill Fields"):
    if not question.strip() or not answer.strip():
        st.error("Please input both question and answer.")
    else:
        result = extract_fields_from_qa(question, answer)
        if "error" in result:
            st.error(result["error"])
            with st.expander("Raw LLM Output"):
                st.code(result.get("raw", ""))
        else:
            st.session_state["symptom"] = result.get("symptom", "")
            st.session_state["solution"] = result.get("solution", "")
            st.session_state["tags"] = ", ".join(result.get("tags", []))
            st.session_state["category"] = result.get("category", "")
            st.session_state["timestamp"] = datetime.now(timezone.utc).isoformat(timespec='seconds')

symptom = st.text_area("Symptom", key="symptom")
solution = st.text_area("Solution", key="solution")
tags = st.text_input("Tags (comma-separated)", key="tags")
category = st.text_input("Category", key="category")

 
st.text_input("Timestamp", value=st.session_state.get("timestamp", ""), disabled=True)

if st.button("✅ Save to Vector Database"):
    if not question or not answer:
        st.error("Please input and analyze a question and answer first.")
    else:
        vector_text = f"{question}\n{answer}\n{symptom}\n{solution}\n{tags}\n{category}"
        vector = model.encode(vector_text).tolist()

        payload = {
            "question": question,
            "answer": answer,
            "symptom": symptom,
            "solution": solution,
            "tags": [t.strip() for t in tags.split(",") if t.strip()],
            "category": category,
            "timestamp": st.session_state.get("timestamp", datetime.utcnow().isoformat() + "Z"),
            "doc_type": "faq"
        }

        point_id = str(uuid.uuid4())
        point = PointStruct(id=point_id, vector=vector, payload=payload)
        qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])
        st.success(f"✅ Saved successfully, ID={point_id}")
