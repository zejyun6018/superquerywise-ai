import streamlit as st
import pandas as pd
import json
import httpx
import re
from uuid import uuid4
from datetime import datetime, timezone

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models


COLLECTION_NAME = "supermicro_spec"
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
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
    )


PROMPT_TEMPLATE = """
You are a technical assistant that converts product specification data into Q&A format.

Generate clear, concise, and technically accurate Questions and Answers based on each field of the specification data.

 
The System family is: {series_code_name}
The System name is : {system_name}

The full specification data is shown below:
{data_row}

For each specification field, try to generate one or more QA pairs to help the user understand the system or its features.

Output the QAs in the following strict JSON format:

[
  {{
    "content": "Q: <question>? A: <answer>",
    "metadata": {{
      "product_brand": "Supermicro",
      "product_family": "{series_code_name}",
      "product_model": "{system_name}",
      "product_type": "System",
      "Component_motherboard":"{motherboard_name}",
      "doc_type": "specification",
      "tags": ["list", "of", "relevant", "technical", "keywords"]
    }}
  }},
  ...
]

Return only valid JSON array. No explanation, no markdown, no comments.
"""

def extract_json(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    return match.group(0).strip() if match else text

def call_llm(prompt: str, timeout_sec=60) -> str:
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 10000,
    }
    try:
        response = httpx.post(VLLM_ENDPOINT, json=payload, timeout=timeout_sec)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.error(f"❌ LLM 請求失敗: {e}")
        return ""
 
def safe_get(row: dict, key: str):
    for k in row.keys():
        if k.lower() == key.lower():
            return row[k]
    return None


def embed_text(text: str) -> list[float]:
    emb = model.encode(text)
    return emb.tolist()


def save_qa_to_qdrant(qa_list: list):
    points = []
    for item in qa_list:
        vector = embed_text(item["content"])
        point = models.PointStruct(
            id=item["id"],
            vector=vector,
            payload={
                "content": item["content"],
                "metadata": item.get("metadata", {}),
                "timestamp": item.get("timestamp", "")
            }
        )
        points.append(point)
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


def generate_qa_from_row(row: dict, timeout_sec=60) -> list:
    motherboard_name = safe_get(row, "motherboard_name") or ""
    series_code_name = safe_get(row, "series_code_name") or ""
    system_name = safe_get(row, "system_name") or ""

    row_text = "\n".join(f"- {k}: {v}" for k, v in row.items() if pd.notna(v))
    prompt = PROMPT_TEMPLATE.format(
        motherboard_name=motherboard_name,
        series_code_name=series_code_name,
        system_name = system_name,
        data_row=row_text,
    )

    llm_output = call_llm(prompt, timeout_sec=timeout_sec)
    st.code(llm_output, language="json")  # debug 用

    try:
        clean_json = extract_json(llm_output)
        qa_list = json.loads(clean_json)
        for item in qa_list:
            item["id"] = str(uuid4())
            item["timestamp"] = datetime.now(timezone.utc).isoformat(timespec='seconds')
        save_qa_to_qdrant(qa_list)
        return qa_list
    except Exception as e:
        st.warning(f"⚠️ 解析 LLM 回傳 JSON 失敗: {e}")
        return []




        

st.set_page_config(page_title="📄 自動 QA 生成器", layout="wide")
st.title("📄 自動 QA 生成器（vLLM + Qdrant）")

uploaded_file = st.file_uploader("請上傳產品規格 CSV 檔案", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 讀取的欄位名稱")
    st.write(df.columns.tolist())

    st.subheader("📋 預覽前 5 筆資料")
    st.dataframe(df.head(5))

    timeout_per_row = st.number_input("LLM 呼叫 timeout (秒)", min_value=5, max_value=120, value=60, step=5)

    if st.button("🚀 開始生成並存入 Qdrant"):
        all_qas = []
        progress_bar = st.progress(0)
        total = len(df)

        with st.spinner("生成中... 請稍候"):
            for idx, row in df.iterrows():
                qa_items = generate_qa_from_row(row.to_dict(), timeout_sec=timeout_per_row)
                all_qas.extend(qa_items)
                progress_bar.progress((idx + 1) / total)

        st.success(f"✅ 共產生 {len(all_qas)} 筆 QA 並存入 Qdrant")

        st.subheader("📄 前 5 筆範例")
        for qa in all_qas[:5]:
            st.json(qa)

        st.download_button(
            label="📥 下載全部 QA（JSON）",
            data=json.dumps(all_qas, indent=2, ensure_ascii=False),
            file_name="qa_output.json",
            mime="application/json",
        )
