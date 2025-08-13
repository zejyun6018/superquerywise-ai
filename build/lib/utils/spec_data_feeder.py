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


COLLECTION_NAME = "supermicro_product_spec"
EMBEDDING_MODEL_PATH = "/meta/all-mpnet-base-v2"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


VLLM_ENDPOINT = "http://localhost:9999/v1/chat/completions"
LLM_MODEL_NAME = "gpt-oss-120b"


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

Generate clear, concise, and technically accurate Questions and Answers based on **every single key and value** in the specification data, without skipping any fields.
 
The product name is: {product_name}
The product generation is: {product_generation}
the product family is: {product_family}


The full specification data is shown below:
{data_row}

For each specification field in the data row, generate QA pairs that help the user understand the system or its features. Every QA must explicitly mention the product name ({product_name}) to improve searchability.

Output the QAs as much as possble in the following strict JSON format:

[
  {{
    "content": "Q: <question>? A: <answer>",
    "metadata": {{
      "product_brand": "Supermicro",
      "product_generation": "{product_generation}",
      "product_type": {product_type},
      "product_family": "{product_family}",
      "product_name":"{product_name}"
      "doc_type": "specification",
      "tags": ["list", "of", "relevant", "technical", "keywords"]
    }}
  }},
  ...
] 

Notes:
- Return only valid JSON array. No explanation, no markdown, no comments.
- Make sure every QA explicitly references the product name {product_name}.
- Include all keys and values from the specification data, no omissions.
- Use relevant and precise technical tags in the metadata "tags" array for each QA.

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
        "max_tokens": 100000,
    }
    try:
        response = httpx.post(VLLM_ENDPOINT, json=payload, timeout=timeout_sec)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.error(f"❌ LLM 請求失敗: {e}")
        return ""

def extract_data_value(input):
    # 避免覆蓋內建 str
    if isinstance(input, str):
        try:
            # 先解析 JSON 字串
            data = json.loads(input)
        except json.JSONDecodeError:
            return None
    elif isinstance(input, list):
        data = input
    else:
        return None


    
    if data and isinstance(data, list) and "generation" in data[0]:
        return data[0]["generation"]
    elif data and isinstance(data, list) and "product_family" in data[0]:
        return data[0]["product_family"]
    elif data and isinstance(data, list) and "product_type" in data[0]:
        return data[0]["product_type"]
    return None




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

    product_name = safe_get(row, "motherboard_name") or ""
    product_type = extract_data_value(safe_get(row, "motherboard_type") or "")
    product_family = extract_data_value(safe_get(row, "product_family") or "")  
    product_generation = extract_data_value(safe_get(row, "motherboard_generation") or "")

    row_text = "\n".join(f"- {k}: {v}" for k, v in row.items() if pd.notna(v))


    print("row_text=============================================================",row_text)


    prompt = PROMPT_TEMPLATE.format(
        product_name=product_name,
        product_type=product_type,
        product_family=product_family,
        product_generation = product_generation,
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
