import json
import re
import inspect
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http import models



def extract_fail_flag(text: str) -> bool:
    text = text.strip()
    match = re.search(r"```?\s*FAIL\s*```?", text, re.IGNORECASE)
    return bool(match)


def extract_json(text: str) -> str:
 
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        raise ValueError("No JSON object found in text")



def clean_response(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()



def extract_json_array(text: str) -> str:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        raise ValueError("No JSON array found in text")



def print_fun_name():
    print(f"Function call: ______________{inspect.currentframe().f_back.f_code.co_name}()______________\n")



async def qdrant_collection_exists(qdrant_client, collection_name: str, vector_size: int):
    try:
        await qdrant_client.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except UnexpectedResponse as e:
        if e.status_code == 404:
            print(f"Collection '{collection_name}' not found. Creating...")
            await qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Collection '{collection_name}' created.")
        else:
            raise