import json
import re
import inspect
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http import models
import random


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




def get_professional_response(matched_intent):
    if matched_intent == "greeting":
        openings = [
            "Hello! 👋",
            "Welcome to Supermicro support. 🎉",
            "Hi there! 😊",
            "Good day! ☀️",
            "Greetings! 🙌",
            "Thank you for contacting Supermicro support. 🙏",
            "Hello! 👋",
            "Hi! 👋"
        ]
        greetings = [
            "I’m here to assist you with any system or hardware inquiries.",
            "Ready to help you with your Supermicro product today.",
            "Feel free to ask any questions regarding Supermicro products.",
            "I’m ready to help with your system or hardware questions.",
            "Here to support your Supermicro needs.",
            "Happy to assist with any Supermicro-related questions.",
            "At your service for all Supermicro product inquiries.",
            "I’m here to help you get the best out of your Supermicro system."
        ]
        questions = [
            "How can I assist you today?",
            "What can I help you with?",
            "Please let me know how I can be of service.",
            "How may I assist you further?",
            "Is there something specific you'd like to know?",
            "What would you like to discuss today?",
            "How can I support your Supermicro needs?",
            "Feel free to ask any questions you have."
        ]
    elif matched_intent == "non-product":
        openings = [
            "I’m sorry. 😔",
            "Apologies. 🙏",
            "Unfortunately, 😕",
            "Regrettably, 😞",
            "I’m specialized in Supermicro systems. 🔧",
            "My expertise is limited to Supermicro products. 💻",
            "Please note, 🚫",
            "I’m here to help with Supermicro product questions only. 📚"
        ]
        greetings = [
            "I can only assist with Supermicro product or system-related questions.",
            "I’m unable to answer that topic, please try asking about Supermicro hardware.",
            "That question is outside my scope, but I’d be happy to help with any Supermicro-related inquiry.",
            "Could you rephrase your question to focus on our products?",
            "Please direct your question towards Supermicro hardware or system specifications.",
            "Can you please ask a product-specific question?",
            "Please provide a relevant question about Supermicro systems.",
            "Ask about Supermicro products or system configurations for support."
        ]
        questions = [
            "Would you like help with a product-related question?",
            "Can I assist you with any Supermicro system inquiries?",
            "Please let me know if you have any questions about Supermicro products.",
            "Is there a Supermicro product you want to know more about?",
            "Feel free to ask about our Supermicro hardware or systems.",
            "How can I support you with Supermicro product information?",
            "Are there any Supermicro system details I can help clarify?",
            "Please ask a product or system related question."
        ]
    elif matched_intent:
        openings = [
            "Sure! 😊",
            "I’m happy to assist. 🙌",
            "Glad to help! 👍",
            "Certainly! 👌",
            "I’m here to support you. 🤝",
            "Happy to help you out. 😊",
            "At your service! ",
            "Ready to assist! "
        ]
        greetings = [
            "Could you provide more details about your request?",
            "Please share specifics so I can give accurate support.",
            "Tell me more so I can assist better.",
            "I’m here to help; could you elaborate?",
            "To assist you better, please provide additional information.",
            "Feel free to give more context about your question.",
            "Please provide further details for accurate assistance.",
            "Let me know more so I can support you effectively."
        ]
        questions = [
            "What else can I help you with?",
            "Is there anything specific you want to discuss?",
            "How can I further assist you?",
            "Please let me know how I can help.",
            "What more information can I provide?",
            "Is there another question I can answer?",
            "Feel free to ask more questions.",
            "How else can I support you?"
        ]
    else:
        openings = [
            "I’m not sure I understand. 🤔",
            "Could you clarify please? 🤨",
            "I’m having trouble understanding. 😕",
            "Apologies, can you explain further? ",
            "I want to assist you better. 🤝",
            "Please help me understand better. 🧐",
            "Sorry, I didn’t quite catch that. 😅",
            "Could you elaborate, please? 😊"
        ]
        greetings = [
            "Could you rephrase your question?",
            "Please provide more details so I can assist.",
            "Can you clarify your request?",
            "I need more info to assist you properly.",
            "Could you give more context?",
            "Help me understand your question better.",
            "Please explain further for accurate support.",
            "Let me know more so I can help."
        ]
        questions = [
            "How can I assist you further?",
            "What would you like to know?",
            "Please provide more information.",
            "Can you give additional details?",
            "How may I help you today?",
            "Feel free to ask more questions.",
            "What else can I do for you?",
            "Is there something else I can assist with?"
        ]

    return f"{random.choice(openings)} {random.choice(greetings)} {random.choice(questions)}"