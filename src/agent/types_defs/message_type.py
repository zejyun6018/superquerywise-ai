from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple




class Message(BaseModel):
    role: str
    content: str

 
class ChatRequest(BaseModel):
    model: Optional[str] =  Field(default=None, description="Model ID (optional)")
    messages: List[Message]
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 10000
    stream: Optional[bool] = True


