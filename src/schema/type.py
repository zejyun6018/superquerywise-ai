from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum

class ErrorCode(str, Enum):
    DATA_NOT_FOUND = "001_DATA_NOT_FOUND"
    INVALID_INPUT = "002_INVALID_INPUT"
    TIMEOUT = "003_TIMEOUT"
    UNKNOWN_ERROR = "004_UNKNOWN"



class ToolInput(BaseModel):
    query_id: str = Field(..., description="Task tracking ID returned by the LLM")
    query: str = Field(..., description="User input question")
    context: Optional[str] = None

class ToolOutput(BaseModel):
    success: bool = Field(..., description="Whether the operation was successful")
    result: Optional[Any] = Field(None, description="Returned result on success, flexible type")
    message: Optional[str] = Field(None, description="Error message or operation description")
    error_code: Optional[str] = Field(None, description="Error code if operation failed")
    query_id: Optional[str] = Field(None, description="Unique identifier of the query")
    tool_name: Optional[str] = Field(None, description="Name of the invoked tool")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional info for extensibility")

