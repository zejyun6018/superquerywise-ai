from typing import List, Optional, Dict, Any, Annotated
from pydantic import BaseModel, field_validator, Field
import operator

 
class SlotInfo(BaseModel):
    product_model: Optional[List[str]] = Field(default_factory=list)
    feature: Optional[str] = None
    error_code: Optional[str] = None
    time: Optional[str] = None
    other: Optional[str] = None

    def __add__(self, other: Any) -> "SlotInfo":
        if isinstance(other, list):
            other = other[0] if other else SlotInfo()
        if not isinstance(other, SlotInfo):
            raise TypeError(f"Can only add SlotInfo with SlotInfo, got {type(other)}")
        return SlotInfo(
            product_model=list(set((self.product_model or []) + (other.product_model or []))),
            feature=self.feature or other.feature,
            error_code=self.error_code or other.error_code,
            time=self.time or other.time,
            other=self.other or other.other,
        )

    @field_validator("feature", "error_code", "time", "other", mode="before")
    @classmethod
    def ensure_str(cls, v):
        if isinstance(v, list):
            return v[0] if v else ""
        return v or ""

 
class SubQueryInfo(BaseModel):
    query: str
    slots: List[SlotInfo] = Field(default_factory=list)
    query_type: Optional[str] = None  # "faq", "spec", "graph"
    result: Optional[str] = None   
    sub_answer: Optional[str] = None
    intent: Optional[str] = None  
    query_sources: Optional[List[str]] = None
    step_log: List[str] = Field(default_factory=list)  
    metadata: Optional[Dict[str, Any]] = None

class SubQueryWrapper(BaseModel):
    subquery: SubQueryInfo

 
class OverallState(BaseModel):
    messages: List[dict]
    #query_types: List[str] = Field(default_factory=list)
    slots: Annotated[SlotInfo, operator.add] = Field(default_factory=SlotInfo)
    sub_queries: Annotated[List[SubQueryInfo], operator.add] = Field(default_factory=list)
    sub_answers: List[str] = Field(default_factory=list)
    final_answer: Optional[str] = None
    intents: Optional[List[Dict[str, Any]]] = None

    #semantic_result: Optional[dict] = None  # 欄位儲存原始語意分析，便於 debug/log 重建語意歷程
    #metadata: Optional[dict] = Field(default_factory=dict)  # 例如 timestamp、source、user_id 之類的通用欄位

