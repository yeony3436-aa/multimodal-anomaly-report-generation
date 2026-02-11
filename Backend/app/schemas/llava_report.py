from pydantic import BaseModel
from typing import Optional, List

class ReportDTO(BaseModel):
    id: int
    filename: str
    image_path: str
    heatmap_path: Optional[str] = None
    overlay_path: Optional[str] = None

    dataset: str
    category: str
    ground_truth: Optional[str] = None
    decision: str
    confidence: Optional[float] = None

    has_defect: int
    defect_type: str
    location: str
    severity: str

    defect_description: str
    possible_cause: str
    product_description: str

    summary: str
    impact: str
    recommendation: str

    inference_time: Optional[float] = None
    datetime: str

class ReportListDTO(BaseModel):
    items: List[ReportDTO]
    total: int
