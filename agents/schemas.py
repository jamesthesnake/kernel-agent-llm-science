from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel

DType = Literal["fp32", "bf16", "fp16"]

class ShapeSoftmax(BaseModel):
    B: int
    N: int

class ShapeStencil(BaseModel):
    H: int
    W: int

class TritonPlan(BaseModel):
    experiment_id: str
    backend: Literal["triton"]
    op: Literal["row_softmax"]
    dtype: DType
    shapes: List[ShapeSoftmax]
    hypothesis: str
    metrics: List[str]
    tolerance: Dict[str, Dict[DType, float]]
    param_grid: Dict[str, List[int]]
    triton_kernel: str

class CudaPlan(BaseModel):
    experiment_id: str
    backend: Literal["cuda"]
    op: Literal["stencil3x3"]
    dtype: DType
    shapes: List[ShapeStencil]
    hypothesis: str
    metrics: List[str]
    tolerance: Dict[str, Dict[DType, float]]
    param_grid: Dict[str, List[int]]
    iters: int
    cuda_kernel: str

class ConfigResult(TypedDict, total=False):
    config: Dict[str, Any]
    shape: Dict[str, int]
    latency_ms: float
    throughput_gbps: Optional[float]
    achieved_occupancy: Optional[float]
    l_inf_error: float
    ulp_error: Optional[float]
    baseline_latency_ms: Optional[float]
    speedup_vs_baseline: Optional[float]
    passed: bool
    notes: Optional[str]

class Results(BaseModel):
    experiment_id: str
    backend: Literal["triton", "cuda"]
    op: str
    dtype: DType
    shapes: List[Dict[str, int]]
    hypothesis: str
    tolerance: Dict[str, Dict[DType, float]]
    tested: List[ConfigResult]
    best: Optional[ConfigResult]
    executor_info: Dict[str, Any]
    recheck: Optional[Dict[str, Any]] = None

# Analysis Agent Schemas
class Pattern(BaseModel):
    pattern_type: str
    description: str
    frequency: int
    avg_speedup: float
    examples: List[str]
    confidence: float

class Insight(BaseModel):
    title: str
    description: str
    significance: str
    evidence: str
    confidence: float

class Hypothesis(BaseModel):
    statement: str
    rationale: str
    testable_predictions: List[str]
    experimental_design: str
    confidence: float

class PerformanceMetrics(BaseModel):
    mean_speedup: float
    max_speedup: float
    success_rate: float
    convergence_steps: int
    optimization_patterns: List[str]

class AnalysisResult(BaseModel):
    performance_metrics: PerformanceMetrics
    discovered_patterns: List[Pattern]
    generated_insights: List[Insight]
    novel_hypotheses: List[Hypothesis]
    analysis_timestamp: str
    confidence_score: float

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
