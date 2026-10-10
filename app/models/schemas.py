"""Pydantic models/schemas for ClaimLens AI."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Health Response
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    ollama_available: bool
    configured_model: str
    florence_available: bool
    tesseract_available: bool


# =============================================================================
# Text-Only Analysis Request
# =============================================================================

class AnalyzeTextOnlyRequest(BaseModel):
    """Request for text-only claim analysis."""
    user_text: str = Field(..., description="User-provided claim description in Dutch")
    language: Optional[str] = Field("nl", description="Language code (default: nl for Dutch)")


# =============================================================================
# Extraction Outputs
# =============================================================================

class OCRResult(BaseModel):
    """Tesseract OCR extraction result."""
    source: str = "tesseract"
    raw_text: str
    language_used: str
    warnings: List[str] = Field(default_factory=list)


class VisionStructuredOutput(BaseModel):
    """Structured output from a vision model (reserved for future use).

    Florence-2 was removed from the active pipeline; the schema is kept
    so vision-pipeline response payloads stay forward-compatible.
    """
    visible_damage_area: Optional[str] = None
    accident_diagram_summary: Optional[str] = None
    checkboxes_or_marked_fields: List[str] = Field(default_factory=list)
    visible_notes: List[str] = Field(default_factory=list)
    detected_objects_or_regions: List[str] = Field(default_factory=list)


class VisionResult(BaseModel):
    """Vision-model extraction result. Reserved; always ``None`` from the
    OCR pipeline. See README §"Model Evaluation History" — the active
    vision path is ``/claims/analyze-vision`` (qwen3-vl:4b)."""
    source: str = "florence-2"
    raw_output: str
    structured_output: VisionStructuredOutput
    warnings: List[str] = Field(default_factory=list)


# =============================================================================
# Fusion Layer
# =============================================================================

class VehicleInfo(BaseModel):
    """Vehicle information extracted from claim."""
    make: Optional[str] = None
    model: Optional[str] = None
    license_plate: Optional[str] = None
    damage_description: Optional[str] = None


class NormalizedClaimInput(BaseModel):
    """Normalized claim input after fusion."""
    incident_date: Optional[str] = None
    location: Optional[str] = None
    vehicle_a: Dict[str, Any] = Field(default_factory=dict)
    vehicle_b: Dict[str, Any] = Field(default_factory=dict)
    damage_description: Optional[str] = None
    damage_location: Optional[str] = None
    accident_type: Optional[str] = None
    user_description: Optional[str] = None
    document_notes: List[str] = Field(default_factory=list)


class SourceConflict(BaseModel):
    """Conflict between different data sources."""
    field: str
    sources: Dict[str, str]
    message: str


class FusionResult(BaseModel):
    """Result from the fusion layer."""
    normalized_claim_input: NormalizedClaimInput
    missing_fields: List[str]
    conflicts: List[SourceConflict]
    source_map: Dict[str, List[str]]
    raw_sources: Dict[str, Any]


# =============================================================================
# AI Reasoning
# =============================================================================

class ClaimInterpretation(BaseModel):
    """Phi-4 claim interpretation output."""
    claim_type: Optional[str] = None
    accident_summary: Optional[str] = None
    damage_summary: Optional[str] = None
    damage_location: Optional[str] = None
    possible_fault_indicators: List[str] = Field(default_factory=list)
    fraud_indicators: List[str] = Field(default_factory=list)
    requires_additional_information: List[str] = Field(default_factory=list)
    source_conflicts_noted: List[str] = Field(default_factory=list)
    language_notes: Optional[str] = None


class AIReasoningResult(BaseModel):
    """Result from Phi-4 reasoning layer."""
    model: str
    claim_interpretation: Optional[ClaimInterpretation] = None
    error: Optional[str] = None


# =============================================================================
# Rules Engine
# =============================================================================

class AppliedRule(BaseModel):
    """A rule that was applied."""
    rule_id: str
    severity: str
    message: str


class RulesResult(BaseModel):
    """Result from the rules engine."""
    applied_rules: List[AppliedRule]
    rule_flags: List[str]


# =============================================================================
# Decision Engine
# =============================================================================

class DecisionResult(BaseModel):
    """Final decision from the decision engine."""
    risk_level: str  # low, medium, high, unknown
    manual_review_required: bool
    recommended_action: str  # auto_process, request_more_info, manual_review, reject_not_supported
    decision_reasons: List[str]


# =============================================================================
# Audit
# =============================================================================

class AuditInfo(BaseModel):
    """Audit information for the claim."""
    pipeline_version: str
    steps_completed: List[str]
    warnings: List[str]


# =============================================================================
# Input Summary
# =============================================================================

class InputSummary(BaseModel):
    """Summary of inputs received."""
    file_name: Optional[str] = None
    file_type: Optional[str] = None
    user_text_provided: bool
    language: str
    mode: Optional[str] = None


# =============================================================================
# Extraction Container
# =============================================================================

class ExtractionContainer(BaseModel):
    """Container for OCR and Vision results."""
    ocr: OCRResult
    vision: VisionResult


# =============================================================================
# Full Response Models
# =============================================================================

class ClaimAnalysisResponse(BaseModel):
    """Full response for claim analysis with file upload."""
    claim_id: str
    status: str
    input_summary: Dict[str, Any]
    extraction: Dict[str, Any]
    fusion: Dict[str, Any]
    ai_reasoning: Dict[str, Any]
    rules: Dict[str, Any]
    decision: Dict[str, Any]
    audit: Dict[str, Any]


class AnalyzeTextOnlyResponse(BaseModel):
    """Full response for text-only claim analysis."""
    claim_id: str
    status: str
    input_summary: Dict[str, Any]
    extraction: Dict[str, Any]
    fusion: Dict[str, Any]
    ai_reasoning: Dict[str, Any]
    rules: Dict[str, Any]
    decision: Dict[str, Any]
    audit: Dict[str, Any]
