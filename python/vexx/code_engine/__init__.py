"""Vexx 2.1 code engine — constrained code generation and research verification."""

from __future__ import annotations

from .candidate import CandidateReport, CandidateVerifier
from .config import CodeEngineConfig
from .conflict import BoundCut, CodeCheckpoint, contract_single_row
from .contracts import (
    KEYWORD_ONLY,
    POS_ONLY,
    POS_OR_KEYWORD,
    UNKNOWN_DYNAMIC_DISPATCH,
    VAR_KEYWORD,
    VAR_POSITIONAL,
    CallableContract,
    CallCheck,
    ParameterContract,
    build_contracts,
    check_call,
    header_contract_from_text,
)
from .controller import VexxCodeController
from .controller import VexxCodeController as _VC  # noqa: F401
from .experiments import ExperimentEpisode, ExperimentResult
from .lexer import LexerState, advance_lexer_state
from .masking import CodeSession, DeadEnd
from .policy import CodePolicyError, check_ast_policy
from .protocol import ProtocolConfig, ProtocolPhase, ProtocolTracker
from .sandbox import ExecutionResult, SandboxRunner

__all__ = [
    "CodeEngineConfig",
    "VexxCodeController",
    "CodeSession",
    "LexerState",
    "advance_lexer_state",
    "ParameterContract",
    "CallableContract",
    "CallCheck",
    "build_contracts",
    "header_contract_from_text",
    "check_call",
    "POS_ONLY",
    "POS_OR_KEYWORD",
    "VAR_POSITIONAL",
    "KEYWORD_ONLY",
    "VAR_KEYWORD",
    "UNKNOWN_DYNAMIC_DISPATCH",
    "DeadEnd",
    "ProtocolConfig",
    "ProtocolPhase",
    "ProtocolTracker",
    "check_ast_policy",
    "CodePolicyError",
    "SandboxRunner",
    "ExecutionResult",
    "CandidateReport",
    "CandidateVerifier",
    "ExperimentEpisode",
    "ExperimentResult",
    "CodeCheckpoint",
    "BoundCut",
    "contract_single_row",
]