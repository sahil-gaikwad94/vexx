"""Vexx 2.1 code engine — constrained code generation and research verification."""

from __future__ import annotations

from .config import CodeEngineConfig
from .contracts import (
    POS_ONLY,
    POS_OR_KEYWORD,
    UNKNOWN_DYNAMIC_DISPATCH,
    VAR_KEYWORD,
    VAR_POSITIONAL,
    KEYWORD_ONLY,
    CallCheck,
    CallableContract,
    ParameterContract,
    build_contracts,
    check_call,
    header_contract_from_text,
)
from .lexer import LexerState, advance_lexer_state
from .masking import CodeSession, DeadEnd
from .controller import VexxCodeController
from .policy import CodePolicyError, check_ast_policy
from .protocol import ProtocolConfig, ProtocolPhase, ProtocolTracker
from .sandbox import SandboxRunner, ExecutionResult
from .candidate import CandidateReport, CandidateVerifier
from .experiments import ExperimentEpisode, ExperimentResult
from .conflict import CodeCheckpoint, BoundCut, contract_single_row
from .controller import VexxCodeController as _VC  # noqa: F401

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