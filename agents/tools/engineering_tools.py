"""
Engineering Assistant Tools for the ReAct Agent.

Tools available to the agent for Bosch-style engineering assistance:
  - code_search: search codebase / docs for relevant snippets
  - document_lookup: retrieve technical documentation
  - information_extractor: extract structured info from unstructured text
  - code_explainer: explain code snippets
  - error_debugger: diagnose error messages
  - spec_parser: parse technical specs into structured format

These mirror real industrial AI assistant capabilities (Bosch, Siemens, BMW R&D).
"""
import re
import ast
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CodeSearchTool:
    """Search indexed codebase or documentation for relevant snippets."""
    name = "code_search"
    description = (
        "Search the engineering codebase or documentation for relevant code snippets, "
        "functions, or technical references. Input: search query string."
    )

    def __init__(self, retriever=None):
        self.retriever = retriever

    def run(self, query: str) -> str:
        if self.retriever:
            docs = self.retriever.get_relevant_documents(query)
            if docs:
                return "\n\n".join(f"[Source: {d.metadata.get('source','unknown')}]\n{d.page_content}"
                                   for d in docs[:3])
        return f"[code_search] No indexed codebase found. Query: '{query}'. In production, this would search your repo/docs."


class DocumentLookupTool:
    """Retrieve technical documentation sections."""
    name = "document_lookup"
    description = (
        "Look up technical documentation, API references, or engineering specs. "
        "Input: topic or keyword to look up."
    )

    def run(self, query: str) -> str:
        # Simulated documentation lookup — in production connects to RAG retriever
        mock_docs = {
            "can bus": "CAN Bus (Controller Area Network): Serial communication protocol for ECU-to-ECU communication. Max speed: 1Mbit/s. Frame format: 11-bit or 29-bit identifier.",
            "autosar": "AUTOSAR (Automotive Open System Architecture): Standardized automotive software framework. Layers: Application, RTE, BSW, MCAL.",
            "misra": "MISRA C: Guidelines for C language use in safety-critical systems. Covers 143 rules across 10 categories.",
            "pid controller": "PID Controller: Proportional-Integral-Derivative control. Output = Kp*e + Ki*∫e dt + Kd*de/dt. Used in motor control, thermal management.",
            "ota update": "OTA (Over-the-Air) Update: Wireless firmware delivery system. Requires differential update, rollback capability, signature verification.",
        }
        query_lower = query.lower()
        for key, content in mock_docs.items():
            if key in query_lower or any(w in query_lower for w in key.split()):
                return f"[document_lookup] {content}"
        return f"[document_lookup] No direct match for '{query}'. Try rephrasing or use code_search."


class InformationExtractorTool:
    """Extract structured information from unstructured engineering text."""
    name = "information_extractor"
    description = (
        "Extract structured information from unstructured engineering text. "
        "Identifies entities like components, specifications, error codes, and requirements. "
        "Input: raw text to extract from."
    )

    def run(self, text: str) -> str:
        extracted = {
            "error_codes": re.findall(r'\b[A-Z]{1,3}[-_]?\d{3,6}\b', text),
            "version_numbers": re.findall(r'v?\d+\.\d+(?:\.\d+)?', text),
            "temperatures": re.findall(r'-?\d+(?:\.\d+)?\s*°?[CF]', text),
            "voltages": re.findall(r'\d+(?:\.\d+)?\s*V(?:DC|AC)?', text),
            "frequencies": re.findall(r'\d+(?:\.\d+)?\s*(?:Hz|kHz|MHz|GHz)', text),
            "memory_sizes": re.findall(r'\d+(?:\.\d+)?\s*(?:KB|MB|GB|TB)', text),
            "requirements": [s.strip() for s in re.split(r'[.;]', text)
                           if any(w in s.lower() for w in ['shall', 'must', 'required', 'should'])],
        }
        # Remove empty
        extracted = {k: v for k, v in extracted.items() if v}
        if extracted:
            return f"[information_extractor]\n{json.dumps(extracted, indent=2)}"
        return "[information_extractor] No structured entities detected in input."


class CodeExplainerTool:
    """Explain code snippets in plain language."""
    name = "code_explainer"
    description = (
        "Explain what a code snippet does in plain engineering language. "
        "Input: code snippet as a string."
    )

    def run(self, code: str) -> str:
        lines = [l for l in code.strip().split('\n') if l.strip()]
        n_lines = len(lines)

        # Detect language
        lang = "Python" if any(kw in code for kw in ['def ', 'import ', 'class ']) else \
               "C/C++" if any(kw in code for kw in ['#include', 'void ', 'int main']) else \
               "unknown"

        # Count functions/classes
        funcs = re.findall(r'(?:def |void |int |float )\s*(\w+)\s*\(', code)
        loops = len(re.findall(r'\b(?:for|while)\b', code))
        conditions = len(re.findall(r'\bif\b', code))

        summary = (
            f"[code_explainer] Language: {lang} | Lines: {n_lines} | "
            f"Functions: {funcs} | Loops: {loops} | Conditionals: {conditions}\n"
            f"This snippet {'defines the function(s): ' + str(funcs) if funcs else 'contains inline logic'} "
            f"with {loops} loop(s) and {conditions} conditional branch(es). "
            f"For full semantic explanation, pass to the LLM via the main agent."
        )
        return summary


class ErrorDebuggerTool:
    """Diagnose error messages and suggest fixes."""
    name = "error_debugger"
    description = (
        "Diagnose engineering error messages, stack traces, or fault codes. "
        "Input: error message or fault code string."
    )

    def run(self, error: str) -> str:
        error_lower = error.lower()
        diagnoses = {
            "segmentation fault": "Memory access violation. Check: null pointer dereference, buffer overflow, stack corruption, use-after-free.",
            "timeout": "Operation exceeded time limit. Check: network latency, deadlocks, infinite loops, resource starvation.",
            "null pointer": "Attempted to dereference a null/None pointer. Add null checks before dereferencing.",
            "out of memory": "Memory exhaustion. Check: memory leaks, large allocations, missing deallocation.",
            "stack overflow": "Recursive call depth exceeded. Check: infinite recursion, deep call chains.",
            "import error": "Module not found. Check: package installation, PYTHONPATH, virtual environment activation.",
            "connection refused": "Network connection rejected. Check: service running, port open, firewall rules.",
            "permission denied": "Insufficient privileges. Check: file permissions, sudo requirements, user roles.",
        }
        for key, diagnosis in diagnoses.items():
            if key in error_lower:
                return f"[error_debugger] Diagnosis: {diagnosis}"

        # Generic analysis
        return (
            f"[error_debugger] Error pattern: '{error[:100]}'. "
            "Recommendation: Check logs for full stack trace, validate inputs, "
            "verify system state and resource availability."
        )


class SpecParserTool:
    """Parse technical specifications into structured format."""
    name = "spec_parser"
    description = (
        "Parse technical specifications or requirements documents into structured JSON. "
        "Input: specification text."
    )

    def run(self, spec_text: str) -> str:
        # Extract requirement IDs
        req_ids = re.findall(r'\b(?:REQ|SRS|SYS|HW|SW)[-_]\d+\b', spec_text)

        # Extract parameter definitions (Name: Value unit)
        params = re.findall(r'([A-Z][a-zA-Z\s]+?):\s*([0-9.,\-±]+\s*(?:ms|V|A|Hz|°C|KB|MB|rpm|Nm)?)', spec_text)

        # Extract boolean properties
        shall_reqs = [s.strip() for s in re.split(r'[.;\n]', spec_text)
                     if 'shall' in s.lower() or 'must' in s.lower()]

        structured = {
            "requirement_ids": req_ids,
            "parameters": {name.strip(): value.strip() for name, value in params[:10]},
            "shall_requirements": shall_reqs[:5],
            "word_count": len(spec_text.split()),
        }
        return f"[spec_parser]\n{json.dumps(structured, indent=2, default=str)}"


def get_all_tools() -> List[Any]:
    """Return all engineering assistant tools."""
    return [
        CodeSearchTool(),
        DocumentLookupTool(),
        InformationExtractorTool(),
        CodeExplainerTool(),
        ErrorDebuggerTool(),
        SpecParserTool(),
    ]
