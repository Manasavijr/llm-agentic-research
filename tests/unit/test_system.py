"""Unit tests for LLM Agentic Research Platform."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ── Tool Tests ────────────────────────────────────────────────────────────────

def test_information_extractor():
    from agents.tools.engineering_tools import InformationExtractorTool
    tool = InformationExtractorTool()
    result = tool.run("The ECU operates at 12V and 100MHz. Error code: ECU-12345. Temperature: -40°C to 85°C.")
    assert "information_extractor" in result
    assert "12" in result or "voltage" in result.lower() or "100" in result


def test_error_debugger_known():
    from agents.tools.engineering_tools import ErrorDebuggerTool
    tool = ErrorDebuggerTool()
    result = tool.run("Segmentation fault in main.c:42")
    assert "null pointer" in result.lower() or "memory" in result.lower()


def test_error_debugger_unknown():
    from agents.tools.engineering_tools import ErrorDebuggerTool
    tool = ErrorDebuggerTool()
    result = tool.run("Unknown custom error XYZ-999")
    assert "error_debugger" in result


def test_document_lookup_known():
    from agents.tools.engineering_tools import DocumentLookupTool
    tool = DocumentLookupTool()
    result = tool.run("what is CAN bus protocol")
    assert "CAN" in result


def test_spec_parser():
    from agents.tools.engineering_tools import SpecParserTool
    tool = SpecParserTool()
    result = tool.run("REQ-001: The system shall respond within 10ms. Voltage: 12V. Frequency: 100Hz.")
    data = json.loads(result.replace("[spec_parser]\n", ""))
    assert "requirement_ids" in data
    assert "REQ-001" in data["requirement_ids"]


def test_code_explainer():
    from agents.tools.engineering_tools import CodeExplainerTool
    tool = CodeExplainerTool()
    code = "def process_sensor(data):\n    for i in range(len(data)):\n        if data[i] > threshold:\n            trigger_alarm()\n"
    result = tool.run(code)
    assert "Python" in result
    assert "code_explainer" in result


def test_get_all_tools():
    from agents.tools.engineering_tools import get_all_tools
    tools = get_all_tools()
    assert len(tools) == 6
    names = [t.name for t in tools]
    assert "code_search" in names
    assert "information_extractor" in names
    assert "error_debugger" in names


# ── Document Chunker Tests ────────────────────────────────────────────────────

def test_chunker_basic():
    from rag.pipeline.rag_engine import DocumentChunker
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
    text = "This is a test. " * 50
    chunks = chunker.chunk(text, metadata={"source": "test"})
    assert len(chunks) > 1
    assert all("text" in c for c in chunks)
    assert all("source" in c for c in chunks)


def test_chunker_metadata():
    from rag.pipeline.rag_engine import DocumentChunker
    chunker = DocumentChunker()
    chunks = chunker.chunk("Hello world. " * 20, metadata={"source": "manual.pdf", "page": 1})
    assert all(c["source"] == "manual.pdf" for c in chunks)


# ── Benchmark Metric Tests ────────────────────────────────────────────────────

def test_rouge_l():
    from rag.benchmarks.evaluator import compute_rouge_l
    score = compute_rouge_l("The CAN bus speed is 1 Mbit/s", "CAN bus maximum speed is 1 Mbit/s")
    assert 0.0 <= score <= 1.0
    assert score > 0.3  # should have good overlap


def test_faithfulness():
    from rag.benchmarks.evaluator import compute_faithfulness
    answer = "CAN bus is a serial protocol"
    context = "CAN bus is a serial communication protocol developed by Bosch"
    score = compute_faithfulness(answer, context)
    assert 0.0 <= score <= 1.0
    assert score > 0.3


def test_retrieval_precision():
    from rag.benchmarks.evaluator import compute_retrieval_precision
    retrieved = [
        "CAN bus is a serial communication protocol for ECU communication",
        "Completely unrelated text about cooking recipes",
    ]
    reference = "CAN bus serial communication protocol ECU"
    prec = compute_retrieval_precision(retrieved, "CAN bus", reference)
    assert 0.0 <= prec <= 1.0


# ── LoRA Config Tests ─────────────────────────────────────────────────────────

def test_lora_config_defaults():
    from finetuning.lora_demo import LoRAConfig
    config = LoRAConfig()
    assert config.r == 16
    assert config.lora_alpha == 32
    assert "q_proj" in config.target_modules
    assert "trainable" in config.trainable_params_estimate.lower()


def test_dataset_prep():
    from finetuning.lora_demo import EngineeringDatasetPrep
    prep = EngineeringDatasetPrep()
    formatted = prep.prepare_dataset()
    assert len(formatted) >= 5
    assert all("[INST]" in f for f in formatted)
    assert all("[/INST]" in f for f in formatted)
    stats = prep.compute_dataset_stats(formatted)
    assert stats["n_examples"] >= 5
    assert stats["avg_length_tokens"] > 0


def test_lora_training_simulation():
    from finetuning.lora_demo import LoRAConfig, LoRATrainer
    config = LoRAConfig(r=8)
    trainer = LoRATrainer(lora_config=config)
    result = trainer.simulate_training_run(n_epochs=2, n_steps=5)
    assert "final_loss" in result
    assert result["final_loss"] < 2.5
    assert len(result["loss_history"]) == 2


# ── ReAct Agent Structure Tests ───────────────────────────────────────────────

def test_react_agent_parse():
    from agents.react.react_agent import ReActAgent
    from agents.tools.engineering_tools import get_all_tools

    with patch("agents.react.react_agent.OllamaLLM"):
        agent = ReActAgent(tools=get_all_tools())
        thought, action, action_input, final = agent._parse_llm_output(
            "Thought: I need to look up CAN bus\nAction: document_lookup\nAction Input: CAN bus protocol\n"
        )
        assert thought is not None
        assert action == "document_lookup"
        assert "CAN" in action_input


def test_react_agent_tool_execution():
    from agents.react.react_agent import ReActAgent
    from agents.tools.engineering_tools import get_all_tools

    with patch("agents.react.react_agent.OllamaLLM"):
        agent = ReActAgent(tools=get_all_tools())
        obs = agent._run_tool("error_debugger", "segmentation fault in sensor.c")
        assert "memory" in obs.lower() or "null" in obs.lower()


def test_react_agent_unknown_tool():
    from agents.react.react_agent import ReActAgent
    from agents.tools.engineering_tools import get_all_tools

    with patch("agents.react.react_agent.OllamaLLM"):
        agent = ReActAgent(tools=get_all_tools())
        obs = agent._run_tool("nonexistent_tool", "input")
        assert "Unknown tool" in obs
