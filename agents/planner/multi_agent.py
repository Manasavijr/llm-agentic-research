"""
Multi-Agent System: Planner + Executor Architecture.

The Planner agent decomposes complex engineering questions into subtasks.
The Executor agent handles each subtask using the ReAct framework.
Results are synthesized into a final report.

This mirrors Bosch's agentic AI prototype architecture for:
- Complex technical document analysis
- Multi-step engineering problem solving
- Automated specification review
"""
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)

PLANNER_PROMPT = """You are a senior engineering AI planner.
Your job is to decompose complex engineering questions into clear, executable subtasks.

Question: {question}
Context: {context}

Decompose this into 2-4 specific subtasks. Format EXACTLY as:
SUBTASK 1: [specific actionable subtask]
SUBTASK 2: [specific actionable subtask]
SUBTASK 3: [specific actionable subtask] (if needed)
SUBTASK 4: [specific actionable subtask] (if needed)

Each subtask should be self-contained and answerable independently.
Focus on technical precision."""

SYNTHESIZER_PROMPT = """You are a senior engineering AI synthesizer.
Combine the following subtask results into a comprehensive, well-structured technical answer.

Original Question: {question}

Subtask Results:
{subtask_results}

Produce a final answer that:
1. Directly answers the original question
2. Integrates all relevant findings
3. Uses precise technical language
4. Highlights key insights and recommendations
5. Notes any limitations or uncertainties

Final Comprehensive Answer:"""


@dataclass
class SubtaskResult:
    subtask: str
    result: str
    agent_steps: int
    tools_used: List[str]
    latency_ms: float


@dataclass
class MultiAgentResult:
    question: str
    plan: List[str]
    subtask_results: List[SubtaskResult]
    final_synthesis: str
    total_latency_ms: float
    success: bool


class PlannerAgent:
    """Decomposes complex questions into subtasks."""

    def __init__(self, llm: OllamaLLM):
        self.llm = llm

    def plan(self, question: str, context: str = "") -> List[str]:
        prompt = PLANNER_PROMPT.format(question=question, context=context)
        try:
            output = self.llm.invoke(prompt)
            subtasks = []
            for line in output.split('\n'):
                if line.strip().startswith('SUBTASK'):
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        subtasks.append(parts[1].strip())
            return subtasks if subtasks else [question]
        except Exception as e:
            logger.error(f"Planner failed: {e}")
            return [question]


class SynthesizerAgent:
    """Combines subtask results into a final answer."""

    def __init__(self, llm: OllamaLLM):
        self.llm = llm

    def synthesize(self, question: str, subtask_results: List[SubtaskResult]) -> str:
        results_text = "\n\n".join(
            f"Subtask {i+1}: {r.subtask}\nResult: {r.result}"
            for i, r in enumerate(subtask_results)
        )
        prompt = SYNTHESIZER_PROMPT.format(
            question=question,
            subtask_results=results_text,
        )
        try:
            return self.llm.invoke(prompt).strip()
        except Exception as e:
            logger.error(f"Synthesizer failed: {e}")
            return "\n".join(f"• {r.result}" for r in subtask_results)


class MultiAgentOrchestrator:
    """
    Orchestrates Planner → Executor → Synthesizer pipeline.

    Flow:
      1. Planner decomposes question into subtasks
      2. ReAct executor handles each subtask with tools
      3. Synthesizer combines results into final answer
    """

    def __init__(
        self,
        react_agent,
        model: str = "llama3.2",
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.react_agent = react_agent
        llm = OllamaLLM(base_url=ollama_base_url, model=model, temperature=0.1)
        self.planner = PlannerAgent(llm)
        self.synthesizer = SynthesizerAgent(llm)

    def run(self, question: str, context: str = "") -> MultiAgentResult:
        t0 = time.perf_counter()
        logger.info(f"Multi-agent orchestrator starting: '{question[:60]}...'")

        # Step 1: Plan
        logger.info("Step 1: Planner decomposing question...")
        plan = self.planner.plan(question, context)
        logger.info(f"  Plan: {plan}")

        # Step 2: Execute each subtask
        subtask_results = []
        for i, subtask in enumerate(plan):
            logger.info(f"Step 2.{i+1}: Executing subtask: '{subtask[:50]}'")
            agent_result = self.react_agent.run(subtask)
            subtask_results.append(SubtaskResult(
                subtask=subtask,
                result=agent_result.final_answer,
                agent_steps=agent_result.total_steps,
                tools_used=agent_result.tools_used,
                latency_ms=agent_result.latency_ms,
            ))

        # Step 3: Synthesize
        logger.info("Step 3: Synthesizing results...")
        synthesis = self.synthesizer.synthesize(question, subtask_results)

        total_latency = (time.perf_counter() - t0) * 1000
        logger.info(f"Multi-agent complete in {total_latency:.0f}ms, {len(plan)} subtasks")

        return MultiAgentResult(
            question=question,
            plan=plan,
            subtask_results=subtask_results,
            final_synthesis=synthesis,
            total_latency_ms=round(total_latency, 2),
            success=True,
        )
