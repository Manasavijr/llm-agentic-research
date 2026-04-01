"""
ReAct (Reasoning + Acting) Agent for Engineering Assistance.

Implements the ReAct framework (Yao et al. 2022):
  Thought → Action → Observation → Thought → ... → Final Answer

The agent reasons step-by-step, selects tools, observes results,
and iterates until it can produce a confident final answer.

Use cases:
  - Engineering doc Q&A
  - Code explanation and debugging
  - Specification parsing and extraction
  - Technical report generation
"""
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)

REACT_SYSTEM_PROMPT = """You are an expert engineering AI assistant using the ReAct framework.
You have access to the following tools:

{tool_descriptions}

Use this EXACT format for every response:

Thought: [Your reasoning about what to do next]
Action: [tool_name]
Action Input: [input to the tool]
Observation: [tool result will be inserted here]
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information to answer.
Final Answer: [your complete, detailed answer]

Rules:
- Always start with a Thought
- Use tools to gather information before answering
- Never make up information — use tools or say you don't know
- For engineering questions, be precise and technical
- Final Answer must be comprehensive and well-structured

Question: {question}
"""

REACT_CONTINUE_PROMPT = """Continue from where you left off.

Previous steps:
{history}

Observation from last action: {observation}

Continue with the next Thought:"""


@dataclass
class AgentStep:
    thought: str
    action: str
    action_input: str
    observation: str
    step_num: int


@dataclass
class AgentResult:
    question: str
    final_answer: str
    steps: List[AgentStep]
    total_steps: int
    latency_ms: float
    tools_used: List[str]
    success: bool
    error: Optional[str] = None


class ReActAgent:
    """
    ReAct Agent with tool use for engineering assistance.
    Implements iterative Thought → Action → Observation loop.
    """

    def __init__(
        self,
        tools: List[Any],
        model: str = "llama3.2",
        ollama_base_url: str = "http://localhost:11434",
        max_steps: int = 6,
        temperature: float = 0.1,
    ):
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.llm = OllamaLLM(
            base_url=ollama_base_url,
            model=model,
            temperature=temperature,
        )
        logger.info(f"ReActAgent initialized with {len(tools)} tools: {list(self.tools.keys())}")

    def _build_tool_descriptions(self) -> str:
        return "\n".join(
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        )

    def _parse_llm_output(self, text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Parse LLM output for Thought, Action, Action Input, Final Answer."""
        thought = re.search(r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)", text, re.DOTALL)
        action = re.search(r"Action:\s*(\w+)", text)
        action_input = re.search(r"Action Input:\s*(.+?)(?=Observation:|Thought:|Final Answer:|$)", text, re.DOTALL)
        final_answer = re.search(r"Final Answer:\s*(.+?)$", text, re.DOTALL)

        return (
            thought.group(1).strip() if thought else None,
            action.group(1).strip() if action else None,
            action_input.group(1).strip() if action_input else None,
            final_answer.group(1).strip() if final_answer else None,
        )

    def _run_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute a tool and return the observation."""
        if tool_name not in self.tools:
            return f"[ERROR] Unknown tool '{tool_name}'. Available: {list(self.tools.keys())}"
        try:
            return self.tools[tool_name].run(tool_input)
        except Exception as e:
            return f"[ERROR] Tool '{tool_name}' failed: {str(e)}"

    def run(self, question: str) -> AgentResult:
        """Run the ReAct loop for a given question."""
        t0 = time.perf_counter()
        steps = []
        tools_used = []
        history = ""

        prompt = REACT_SYSTEM_PROMPT.format(
            tool_descriptions=self._build_tool_descriptions(),
            question=question,
        )

        for step_num in range(1, self.max_steps + 1):
            logger.info(f"ReAct step {step_num}/{self.max_steps}")

            try:
                if step_num == 1:
                    llm_output = self.llm.invoke(prompt)
                else:
                    continue_prompt = REACT_CONTINUE_PROMPT.format(
                        history=history,
                        observation=steps[-1].observation if steps else "",
                    )
                    llm_output = self.llm.invoke(continue_prompt)

                thought, action, action_input, final_answer = self._parse_llm_output(llm_output)

                if final_answer:
                    latency = (time.perf_counter() - t0) * 1000
                    logger.info(f"Agent reached final answer in {step_num} steps")
                    return AgentResult(
                        question=question,
                        final_answer=final_answer,
                        steps=steps,
                        total_steps=step_num,
                        latency_ms=round(latency, 2),
                        tools_used=list(set(tools_used)),
                        success=True,
                    )

                if action and action_input:
                    observation = self._run_tool(action, action_input)
                    tools_used.append(action)
                    agent_step = AgentStep(
                        thought=thought or "",
                        action=action,
                        action_input=action_input,
                        observation=observation,
                        step_num=step_num,
                    )
                    steps.append(agent_step)
                    history += f"\nStep {step_num}:\nThought: {thought}\nAction: {action}\nAction Input: {action_input}\nObservation: {observation}\n"
                    logger.info(f"  Action: {action} → {observation[:80]}...")
                else:
                    # No clear action — try to extract final answer from raw output
                    if llm_output.strip():
                        latency = (time.perf_counter() - t0) * 1000
                        return AgentResult(
                            question=question,
                            final_answer=llm_output.strip(),
                            steps=steps,
                            total_steps=step_num,
                            latency_ms=round(latency, 2),
                            tools_used=list(set(tools_used)),
                            success=True,
                        )

            except Exception as e:
                logger.error(f"Agent step {step_num} failed: {e}")
                latency = (time.perf_counter() - t0) * 1000
                return AgentResult(
                    question=question,
                    final_answer="",
                    steps=steps,
                    total_steps=step_num,
                    latency_ms=round(latency, 2),
                    tools_used=list(set(tools_used)),
                    success=False,
                    error=str(e),
                )

        # Max steps reached
        latency = (time.perf_counter() - t0) * 1000
        last_obs = steps[-1].observation if steps else "No observations collected."
        return AgentResult(
            question=question,
            final_answer=f"[Max steps reached] Best available answer based on observations:\n{last_obs}",
            steps=steps,
            total_steps=self.max_steps,
            latency_ms=round(latency, 2),
            tools_used=list(set(tools_used)),
            success=False,
            error="Max steps exceeded",
        )
