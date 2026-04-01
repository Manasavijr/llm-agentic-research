"""
LLM Agentic AI Research Platform — FastAPI Interface.

Exposes:
  - ReAct engineering agent
  - Multi-agent orchestrator
  - RAG pipeline Q&A
  - Benchmark runner
  - Fine-tuning demo
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agents.tools.engineering_tools import get_all_tools
from agents.react.react_agent import ReActAgent
from agents.planner.multi_agent import MultiAgentOrchestrator
from rag.pipeline.rag_engine import RAGEngine
from finetuning.lora_demo import LoRAConfig, LoRATrainer, EngineeringDatasetPrep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

_react_agent: ReActAgent = None
_orchestrator: MultiAgentOrchestrator = None
_rag_engine: RAGEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _react_agent, _orchestrator, _rag_engine
    logger.info("Initializing LLM Agentic Research Platform...")

    tools = get_all_tools()
    _react_agent = ReActAgent(tools=tools, model=MODEL, ollama_base_url=OLLAMA_URL)
    _orchestrator = MultiAgentOrchestrator(_react_agent, model=MODEL, ollama_base_url=OLLAMA_URL)

    _rag_engine = RAGEngine(llm_model=MODEL, ollama_base_url=OLLAMA_URL)

    logger.info("Platform ready")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="LLM Agentic AI Research Platform",
    description="ReAct agents, multi-agent orchestration, RAG pipeline, LoRA fine-tuning — engineering assistance focus",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Request/Response models ───────────────────────────────────────────────────

class AgentQuery(BaseModel):
    question: str
    use_multi_agent: bool = False

class RAGQuery(BaseModel):
    question: str
    top_k: int = 5

class IngestRequest(BaseModel):
    text: str
    source: str = "user_upload"

class LoRARequest(BaseModel):
    n_epochs: int = 3
    lora_r: int = 16
    lora_alpha: int = 32


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agents": {"react": _react_agent is not None, "orchestrator": _orchestrator is not None},
        "rag": _rag_engine.get_stats() if _rag_engine else None,
        "model": MODEL,
    }


@app.post("/api/agent/query", tags=["Agent"])
async def agent_query(req: AgentQuery):
    """Run ReAct agent or multi-agent orchestrator for engineering Q&A."""
    if not _react_agent:
        raise HTTPException(503, "Agent not initialized")
    if req.use_multi_agent:
        result = _orchestrator.run(req.question)
        return {
            "mode": "multi_agent",
            "question": result.question,
            "plan": result.plan,
            "final_answer": result.final_synthesis,
            "subtasks": len(result.subtask_results),
            "total_latency_ms": result.total_latency_ms,
        }
    else:
        result = _react_agent.run(req.question)
        return {
            "mode": "react",
            "question": result.question,
            "final_answer": result.final_answer,
            "steps": result.total_steps,
            "tools_used": result.tools_used,
            "latency_ms": result.latency_ms,
            "success": result.success,
        }


@app.post("/api/rag/ingest", tags=["RAG"])
async def rag_ingest(req: IngestRequest):
    """Ingest engineering text into the RAG vector store."""
    n_chunks = _rag_engine.ingest_text(req.text, req.source)
    return {"chunks_indexed": n_chunks, "stats": _rag_engine.get_stats()}


@app.post("/api/rag/query", tags=["RAG"])
async def rag_query(req: RAGQuery):
    """Query the RAG pipeline with an engineering question."""
    if not _rag_engine:
        raise HTTPException(503, "RAG engine not initialized")
    return _rag_engine.query(req.question, top_k=req.top_k)


@app.post("/api/rag/retrieve", tags=["RAG"])
async def rag_retrieve(req: RAGQuery):
    """Direct FAISS retrieval without generation."""
    if not _rag_engine:
        raise HTTPException(503, "RAG engine not initialized")
    return {"chunks": _rag_engine.retrieve(req.question, top_k=req.top_k)}


@app.get("/api/rag/stats", tags=["RAG"])
async def rag_stats():
    return _rag_engine.get_stats() if _rag_engine else {}


@app.post("/api/benchmark/run", tags=["Benchmark"])
async def run_benchmark():
    """Run RAG benchmark and return comparative metrics."""
    from rag.benchmarks.evaluator import BenchmarkConfig, DEFAULT_CONFIGS

    def engine_factory(config: BenchmarkConfig):
        return RAGEngine(
            chunk_size=config.chunk_size,
            top_k=config.top_k,
            llm_model=MODEL,
            ollama_base_url=OLLAMA_URL,
        )

    from rag.benchmarks.evaluator import RAGBenchmarker
    benchmarker = RAGBenchmarker(engine_factory)

    # Run 2 configs for speed (full 4 can take 10+ min)
    results = []
    for config in DEFAULT_CONFIGS[:2]:
        r = benchmarker.run_single_config(config)
        results.append({
            "config": config.name,
            "avg_rouge_l": r.avg_rouge_l,
            "avg_faithfulness": r.avg_faithfulness,
            "avg_latency_ms": r.avg_latency_ms,
            "avg_retrieval_precision": r.avg_retrieval_precision,
        })
    return {"benchmark_results": results}


@app.post("/api/finetune/lora", tags=["FineTuning"])
async def run_lora_demo(req: LoRARequest):
    """Demonstrate LoRA fine-tuning pipeline for engineering domain adaptation."""
    config = LoRAConfig(r=req.lora_r, lora_alpha=req.lora_alpha)
    trainer = LoRATrainer(lora_config=config)
    dataset_prep = EngineeringDatasetPrep()
    formatted = dataset_prep.prepare_dataset()
    stats = dataset_prep.compute_dataset_stats(formatted)
    training_result = trainer.simulate_training_run(n_epochs=req.n_epochs)
    return {
        "dataset_stats": stats,
        "training_result": training_result,
        "peft_config": {
            "r": config.r,
            "alpha": config.lora_alpha,
            "target_modules": config.target_modules,
            "trainable_params": config.trainable_params_estimate,
        },
    }


@app.get("/api/tools", tags=["Agent"])
async def list_tools():
    tools = get_all_tools()
    return {"tools": [{"name": t.name, "description": t.description} for t in tools]}


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8084, reload=True)

from fastapi import Form

def wrap(title, body):
    return f"<html><head><meta charset=UTF-8><title>{title}</title><style>body{{background:#0f1117;color:#e2e8f0;font-family:system-ui;padding:32px;max-width:860px;margin:0 auto}}h1{{font-size:20px;font-weight:700}}h2{{color:#b794f4;margin:28px 0 12px}}label{{display:block;font-size:11px;color:#718096;margin-bottom:5px;text-transform:uppercase;font-weight:700}}textarea,input,select{{width:100%;background:#1a1f2e;border:1px solid #2d3748;border-radius:6px;padding:9px;color:#e2e8f0;font-size:13px;margin-bottom:12px;box-sizing:border-box}}input[type=number]{{width:80px}}button{{background:#b794f4;color:#1a1f2e;border:none;padding:10px 24px;border-radius:6px;font-weight:700;cursor:pointer;font-size:13px}}pre{{background:#1a1f2e;border:1px solid #2d3748;border-radius:6px;padding:16px;white-space:pre-wrap;font-size:13px;line-height:1.7}}a{{color:#63b3ed}}hr{{border:none;border-top:1px solid #2d3748;margin:28px 0}}.info{{background:#1a2a3a;border:1px solid #2d5a8a;color:#90cdf4;padding:10px;border-radius:6px;font-size:13px;margin-bottom:12px}}</style></head><body>{body}</body></html>"

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    body = """
<h1>LLM Agentic AI &amp; RAG Research Platform</h1>
<p style="color:#718096;font-size:13px;margin-bottom:24px">ReAct Agents &middot; RAG Pipeline &middot; LoRA Fine-Tuning &middot; Benchmarking &middot; Engineering Assistance</p>
<hr>
<h2>LoRA Fine-Tuning Demo</h2>
<form action="/ui/lora" method="post">
<div style="display:flex;gap:16px;flex-wrap:wrap">
<div><label>Epochs</label><input type="number" name="n_epochs" value="3" min="1" max="5"></div>
<div><label>LoRA Rank</label><input type="number" name="lora_r" value="16" min="4" max="64"></div>
<div><label>Alpha</label><input type="number" name="lora_alpha" value="32" min="8" max="128"></div>
</div>
<button type="submit">Run LoRA Demo</button>
</form>
<hr>
<h2>RAG Pipeline Query</h2>
<div class="info">Requires Ollama running in a separate terminal: ollama serve</div>
<form action="/ui/rag" method="post">
<label>Question</label>
<textarea name="question" rows="3" placeholder="e.g. What is CAN FD and how does it differ from classical CAN?"></textarea>
<button type="submit">Query RAG</button>
</form>
<hr>
<h2>Ingest Document into FAISS</h2>
<form action="/ui/ingest" method="post">
<label>Document Text</label>
<textarea name="text" rows="5" placeholder="Paste technical documentation here..."></textarea>
<label>Source Name</label>
<input type="text" name="source" value="my_document">
<button type="submit">Ingest into FAISS</button>
</form>
<hr>
<h2>Test Engineering Tool</h2>
<form action="/ui/tool" method="post">
<label>Tool</label>
<select name="tool" style="width:auto">
<option value="error_debugger">error_debugger</option>
<option value="document_lookup">document_lookup</option>
<option value="information_extractor">information_extractor</option>
<option value="code_explainer">code_explainer</option>
<option value="spec_parser">spec_parser</option>
</select>
<label style="margin-top:8px">Input</label>
<textarea name="inp" rows="3" placeholder="e.g. segmentation fault in sensor.c line 42"></textarea>
<button type="submit">Run Tool</button>
</form>
<hr>
<p style="color:#718096;font-size:12px"><a href="/docs">/docs</a> &middot; <a href="/health">/health</a></p>
"""
    return wrap("LLM Agentic Platform", body)

@app.post("/ui/lora", response_class=HTMLResponse)
async def ui_lora(n_epochs: int = Form(3), lora_r: int = Form(16), lora_alpha: int = Form(32)):
    from finetuning.lora_demo import LoRAConfig, LoRATrainer, EngineeringDatasetPrep
    config = LoRAConfig(r=lora_r, lora_alpha=lora_alpha)
    trainer = LoRATrainer(lora_config=config)
    stats = EngineeringDatasetPrep().compute_dataset_stats(EngineeringDatasetPrep().prepare_dataset())
    result = trainer.simulate_training_run(n_epochs=n_epochs)
    out = f"Base model: {result['base_model']}\nLoRA r={result['lora_r']}, alpha={result['lora_alpha']}\nTrainable: {result['trainable_params']}\nQLoRA 4-bit: {result['qlora_enabled']}\n\nDataset: {stats['n_examples']} examples\n\nLoss:\n" + "\n".join(f"  Epoch {h['epoch']}: {h['train_loss']}" for h in result['loss_history'])
    return wrap("LoRA Results", f"<a href='/'>Back</a><h2>LoRA Fine-Tuning Results</h2><pre>{out}</pre>")

@app.post("/ui/rag", response_class=HTMLResponse)
async def ui_rag(question: str = Form(...)):
    result = _rag_engine.query(question)
    return wrap("RAG Result", f"<a href='/'>Back</a><h2>RAG Result</h2><pre><b>Q:</b> {question}\n\n<b>A:</b> {result['answer']}\n\nChunks: {result['retrieved_chunks']} | Latency: {result['latency_ms']}ms</pre>")

@app.post("/ui/ingest", response_class=HTMLResponse)
async def ui_ingest(text: str = Form(...), source: str = Form("doc")):
    n = _rag_engine.ingest_text(text, source)
    return wrap("Ingest Result", f"<a href='/'>Back</a><h2>Ingest Result</h2><pre>Ingested {n} chunks from '{source}'.\nTotal: {_rag_engine.get_stats()['total_chunks']} chunks in index.</pre>")

@app.post("/ui/tool", response_class=HTMLResponse)
async def ui_tool(tool: str = Form(...), inp: str = Form(...)):
    from agents.tools.engineering_tools import get_all_tools
    tools = {t.name: t for t in get_all_tools()}
    result = tools[tool].run(inp) if tool in tools else f"Unknown tool: {tool}"
    return wrap(f"Tool: {tool}", f"<a href='/'>Back</a><h2>{tool}</h2><pre>{result}</pre>")
