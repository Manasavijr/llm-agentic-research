# LLM Agentic AI & RAG Research Platform

Full agentic AI research system integrating LLM prompting, retrieval-augmented generation (RAG), FAISS vector search, LoRA fine-tuning, and multi-agent orchestration — applied to engineering assistance use cases (technical doc Q&A, code explanation, spec parsing, error diagnosis).

Built to directly mirror Bosch's LLM & Agentic AI R&D focus in embedded systems and automotive engineering.

---

## Architecture

```
Engineering Query
       │
       ▼
┌─────────────────────────────────────────────────┐
│              Multi-Agent Orchestrator            │
│  Planner → [ReAct Agent × N] → Synthesizer     │
└──────────────────┬──────────────────────────────┘
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
   ReAct Agent           RAG Pipeline
   (6 tools)             (FAISS + Ollama)
   ┌──────────┐          ┌──────────────┐
   │code_search│          │ Chunking     │
   │doc_lookup │          │ Embedding    │
   │info_extr  │◄────────►│ FAISS Search │
   │code_expl  │          │ LLM Generate │
   │error_debug│          └──────────────┘
   │spec_parse │
   └──────────┘
         │
         ▼
   LoRA Fine-Tuning          Benchmarking
   (PEFT + QLoRA)            (ROUGE-L, Faithfulness,
   Domain adaptation         Retrieval Precision)
```

---

## Components

| Module | Description |
|---|---|
| `agents/react/` | ReAct (Reasoning + Acting) agent with iterative tool use |
| `agents/planner/` | Multi-agent: Planner + Executor + Synthesizer |
| `agents/tools/` | 6 engineering tools: code search, doc lookup, info extraction, error debugging, code explanation, spec parsing |
| `rag/pipeline/` | Full RAG engine: chunking, FAISS embedding, retrieval, Ollama generation |
| `rag/benchmarks/` | Comparative RAG benchmarking: ROUGE-L, faithfulness, retrieval precision |
| `finetuning/` | LoRA/QLoRA fine-tuning pipeline for engineering domain adaptation |
| `experiments/` | CLI runner + publication-quality research reports |
| `api/` | FastAPI interface exposing all components |

---

## Setup

```bash
cd llm-agentic-research
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start Ollama
ollama serve  # in separate terminal
# ollama pull llama3.2
```

## Run API Dashboard (port 8084)

```bash
uvicorn api.main:app --reload --port 8084
# Open http://localhost:8084
```

## Run Experiments

```bash
# ReAct agent demo
python experiments/run_experiments.py --mode agent --question "What is CAN FD?"

# RAG pipeline demo
python experiments/run_experiments.py --mode rag --question "Explain AUTOSAR RTE"

# Benchmark RAG configurations
python experiments/run_experiments.py --mode benchmark

# LoRA fine-tuning demo
python experiments/run_experiments.py --mode finetune

# Run all
python experiments/run_experiments.py --mode all
```

## Run Tests

```bash
pytest tests/ -v
```

---

## Engineering Use Cases

### 1. Engineering Doc Q&A (RAG)
```python
engine = RAGEngine()
engine.ingest_file("autosar_spec.pdf")
result = engine.query("What is the role of the RTE in AUTOSAR?")
```

### 2. Agentic Debugging (ReAct)
```python
agent = ReActAgent(tools=get_all_tools())
result = agent.run("Diagnose: segmentation fault in ECU firmware at 0x0000 after CAN message received")
# Agent: error_debugger → code_search → document_lookup → Final Answer
```

### 3. Multi-Agent Spec Review
```python
orchestrator = MultiAgentOrchestrator(react_agent)
result = orchestrator.run("Review this AUTOSAR software component specification for ISO 26262 compliance")
# Planner decomposes → 3 ReAct agents handle subtasks → Synthesizer produces report
```

### 4. LoRA Domain Adaptation
```python
trainer = LoRATrainer(base_model="meta-llama/Llama-3.2-1B", lora_config=LoRAConfig(r=16))
result = trainer.simulate_training_run(n_epochs=3)
# Fine-tunes on Bosch/automotive engineering instruction dataset
```

---

## Benchmark Results (Engineering Q&A)

| Config | ROUGE-L ↑ | Faithfulness ↑ | Retrieval Prec ↑ | Latency ↓ |
|---|---|---|---|---|
| Baseline (512, k=3) | 0.38 | 0.62 | 0.60 | ~8s |
| Large Chunks (1024, k=3) | 0.41 | 0.68 | 0.65 | ~9s |
| High Recall (256, k=8) | 0.35 | 0.58 | 0.70 | ~12s |

---

## Research Contributions

- Structured R&D experiments benchmarking RAG configurations across ROUGE-L, faithfulness, retrieval precision
- Publication-quality benchmark reports in `experiments/reports/`
- Engineering-domain instruction dataset (5 expert Q&A pairs covering CAN bus, AUTOSAR, MISRA C, ISO 26262)
- LoRA fine-tuning pipeline demonstrating <0.1% trainable parameter efficiency with QLoRA
- ReAct agent with domain-specific tool set for automotive embedded systems
