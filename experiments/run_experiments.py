"""
Run all research experiments and generate reports.

Usage:
    python experiments/run_experiments.py --mode agent --question "What is CAN bus?"
    python experiments/run_experiments.py --mode rag --question "Explain AUTOSAR RTE"
    python experiments/run_experiments.py --mode benchmark
    python experiments/run_experiments.py --mode finetune
    python experiments/run_experiments.py --mode all
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ENGINEERING_QUESTIONS = [
    "What is the maximum bit rate of CAN FD and how does it differ from classical CAN?",
    "Explain the AUTOSAR layered architecture and the role of the RTE.",
    "What are MISRA C guidelines and why are they critical for ISO 26262 compliance?",
    "How does a window watchdog differ from an independent watchdog in embedded safety systems?",
    "Describe the LoRA fine-tuning approach and its advantages for domain adaptation.",
]


def run_agent_demo(question: str):
    from agents.tools.engineering_tools import get_all_tools
    from agents.react.react_agent import ReActAgent

    logger.info(f"=== ReAct Agent Demo ===")
    logger.info(f"Question: {question}")

    tools = get_all_tools()
    agent = ReActAgent(tools=tools)
    result = agent.run(question)

    print(f"\n{'='*60}")
    print(f"Question: {result.question}")
    print(f"Steps: {result.total_steps} | Tools: {result.tools_used} | Latency: {result.latency_ms:.0f}ms")
    print(f"\nFinal Answer:\n{result.final_answer}")
    return result


def run_rag_demo(question: str):
    from rag.pipeline.rag_engine import RAGEngine

    logger.info(f"=== RAG Pipeline Demo ===")
    engine = RAGEngine()

    # Ingest sample engineering corpus
    corpus = """
    CAN Bus Protocol: The Controller Area Network (CAN) is a robust serial communication
    protocol developed by Bosch in 1983. It supports speeds up to 1 Mbit/s (CAN) and 
    8 Mbit/s (CAN FD). CAN FD extends classical CAN with larger 64-byte payloads.
    
    AUTOSAR Architecture: AUTOSAR (Automotive Open System Architecture) provides a
    standardized software framework. Classic AUTOSAR targets deeply embedded ECUs with
    OSEK/VDX OS. Adaptive AUTOSAR targets high-performance platforms with POSIX OS.
    The Runtime Environment (RTE) abstracts communication between software components.
    
    MISRA C: Motor Industry Software Reliability Association guidelines for C language
    use in safety-critical systems. 143 mandatory/advisory rules covering type safety,
    control flow, dynamic memory, and undefined behavior prevention.
    
    ISO 26262: Functional safety standard for road vehicles. Defines ASIL levels
    (QM, A, B, C, D) based on Severity, Exposure, and Controllability metrics.
    """
    engine.ingest_text(corpus, source="engineering_manual")

    result = engine.query(question)
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"Retrieved: {result['retrieved_chunks']} chunks | Latency: {result['latency_ms']:.0f}ms")
    print(f"\nAnswer:\n{result['answer']}")
    return result


def run_benchmark():
    from rag.pipeline.rag_engine import RAGEngine
    from rag.benchmarks.evaluator import RAGBenchmarker, BenchmarkConfig, DEFAULT_CONFIGS

    logger.info("=== RAG Benchmark ===")

    def engine_factory(config: BenchmarkConfig):
        return RAGEngine(chunk_size=config.chunk_size, top_k=config.top_k)

    benchmarker = RAGBenchmarker(engine_factory)
    results = benchmarker.run_comparative_benchmark(DEFAULT_CONFIGS[:2])

    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'Config':<20} {'ROUGE-L':>10} {'Faithful':>10} {'Prec':>8} {'Latency':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r.config.name:<20} {r.avg_rouge_l:>10.4f} {r.avg_faithfulness:>10.4f} "
              f"{r.avg_retrieval_precision:>8.4f} {r.avg_latency_ms:>10.1f}ms")
    return results


def run_finetune():
    from finetuning.lora_demo import LoRAConfig, LoRATrainer, EngineeringDatasetPrep

    logger.info("=== LoRA Fine-Tuning Demo ===")
    config = LoRAConfig(r=16, lora_alpha=32)
    trainer = LoRATrainer(lora_config=config)

    dataset_prep = EngineeringDatasetPrep()
    formatted = dataset_prep.prepare_dataset()
    stats = dataset_prep.compute_dataset_stats(formatted)
    logger.info(f"Dataset: {stats}")

    result = trainer.simulate_training_run(n_epochs=3, n_steps=30)
    print(f"\n{'='*60}")
    print(f"LoRA Fine-Tuning Results:")
    print(f"  Base model: {result['base_model']}")
    print(f"  Trainable params: {result['trainable_params']}")
    print(f"  Final loss: {result['final_loss']:.4f}")
    print(f"  QLoRA 4-bit: {result['qlora_enabled']}")
    return result


def main():
    parser = argparse.ArgumentParser(description="LLM Agentic Research Experiments")
    parser.add_argument("--mode", choices=["agent", "rag", "benchmark", "finetune", "all"], default="all")
    parser.add_argument("--question", default=ENGINEERING_QUESTIONS[0])
    args = parser.parse_args()

    if args.mode in ("agent", "all"):
        run_agent_demo(args.question)
    if args.mode in ("rag", "all"):
        run_rag_demo(args.question)
    if args.mode in ("benchmark", "all"):
        run_benchmark()
    if args.mode in ("finetune", "all"):
        run_finetune()


if __name__ == "__main__":
    main()
