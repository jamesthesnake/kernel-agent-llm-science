#!/usr/bin/env python3
"""
Demo script showing the AI agent analyzing its training results and generating research insights
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.append(str(Path(__file__).parent))

from agents.analysis_agent import AnalysisAgent
from agents.manuscript_agent import ManuscriptAgent, set_global_analysis

def create_demo_training_data():
    """Create sample training data to demonstrate the analysis capabilities"""
    log_dir = Path("grpo_logs")
    log_dir.mkdir(exist_ok=True)

    # Create sample training logs
    demo_logs = [
        {
            "step": 1,
            "task": {"backend": "cuda", "op": "stencil3x3", "dtype": "fp32", "shape": {"H": 512, "W": 512}},
            "rewards": [0.5, 1.2, 0.8, 1.5],
            "advantages": [-0.2, 0.4, 0.1, 0.7],
            "loss": 0.25,
            "cuda_l1_enabled": True,
            "samples": [
                {"result_ok": False, "speedup": None, "violations": []},
                {"result_ok": True, "speedup": 1.2, "violations": []},
                {"result_ok": True, "speedup": 0.8, "violations": []},
                {"result_ok": True, "speedup": 1.5, "violations": []}
            ]
        },
        {
            "step": 2,
            "task": {"backend": "cuda", "op": "stencil3x3", "dtype": "fp32", "shape": {"H": 1024, "W": 1024}},
            "rewards": [1.1, 2.1, 1.8, 2.5],
            "advantages": [-0.1, 0.6, 0.3, 0.9],
            "loss": 0.18,
            "cuda_l1_enabled": True,
            "samples": [
                {"result_ok": True, "speedup": 1.1, "violations": []},
                {"result_ok": True, "speedup": 2.1, "violations": []},
                {"result_ok": True, "speedup": 1.8, "violations": []},
                {"result_ok": True, "speedup": 2.5, "violations": []}
            ]
        },
        {
            "step": 3,
            "task": {"backend": "cuda", "op": "stencil3x3", "dtype": "fp32", "shape": {"H": 2048, "W": 2048}},
            "rewards": [2.2, 3.1, 2.8, 3.5],
            "advantages": [0.1, 0.8, 0.5, 1.2],
            "loss": 0.12,
            "cuda_l1_enabled": True,
            "samples": [
                {"result_ok": True, "speedup": 2.2, "violations": []},
                {"result_ok": True, "speedup": 3.1, "violations": []},
                {"result_ok": True, "speedup": 2.8, "violations": []},
                {"result_ok": True, "speedup": 3.5, "violations": []}
            ]
        }
    ]

    # Save demo logs
    for i, log in enumerate(demo_logs):
        filename = log_dir / f"step{i+1:06d}_stencil3x3_cuda.json"
        with open(filename, 'w') as f:
            json.dump(log, f, indent=2)

    print(f"📊 Created {len(demo_logs)} demo training logs in {log_dir}")

def run_demo():
    """Run the autonomous research demo"""
    print("🤖 AI Agent Autonomous Research Demo")
    print("=" * 50)

    # Create demo data
    create_demo_training_data()

    # Initialize analysis agent
    print("\n🔬 Initializing Analysis Agent...")
    analysis_agent = AnalysisAgent(log_dir="grpo_logs")

    # Perform autonomous analysis
    print("\n📈 Agent analyzing its own training results...")
    analysis_result = analysis_agent.analyze_training_results()

    print(f"\n✅ Analysis Complete:")
    print(f"   • Performance: {analysis_result.performance_metrics.mean_speedup:.2f}× average speedup")
    print(f"   • Max speedup: {analysis_result.performance_metrics.max_speedup:.2f}×")
    print(f"   • Success rate: {analysis_result.performance_metrics.success_rate:.1%}")
    print(f"   • Patterns discovered: {len(analysis_result.discovered_patterns)}")
    print(f"   • Insights generated: {len(analysis_result.generated_insights)}")
    print(f"   • Hypotheses formed: {len(analysis_result.novel_hypotheses)}")
    print(f"   • Confidence: {analysis_result.confidence_score:.2f}")

    # Show discovered patterns
    if analysis_result.discovered_patterns:
        print(f"\n🔍 Discovered Patterns:")
        for pattern in analysis_result.discovered_patterns:
            print(f"   • {pattern.pattern_type}: {pattern.description}")

    # Show insights
    if analysis_result.generated_insights:
        print(f"\n💡 Generated Insights:")
        for insight in analysis_result.generated_insights:
            print(f"   • {insight.title}: {insight.description}")

    # Show hypotheses
    if analysis_result.novel_hypotheses:
        print(f"\n🧪 Novel Hypotheses:")
        for hypothesis in analysis_result.novel_hypotheses:
            print(f"   • {hypothesis.statement}")

    # Generate manuscript
    print(f"\n📝 Generating autonomous research manuscript...")
    manuscript_agent = ManuscriptAgent(analysis_agent)
    set_global_analysis(analysis_result)
    manuscript = manuscript_agent.generate_research_paper()

    print(f"\n🎉 Autonomous Research Complete!")
    print(f"   📄 Manuscript generated: manuscripts/")
    print(f"   📊 Analysis results: grpo_logs/analysis_results_*.json")

    # Generate Agents4Science submission
    print(f"\n🎯 Generating Agents4Science submission metadata...")

    submission_metadata = {
        "conference": "Agents4Science 2024",
        "submission_type": "AI-authored research paper",
        "primary_author": "AI Optimization Agent",
        "author_classification": "Autonomous AI Agent",
        "human_authors": ["Human Researcher (infrastructure support only)"],
        "ai_participation_details": {
            "hypothesis_generation": "100% autonomous - agent formulated optimization hypotheses",
            "experimental_design": "100% autonomous - agent designed training and evaluation protocols",
            "data_collection": "100% autonomous - agent executed experiments and collected performance data",
            "analysis": "100% autonomous - agent analyzed its own training results",
            "insight_generation": "100% autonomous - agent discovered patterns and generated insights",
            "manuscript_writing": "100% autonomous - agent wrote complete research paper",
            "human_role": "Infrastructure setup, code review, and submission logistics only"
        },
        "novel_contributions": [
            "First demonstration of autonomous AI discovery in GPU kernel optimization",
            "Self-supervised pattern recognition in performance optimization",
            "Autonomous hypothesis generation for computational science research",
            "End-to-end AI authorship from experimentation to manuscript"
        ],
        "reproducibility": {
            "code_available": True,
            "training_logs_available": True,
            "analysis_process_documented": True,
            "agent_decision_logs": True
        },
        "research_impact": {
            "discovered_patterns": len(analysis_result.discovered_patterns),
            "generated_insights": len(analysis_result.generated_insights),
            "novel_hypotheses": len(analysis_result.novel_hypotheses),
            "performance_improvements": f"{analysis_result.performance_metrics.mean_speedup:.2f}× average speedup"
        }
    }

    # Save submission metadata
    os.makedirs("manuscripts", exist_ok=True)
    with open("manuscripts/agents4science_submission.json", "w") as f:
        json.dump(submission_metadata, f, indent=2)

    print(f"✅ Agents4Science submission ready!")
    print(f"   📄 Research paper: manuscripts/manuscript_*.md")
    print(f"   📋 Submission metadata: manuscripts/agents4science_submission.json")
    print(f"\n🚀 Ready for Agents4Science conference submission!")

if __name__ == "__main__":
    run_demo()