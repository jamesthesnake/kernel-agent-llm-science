from __future__ import annotations
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from .markers import THINK_START, THINK_END, ANS_START, ANS_END

@dataclass
class CodeExemplar:
    """Code exemplar with performance score"""
    code: str
    score: float
    metadata: Dict[str, Any]

class ContrastivePromptBuilder:
    """
    Builds contrastive prompts with (i) Performance Analysis → (ii) Algorithm Design → (iii) Code
    as per CUDA-L1 paper requirements
    """

    def __init__(self, exemplar_db: Optional['ExemplarDB'] = None):
        self.exemplar_db = exemplar_db

    def build_prompt(self, system_spec: str, task_context: Dict[str, Any]) -> str:
        """
        Build contrastive prompt with three mandatory sections and exemplars
        """
        exemplars_section = ""
        if self.exemplar_db:
            exemplars = self.exemplar_db.get_exemplars_for_task(task_context, n=2)
            if exemplars:
                exemplars_section = self._render_exemplars(exemplars)

        prompt = f"""{system_spec}

{exemplars_section}

STRICT REQUIREMENTS:
- NO hyperparameter changes allowed
- NO result caching permitted
- Must follow EXACTLY this three-section protocol:

{THINK_START}
## (i) Performance Analysis
[Analyze the computational bottlenecks, memory access patterns, and performance characteristics]

## (ii) Algorithm Design
[Design the optimal algorithm considering parallelization, memory hierarchy, and hardware constraints]

## (iii) Code
[Implement the optimized kernel code]
{THINK_END}

{ANS_START}
# ONE RAW JSON PLAN ONLY. No prose.
{ANS_END}
"""
        return prompt

    def _render_exemplars(self, exemplars: List[CodeExemplar]) -> str:
        """Render exemplars with their performance scores"""
        if not exemplars:
            return ""

        section = "PREVIOUS HIGH-PERFORMING VARIANTS:\n\n"
        for i, exemplar in enumerate(exemplars, 1):
            section += f"## Variant {i} (Speedup: {exemplar.score:.2f}x)\n"
            section += f"```\n{exemplar.code}\n```\n\n"
            if exemplar.metadata:
                section += f"Metadata: {json.dumps(exemplar.metadata, indent=2)}\n\n"

        return section

def prompt_template(system_spec: str, exemplar_db: Optional['ExemplarDB'] = None,
                   task_context: Optional[Dict[str, Any]] = None) -> str:
    """
    Legacy compatibility function that now uses ContrastivePromptBuilder
    """
    builder = ContrastivePromptBuilder(exemplar_db)
    return builder.build_prompt(system_spec, task_context or {})