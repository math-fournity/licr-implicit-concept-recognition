# LICR: LLM-based Implicit Concept Recognition

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**LICR** is a hybrid architecture for recognizing implicit concepts in text and linking them to Wikidata knowledge base.

## What is Implicit Concept Recognition?

Traditional entity linking systems can only identify **explicitly mentioned** entities. LICR goes further by recognizing **implicit concepts**—methodologies, principles, and patterns that texts embody but do not explicitly name.

### Example

Given this mathematical proof:

> "Suppose √2 = p/q where p and q are coprime. Then 2q² = p², so p² is even, thus p is even. Let p = 2k, then q² = 2k², so q is also even. But this contradicts our assumption. Therefore √2 is irrational."

**Traditional Entity Linking finds:** √2, rational number, coprime, irrational number

**LICR additionally finds:**
- `proof by contradiction` (Q184899) — the proof method used
- `reductio ad absurdum` (Q184899) — Latin synonym  
- `mathematical proof` (Q11538) — the general methodology

## Architecture

```
Text → LLM (semantic recognition) → Concept Names → Wikidata API → Q-identifiers
```

LICR combines:
- **LLM's strength**: Semantic pattern recognition (100% recall on core concepts)
- **API's strength**: Precise identifier lookup (arbitrary Q-numbers)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Using Mock Backend (No API Key Required)

```bash
# Run a single test case
python licr.py --mock --test math_proof

# Run all test cases
python licr.py --mock --test all
```

### Using OpenAI

```bash
python licr.py --openai YOUR_API_KEY --text "Your text to analyze"
```

### Using Anthropic Claude

```bash
python licr.py --anthropic YOUR_API_KEY --test algorithm
```

### As a Python Library

```python
from licr import LICR, MockBackend, OpenAIBackend, WikidataClient

# Initialize with mock backend (for testing)
backend = MockBackend()
licr = LICR(backend)

# Or with real LLM
# backend = OpenAIBackend("your-api-key")
# licr = LICR(backend)

# Extract implicit concepts
text = """
Suppose √2 = p/q where p and q are coprime. Then 2q² = p², 
so p² is even, thus p is even. Let p = 2k, then q² = 2k², 
so q is also even. But this contradicts our assumption. 
Therefore √2 is irrational.
"""

result = licr.extract(text)

# Access results
print("Explicit concepts:", result.explicit_concepts)
for concept in result.implicit_concepts:
    print(f"Implicit: {concept.name} [{concept.wikidata_id}]")
    print(f"  Evidence: {concept.evidence}")
```

## Test Cases

LICR includes 5 cross-domain test cases from the research paper:

| Test Case | Domain | Key Implicit Concepts |
|-----------|--------|----------------------|
| `math_proof` | Mathematics | proof by contradiction, reductio ad absurdum |
| `algorithm` | Computer Science | quicksort, divide and conquer |
| `economics` | Economics | wage-price spiral, inflation |
| `game_theory` | Game Theory | prisoner's dilemma, Nash equilibrium |
| `physics` | Quantum Physics | double-slit experiment, wave-particle duality |

## Research Results

From our empirical study across 5 domains:

- **Core Implicit Concept Recall**: 100%
- **Overall Accuracy**: 96%
- **LLM Q-Number Reliability**: Low (hence the hybrid architecture)

## API Reference

### LICR Class

```python
class LICR:
    def __init__(self, llm_backend: LLMBackend, wikidata_client: WikidataClient = None)
    def extract(self, text: str, resolve_wikidata: bool = True) -> LICRResult
    def to_dict(self, result: LICRResult) -> dict
```

### Data Classes

```python
@dataclass
class ImplicitConcept:
    name: str                    # Concept name
    evidence: str                # Text evidence
    hierarchy: str               # methodology / domain / entity
    wikidata_id: Optional[str]   # Q-identifier
    wikidata_label: Optional[str]

@dataclass  
class LICRResult:
    text: str
    explicit_concepts: list[str]
    implicit_concepts: list[ImplicitConcept]
```

### LLM Backends

- `MockBackend()` — For testing without API calls
- `OpenAIBackend(api_key, model="gpt-4")` — OpenAI API
- `AnthropicBackend(api_key, model="claude-sonnet-4-20250514")` — Anthropic API

## Integration with RAG Frameworks

LICR is designed to integrate with existing knowledge graph frameworks:

- **LlamaIndex**: As a custom `PathExtractor`
- **LangChain**: As an extension to `LLMGraphTransformer`
- **Microsoft GraphRAG**: As an entity extraction enhancement

See the [integration analysis](LICR_integration_analysis.md) for detailed architecture proposals.

## Citation

If you use LICR in your research, please cite:

```bibtex
@misc{licr2025,
  title={Beyond Named Entities: LICR - An Empirical Study of LLM-based Implicit Concept Recognition},
  author={Independent Researcher},
  year={2025},
  publisher={Zenodo},
  doi={10.5281/zenodo.XXXXXXX}
}
```

## License

This project is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Prior Art Notice

This repository establishes prior art for the LICR methodology and architecture, preventing patent claims on:
- Implicit concept recognition using LLMs
- Hybrid LLM-API architecture for knowledge base linking
- The specific prompt engineering techniques described herein
