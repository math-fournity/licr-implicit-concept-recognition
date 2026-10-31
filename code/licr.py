"""
LICR: LLM-based Implicit Concept Recognition

A hybrid architecture for recognizing implicit concepts in text and linking them to Wikidata.

Architecture: Text → LLM (semantic recognition) → Concept Names → Wikidata API → Q-identifiers

Author: Independent Researcher
License: CC-BY 4.0
Version: 1.0.0
"""

import json
import requests
from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ImplicitConcept:
    """Represents an implicit concept identified in text."""
    name: str                    # Concept name (e.g., "proof by contradiction")
    evidence: str                # Text evidence supporting the inference
    hierarchy: str               # methodology / domain / entity
    wikidata_id: Optional[str] = None  # Q-identifier (e.g., "Q184899")
    wikidata_label: Optional[str] = None  # Verified label from Wikidata


@dataclass
class LICRResult:
    """Complete LICR analysis result."""
    text: str
    explicit_concepts: list[str]
    implicit_concepts: list[ImplicitConcept]


# ============================================================================
# Wikidata API Client
# ============================================================================

class WikidataClient:
    """Client for Wikidata API operations."""
    
    SEARCH_URL = "https://www.wikidata.org/w/api.php"
    
    def search(self, concept_name: str, language: str = "en") -> Optional[str]:
        """
        Search Wikidata for a concept and return its Q-identifier.
        
        Args:
            concept_name: The concept name to search for
            language: Language code (default: "en")
            
        Returns:
            Q-identifier string (e.g., "Q184899") or None if not found
        """
        params = {
            "action": "wbsearchentities",
            "search": concept_name,
            "language": language,
            "format": "json",
            "limit": 1
        }
        
        try:
            response = requests.get(self.SEARCH_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("search"):
                return data["search"][0]["id"]
            return None
            
        except requests.RequestException as e:
            print(f"Wikidata API error: {e}")
            return None
    
    def get_label(self, qid: str, language: str = "en") -> Optional[str]:
        """Get the label for a Q-identifier."""
        params = {
            "action": "wbgetentities",
            "ids": qid,
            "props": "labels",
            "languages": language,
            "format": "json"
        }
        
        try:
            response = requests.get(self.SEARCH_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "entities" in data and qid in data["entities"]:
                labels = data["entities"][qid].get("labels", {})
                if language in labels:
                    return labels[language]["value"]
            return None
            
        except requests.RequestException:
            return None


# ============================================================================
# LLM Backend Interface
# ============================================================================

class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Send prompt to LLM and return response."""
        pass


class OpenAIBackend(LLMBackend):
    """OpenAI API backend."""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"
    
    def complete(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        }
        
        response = requests.post(self.api_url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class AnthropicBackend(LLMBackend):
    """Anthropic API backend."""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"
    
    def complete(self, prompt: str) -> str:
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        data = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        response = requests.post(self.api_url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        return response.json()["content"][0]["text"]


class MockBackend(LLMBackend):
    """Mock backend for testing without API calls."""
    
    def __init__(self):
        # Pre-defined responses for test cases
        self.responses = {
            "sqrt2": json.dumps({
                "explicit_concepts": ["square root of 2", "rational number", "coprime integers", "irrational number"],
                "implicit_concepts": [
                    {"concept": "proof by contradiction", "evidence": "Assume proposition, derive contradiction, negate assumption", "hierarchy": "methodology"},
                    {"concept": "reductio ad absurdum", "evidence": "Latin term for proof by contradiction", "hierarchy": "methodology"},
                    {"concept": "mathematical proof", "evidence": "Formal logical reasoning process", "hierarchy": "methodology"}
                ]
            }),
            "quicksort": json.dumps({
                "explicit_concepts": ["array", "pivot element", "partition", "recursion", "subarray"],
                "implicit_concepts": [
                    {"concept": "quicksort", "evidence": "Pivot selection, partitioning, recursive processing", "hierarchy": "domain"},
                    {"concept": "divide and conquer", "evidence": "Decompose problem into subproblems", "hierarchy": "methodology"}
                ]
            })
        }
    
    def complete(self, prompt: str) -> str:
        # Return appropriate mock response based on prompt content
        if "√2" in prompt or "sqrt" in prompt.lower() or "coprime" in prompt.lower():
            return self.responses["sqrt2"]
        elif "pivot" in prompt.lower() or "partition" in prompt.lower():
            return self.responses["quicksort"]
        else:
            return json.dumps({"explicit_concepts": [], "implicit_concepts": []})


# ============================================================================
# LICR Core Engine
# ============================================================================

class LICR:
    """
    LICR: LLM-based Implicit Concept Recognition
    
    Main class implementing the hybrid architecture for implicit concept recognition.
    """
    
    PROMPT_TEMPLATE = '''You are a Wikidata concept annotation expert. Analyze the following text and identify all concepts linkable to Wikidata.

Requirements:
1. Identify [Explicit Concepts]: Entities directly mentioned in text (names, places, terms, etc.)
2. Identify [Implicit Concepts]: Methods, principles, patterns that text embodies but does not explicitly name
   - This is the KEY task: identify what the text "does" rather than what it "says"
3. Use Wikidata standard English concept names
4. Classify by hierarchy:
   - methodology: Proof methods, analytical frameworks, research paradigms
   - domain: Disciplinary terms, theoretical models
   - entity: People, objects, events

Output ONLY valid JSON (no markdown, no explanation):
{{
  "explicit_concepts": ["concept1", "concept2", ...],
  "implicit_concepts": [
    {{
      "concept": "concept name",
      "evidence": "specific text evidence supporting inference",
      "hierarchy": "methodology/domain/entity"
    }}
  ]
}}

Text to analyze:
{text}'''

    def __init__(self, llm_backend: LLMBackend, wikidata_client: Optional[WikidataClient] = None):
        """
        Initialize LICR engine.
        
        Args:
            llm_backend: LLM backend for concept extraction
            wikidata_client: Wikidata client for Q-ID resolution (optional)
        """
        self.llm = llm_backend
        self.wikidata = wikidata_client or WikidataClient()
    
    def extract(self, text: str, resolve_wikidata: bool = True) -> LICRResult:
        """
        Extract implicit concepts from text.
        
        Args:
            text: Input text to analyze
            resolve_wikidata: Whether to resolve Wikidata Q-identifiers
            
        Returns:
            LICRResult containing explicit and implicit concepts
        """
        # Step 1: LLM extracts concepts (semantic understanding)
        prompt = self.PROMPT_TEMPLATE.format(text=text)
        response = self.llm.complete(prompt)
        
        # Step 2: Parse LLM response
        try:
            # Handle potential markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            data = json.loads(response.strip())
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Raw response: {response}")
            return LICRResult(text=text, explicit_concepts=[], implicit_concepts=[])
        
        explicit = data.get("explicit_concepts", [])
        
        # Step 3: Build ImplicitConcept objects
        implicit_concepts = []
        for item in data.get("implicit_concepts", []):
            concept = ImplicitConcept(
                name=item.get("concept", ""),
                evidence=item.get("evidence", ""),
                hierarchy=item.get("hierarchy", "domain")
            )
            
            # Step 4: Resolve Wikidata Q-identifiers (API precision)
            if resolve_wikidata:
                qid = self.wikidata.search(concept.name)
                if qid:
                    concept.wikidata_id = qid
                    concept.wikidata_label = self.wikidata.get_label(qid)
            
            implicit_concepts.append(concept)
        
        return LICRResult(
            text=text,
            explicit_concepts=explicit,
            implicit_concepts=implicit_concepts
        )
    
    def to_dict(self, result: LICRResult) -> dict:
        """Convert LICRResult to dictionary for JSON serialization."""
        return {
            "text": result.text,
            "explicit_concepts": result.explicit_concepts,
            "implicit_concepts": [
                {
                    "name": c.name,
                    "evidence": c.evidence,
                    "hierarchy": c.hierarchy,
                    "wikidata_id": c.wikidata_id,
                    "wikidata_label": c.wikidata_label
                }
                for c in result.implicit_concepts
            ]
        }


# ============================================================================
# Test Cases from Paper
# ============================================================================

TEST_CASES = {
    "math_proof": {
        "text": """Suppose √2 = p/q where p and q are coprime. Then 2q² = p², so p² is even, 
thus p is even. Let p = 2k, then q² = 2k², so q is also even. But this 
contradicts our assumption that p and q are coprime. Therefore √2 is irrational.""",
        "expected_implicit": ["proof by contradiction", "reductio ad absurdum", "mathematical proof"]
    },
    
    "algorithm": {
        "text": """To sort an array, pick a pivot element. Partition the array so elements 
less than pivot go left, greater go right. Then recursively apply the 
same process to the left and right subarrays until each subarray has 
one element.""",
        "expected_implicit": ["quicksort", "divide and conquer", "recursion"]
    },
    
    "economics": {
        "text": """Workers demanded higher wages because prices kept rising. Companies raised 
prices to cover increased labor costs. This led workers to demand even 
higher wages. The cycle continued, with each round of increases triggering 
the next.""",
        "expected_implicit": ["wage-price spiral", "inflation", "positive feedback loop"]
    },
    
    "game_theory": {
        "text": """Two suspects are interrogated separately. If both stay silent, each gets 
1 year. If one confesses while the other stays silent, the confessor goes 
free while the other gets 10 years. If both confess, each gets 5 years. 
Rationally, each suspect finds that confessing is better regardless of 
what the other does.""",
        "expected_implicit": ["prisoner's dilemma", "game theory", "Nash equilibrium", "dominant strategy"]
    },
    
    "physics": {
        "text": """A beam of light passes through two narrow parallel slits. On the screen 
behind, instead of two bright lines, we observe alternating bright and 
dark bands. The pattern changes when we try to detect which slit each 
photon passes through.""",
        "expected_implicit": ["double-slit experiment", "wave interference", "wave-particle duality"]
    }
}


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Command-line interface for LICR."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LICR: LLM-based Implicit Concept Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with mock backend (no API key needed)
  python licr.py --mock --test math_proof
  
  # Run with OpenAI
  python licr.py --openai YOUR_API_KEY --text "Your text here"
  
  # Run with Anthropic
  python licr.py --anthropic YOUR_API_KEY --test algorithm
  
  # Run all test cases
  python licr.py --mock --test all
        """
    )
    
    # Backend selection
    backend_group = parser.add_mutually_exclusive_group(required=True)
    backend_group.add_argument("--mock", action="store_true", help="Use mock backend (for testing)")
    backend_group.add_argument("--openai", metavar="API_KEY", help="Use OpenAI backend")
    backend_group.add_argument("--anthropic", metavar="API_KEY", help="Use Anthropic backend")
    
    # Input selection
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="Text to analyze")
    input_group.add_argument("--test", choices=list(TEST_CASES.keys()) + ["all"], help="Run test case")
    
    # Options
    parser.add_argument("--no-wikidata", action="store_true", help="Skip Wikidata resolution")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    # Initialize backend
    if args.mock:
        backend = MockBackend()
    elif args.openai:
        backend = OpenAIBackend(args.openai)
    else:
        backend = AnthropicBackend(args.anthropic)
    
    # Initialize LICR
    licr = LICR(backend)
    
    # Process input
    if args.text:
        texts = [("custom", args.text)]
    elif args.test == "all":
        texts = [(name, case["text"]) for name, case in TEST_CASES.items()]
    else:
        texts = [(args.test, TEST_CASES[args.test]["text"])]
    
    # Run extraction
    for name, text in texts:
        print(f"\n{'='*60}")
        print(f"Test Case: {name}")
        print(f"{'='*60}")
        
        result = licr.extract(text, resolve_wikidata=not args.no_wikidata)
        
        if args.json:
            print(json.dumps(licr.to_dict(result), indent=2, ensure_ascii=False))
        else:
            print(f"\nInput Text:\n{text[:200]}...")
            print(f"\nExplicit Concepts: {result.explicit_concepts}")
            print(f"\nImplicit Concepts:")
            for c in result.implicit_concepts:
                qid_str = f" [{c.wikidata_id}]" if c.wikidata_id else ""
                print(f"  • {c.name}{qid_str}")
                print(f"    Hierarchy: {c.hierarchy}")
                print(f"    Evidence: {c.evidence}")
                if c.wikidata_label:
                    print(f"    Wikidata Label: {c.wikidata_label}")


if __name__ == "__main__":
    main()
