# 4.1 Search & Research Agent

Search agents reveal a fundamental contradiction in agentic RL: external knowledge access provides comprehensive information but creates unstable training conditions and scalability bottlenecks, while internal knowledge approaches offer controlled training but risk hallucination and knowledge staleness. The survey evidence suggests current methods excel at single-hop retrieval but struggle with the multi-hop reasoning and cross-source verification that define meaningful research tasks.

## Key Takeaways
- **Training Instability**: External API-based training faces uncontrolled document quality and prohibitive costs
- **Performance Gap**: Open-source methods largely fail BrowseComp (OpenAI Deep Research achieves only 51.5% pass@1)
- **Multi-Hop Challenge**: Long-horizon search (40+ tool calls) requires sophisticated credit assignment
- **Verification Crisis**: Cross-source information synthesis lacks reliable automated evaluation

## Prerequisites Check

```bash
# Survey-validated libraries and APIs
python -c "import requests, json, torch; print('Search APIs and RL stack ready')"
python -c "import transformers, numpy; print('LLM integration ready')" 

# Critical understanding check
echo "Do you understand why API costs limit RL scalability?"
echo "Can you identify multi-hop reasoning failure modes?"
echo "Have you analyzed the external vs internal knowledge trade-offs?"
```

## Table of Contents

- [4.1.1 Open Source RL Methods](4.1.1_Open_Source_RL_Methods.md) - External API and internal knowledge approaches
- [4.1.2 Closed Source RL Methods](4.1.2_Closed_Source_RL_Methods.md) - Industry systems and performance benchmarks

## Hands-On: Search Agent Implementation Analysis

### External vs Internal Knowledge Contradiction
```python
import torch
import json
import requests
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class SearchResult:
    """Search result with quality metrics"""
    content: str
    relevance_score: float
    source_credibility: float
    retrieval_cost: float  # API calls or compute units
    timestamp: float

class ExternalSearchAgent:
    """Survey finding: "uncontrolled document quality brings instability to training process"""
    
    def __init__(self, api_budget: float = 100.0):
        self.api_budget = api_budget
        self.api_calls_made = 0
        self.total_cost = 0.0
        
        # Survey insight: "API cost is too high and severely limits scalability"
        self.cost_per_call = 0.05  # Realistic API pricing
        
    def search_external(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """External search with cost tracking and quality variance"""
        
        if self.total_cost + (self.cost_per_call * max_results) > self.api_budget:
            raise RuntimeError(f"API budget exceeded: {self.total_cost:.2f}/{self.api_budget}")
        
        # Simulate external search with realistic quality variance
        results = []
        for i in range(max_results):
            self.api_calls_made += 1
            self.total_cost += self.cost_per_call
            
            # Survey finding: uncontrolled quality creates training instability
            quality_variance = np.random.normal(0.6, 0.3)  # High variance
            relevance = max(0.1, min(0.9, quality_variance))
            
            # Credibility varies significantly with external sources
            credibility_variance = np.random.normal(0.5, 0.4)
            credibility = max(0.1, min(1.0, credibility_variance))
            
            results.append(SearchResult(
                content=f"External result {i+1} for '{query}' - Variable quality content...",
                relevance_score=relevance,
                source_credibility=credibility,
                retrieval_cost=self.cost_per_call,
                timestamp=time.time()
            ))
            
        return results
    
    def get_training_stability_metrics(self) -> Dict:
        """Analyze training stability issues from external search"""
        return {
            'api_budget_utilization': self.total_cost / self.api_budget,
            'calls_per_episode': self.api_calls_made,
            'cost_scaling_factor': self.api_calls_made * self.cost_per_call,
            'training_episodes_affordable': int(self.api_budget / (self.api_calls_made * self.cost_per_call)) if self.api_calls_made > 0 else 0,
            'stability_risk': 'HIGH - uncontrolled document quality'
        }

class InternalSearchAgent:
    """Survey finding: ZeroSearch/SSRL approaches using "LLM internal knowledge"""
    
    def __init__(self, knowledge_base_size: int = 1000):
        self.knowledge_base_size = knowledge_base_size
        self.hallucination_rate = 0.15  # Realistic internal knowledge limitation
        self.knowledge_staleness = 0.25  # Information may be outdated
        
    def search_internal(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Internal search with controlled quality but knowledge limitations"""
        
        results = []
        for i in range(max_results):
            # Survey benefit: controllable and stable quality
            base_relevance = 0.7  # More consistent than external
            relevance_variance = np.random.normal(0, 0.1)  # Lower variance
            relevance = max(0.3, min(0.9, base_relevance + relevance_variance))
            
            # Internal knowledge limitations
            is_hallucinated = np.random.random() < self.hallucination_rate
            is_stale = np.random.random() < self.knowledge_staleness
            
            credibility = 0.9 if not (is_hallucinated or is_stale) else 0.3
            
            results.append(SearchResult(
                content=f"Internal result {i+1} for '{query}' - {'[HALLUCINATED]' if is_hallucinated else ''}{'[STALE]' if is_stale else ''} Controlled quality content...",
                relevance_score=relevance,
                source_credibility=credibility,
                retrieval_cost=0.001,  # Much lower computational cost
                timestamp=time.time()
            ))
            
        return results
    
    def get_training_stability_metrics(self) -> Dict:
        """Analyze internal search training characteristics"""
        return {
            'hallucination_risk': self.hallucination_rate,
            'knowledge_staleness': self.knowledge_staleness,
            'cost_scalability': 'HIGH - no API limits',
            'quality_consistency': 'HIGH - controlled environment',
            'stability_risk': 'MEDIUM - hallucination and staleness issues'
        }

class MultiHopReasoningAgent:
    """Implementation of multi-hop reasoning with credit assignment challenges"""
    
    def __init__(self, search_agent, max_hops: int = 4):
        self.search_agent = search_agent
        self.max_hops = max_hops
        self.reasoning_chain = []
        
    def multi_hop_search(self, initial_query: str, target_depth: int = 3) -> Dict:
        """Survey challenge: long-horizon search requires sophisticated credit assignment"""
        
        self.reasoning_chain = []
        current_query = initial_query
        accumulated_evidence = []
        hop_rewards = []
        
        for hop in range(min(target_depth, self.max_hops)):
            # Search for current hop
            search_results = self.search_agent.search_external(current_query, max_results=3) \
                            if hasattr(self.search_agent, 'search_external') \
                            else self.search_agent.search_internal(current_query, max_results=3)
            
            # Evaluate hop quality (simplified reward)
            hop_quality = np.mean([r.relevance_score * r.source_credibility for r in search_results])
            hop_rewards.append(hop_quality)
            
            # Credit assignment challenge: how much does this hop contribute to final answer?
            credit_assignment_uncertainty = 0.3 * (hop + 1) / target_depth  # Increases with distance
            
            self.reasoning_chain.append({
                'hop': hop + 1,
                'query': current_query,
                'results': search_results,
                'hop_reward': hop_quality,
                'credit_uncertainty': credit_assignment_uncertainty
            })
            
            accumulated_evidence.extend([r.content for r in search_results])
            
            # Generate next hop query (simplified)
            if hop < target_depth - 1:
                current_query = self._generate_followup_query(current_query, search_results)
        
        # Final synthesis challenge
        synthesis_quality = self._synthesize_evidence(accumulated_evidence)
        
        return {
            'reasoning_chain': self.reasoning_chain,
            'final_synthesis': synthesis_quality,
            'total_hops': len(self.reasoning_chain),
            'credit_assignment_difficulty': np.mean([step['credit_uncertainty'] for step in self.reasoning_chain]),
            'temporal_reward_decay': self._calculate_reward_decay(hop_rewards)
        }
    
    def _generate_followup_query(self, current_query: str, results: List[SearchResult]) -> str:
        """Generate next hop query based on current results"""
        # Simplified: extract key terms and create follow-up
        return f"detailed analysis of {current_query} implications"
    
    def _synthesize_evidence(self, evidence: List[str]) -> Dict:
        """Challenge: cross-source verification lacks reliable automated evaluation"""
        
        # Simulate synthesis difficulty
        evidence_consistency = np.random.normal(0.6, 0.2)  # Variable consistency
        synthesis_coverage = min(len(evidence) / 10.0, 1.0)  # Coverage based on evidence volume
        
        return {
            'consistency_score': max(0.1, min(1.0, evidence_consistency)),
            'coverage_score': synthesis_coverage,
            'verification_confidence': evidence_consistency * synthesis_coverage,
            'synthesis_challenge': 'HIGH - no reliable automated cross-source verification'
        }
    
    def _calculate_reward_decay(self, hop_rewards: List[float]) -> Dict:
        """Survey insight: temporal credit assignment in long-horizon tasks"""
        
        if not hop_rewards:
            return {'decay_pattern': 'N/A', 'early_vs_late_contribution': 0.0}
        
        # Calculate reward decay across hops
        decay_weights = [0.9 ** i for i in range(len(hop_rewards))]
        weighted_rewards = [r * w for r, w in zip(hop_rewards, decay_weights)]
        
        early_contribution = sum(weighted_rewards[:len(weighted_rewards)//2]) if len(weighted_rewards) > 1 else weighted_rewards[0]
        late_contribution = sum(weighted_rewards[len(weighted_rewards)//2:]) if len(weighted_rewards) > 1 else 0
        
        return {
            'decay_pattern': decay_weights,
            'early_vs_late_contribution': early_contribution / (late_contribution + 1e-6),
            'temporal_bias': 'Early hops receive disproportionate credit'
        }

# Demo: External vs Internal Contradiction
def demonstrate_search_contradictions():
    """Show the fundamental trade-offs in search agent training"""
    
    print("=== Search Agent Training Contradictions ===")
    
    # External search analysis
    external_agent = ExternalSearchAgent(api_budget=10.0)  # Limited budget
    
    try:
        external_results = external_agent.search_external("reinforcement learning for search agents", max_results=3)
        external_metrics = external_agent.get_training_stability_metrics()
        
        print(f"External Search Results:")
        print(f"  Quality variance: {np.std([r.relevance_score for r in external_results]):.3f}")
        print(f"  Training episodes affordable: {external_metrics['training_episodes_affordable']}")
        print(f"  Stability risk: {external_metrics['stability_risk']}")
        
    except RuntimeError as e:
        print(f"External search failed: {e}")
    
    # Internal search analysis  
    internal_agent = InternalSearchAgent()
    internal_results = internal_agent.search_internal("reinforcement learning for search agents", max_results=3)
    internal_metrics = internal_agent.get_training_stability_metrics()
    
    print(f"\nInternal Search Results:")
    print(f"  Quality consistency: {internal_metrics['quality_consistency']}")
    print(f"  Hallucination risk: {internal_metrics['hallucination_risk']:.1%}")
    print(f"  Knowledge staleness: {internal_metrics['knowledge_staleness']:.1%}")
    
    # Multi-hop reasoning challenge
    multi_hop_agent = MultiHopReasoningAgent(internal_agent)
    reasoning_result = multi_hop_agent.multi_hop_search("How do search agents handle multi-hop reasoning?", target_depth=3)
    
    print(f"\nMulti-Hop Reasoning Analysis:")
    print(f"  Credit assignment difficulty: {reasoning_result['credit_assignment_difficulty']:.3f}")
    print(f"  Verification confidence: {reasoning_result['final_synthesis']['verification_confidence']:.3f}")
    print(f"  Temporal bias: {reasoning_result['temporal_reward_decay']['temporal_bias']}")

# Run demonstration
demonstrate_search_contradictions()
```

## Critical Analysis: Survey Findings vs Implementation Reality

The survey reveals three fundamental problems current search agents haven't solved:

**Problem 1: The Scalability-Quality Dilemma**
- External methods (DeepRetrieval, Search-R1, WebSailor) provide comprehensive information but face prohibitive training costs
- Internal methods (ZeroSearch, SSRL) offer scalable training but risk systematic hallucination and knowledge gaps
- No current approach successfully bridges this divide

**Problem 2: Multi-Hop Credit Assignment Failure**  
- Long-horizon search (40+ tool calls, per ASearcher findings) requires temporal credit assignment across reasoning chains
- Current reward functions struggle to attribute final success to early reasoning steps
- StepSearch's "intermediate step-level rewards" attempts to address this but lacks theoretical grounding

**Problem 3: Verification Without Ground Truth**
- Cross-source synthesis lacks reliable automated evaluation (survey identifies this gap)
- Human evaluation doesn't scale to the interaction volumes required for RL training
- BrowseComp benchmark exposes these limitations: only 51.5% success rate for leading systems

## Survey Method Analysis

| Method Category | Approach | Key Innovation | Fundamental Limitation |
|-----------------|----------|----------------|----------------------|
| **External API** | DeepRetrieval | GRPO-trained query generation | API cost scalability crisis |
| **Multi-turn** | R1-Searcher | Two-stage PPO (when → how) | Credit assignment across turns |
| **End-to-end** | ReSearch | PPO without supervised trajectories | Training stability issues |
| **Internal Knowledge** | ZeroSearch | Pseudo search engine from LLM | Hallucination and staleness |
| **Offline Training** | SSRL | Self-search without external APIs | Reality gap at deployment |

## Resources

- **Primary Survey**: [Section 4.1, arXiv:2509.02547](https://arxiv.org/abs/2509.02547)
- **BrowseComp Benchmark**: OpenAI's challenging information location benchmark
- **GAIA Evaluation**: WebDancer and other methods performance reference
- **Open Source Methods**: 15+ implementations listed in survey Table 4
- **Closed Source Systems**: OpenAI Deep Research, Perplexity DeepResearch, Google Gemini DeepResearch

## Next Steps

- **[4.1.1 Open Source RL Methods](4.1.1_Open_Source_RL_Methods.md)**: Detailed analysis of external vs internal approaches
- **[4.1.2 Closed Source RL Methods](4.1.2_Closed_Source_RL_Methods.md)**: Industry performance benchmarks and system architectures
- **Integration Challenge**: Design hybrid approaches addressing scalability-quality trade-offs

---

*Search agents expose the fundamental tensions between training scalability and information quality that current agentic RL approaches address inadequately—requiring novel architectures that the survey identifies as critical open research questions.*
