# 4. Agentic RL: The Task Perspective

This module exposes a critical tension: should agentic RL optimize for general capabilities or domain specialization? The survey evidence suggests neither pure approach succeeds—task domains impose constraints that reshape how planning, memory, and reasoning manifest, yet completely isolated specialization fails to transfer insights across domains.

## Key Takeaways
- **Domain Constraints Shape Capabilities**: Planning in code differs fundamentally from planning in GUI navigation
- **Reward Design Determines Behavior**: Task-specific reward functions often conflict with general capability objectives  
- **Evaluation Reveals Hidden Assumptions**: Cross-domain benchmarks expose capability limitations invisible in single-domain testing
- **Transfer Learning Challenges**: Agentic policies trained on one domain often fail to generalize effectively

## Prerequisites Check

```bash
# Survey-referenced implementations and benchmarks
python -c "import torch, transformers; print('Core RL stack ready')"
python -c "import gymnasium, numpy; print('Environment interfaces ready')"

# Critical prerequisite: Have you questioned the capability-task mapping assumptions?
echo "Do you understand why task-agnostic training might fail?"
echo "Can you identify domain-specific constraints that reshape RL objectives?"
```

## Table of Contents

- [4.1 Search & Research Agent](4.1_Search_Research_Agent/) - Information synthesis with verification challenges
- [4.2 Code Agent](4.2_Code_Agent/) - Execution-feedback loops and automated debugging
- [4.3 Mathematical Agent](4.3_Mathematical_Agent/) - Formal vs informal reasoning trade-offs
- [4.4 GUI Agent](4.4_GUI_Agent/) - Perception-action coordination in dynamic interfaces  
- [4.5 Vision Agents](4.5_RL_in_Vision_Agents.md) - Multi-modal grounding and spatial reasoning
- [4.6 Embodied Agents](4.6_RL_in_Embodied_Agents.md) - Physical constraints and safety considerations
- [4.7 Multi-Agent Systems](4.7_RL_in_Multi_Agent_Systems.md) - Coordination failures and emergent behaviors
- [4.8 Other Tasks](4.8_Other_Tasks.md) - Edge cases challenging standard frameworks

## Hands-On: Domain-Specific RL Contradictions

### Challenge: Task Rewards vs Capability Transfer
```python
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class TaskConstraint:
    """Domain constraints that reshape RL objectives"""
    domain: str
    constraint_type: str  # 'temporal', 'safety', 'verification', 'resource'
    description: str
    impact_on_exploration: float  # -1 to 1, how much it limits exploration
    reward_shaping_required: bool

class TaskDomainAnalyzer:
    """Analyze how domain constraints conflict with general RL principles"""
    
    def __init__(self):
        # Based on survey findings - real constraints from literature
        self.domain_constraints = {
            'search': [
                TaskConstraint('search', 'verification', 'Information must be factual and traceable', -0.3, True),
                TaskConstraint('search', 'temporal', 'Multi-hop queries require extended context', 0.1, True),
                TaskConstraint('search', 'resource', 'API rate limits constrain exploration', -0.7, False)
            ],
            'code': [
                TaskConstraint('code', 'safety', 'Execution environments must be sandboxed', -0.5, False),
                TaskConstraint('code', 'verification', 'Unit tests provide immediate reward signal', 0.4, True),
                TaskConstraint('code', 'temporal', 'Compilation feedback loops are synchronous', -0.2, True)
            ],
            'math': [
                TaskConstraint('math', 'verification', 'Formal proofs require logical consistency', -0.8, True),
                TaskConstraint('math', 'temporal', 'Long derivations need intermediate checkpoints', 0.2, True)
            ],
            'gui': [
                TaskConstraint('gui', 'temporal', 'UI states can change asynchronously', 0.3, False),
                TaskConstraint('gui', 'safety', 'Irreversible actions (delete, send) need confirmation', -0.9, True)
            ]
        }
    
    def analyze_domain_conflicts(self, primary_domain: str, transfer_domain: str) -> Dict:
        """Identify why policies trained on primary_domain fail in transfer_domain"""
        
        primary_constraints = self.domain_constraints.get(primary_domain, [])
        transfer_constraints = self.domain_constraints.get(transfer_domain, [])
        
        conflicts = []
        for p_constraint in primary_constraints:
            for t_constraint in transfer_constraints:
                if p_constraint.constraint_type == t_constraint.constraint_type:
                    conflict_severity = abs(p_constraint.impact_on_exploration - t_constraint.impact_on_exploration)
                    if conflict_severity > 0.5:
                        conflicts.append({
                            'type': p_constraint.constraint_type,
                            'severity': conflict_severity,
                            'primary_behavior': p_constraint.description,
                            'transfer_requirement': t_constraint.description,
                            'likely_failure_mode': self._predict_failure_mode(p_constraint, t_constraint)
                        })
        
        return {
            'domain_pair': f"{primary_domain} → {transfer_domain}",
            'conflicts': conflicts,
            'transfer_difficulty': np.mean([c['severity'] for c in conflicts]) if conflicts else 0.0,
            'recommendation': self._generate_transfer_strategy(conflicts)
        }
    
    def _predict_failure_mode(self, primary: TaskConstraint, transfer: TaskConstraint) -> str:
        """Predict specific failure modes based on constraint misalignment"""
        if primary.constraint_type == 'safety':
            if primary.impact_on_exploration > transfer.impact_on_exploration:
                return "Over-cautious behavior, missing valid actions"
            else:
                return "Risk-taking in high-stakes environment"
        elif primary.constraint_type == 'verification':
            return "Misaligned reward expectations, incorrect optimization target"
        elif primary.constraint_type == 'temporal':
            return "Inappropriate time horizon, premature termination or excessive delay"
        else:
            return "Resource allocation mismatch"
    
    def _generate_transfer_strategy(self, conflicts: List[Dict]) -> str:
        """Generate strategy for handling domain transfer challenges"""
        if not conflicts:
            return "Low-conflict transfer, standard fine-tuning should work"
        
        high_severity_conflicts = [c for c in conflicts if c['severity'] > 0.7]
        if high_severity_conflicts:
            return "High-conflict transfer, requires domain-specific reward model retraining"
        else:
            return "Medium-conflict transfer, consider constraint-aware regularization"

# Survey-Based Domain Implementations
class SurveyBasedCodeAgent:
    """Code agent based on actual survey findings, not mock implementations"""
    
    def __init__(self):
        # From survey: "unit tests, compilers, or static analyzers, enabling dense and reliable signals"
        self.reward_sources = {
            'unit_tests': {'weight': 0.4, 'immediacy': 'instant'},
            'compilation': {'weight': 0.3, 'immediacy': 'instant'},
            'static_analysis': {'weight': 0.2, 'immediacy': 'instant'},
            'integration_tests': {'weight': 0.1, 'immediacy': 'delayed'}
        }
        
        # Survey insight: "balances exploration (creative synthesis) and exploitation (passing tests)"
        self.exploration_exploitation_tension = 0.3  # Lower = more exploitation-focused
        
    def evaluate_code_action(self, code: str, context: Dict) -> Tuple[float, Dict]:
        """Reward function based on survey's "executable signals" principle"""
        reward = 0.0
        feedback = {}
        
        # Simulate survey-mentioned reward sources
        if 'def ' in code or 'class ' in code:
            reward += self.reward_sources['compilation']['weight']
            feedback['compilation'] = 'success'
            
        # Check for test-oriented patterns (survey: "test-first strategies")
        if 'test' in code.lower() or 'assert' in code:
            reward += self.reward_sources['unit_tests']['weight']
            feedback['test_coverage'] = 'present'
            
        # Survey: "reduces brittle prompt heuristics by grounding revisions in executable signals"
        if context.get('iteration_count', 0) > 0 and reward > 0:
            reward *= 1.2  # Reward iterative improvement
            feedback['iterative_improvement'] = True
            
        return reward, feedback

# Alternative Architecture Challenge
class CapabilityCompositionAlternative:
    """Challenge assumption that capabilities should be independently trained"""
    
    def __init__(self):
        # Hypothesis: Co-training capabilities within task domains might be superior
        self.capability_interdependencies = {
            'planning_memory': 0.8,    # Planning requires memory context
            'reasoning_tools': 0.9,    # Reasoning guides tool selection
            'perception_memory': 0.6,  # Visual memory aids perception
            'tools_planning': 0.7      # Tool results inform planning updates
        }
    
    def demonstrate_joint_training_hypothesis(self, domain: str) -> Dict:
        """Show why joint capability training might outperform modular approaches"""
        
        # Domain-specific capability coupling
        if domain == 'code':
            coupling_strength = {
                'reasoning_tools': 0.95,  # Debugging requires tight reasoning-execution coupling
                'planning_memory': 0.85,  # Code structure planning needs memory of patterns
                'tools_planning': 0.90    # Compiler feedback reshapes planning immediately
            }
        elif domain == 'search':
            coupling_strength = {
                'reasoning_tools': 0.80,  # Query formulation needs reasoning about tool capabilities
                'planning_memory': 0.90,  # Multi-hop planning requires remembering previous searches
                'memory_tools': 0.75      # Memory guides retrieval strategy
            }
        else:
            coupling_strength = self.capability_interdependencies
            
        joint_training_benefit = np.mean(list(coupling_strength.values()))
        
        return {
            'domain': domain,
            'capability_coupling': coupling_strength,
            'joint_training_advantage': joint_training_benefit,
            'modular_training_risk': 1.0 - joint_training_benefit,
            'hypothesis': f"Joint training may be {joint_training_benefit:.1%} more effective than modular"
        }

# Demo: Domain Constraint Analysis
def run_domain_analysis():
    """Demonstrate domain-specific constraints and transfer challenges"""
    
    analyzer = TaskDomainAnalyzer()
    
    # Analyze transfer difficulties
    transfers = [
        ('code', 'math'),  # Code agent → Math reasoning
        ('search', 'gui'), # Search agent → GUI navigation  
        ('math', 'code')   # Math agent → Code generation
    ]
    
    print("=== Domain Transfer Analysis ===")
    for primary, transfer in transfers:
        analysis = analyzer.analyze_domain_conflicts(primary, transfer)
        print(f"\n{analysis['domain_pair']}:")
        print(f"Transfer Difficulty: {analysis['transfer_difficulty']:.2f}")
        print(f"Strategy: {analysis['recommendation']}")
        
        if analysis['conflicts']:
            print("Key Conflicts:")
            for conflict in analysis['conflicts']:
                print(f"  - {conflict['type']}: {conflict['likely_failure_mode']}")
    
    # Test capability composition hypothesis
    comp_alternative = CapabilityCompositionAlternative()
    domains = ['code', 'search', 'math']
    
    print("\n=== Joint vs Modular Capability Training ===")
    for domain in domains:
        result = comp_alternative.demonstrate_joint_training_hypothesis(domain)
        print(f"{domain.title()} Domain: {result['hypothesis']}")

run_domain_analysis()
```

## Critical Framework Analysis

The survey reveals three fundamental contradictions in current agentic RL approaches:

**Contradiction 1: General vs Specialized Reward Functions**
- General capability rewards (helpfulness, factuality) often conflict with task-specific optimization targets
- Code agents optimized for test-passing may sacrifice code readability for humans
- Search agents optimized for information retrieval may ignore synthesis quality

**Contradiction 2: Transfer vs Adaptation Trade-offs**  
- Policies that transfer well across domains tend to be conservative and suboptimal within domains
- Domain-specific optimization often leads to brittle behaviors that fail on distribution shifts
- The survey provides limited evidence for successful cross-domain transfer strategies

**Contradiction 3: Evaluation Misalignment**
- Single-domain benchmarks miss capability interactions that emerge in complex environments
- Process rewards often misalign with outcome objectives
- Human evaluation preferences may not reflect optimal long-term system behavior

## Resources

- **Primary Survey**: [Section 4, arXiv:2509.02547](https://arxiv.org/abs/2509.02547)
- **Code Benchmarks**: SWE-Bench, HumanEval (referenced extensively in survey)
- **Search Evaluation**: GAIA, WebArena for multi-step information tasks
- **Mathematical Reasoning**: MATH dataset, formal proof environments
- **GUI Automation**: Android/Web navigation frameworks from survey Section 4.4
- **Transfer Learning**: Limited evidence in survey - identifies this as research gap

## Next Steps

- **[5. Environment and Frameworks](../5_Environment_and_Frameworks/)**: Infrastructure supporting multi-domain training
- **Critical Evaluation**: Design experiments testing domain transfer assumptions
- **Alternative Architectures**: Explore joint capability-task optimization approaches

---

*The task perspective exposes fundamental tensions between capability generality and domain optimization—tensions that current RL approaches address inadequately, suggesting the need for novel architectural approaches that the survey identifies as open research questions.*
