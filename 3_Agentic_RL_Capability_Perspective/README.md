# 3. Agentic RL: The Model Capability Perspective

This module organizes Agentic RL around core capabilities: planning, tool using, memory, self-improvement, reasoning, and perception. Rather than treating these as fixed heuristics, RL transforms them into trainable policies that interact and co-adapt, creating emergent intelligent behaviors through optimization.

## Key Takeaways
- **Capability-Driven Architecture**: RL enables learning of planning, tool use, memory, and reasoning as trainable components
- **Co-Evolution**: Capabilities develop together through shared optimization rather than isolated engineering
- **Emergent Intelligence**: Complex behaviors emerge from simple capability combinations under RL training
- **Practical Implementation**: Each capability maps to concrete algorithmic patterns and training strategies

## Prerequisites Check

```bash
# Verify core libraries for capability implementation
python -c "import torch, transformers; print('Deep learning stack ready')"
python -c "import gymnasium; print('RL environment tools ready')"
python -c "import numpy as np, json; print('Data processing tools ready')"

# Conceptual prerequisites
echo "Have you completed Module 2 (MDP/POMDP foundations)?"
echo "Do you understand the difference between heuristics and learned policies?"
echo "Are you ready to implement multi-component agentic systems?"
```

## Table of Contents

- [3.1 Planning](3.1_Planning.md)
- [3.2 Tool Using](3.2_Tool_Using.md)
- [3.3 Memory](3.3_Memory.md)
- [3.4 Self-Improvement](3.4_Self_Improvement.md)
- [3.5 Reasoning](3.5_Reasoning.md)
- [3.6 Perception](3.6_Perception.md)
- [3.7 Others](3.7_Others.md)

## Capability Framework Overview

### From Heuristics to Learned Policies
```python
# Traditional approach: Fixed heuristics
class TraditionalAgent:
    def plan(self, goal):
        return self.hardcoded_planning_algorithm(goal)
    
    def use_tool(self, task):
        return self.rule_based_tool_selection(task)
    
    def remember(self, info):
        return self.fixed_memory_structure.store(info)

# Agentic RL approach: Learned capabilities
class AgenticRLAgent:
    def __init__(self):
        self.planning_policy = LearnedPlanningPolicy()
        self.tool_policy = LearnedToolPolicy()
        self.memory_policy = LearnedMemoryPolicy()
        
    def plan(self, goal, context):
        return self.planning_policy.act(goal, context, self.experience)
        
    def use_tool(self, task, available_tools):
        return self.tool_policy.select_and_execute(task, available_tools, self.memory)
        
    def remember(self, info, context):
        return self.memory_policy.decide_storage(info, context, self.current_capacity)
```

### Capability Integration Patterns
```python
class IntegratedCapabilitySystem:
    """Capabilities working together through shared optimization"""
    
    def __init__(self):
        self.capabilities = {
            'planning': PlanningModule(),
            'tools': ToolModule(), 
            'memory': MemoryModule(),
            'reasoning': ReasoningModule()
        }
        self.shared_state = {}
        
    def step(self, observation, goal):
        """Coordinated capability execution"""
        
        # Memory informs planning
        relevant_memory = self.capabilities['memory'].retrieve(observation)
        
        # Planning considers available tools
        plan = self.capabilities['planning'].create_plan(
            goal, observation, relevant_memory, 
            available_tools=self.capabilities['tools'].get_available()
        )
        
        # Execute plan step with reasoning
        next_action = self.capabilities['reasoning'].evaluate_plan_step(
            plan.current_step, observation, self.shared_state
        )
        
        # Update memory with results
        self.capabilities['memory'].update(observation, next_action, plan.current_step)
        
        return next_action, plan
```

## ASCII Diagram: Capability Architecture

```
Traditional Fixed Heuristics:
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Planning   │  │ Tool Using  │  │   Memory    │
│ (hardcoded) │  │ (rule-based)│  │ (fixed DB)  │
└─────────────┘  └─────────────┘  └─────────────┘
       │                │                │
       └────── No Learning ──────────────┘

Agentic RL Learned Capabilities:
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Planning   │◄─┤ Tool Using  │◄─┤   Memory    │
│   Policy    │  │   Policy    │  │   Policy    │
│   π_plan    │  │   π_tool    │  │   π_mem     │
└─────────────┘  └─────────────┘  └─────────────┘
       │                │                │
       │                │                │
       └────────────────┼────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  Shared RL      │
              │  Optimization   │ 
              │  ∇J(θ_all)      │
              └─────────────────┘

Co-Evolution Through Joint Training:
       Planning ◄──────────► Tool Use
          ▲                    ▲
          │                    │
          │     Memory         │
          └─────────◄─────────┘
               ▲
               │
         ┌─────────────┐
         │ Shared      │
         │ Experience  │
         │ Buffer      │
         └─────────────┘
```

## Learning Path Through Capabilities

### Progressive Capability Development
```python
def capability_learning_curriculum():
    """Progressive development of agentic capabilities"""
    
    curriculum = {
        'stage_1_basic': {
            'focus': 'Individual capability learning',
            'capabilities': ['planning', 'tool_use'],
            'integration': 'minimal',
            'complexity': 'simple_tasks'
        },
        'stage_2_coordination': {
            'focus': 'Capability interaction',
            'capabilities': ['planning + tools', 'memory + planning'],
            'integration': 'pairwise',
            'complexity': 'multi_step_tasks'
        },
        'stage_3_emergence': {
            'focus': 'Full system behaviors',
            'capabilities': ['all_capabilities'],
            'integration': 'full_system',
            'complexity': 'complex_reasoning'
        }
    }
    
    print("=== Capability Learning Curriculum ===")
    for stage, details in curriculum.items():
        print(f"\n{stage.upper()}:")
        print(f"  Focus: {details['focus']}")
        print(f"  Capabilities: {details['capabilities']}")
        print(f"  Integration: {details['integration']}")
        print(f"  Task Complexity: {details['complexity']}")

capability_learning_curriculum()
```

## Module Learning Objectives

By completing this module, you will:

1. **Understand Capability Decomposition**: Break down agentic behavior into learnable components
2. **Implement Individual Capabilities**: Build planning, tool use, memory, and reasoning modules
3. **Design Capability Integration**: Create systems where capabilities reinforce each other
4. **Apply RL Training**: Use reinforcement learning to optimize capability policies
5. **Evaluate Emergent Behaviors**: Assess how simple capabilities combine into complex intelligence

## Research Foundations

This module synthesizes research across multiple domains:
- **Planning**: RAP, LATS, tree search integration with RL
- **Tool Integration**: ReAct, Toolformer, function calling optimization
- **Memory Systems**: RAG optimization, episodic memory, working memory policies
- **Self-Improvement**: Reflexion, self-correction, autonomous learning loops
- **Reasoning**: Chain-of-thought optimization, System 1/2 thinking, verification

## Implementation Strategy

### Development Approach
1. **Individual Components First**: Master each capability in isolation
2. **Pairwise Integration**: Learn how capabilities interact in pairs
3. **System-Level Emergence**: Combine all capabilities into unified agents
4. **Optimization and Scaling**: Improve efficiency and handle larger problems

### Evaluation Framework
- **Component-Level**: Test individual capability performance
- **Integration-Level**: Measure capability coordination effectiveness  
- **System-Level**: Evaluate emergent behaviors and task completion
- **Scaling-Level**: Assess performance on increasingly complex problems

## Resources

- **Primary Survey**: [Section 3, arXiv:2509.02547](https://arxiv.org/abs/2509.02547)
- **Planning Research**: RAP, LATS, and tree search methodologies
- **Tool Integration**: ReAct framework and function calling patterns
- **Memory Systems**: RAG optimization and episodic memory architectures
- **Implementation Examples**: PyTorch RL + Transformers integration patterns

## Next Steps

- **Start with [3.1 Planning](3.1_Planning.md)**: Learn policy-driven planning vs heuristic search
- **Implementation Order**: Follow the curriculum progression from basic to emergent capabilities
- **Integration Focus**: Pay attention to how capabilities interact and reinforce each other

---

*This capability perspective transforms static heuristics into dynamic, learned behaviors. The magic happens not in individual capabilities, but in their coordinated evolution through shared RL optimization.*
