# 2. Preliminary: From LLM RL to Agentic RL

This module formalizes the fundamental paradigm shift from preference-based reinforcement learning fine-tuning (PBRFT) to Agentic RL. You'll understand how moving from degenerate single-step settings to long-horizon, partially observable environments transforms both training dynamics and emergent capabilities, enabling planning, tool use, memory, and strategic reasoning.

## Key Takeaways
- **Mathematical Shift**: Single-step MDP (PBRFT) → Multi-step POMDP (Agentic RL)
- **Action Evolution**: Pure text generation → Text + structured tool calls
- **Learning Objective**: Immediate response quality → Long-term trajectory rewards
- **Environment**: Static prompt-response → Dynamic, evolving world states
- **Formalization**: Connect MDP components to practical agentic behaviors

## Table of Contents
- [2.1 Markov Decision Processes](2.1_Markov_Decision_Processes.md)
- [2.2 Environment State](2.2_Environment_State.md)
- [2.3 Action Space](2.3_Action_Space.md)
- [2.4 Transition Dynamics](2.4_Transition_Dynamics.md)
- [2.5 Reward Function](2.5_Reward_Function.md)
- [2.6 Learning Objective](2.6_Learning_Objective.md)
- [2.7 RL Algorithms](2.7_RL_Algorithms.md)

## Quick Start: Side-by-Side Comparison

### PBRFT (Traditional Approach)
```python
# Single-step, terminal decision process
class PBRFTAgent:
    def __init__(self, model):
        self.model = model
    
    def generate_response(self, prompt):
        """One shot: prompt in, text out, done"""
        response = self.model.generate(prompt)
        reward = self.preference_model.score(prompt, response)
        return response, reward  # Episode terminates

# Usage: Each interaction is independent
agent = PBRFTAgent(model)
response, score = agent.generate_response("Solve this math problem: 2+2")
# No memory, no tools, no multi-step reasoning
```

### Agentic RL (New Paradigm)
```python
# Multi-step, continuing decision process
class AgenticRLAgent:
    def __init__(self, model, tools):
        self.model = model
        self.tools = tools
        self.memory = []
        self.state = {}
    
    def step(self, observation):
        """Continuing interaction with environment"""
        # Update internal state from observation
        self.state.update({'current_obs': observation})
        
        # Decide on action type
        action_type = self.choose_action_type(observation)
        
        if action_type == 'tool_use':
            tool_result = self.tools['calculator'].compute("2+2")
            action = f"Using calculator: {tool_result}"
            self.memory.append({'tool_use': tool_result})
        else:
            action = self.model.generate(observation, context=self.memory)
        
        return action, self.get_reward(action), False  # done=False (continuing)

# Usage: Multi-step interaction with memory and tools
agent = AgenticRLAgent(model, {'calculator': Calculator()})
for step in range(max_steps):
    action, reward, done = agent.step(current_observation)
    if done: break
```

## The Mathematical Foundation

### PBRFT Formalization
```
State Space (S):     Single prompt state
Action Space (A):    Text response only
Reward (R):          Preference model score (terminal)
Transition (T):      Deterministic, episode ends
Horizon:             H = 1 (single step)
Objective:           max E[R(s,a)] - immediate reward
```

### Agentic RL Formalization  
```
State Space (S):     Dynamic environment + agent memory
Action Space (A):    A_text ∪ A_tools (hybrid actions)
Reward (R):          Multi-step rewards (sparse + dense)
Transition (T):      Stochastic, environment evolves
Horizon:             H > 1 (multi-step trajectories)
Objective:           max E[∑γᵗR(s_t,a_t)] - discounted returns
```

## Core Conceptual Shifts

### 1. From Static to Dynamic
```python
# PBRFT: Static interaction
def pbrft_interaction():
    prompt = "What should I do?"
    response = model(prompt)
    # No follow-up, no memory, episode done
    return response

# Agentic RL: Dynamic interaction  
def agentic_interaction():
    observation = env.reset()
    trajectory = []
    
    for t in range(horizon):
        action = agent.act(observation, memory=trajectory)
        observation, reward, done = env.step(action)
        trajectory.append((observation, action, reward))
        
        if done: break
    
    return trajectory  # Full interaction sequence
```

### 2. From Simple to Compound Actions
```python
# PBRFT: Single action type
class PBRFTAction:
    def __init__(self, text_response):
        self.text = text_response
        self.type = "text_only"

# Agentic RL: Compound action types
class AgenticAction:
    def __init__(self, text_part=None, tool_call=None, memory_op=None):
        self.text = text_part or ""
        self.tool_call = tool_call  # {"tool": "calculator", "args": [2, 2]}
        self.memory_op = memory_op  # {"op": "store", "data": "..."}
        self.type = "compound"
```

### 3. From Immediate to Sequential Rewards
```python
# PBRFT: Single reward signal
def pbrft_reward(prompt, response):
    return preference_model.score(response)  # One number, done

# Agentic RL: Multi-step reward design
def agentic_reward(trajectory):
    rewards = []
    for step in trajectory:
        step_reward = 0
        
        # Task completion (sparse)
        if step.achieves_goal():
            step_reward += 100
            
        # Tool use efficiency (dense)
        if step.uses_tools_appropriately():
            step_reward += 10
            
        # Memory utilization (process)
        if step.references_relevant_memory():
            step_reward += 5
            
        rewards.append(step_reward)
    
    return rewards  # Sequence of rewards
```

## Prerequisites Check

```bash
# Mathematical prerequisites
echo "Familiarity with basic probability and linear algebra?"
echo "Understanding of Markov chains and decision processes?"

# Technical setup
python -c "import numpy as np; print('NumPy available for mathematical operations')"
python -c "import matplotlib.pyplot as plt; print('Matplotlib ready for visualizations')"
```

## ASCII Diagram: The Paradigm Shift

```
PBRFT (Single-step MDP):
┌─────────┐    ┌─────────┐    ┌─────────┐
│ Prompt  │───►│  Model  │───►│Response │
│   (S)   │    │  π(a|s) │    │   (A)   │
└─────────┘    └─────────┘    └─────────┘
                      │
                      ▼
                 ┌─────────┐
                 │ Reward  │ (Terminal)
                 │   R(s,a)│
                 └─────────┘

Agentic RL (Multi-step POMDP):
┌─────────┐    ┌─────────┐    ┌─────────────┐    ┌─────────┐
│  Obs    │───►│  Agent  │───►│ Action      │───►│  Env    │
│  O_t    │    │ π(a|o,h)│    │ A_t∪Tools   │    │  Update │
└─────────┘    └─────────┘    └─────────────┘    └─────────┘
     ▲              │                                   │
     │              ▼                                   │
     │         ┌─────────┐                             │
     │         │ Memory  │◄────────────────────────────┘
     │         │   H_t   │
     │         └─────────┘
     │              │
     │              ▼
     │         ┌─────────┐
     └─────────┤ Reward  │ (Continuing)
               │ R(s,a,t)│
               └─────────┘
```

## Module Learning Path

1. **[MDP Fundamentals](2.1_Markov_Decision_Processes.md)**: Mathematical framework comparison
2. **[Environment Design](2.2_Environment_State.md)**: State representation in agentic settings  
3. **[Action Spaces](2.3_Action_Space.md)**: Text + tools hybrid action design
4. **[Dynamics](2.4_Transition_Dynamics.md)**: Environment evolution and stochasticity
5. **[Rewards](2.5_Reward_Function.md)**: Multi-step reward engineering
6. **[Objectives](2.6_Learning_Objective.md)**: Long-term vs immediate optimization
7. **[Algorithms](2.7_RL_Algorithms.md)**: PPO, GRPO, DPO adaptations for agents

## Resources

- **Survey Reference**: [Section 2, arXiv:2509.02547](https://arxiv.org/abs/2509.02547)
- **MDP Tutorial**: Sutton & Barto, Reinforcement Learning: An Introduction
- **Implementation Examples**: PyTorch RL tutorials and Gymnasium environments
- **Next Module**: [3. Agentic Capabilities](../3_Agentic_RL_Capability_Perspective/)

---

*This mathematical foundation enables everything that follows. Master these concepts to understand how RL transforms static LLMs into dynamic agents.*
