# RL101: Reinforcement Learning 101

A pragmatic, hands-on course transforming Large Language Models from passive sequence generators into autonomous decision-making agents. Based on "The Landscape of Agentic Reinforcement Learning for LLMs: A Survey" (arXiv:2509.02547), this course bridges theory to implementation with runnable code, practical examples, and industry-standard security practices.

##  Key Takeaways
- **Paradigm Shift**: From single-step preference-based reinforcement fine-tuning (PBRFT) to multi-step agent training (Agentic RL)
- **Technical Foundation**: Partially observable Markov Decision Process (POMDP) formalism enables planning, tool use, memory, and self-improvement
- **Practical Focus**: Every concept includes runnable implementation and evaluation benchmarks
- **Security-First**: Zero-trust architecture with secure-by-design patterns throughout

##  Quick Start

### Prerequisites Check
```bash
# Check Python environment
python --version  # Requires Python 3.8+
pip --version     # Package management

# Install core dependencies
pip install torch transformers gymnasium numpy matplotlib

# Verify GPU availability (optional but recommended)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Quick environment test
python -c "import gym; print('Environment setup complete!')"
```

### 5-Minute Demo: Your First Agentic RL Agent
```python
import gymnasium as gym
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

class SimpleAgenticAgent:
    """Minimal agentic RL agent demonstrating core concepts"""
    def __init__(self, model_name="distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.memory = []  # Simple episodic memory
        
    def act(self, observation, tools_available=None):
        """Core agentic decision: text generation + tool selection"""
        # Simple planning: consider observation + memory
        context = f"Observation: {observation}\nMemory: {self.memory[-3:]}"
        
        # Tokenize with input validation (security)
        if len(context) > 512:
            context = context[-512:]  # Truncate safely
            
        inputs = self.tokenizer(context, return_tensors="pt", truncation=True)
        
        # Generate response (simplified)
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Action selection: [text_response, tool_call, confidence]
        action = {
            'text': "Based on observation, I should...",
            'tool': tools_available[0] if tools_available else None,
            'confidence': 0.8
        }
        
        # Update memory (learning)
        self.memory.append({'obs': observation, 'action': action})
        return action

# Demo usage
agent = SimpleAgenticAgent()
result = agent.act("User asks: What's 2+2?", tools_available=['calculator'])
print(f"Agent decision: {result}")
```

##  Learning Architecture

```
Foundation (Weeks 1-4)          Implementation (Weeks 5-8)
┌─────────────────────┐        ┌─────────────────────┐
│  MDP/POMDP Theory  │───────►│   RAG Systems      │
│  Context Assembly  │        │   Memory Agents    │
│  Reward Design     │        │   Tool Integration │
│  Algorithm Basics  │        │   Multi-Agent      │
└─────────────────────┘        └─────────────────────┘
         │                              │
         ▼                              ▼
┌─────────────────────┐        ┌─────────────────────┐
│ Capability Training │        │  Frontier Research  │
│ Planning, Memory    │◄───────┤  Scaling Challenges │
│ Tool Use, Reasoning │        │  Safety & Trust     │
│ Self-Improvement    │        │  Future Directions  │
└─────────────────────┘        └─────────────────────┘
```

##  Course Modules

### Part I: Mathematical Foundations (Weeks 1-4)
#### [1. Introduction](1_Introduction/)
- Paradigm shift from LLM-RL to Agentic RL
- Survey overview and research landscape

#### [2. From LLM RL to Agentic RL](2_Preliminaries_From_LLM_RL_to_Agentic_RL/)
- [2.1 Markov Decision Processes](2_Preliminaries_From_LLM_RL_to_Agentic_RL/2.1_Markov_Decision_Processes.md)
- [2.2 Environment State](2_Preliminaries_From_LLM_RL_to_Agentic_RL/2.2_Environment_State.md)
- [2.3 Action Space](2_Preliminaries_From_LLM_RL_to_Agentic_RL/2.3_Action_Space.md)
- [2.4 Transition Dynamics](2_Preliminaries_From_LLM_RL_to_Agentic_RL/2.4_Transition_Dynamics.md)
- [2.5 Reward Function](2_Preliminaries_From_LLM_RL_to_Agentic_RL/2.5_Reward_Function.md)
- [2.6 Learning Objective](2_Preliminaries_From_LLM_RL_to_Agentic_RL/2.6_Learning_Objective.md)
- [2.7 RL Algorithms](2_Preliminaries_From_LLM_RL_to_Agentic_RL/2.7_RL_Algorithms.md)

### Part II: Agentic Capabilities (Weeks 5-6)
#### [3. Model Capability Perspective](3_Agentic_RL_Capability_Perspective/)
- [3.1 Planning](3_Agentic_RL_Capability_Perspective/3.1_Planning.md)
- [3.2 Tool Using](3_Agentic_RL_Capability_Perspective/3.2_Tool_Using.md)
- [3.3 Memory](3_Agentic_RL_Capability_Perspective/3.3_Memory.md)
- [3.4 Self-Improvement](3_Agentic_RL_Capability_Perspective/3.4_Self_Improvement.md)
- [3.5 Reasoning](3_Agentic_RL_Capability_Perspective/3.5_Reasoning.md)
- [3.6 Perception](3_Agentic_RL_Capability_Perspective/3.6_Perception.md)
- [3.7 Others](3_Agentic_RL_Capability_Perspective/3.7_Others.md)

### Part III: Task Applications (Weeks 7-8)
#### [4. Task Perspective](4_Agentic_RL_Task_Perspective/)
- [4.1 Search & Research Agent](4_Agentic_RL_Task_Perspective/4.1_Search_Research_Agent/)
- [4.2 Code Agent](4_Agentic_RL_Task_Perspective/4.2_Code_Agent/)
- [4.3 Mathematical Agent](4_Agentic_RL_Task_Perspective/4.3_Mathematical_Agent/)
- [4.4 GUI Agent](4_Agentic_RL_Task_Perspective/4.4_GUI_Agent/)
- [4.5 Vision Agents](4_Agentic_RL_Task_Perspective/4.5_RL_in_Vision_Agents.md)
- [4.6 Embodied Agents](4_Agentic_RL_Task_Perspective/4.6_RL_in_Embodied_Agents.md)
- [4.7 Multi-Agent Systems](4_Agentic_RL_Task_Perspective/4.7_RL_in_Multi_Agent_Systems.md)
- [4.8 Other Tasks](4_Agentic_RL_Task_Perspective/4.8_Other_Tasks.md)

### Part IV: Systems & Future (Weeks 9-12)
#### [5. Environment and Frameworks](5_Environment_and_Frameworks/)
- [5.1 Environment Simulator](5_Environment_and_Frameworks/5.1_Environment_Simulator/)
- [5.2 RL Framework](5_Environment_and_Frameworks/5.2_RL_Framework.md)

#### [6. Open Challenges and Future Directions](6_Open_Challenges_and_Future_Directions/)
- [6.1 Trustworthiness](6_Open_Challenges_and_Future_Directions/6.1_Trustworthiness.md)
- [6.2 Scaling up Agentic Training](6_Open_Challenges_and_Future_Directions/6.2_Scaling_up_Agentic_Training.md)
- [6.3 Scaling up Agentic Environments](6_Open_Challenges_and_Future_Directions/6.3_Scaling_up_Agentic_Environments.md)

#### [7. Conclusion](7_Conclusion/)
- Synthesis and next steps

##  Learning Objectives

By completion, you will:
- **Formalize** agentic RL using MDP/POMDP mathematics
- **Implement** core capabilities: planning, memory, tool use, reasoning
- **Build** practical agents for code, math, GUI, and search tasks
- **Evaluate** using industry-standard benchmarks and environments
- **Deploy** secure, scalable agentic systems in production

##  Resources

### Primary References
- **Survey Paper**: [The Landscape of Agentic Reinforcement Learning for LLMs](https://arxiv.org/abs/2509.02547)
- **Paper Collection**: [Awesome AgenticLLM-RL Papers](https://github.com/xhyumiracle/Awesome-AgenticLLM-RL-Papers)
- **Institutions**: University of Oxford, Shanghai AI Laboratory, National University of Singapore

### Development Tools
- **Core Libraries**: `torch`, `transformers`, `gymnasium`, `numpy`
- **Evaluation**: Standard benchmarks (SWE-Bench, GAIA, WebArena)
- **Security**: Input validation, sandbox execution, permission systems

##  Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines, security requirements, and submission processes.

##  License

[MIT License](LICENSE) - Open source, industry-standard.

---

*This course bridges 500+ research papers into practical, secure, production-ready implementations. Start with the Quick Start demo above, then proceed to Module 1*
