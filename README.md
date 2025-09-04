# RL101: Reinforcement Learning 101

This repository provides a comprehensive, hands-on course transforming LLMs from passive text generators into autonomous agents through reinforcement learning. Based on "The Landscape of Agentic Reinforcement Learning for LLMs" survey, it bridges theory to implementation across planning, tool use, memory, reasoning, self-improvement, and perception capabilities.

## Course Overview

**Duration**: 12-16 weeks  
**Prerequisites**: Basic ML knowledge, Python programming  
**Focus**: From MDP/POMDP foundations → Agentic capabilities → Real-world applications

This course reframes Large Language Models as **decision-making policies** in dynamic, partially observable environments rather than static sequence predictors. You'll learn how RL enables sophisticated behaviors like multi-step planning, tool coordination, persistent memory, and self-improvement.

## What You'll Build

- **Foundational Understanding**: MDP/POMDP formalizations contrasting PBRFT vs. Agentic RL
- **Capability Modules**: RL-driven planning, tool integration, memory systems, reasoning loops  
- **Task-Specific Agents**: Search/research, coding, mathematical, GUI, vision, and embodied agents
- **Production Systems**: Complete agentic architectures with evaluation frameworks

## Repository Structure

### Core Modules

| Module | Focus | Duration |
|--------|-------|----------|
| [1. Introduction](1_Introduction/) | Paradigm shift from static LLMs to agentic policies | Week 1 |
| [2. Preliminaries](2_Preliminaries_From_LLM_RL_to_Agentic_RL/) | MDP/POMDP foundations, algorithms, objectives | Weeks 2-3 |
| [3. Agentic Capabilities](3_Agentic_RL_Capability_Perspective/) | Planning, tools, memory, reasoning, perception | Weeks 4-7 |
| [4. Task Applications](4_Agentic_RL_Task_Perspective/) | Domain-specific implementations and evaluations | Weeks 8-12 |
| [5. Infrastructure](5_Environment_and_Frameworks/) | Environments, frameworks, benchmarks | Week 13 |
| [6. Research Frontiers](6_Open_Challenges_and_Future_Directions/) | Open challenges, scaling, trustworthiness | Week 14 |
| [7. Integration](7_Conclusion/) | Course synthesis and future directions | Week 15-16 |

### Learning Philosophy

**Theory ↔ Practice**: Every concept includes formal definitions, algorithmic details, and hands-on implementation  
**Progressive Complexity**: From single-step decisions → multi-horizon coordination → emergent behaviors  
**Real-World Focus**: Production-ready patterns, evaluation protocols, and safety considerations

## Quick Start

```bash
# Clone repository
git clone https://github.com/[username]/RL101.git
cd RL101

# Install dependencies
pip install -r requirements.txt

# Start with foundations
cd 1_Introduction
jupyter notebook introduction.ipynb
```

## Key Resources

- **Primary Survey**: [The Landscape of Agentic Reinforcement Learning for LLMs](https://arxiv.org/abs/2509.02547)
- **Paper Collection**: [Awesome Agentic LLM RL Papers](https://github.com/xhyumiracle/Awesome-AgenticLLM-RL-Papers)  
- **Course Discussions**: [GitHub Discussions](../../discussions)
- **Implementation Examples**: See `examples/` in each module

## Scope & Boundaries  

### ✅ Primary Focus
- How RL empowers LLM-based agents in dynamic environments
- MDP/POMDP formalizations for multi-step decision making  
- Capability development through environmental interaction
- Task-specific agent architectures and evaluation

### ❌ Out of Scope  
- Traditional RLHF for human preference alignment
- Pure LLM performance optimization on static benchmarks
- Non-LLM based RL algorithms (though comparisons included)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Adding new modules or improving existing content
- Sharing implementations and benchmarks  
- Reporting issues and suggesting enhancements
- Research paper integrations and updates

## License

This course is released under the [MIT License](LICENSE). Feel free to use, modify, and distribute for educational and research purposes.

## Citation

If you use this course in your research or teaching, please cite:

```bibtex
@misc{rl101-agentic,
  title={RL101: Reinforcement Learning 101 — Agentic RL for LLMs},
  author={[Course Authors]},
  year={2024},
  url={https://github.com/[username]/RL101}
}
```

## Acknowledgments

This course is built upon the comprehensive survey by researchers from multiple institutions. Special thanks to the authors of "The Landscape of Agentic Reinforcement Learning for LLMs" and the broader community advancing this field.

---

**Ready to transform static LLMs into autonomous agents?** Start with [Module 1: Introduction](1_Introduction/) 🚀
