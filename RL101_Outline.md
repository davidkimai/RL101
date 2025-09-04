```markdown
RL101/
  README.md
  1_Introduction/
    README.md
  2_Preliminaries_From_LLM_RL_to_Agentic_RL/
    README.md
    2.1_Markov_Decision_Processes.md
    2.2_Environment_State.md
    2.3_Action_Space.md
    2.4_Transition_Dynamics.md
    2.5_Reward_Function.md
    2.6_Learning_Objective.md
    2.7_RL_Algorithms.md
  3_Agentic_RL_Capability_Perspective/
    README.md
    3.1_Planning.md
    3.2_Tool_Using.md
    3.3_Memory.md
    3.4_Self_Improvement.md
    3.5_Reasoning.md
    3.6_Perception.md
    3.7_Others.md
  4_Agentic_RL_Task_Perspective/
    README.md
    4.1_Search_Research_Agent/
      README.md
      4.1.1_Open_Source_RL_Methods.md
      4.1.2_Closed_Source_RL_Methods.md
    4.2_Code_Agent/
      README.md
      4.2.1_RL_for_Code_Generation.md
      4.2.2_RL_for_Iterative_Code_Refinement.md
      4.2.3_RL_for_Automated_Software_Engineering.md
    4.3_Mathematical_Agent/
      README.md
      4.3.1_RL_for_Informal_Mathematical_Reasoning.md
      4.3.2_RL_for_Formal_Mathematical_Reasoning.md
    4.4_GUI_Agent/
      README.md
      4.4.1_RL_free_Methods.md
      4.4.2_RL_in_Static_GUI_Environments.md
      4.4.3_RL_in_Interactive_GUI_Environments.md
    4.5_RL_in_Vision_Agents.md
    4.6_RL_in_Embodied_Agents.md
    4.7_RL_in_Multi_Agent_Systems.md
    4.8_Other_Tasks.md
  5_Environment_and_Frameworks/
    README.md
    5.1_Environment_Simulator/
      README.md
      5.1.1_Web_Environments.md
      5.1.2_GUI_Environments.md
      5.1.3_Coding_SWE_Environments.md
      5.1.4_Domain_specific_Environments.md
      5.1.5_Simulated_and_Game_Environments.md
      5.1.6_General_Purpose_Environments.md
    5.2_RL_Framework.md
  6_Open_Challenges_and_Future_Directions/
    README.md
    6.1_Trustworthiness.md
    6.2_Scaling_up_Agentic_Training.md
    6.3_Scaling_up_Agentic_Environments.md
  7_Conclusion/
    README.md
```

```markdown
# RL101/README.md

# RL101: Reinforcement Learning 101 — Agentic RL for LLMs
This repository is a pragmatic, structured course based on the survey “The Landscape of Agentic Reinforcement Learning for LLMs: A Survey” (Sept 1, 2025). It reframes LLMs from passive generators to agents trained via RL in dynamic, partially observable settings with planning, tool use, memory, reasoning, self-improvement, and perception. The outline mirrors the survey’s table of contents and logic to make the repo a study-to-implementation bridge for RL practitioners.

## Checklist (What this course will do)
- Map survey sections to hands-on learning modules and lessons with concise goals.  
- Contrast PBRFT (preference-based RL fine-tuning) with Agentic RL through MDP/POMDP formalisms.  
- Organize agentic capabilities (planning, tools, memory, self-improvement, reasoning, perception) into teachable units.  
- Align tasks and environments with capability training and evaluation protocols.  
- Highlight open-source environments/frameworks and surface representative benchmarks from the survey.  
- Summarize open challenges to guide research directions and repo extensions.

## Resources
- Survey (primary reference): https://arxiv.org/abs/2509.02547  
- Awesome list (from survey): https://github.com/xhyumiracle/Awesome-AgenticLLM-RL-Papers

_Citation_: :contentReference[oaicite:0]{index=0}
```

```markdown
# 1_Introduction/README.md

# 1. Introduction
This module motivates the shift from LLM post-training as static sequence modeling to agentic RL, where models become decision-making policies in dynamic environments. It outlines how planning, tool invocation, memory, self-improvement, and perception emerge when RL is applied beyond single-turn alignment. The section frames the survey’s scope and clarifies what is in/out of scope for Agentic RL.

## Resources
- Primary Survey: https://arxiv.org/abs/2509.02547
```

```markdown
# 2_Preliminaries_From_LLM_RL_to_Agentic_RL/README.md

# 2. Preliminary: From LLM RL to Agentic RL
This module formalizes the paradigm shift from preference-based reinforcement fine-tuning (PBRFT) to Agentic RL. It introduces MDP/POMDP abstractions, differentiating degenerate single-step settings from long-horizon, partially observable environments where both text and structured actions matter. You will connect states, actions, transitions, rewards, objectives, and algorithms to agentic behavior.

## Resources
- Survey figures and tables contrasting PBRFT vs. Agentic RL: https://arxiv.org/abs/2509.02547
```

```markdown
# 2_Preliminaries_From_LLM_RL_to_Agentic_RL/2.1_Markov_Decision_Processes.md

# 2.1 Markov Decision Processes
Learn the formal MDP/POMDP setup for LLM training. PBRFT is cast as a single-step decision problem, while Agentic RL adopts a POMDP with observations, longer horizons, and discounting. The goal is to understand how formalism shapes what an LLM “does” during training and deployment.

## Resources
- Survey’s formal definitions and Table 1 (PBRFT vs. Agentic RL).
```

```markdown
# 2_Preliminaries_From_LLM_RL_to_Agentic_RL/2.2_Environment_State.md

# 2.2 Environment State
PBRFT assumes a one-shot prompt state; episodes end immediately after a response. Agentic RL models a changing world-state with partial observations, where the agent gathers feedback over time. This shift enables temporally extended behaviors and interaction with tools/users/environments.
```

```markdown
# 2_Preliminaries_From_LLM_RL_to_Agentic_RL/2.3_Action_Space.md

# 2.3 Action Space
Agentic RL splits actions into free-form text (A_text) and structured commands (A_action), often delimited with special tokens for tool and environment operations. Text communicates; structured actions query tools or modify state. Policies must govern both seamlessly within one learning loop.
```

```markdown
# 2_Preliminaries_From_LLM_RL_to_Agentic_RL/2.4_Transition_Dynamics.md

# 2.4 Transition Dynamics
PBRFT uses deterministic, terminal transitions; Agentic RL embraces stochastic, evolving dynamics dependent on actions. Text may leave state untouched, while structured actions update the environment or fetch new information. This sequential interplay is essential for exploration and adaptation.
```

```markdown
# 2_Preliminaries_From_LLM_RL_to_Agentic_RL/2.5_Reward_Function.md

# 2.5 Reward Function
PBRFT typically relies on final scalar signals from verifiers or preference models. Agentic RL mixes sparse task outcomes with step-level sub-rewards and learned signals (e.g., unit tests, symbolic checkers). Understanding reward granularity is key to stabilizing learning and avoiding reward hacking.
```

```markdown
# 2_Preliminaries_From_LLM_RL_to_Agentic_RL/2.6_Learning_Objective.md

# 2.6 Learning Objective
PBRFT optimizes expected response-level reward without discounting. Agentic RL maximizes discounted returns across trajectories, necessitating exploration and long-term credit assignment. This objective supports planning, reflection, and tool strategy that unfold over multiple steps.
```

```markdown
# 2_Preliminaries_From_LLM_RL_to_Agentic_RL/2.7_RL_Algorithms.md

# 2.7 RL Algorithms
Review core algorithm families used in PBRFT and Agentic RL: policy gradients (e.g., REINFORCE), PPO variants, DPO-style preference optimization, and GRPO/derivatives for group-relative credit signals. Each trades off stability, compute, and signal design—critical when integrating tools, memory, and multi-turn planning.

## Resources
- Algorithm taxonomy and comparison table (per survey).
```

```markdown
# 3_Agentic_RL_Capability_Perspective/README.md

# 3. Agentic RL: The Model Capability Perspective
This module organizes Agentic RL around capabilities: planning, tool using, memory, self-improvement, reasoning, and perception. Rather than treating these as fixed heuristics, RL turns them into trainable policies that interact and co-adapt. You’ll study how RL augments each capability and where research is headed.
```

```markdown
# 3_Agentic_RL_Capability_Perspective/3.1_Planning.md

# 3.1 Planning
Explore RL as (i) an external guide (e.g., reward/heuristic learning guiding search like MCTS) and (ii) an internal driver (directly optimizing the LLM’s planning policy via interaction). The synthesis aims to merge fast plan proposals with deliberate search, optimizing a meta-policy over “how to deliberate.” These approaches enable long-horizon coordination and robustness.

## Resources
- Survey discussion on RAP/LATS-style guided search and policy-centric planning.
```

```markdown
# 3_Agentic_RL_Capability_Perspective/3.2_Tool_Using.md

# 3.2 Tool Using
Trace the evolution from ReAct-style prompting and SFT datasets to tool-integrated RL (TIR). RL teaches when and how to invoke tools, compose them, and recover from errors—moving beyond imitation to outcome-driven control. A frontier is long-horizon TIR with better temporal credit assignment.

## Resources
- The survey’s TIR overview and referenced frameworks; commercial/open examples noted therein.
```

```markdown
# 3_Agentic_RL_Capability_Perspective/3.3_Memory.md

# 3.3 Memory
See how RL transforms memory from static stores into controllable subsystems: when to retrieve/write/forget. Approaches include RL-guided RAG, explicit token memories managed by policies, and latent memory tokens that persist across contexts. Structured memory (graphs/temporal) is emerging but under-explored for RL control.

## Resources
- Survey’s memory table and exemplars (e.g., Memory-R1, MemAgent, MEM1, MemoryLLM, M+).
```

```markdown
# 3_Agentic_RL_Capability_Perspective/3.4_Self_Improvement.md

# 3.4 Self-Improvement
Study a spectrum from verbal self-correction (prompted reflection) to RL-internalized self-correction and fully autonomous self-training loops. RL helps agents persistently learn from critique, verification, and execution feedback—covering self-play, curriculum generation, and collective bootstrapping.

## Resources
- Survey references: Reflexion, Self-Refine, CRITIC; KnowSelf/Reflection-DPO; R-Zero/Absolute Zero/TTRL; SiriuS/MALT.
```

```markdown
# 3_Agentic_RL_Capability_Perspective/3.5_Reasoning.md

# 3.5 Reasoning
Contrast fast (System 1-like) and slow (System 2-like) reasoning. RL and test-time scaling enhance deliberate multi-step reasoning with verification and planning, while hybrid methods seek to adaptively switch depth by task complexity. The challenge is training stable slow-thinking within agentic environments without overthinking.

## Resources
- Survey narrative on slow-reasoning RL, dataset construction, and hybrid strategies.
```

```markdown
# 3_Agentic_RL_Capability_Perspective/3.6_Perception.md

# 3.6 Perception
Multimodal agents progress from passive perception to active visual cognition by aligning vision–language–action with RL. Preference- and groupwise-RL approaches encourage step-wise, vision-grounded reasoning and task-specific partial rewards. Emerging curricula and gyms support scalable multimodal reinforcement finetuning.

## Resources
- Survey examples: Visual-RFT/Reason-RFT/Vision-R1 family and curriculum-based methods.
```

```markdown
# 3_Agentic_RL_Capability_Perspective/3.7_Others.md

# 3.7 Others
This section collects capability threads not covered elsewhere (e.g., niche modalities or interaction patterns). It highlights gaps and cross-cutting techniques to integrate less-explored skills into agentic policies. To be specified due to lack of detailed information in the survey’s subsection.

```

```markdown
# 4_Agentic_RL_Task_Perspective/README.md

# 4. Agentic RL: The Task Perspective
Shift from capabilities to application domains: search/research, code, math, GUI, vision, embodied, multi-agent, and more. For each, the survey outlines how RL interfaces with tools/environments, verification, and reward design. You will align capability training to task-specific evaluation.

```

```markdown
# 4_Agentic_RL_Task_Perspective/4.1_Search_Research_Agent/README.md

# 4.1 Search & Research Agent
Agents that browse, retrieve, and synthesize information benefit from RL to decide queries, follow links, and verify claims. The survey distinguishes open- vs. closed-source RL methods and emphasizes robust credit assignment across multi-step search trajectories. Evaluation often mixes end-task accuracy with process checks.

## Resources
- Survey mentions task suites like Browse-related environments; see also GAIA/BrowseComp noted in figures.
```

```markdown
# 4_Agentic_RL_Task_Perspective/4.1_Search_Research_Agent/4.1.1_Open_Source_RL_Methods.md

# 4.1.1 Open Source RL Methods
Covers open implementations where search strategies, rewards, and environments are transparent. Focus on how agents structure queries, plan hops, and ground synthesis with tool outputs. To be specified where the survey lists particular open baselines or frameworks without full details.

```

```markdown
# 4_Agentic_RL_Task_Perspective/4.1_Search_Research_Agent/4.1.2_Closed_Source_RL_Methods.md

# 4.1.2 Closed Source RL Methods
Summarizes proprietary systems (e.g., commercial research agents) emphasizing tool-integrated reasoning and self-evolving workflows. Discusses observed behaviors and evaluation patterns reported by the survey. Detailed implementation specifics are limited; some items: To be specified due to lack of public details.

```

```markdown
# 4_Agentic_RL_Task_Perspective/4.2_Code_Agent/README.md

# 4.2 Code Agent
Code agents use RL to decide when to draft, execute, test, and repair code via tool calls (interpreters, editors). Rewards may come from unit tests, compilers, or static analyzers, enabling dense and reliable signals. Subsections cover generation, refinement, and automated SWE.

## Resources
- Survey references: SWE-related benchmarks (e.g., SWE-Bench/Verified), editors/interpreters.
```

```markdown
# 4_Agentic_RL_Task_Perspective/4.2_Code_Agent/4.2.1_RL_for_Code_Generation.md

# 4.2.1 RL for Code Generation
Focus on initial program synthesis guided by execution/verifier rewards. RL encourages structured decomposition, test-first strategies, and library/tool selection. The design balances exploration (creative synthesis) and exploitation (passing tests).

```

```markdown
# 4_Agentic_RL_Task_Perspective/4.2_Code_Agent/4.2.2_RL_for_Iterative_Code_Refinement.md

# 4.2.2 RL for Iterative Code Refinement
Agents iteratively debug and patch using test failures and runtime logs as feedback. Policies learn when to run, inspect, or edit, and which parts to modify. RL reduces brittle prompt heuristics by grounding revisions in executable signals.

```

```markdown
# 4_Agentic_RL_Task_Perspective/4.2_Code_Agent/4.2.3_RL_for_Automated_Software_Engineering.md

# 4.2.3 RL for Automated Software Engineering
Beyond single files, agents manage repos, dependencies, and CI flows. Rewards combine end-to-end build/test success with intermediate lints and coverage. To be specified where the survey lists pipelines without granular detail.

```

```markdown
# 4_Agentic_RL_Task_Perspective/4.3_Mathematical_Agent/README.md

# 4.3 Mathematical Agent
Mathematics stresses deliberate reasoning with verifiable outcomes. RL leverages symbolic checkers and step rewards to stabilize long derivations. The survey separates informal (natural language math) and formal (proof assistants) settings.

```

```markdown
# 4_Agentic_RL_Task_Perspective/4.3_Mathematical_Agent/4.3.1_RL_for_Informal_Mathematical_Reasoning.md

# 4.3.1 RL for Informal Mathematical Reasoning
Agents generate human-readable chains of thought and use verification to prune errors. Group-relative and stepwise rewards improve sample efficiency. Curricula and test-time scaling complement RL to manage difficulty.

```

```markdown
# 4_Agentic_RL_Task_Perspective/4.3_Mathematical_Agent/4.3.2_RL_for_Formal_Mathematical_Reasoning.md

# 4.3.2 RL for Formal Mathematical Reasoning
Formal proofs in systems (e.g., proof assistants) yield precise rewards and constraints. RL policies learn search strategies, tactic selection, and backtracking. To be specified where the survey cites systems/tools without exhaustive details.

```

```markdown
# 4_Agentic_RL_Task_Perspective/4.4_GUI_Agent/README.md

# 4.4 GUI Agent
GUI agents act on apps/web UIs using perception, planning, and tool calls. The survey distinguishes RL-free pipelines, static GUI environments, and interactive environments where RL is critical for exploration and robustness. Rewards may include task completion, navigation correctness, and error recovery.

```

```markdown
# 4_Agentic_RL_Task_Perspective/4.4_GUI_Agent/4.4.1_RL_free_Methods.md

# 4.4.1 RL-free Methods
Covers heuristic and SFT-based GUI agents that rely on templates or demonstrations. These set baselines but struggle with novelty and error recovery. RL addresses these limitations in subsequent subsections.

```

```markdown
# 4_Agentic_RL_Task_Perspective/4.4_GUI_Agent/4.4.2_RL_in_Static_GUI_Environments.md

# 4.4.2 RL in Static GUI Environments
In constrained GUIs, RL learns click/scroll/type sequences with visual grounding. Rewards come from scripted goals or page states. The focus is stability, perception-action alignment, and sample efficiency.

```

```markdown
# 4_Agentic_RL_Task_Perspective/4.4_GUI_Agent/4.4.3_RL_in_Interactive_GUI_Environments.md

# 4.4.3 RL in Interactive GUI Environments
Interactive GUIs require credit assignment over longer trajectories and recovery from unexpected states. RL policies coordinate perception, tool queries (e.g., DOM access), and multi-turn strategies. Evaluation emphasizes generalization across tasks/sites.

```

```markdown
# 4_Agentic_RL_Task_Perspective/4.5_RL_in_Vision_Agents.md

# 4.5 RL in Vision Agents
Vision-language agents evolve from passive captioning to active, step-wise reasoning grounded in images/video. RL (e.g., preference/groupwise objectives) shapes spatial consistency, intermediate checks, and task-specific partial rewards. Curricula and gyms further stabilize training.

## Resources
- Survey’s visual RL families (e.g., Visual-RFT/Reason-RFT/Vision-R1), Spatial/partial reward schemes.
```

```markdown
# 4_Agentic_RL_Task_Perspective/4.6_RL_in_Embodied_Agents.md

# 4.6 RL in Embodied Agents
Embodied agents act in simulators or the real world, integrating perception, planning, and control. RL handles exploration, safety, and long-horizon objectives, often with dense auxiliary signals. Tool use (e.g., planners, map builders) is coordinated with language policies.

## Resources
- Survey references to embodied environments and evaluation protocols.
```

```markdown
# 4_Agentic_RL_Task_Perspective/4.7_RL_in_Multi_Agent_Systems.md

# 4.7 RL in Multi-Agent Systems
Agents coordinate, compete, or negotiate in multi-agent settings. RL shapes communication, role assignment, and market/team dynamics. Rewards may be individual or shared, with attention to stability and emergent behaviors.

```

```markdown
# 4_Agentic_RL_Task_Perspective/4.8_Other_Tasks.md

# 4.8 Other Tasks
This catch-all includes domains not covered above (e.g., science discovery, data management, terminal use). The survey notes representative tasks and environments; details vary widely. Some items: To be specified due to limited detail.

```

```markdown
# 5_Environment_and_Frameworks/README.md

# 5. Environment and Frameworks
This module catalogs simulators, benchmarks, and RL frameworks central to Agentic RL experimentation. It groups environments by domain (web/GUI/coding/domain-specific/games/general-purpose) and summarizes RL framework trends. Use it to select training grounds and toolchains for course projects.

## Resources
- Survey’s consolidated compendium of environments and frameworks.
```

```markdown
# 5_Environment_and_Frameworks/5.1_Environment_Simulator/README.md

# 5.1 Environment Simulator
Environments define state, action, and feedback loops where agentic policies are trained and evaluated. The survey organizes them by domain and interaction style, noting reward availability and observability. Choice of environment shapes reward design and algorithm selection.

```

```markdown
# 5_Environment_and_Frameworks/5.1_Environment_Simulator/5.1.1_Web_Environments.md

# 5.1.1 Web Environments
Web tasks involve navigation, retrieval, and multi-hop synthesis. RL agents coordinate browsing actions with verification to mitigate hallucinations. To be specified with concrete environment links where the survey lists names but no URLs.

```

```markdown
# 5_Environment_and_Frameworks/5.1_Environment_Simulator/5.1.2_GUI_Environments.md

# 5.1.2 GUI Environments
GUI simulators support pointing/typing/clicking with visual observations. Rewards range from task completion to structured sub-goals. This enables curriculum design for perception-to-action skills.

```

```markdown
# 5_Environment_and_Frameworks/5.1_Environment_Simulator/5.1.3_Coding_SWE_Environments.md

# 5.1.3 Coding & Software Engineering Environments
Coding environments provide interpreters, compilers, and unit tests for dense, reliable signals. RL agents iterate code-edit–run loops and manage project structure. Benchmarks like SWE-focused suites appear in the survey.

## Resources
- Survey mentions: SWE-Bench/Verified and similar; concrete links: To be specified if not provided.
```

```markdown
# 5_Environment_and_Frameworks/5.1_Environment_Simulator/5.1.4_Domain_specific_Environments.md

# 5.1.4 Domain-specific Environments
Includes scientific workflows, data management, and specialized APIs. Rewards may be partially learned or proxy-based. Emphasis on tool integration and safety constraints.

```

```markdown
# 5_Environment_and_Frameworks/5.1_Environment_Simulator/5.1.5_Simulated_and_Game_Environments.md

# 5.1.5 Simulated & Game Environments
Games/text-worlds enable long-horizon exploration with controllable difficulty and verifiers. RL helps distill reusable skills and evaluate planning/memory. Examples cited in the survey include text games like ALFWorld/TextCraft.

```

```markdown
# 5_Environment_and_Frameworks/5.1_Environment_Simulator/5.1.6_General_Purpose_Environments.md

# 5.1.6 General-Purpose Environments
General platforms aggregate heterogeneous tasks (web, tools, math, code). They support mixed reward schemes and systematic evaluation of integrated capabilities. Selection depends on your target capability mix.

```

```markdown
# 5_Environment_and_Frameworks/5.2_RL_Framework.md

# 5.2 RL Framework
Surveys RL frameworks and training stacks enabling PPO/GRPO/DPO-style optimization for agentic settings. Key considerations include memory and tool APIs, reward model orchestration, logging/rollout tooling, and safety controls. Choose frameworks that simplify long-horizon credit assignment and tool integration.

```

```markdown
# 6_Open_Challenges_and_Future_Directions/README.md

# 6. Open Challenges and Future Directions
Agentic RL faces challenges in trustworthiness, scaling training, and scaling environments. The survey outlines failure modes, alignment issues, and the need for richer simulators and benchmarks. This module frames research projects to push the field forward.

```

```markdown
# 6_Open_Challenges_and_Future_Directions/6.1_Trustworthiness.md

# 6.1 Trustworthiness
Covers safety, reliability, and evaluation integrity. RL must avoid reward hacking and maintain verifiable behaviors under distribution shifts. Tool use and memory introduce new attack surfaces; defenses and audits are needed.

```

```markdown
# 6_Open_Challenges_and_Future_Directions/6.2_Scaling_up_Agentic_Training.md

# 6.2 Scaling up Agentic Training
Discusses compute-efficient rollouts, stable objectives (e.g., group-relative signals), and hybrid supervision. Curriculum design, process rewards, and buffer strategies help with sample efficiency. Practical recipes focus on training stability and cost.

```

```markdown
# 6_Open_Challenges_and_Future_Directions/6.3_Scaling_up_Agentic_Environments.md

# 6.3 Scaling up Agentic Environments
Richer, dynamic, partially observable environments are needed for robust generalization. Standardization and shared benchmarks will improve comparability. Tool ecosystems should support safe, reproducible, multi-modal interaction.

```

```markdown
# 7_Conclusion/README.md

# 7. Conclusion
The survey consolidates theory, algorithms, capabilities, tasks, and infrastructure into a coherent picture of Agentic RL. This course outline turns that picture into a roadmap for study and practical exploration. It closes by pointing to open challenges that learners can address through projects and contributions.

## Resources
- Primary Survey: https://arxiv.org/abs/2509.02547
- Paper List: https://github.com/xhyumiracle/Awesome-AgenticLLM-RL-Papers
```
