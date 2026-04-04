# Long-Horizon Agent Benchmark Survey

**Date**: 2026-04-02
**Purpose**: Comprehensive survey of long-horizon agent benchmarks and evaluation methodologies, with focus on benchmarks that test consequential scored actions, dynamic environments, adaptation, and long-term coherence. Informs the design of a next-generation benchmark.

---

## 1. Taxonomy of Benchmarks

We organize the landscape along two axes: **what the agent does** (action type) and **what the environment does** (dynamism).

### Axis 1: Agent Action Type

| Category | Description | Examples |
|----------|-------------|----------|
| **QA/Retrieval** | Agent answers questions from memory or context | LoCoMo, LongMemEval, InfiniteBench |
| **Tool Calling** | Agent invokes APIs/tools to gather info or act | GAIA, tau-bench, AppWorld |
| **Code/Terminal** | Agent writes and executes code | SWE-bench, OSWorld, TheAgentCompany |
| **Navigation/GUI** | Agent clicks, types, scrolls in a UI | WebArena, VisualWebArena, AndroidWorld |
| **Game Actions** | Agent takes discrete game moves | BALROG, Crafter, NetHack, HeroBench |
| **Social/Chat** | Agent converses with simulated humans/agents | Sotopia, MultiAgentBench, CAMEL |
| **Embodied** | Agent moves/manipulates in 3D/grid space | PARTNR, MineDojo, MiniGrid |

### Axis 2: Environment Dynamism

| Level | Description | Examples |
|-------|-------------|----------|
| **Static** | Environment doesn't change; same for all runs | SWE-bench, GAIA, LoCoMo |
| **Procedurally Generated** | Different each run but static within episode | BALROG, Crafter, ProcGen |
| **Reactive** | State changes in response to agent actions | WebArena, AppWorld, TheAgentCompany |
| **Asynchronous/Dynamic** | Environment changes *independently* of agent | Gaia2/ARE, tau-bench, MEMTRACK |
| **Adversarial/Adaptive** | Environment actively adapts to agent behavior | AbideGym, Sotopia (with strategic NPCs) |

### Axis 3: Horizon Length

| Tier | Scale | Examples |
|------|-------|----------|
| **Short** (<100 steps, <10K tokens) | Single API call or few-shot | GAIA L1, BabyAI |
| **Medium** (100-1K steps, 10K-100K tokens) | Multi-step workflows | WebArena, SWE-bench, tau-bench |
| **Long** (1K-10K steps, 100K-1M tokens) | Extended sessions | UltraHorizon (200K+ tokens), TRIP-Bench (200K tokens) |
| **Ultra-Long** (10K+ steps, 1M+ tokens) | Days/weeks of interaction | Generative Agents (2 simulated days), LongMemEval (1.5M tokens) |

---

## 2. Detailed Benchmark Profiles

### 2.1 Agent/LLM Long-Horizon Benchmarks

---

#### GAIA (General AI Assistants)
- **Paper**: Mialon et al., 2023. "GAIA: A Benchmark for General AI Assistants." arXiv:2311.12983. (Meta AI)
- **Core capability**: Multi-step web research, reasoning, tool use
- **Task structure**: 466 questions at 3 difficulty levels; agent uses web browsing, code execution, file reading to produce a single final answer
- **Action space**: Tool calls (web search, file reader, calculator, code interpreter); output is a short text answer
- **Horizon**: Short-to-medium; most tasks require 5-20 tool calls
- **Scoring**: Deterministic exact-match against gold answers; automated
- **Consequential actions?**: No -- agent gathers information then outputs an answer
- **Dynamic environment?**: No -- web snapshots are static for reproducibility
- **Multi-agent?**: No
- **Key insight**: Tests breadth of tool use but not sustained interaction. Best systems now at ~75%.

---

#### WebArena / VisualWebArena
- **Paper**: Zhou et al., 2023. "WebArena: A Realistic Web Environment for Building Autonomous Agents." ICLR 2024. arXiv:2307.13854. / Koh et al., 2024. VisualWebArena. ACL 2024. arXiv:2401.13649.
- **Core capability**: Autonomous web navigation and task completion
- **Task structure**: Self-hosted web environments (Shopping, Reddit, CMS, GitLab, Maps). Agent receives natural language instruction and must complete tasks by interacting with real web UIs.
- **Action space**: GUI actions -- click(element_id), type(text), scroll, go_back, navigate(url). VisualWebArena adds Set-of-Marks visual grounding where elements are labeled with bounding boxes and IDs.
- **Horizon**: Medium; ~10-30 actions per task. 812 tasks (WebArena), 910 tasks (VisualWebArena)
- **Scoring**: Deterministic programmatic evaluation comparing final page state, URL, or extracted content against gold standard
- **Consequential actions?**: Yes -- agent modifies web application state (posts, orders, edits)
- **Dynamic environment?**: Reactive -- state changes on agent actions but doesn't change independently
- **Multi-agent?**: No
- **Key insight**: Realistic web interaction but short horizon. Best agents ~60-70% on VisualWebArena vs 89% human.

---

#### OSWorld / AndroidWorld
- **Paper**: Xie et al., 2024. "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments." NeurIPS 2024. arXiv:2404.07972.
- **Core capability**: Full desktop/mobile OS task completion
- **Task structure**: 369 tasks across Ubuntu/Windows/macOS. Agent sees screenshots, uses keyboard/mouse to complete real software tasks (file management, web browsing, coding, office apps, multimedia).
- **Action space**: Raw keyboard and mouse actions (click coordinates, type text, keyboard shortcuts). AndroidWorld: 116 tasks on stock Android apps.
- **Horizon**: Medium; varies from 5 to 100+ steps per task
- **Scoring**: Execution-based evaluation via programmatic state checks (file contents, application state, UI state)
- **Consequential actions?**: Yes -- full OS-level state modification
- **Dynamic environment?**: Reactive only
- **Multi-agent?**: No
- **Key insight**: Closest to general computer use. Recent SOTA ~70% (100 steps) vs 72% human. OSWorld-Gold measures efficiency (not just success).

---

#### SWE-bench / SWE-bench Verified
- **Paper**: Jimenez et al., 2024. "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" ICLR 2024. arXiv:2310.06770.
- **Core capability**: Autonomous software bug fixing
- **Task structure**: Real GitHub issues from 12 Python repos. Agent receives issue description + full repo, must produce a git diff that passes hidden test suite. SWE-bench Verified: 500 human-validated instances.
- **Action space**: Code editing (produce git diff), shell commands, file navigation. Frameworks like OpenHands provide bash + code editor + browser.
- **Horizon**: Medium; typical solutions require reading 10-50 files, editing 1-5 files
- **Scoring**: Fully deterministic -- hidden unit tests (FAIL_TO_PASS) must pass. Binary pass@1.
- **Consequential actions?**: Yes -- code changes that must compile and pass tests
- **Dynamic environment?**: Static -- repo snapshot is frozen
- **Multi-agent?**: No (though SWE-bench Pro targets longer tasks)
- **Key insight**: Gold standard for deterministic evaluation. ~75% SOTA on Verified. But each task is independent -- no long-term state accumulation.

---

#### AgentBench
- **Paper**: Liu et al., 2023. "AgentBench: Evaluating LLMs as Agents." ICLR 2024. arXiv:2308.03688.
- **Core capability**: Multi-domain agent evaluation across 8 environments
- **Task structure**: 8 environments in 3 categories: Code (OS, DB, KG), Game (digital card game, lateral thinking), Web (web shopping, web browsing, mind2web)
- **Action space**: Domain-specific -- shell commands, SQL queries, SPARQL, game moves, web clicks, API calls
- **Horizon**: Short-to-medium; varies per environment
- **Scoring**: Environment-specific deterministic metrics (task success, score achieved)
- **Consequential actions?**: Yes in game/OS environments; information gathering in others
- **Dynamic environment?**: Reactive in games; static in others
- **Multi-agent?**: No
- **Key insight**: Breadth over depth. Identified poor long-term reasoning and decision-making as main failure modes.

---

#### tau-bench (and tau2-bench, tau3-bench)
- **Paper**: Sierra Research, 2024. "tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains." arXiv:2406.12045.
- **Core capability**: Customer service task completion with dynamic human interaction
- **Task structure**: Agent receives policy guidelines + API tools. Must converse with LLM-simulated user to gather info and complete transactions (booking changes, returns, etc.) in retail and airline domains.
- **Action space**: Tool/API calls (search_orders, modify_booking, issue_refund) + natural language responses to user. Actions modify a database.
- **Horizon**: Medium; 5-15 conversational turns with multiple tool calls per turn
- **Scoring**: Deterministic database state comparison -- final DB state vs annotated goal state. Novel pass^k metric measures reliability across k repeated trials.
- **Consequential actions?**: Yes -- database modifications (order changes, refunds, bookings)
- **Dynamic environment?**: Yes -- LLM-simulated user introduces ambiguity, changes requests mid-conversation
- **Multi-agent?**: Two-party (agent + simulated user); tau3-bench adds voice
- **Key insight**: pass^k metric is valuable -- even GPT-4o succeeds <50% and pass^8 <25% in retail. Tests reliability, not just peak performance. tau-Knowledge extends to large document collections.

---

#### MLAgentBench / MLE-bench
- **Paper**: Huang et al., 2023. "MLAgentBench." arXiv:2310.03302. / OpenAI, 2024. "MLE-bench." arXiv:2410.07095.
- **Core capability**: ML experimentation and engineering
- **Task structure**: MLAgentBench: 13 tasks (improve CIFAR-10 model, BabyLM, etc.). MLE-bench: 75 Kaggle competition tasks.
- **Action space**: Read/write files, execute Python code, inspect outputs
- **Horizon**: Medium-to-long; multiple experiment iterations
- **Scoring**: Deterministic -- Kaggle-style leaderboard metrics (accuracy, RMSE, etc.) with medal thresholds
- **Consequential actions?**: Yes -- code that trains models, modifies datasets
- **Dynamic environment?**: Static per task
- **Multi-agent?**: No
- **Key insight**: Tests iterative improvement cycle. Best: o1-preview 16.9% bronze-level pass@1, 34.1% pass@8. Performance varies wildly by task novelty.

---

#### Generative Agents (Smallville)
- **Paper**: Park et al., 2023. "Generative Agents: Interactive Simulacra of Human Behavior." UIST 2023. arXiv:2304.03442.
- **Core capability**: Believable autonomous social behavior over extended time
- **Task structure**: 25 agents in sandbox town over 2 simulated days. Each agent has persona, relationships, daily routines. Agents autonomously plan, converse, form relationships, coordinate events.
- **Action space**: Natural language actions (movement, conversation, object interaction) in tile-based environment
- **Horizon**: Ultra-long; 2 full simulated days with continuous autonomous behavior
- **Scoring**: Human evaluation (100 participants, within-subjects study); interview-based probes of memory/reflection/planning
- **Consequential actions?**: Yes -- social actions affect relationship state, information propagation
- **Dynamic environment?**: Yes -- other agents create emergent events; information diffuses through social network
- **Multi-agent?**: Yes -- 25 concurrent agents
- **Key insight**: Demonstrates emergent social dynamics but evaluation is expensive (human judges). Architecture: memory stream + reflection + planning. No deterministic scoring.

---

#### BALROG
- **Paper**: Paglieri et al., 2024. "BALROG: Benchmarking Agentic LLM and VLM Reasoning On Games." ICLR 2025. arXiv:2411.13543.
- **Core capability**: Sequential decision-making in game environments
- **Task structure**: 6 game environments of increasing complexity: BabyAI (grid nav), TextWorld (text adventure), Crafter (2D survival), Baba Is AI (puzzle), MiniHack (dungeon), NetHack (full roguelike)
- **Action space**: Per-game discrete actions. BabyAI: {turn_left, turn_right, forward, pick_up, drop, toggle}. Crafter: {move, craft, place, attack}. NetHack: keyboard commands (~70 actions). All output as natural language strings.
- **Horizon**: Spans orders of magnitude. BabyAI: ~10 steps. TextWorld/BabaIsAI: ~100 steps. Crafter/MiniHack: ~1000 steps. NetHack: 10,000-100,000 steps.
- **Scoring**: Deterministic. Binary completion (0/100) for BabyAI, BabaIsAI, MiniHack. Continuous progression (0-100) for TextWorld, Crafter, NetHack. Custom data-informed progression metric for NetHack.
- **Consequential actions?**: Yes -- every action changes game state
- **Dynamic environment?**: Yes -- procedurally generated (different each run), enemies act autonomously, resources deplete
- **Multi-agent?**: No (single agent vs environment)
- **Key insight**: Best model (DeepSeek-R1) only 34.9% average progression. VLMs perform *worse* with images than text-only. Models can answer game knowledge questions but fail to apply knowledge in practice. Huge gap between knowing and doing.

---

#### Voyager (Minecraft)
- **Paper**: Wang et al., 2023. "Voyager: An Open-Ended Embodied Agent with Large Language Models." arXiv:2305.16291.
- **Core capability**: Open-ended exploration, skill acquisition, lifelong learning
- **Task structure**: Agent explores Minecraft world autonomously, discovers resources, crafts tools, progresses through tech tree. No fixed end-state.
- **Action space**: JavaScript code controlling Minecraft bot (movement, mining, crafting, combat via Mineflayer API). Agent writes code snippets that become reusable "skills" stored in a library.
- **Horizon**: Ultra-long; continuous exploration measured in items discovered and milestones reached
- **Scoring**: Milestone-based: unique items obtained, tech tree levels unlocked, distance traveled. Deterministic counts.
- **Consequential actions?**: Yes -- all actions modify world state, skill library grows
- **Dynamic environment?**: Yes -- Minecraft world is procedurally generated, mobs spawn, day/night cycle
- **Multi-agent?**: No
- **Key insight**: Skill library enables compositional learning and transfer to new worlds. 3.3x more unique items, 15.3x faster milestone unlocking vs baselines. But no adversarial evaluation -- agent never faces regime changes.

---

#### SPRING
- **Paper**: Wu et al., 2023. "SPRING: Studying the Paper and Reasoning to Play Games." NeurIPS 2023. arXiv:2305.15486.
- **Core capability**: Knowledge transfer from documentation to game play
- **Task structure**: LLM reads the Crafter academic paper (LaTeX source), builds a DAG of game-related questions, then uses this knowledge to play the game.
- **Action space**: Crafter game actions (movement, crafting, combat) selected via DAG traversal + LLM reasoning
- **Horizon**: Medium; standard Crafter episode length (~1000 steps)
- **Scoring**: Crafter achievement scores (22 achievements, geometric mean)
- **Consequential actions?**: Yes -- game actions
- **Dynamic environment?**: Yes (Crafter is procedurally generated)
- **Multi-agent?**: No
- **Key insight**: GPT-4 + SPRING outperforms all RL baselines trained for 1M steps, with zero training. Demonstrates LLMs can leverage documentation but doesn't test adaptation to undocumented changes.

---

### 2.2 Memory-Specific Benchmarks

---

#### LoCoMo
- **Paper**: Maharana et al., 2024. "Evaluating Very Long-Term Conversational Memory of LLM Agents." ACL 2024. arXiv:2402.17753.
- **Core capability**: Long-term conversational memory
- **Task structure**: Conversations of ~300 turns across up to 35 sessions (~9K-26K tokens). Tasks: QA, event summarization, dialogue generation.
- **Action space**: Text answers to questions (no tool use, no environment interaction)
- **Horizon**: Medium by today's standards; 26K tokens fits in modern context windows
- **Scoring**: Automated metrics (F1, ROUGE, BERTScore) + human evaluation
- **Consequential actions?**: No -- purely question answering
- **Dynamic?**: No -- conversations are pre-recorded
- **Key insight**: Foundational benchmark but increasingly saturated. Conversations short enough for modern 128K+ context windows.

---

#### LongMemEval
- **Paper**: Wu et al., 2025. "LongMemEval: Benchmarking Chat Assistants on Long-Term Memory." ICLR 2025. arXiv:2410.10813.
- **Core capability**: Five core memory abilities across long chat histories
- **Task structure**: 500 manually created questions testing information extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention. Questions require recalling info from task-oriented dialogues (115K-1.5M tokens).
- **Action space**: Text answers only
- **Horizon**: Long (up to 1.5M tokens of history)
- **Scoring**: LLM-based evaluation (GPT-4 judge) + human validation
- **Consequential actions?**: No
- **Dynamic?**: No
- **Key insight**: Tests genuine long-horizon memory beyond context window capacity. But still tests retrieval of stated facts, not synthesis of patterns across episodes.

---

#### StreamBench
- **Paper**: Wu et al., 2024. "StreamBench: Towards Benchmarking Continuous Improvement of Language Agents." NeurIPS 2024 D&B. arXiv:2406.08747.
- **Core capability**: Continuous online improvement from streaming feedback
- **Task structure**: Agent receives sequential tasks (text-to-SQL, Python coding, tool use, medical diagnosis, QA) with feedback after each attempt. Must improve over time.
- **Action space**: Task-specific outputs (SQL queries, Python code, tool calls, diagnoses)
- **Horizon**: Long; continuous stream of tasks with accumulating feedback
- **Scoring**: Task-specific deterministic metrics (execution accuracy for SQL, pass rates for code)
- **Consequential actions?**: Partially -- outputs are evaluated but don't modify a persistent environment
- **Dynamic?**: Yes -- feedback stream creates evolving context; implicit distribution shift across task stream
- **Key insight**: Only benchmark explicitly testing *improvement over time*. Closest to testing adaptation, but tasks are independent (no shared world state).

---

#### MemoryBench
- **Paper**: Ao et al., 2025. "MemoryBench: A Benchmark for Memory and Continual Learning in LLM Systems." arXiv:2510.17281.
- **Core capability**: Learning from accumulated user feedback
- **Task structure**: ~20,000 task examples across 11 datasets, multi-domain (open, legal, academic), multilingual. Tests both declarative memory (semantic + episodic) and procedural memory via user feedback simulation.
- **Action space**: Task outputs + feedback integration (explicit text critiques, action signals like/dislike/copy, implicit behavioral signals)
- **Horizon**: Sequential batches with on-policy feedback collection
- **Scoring**: Hybrid -- objective metrics (F1, accuracy) for verifiable tasks + LLM-as-judge (1-10 scale) for subjective tasks
- **Consequential actions?**: Partially -- agent's memory updates affect future performance
- **Dynamic?**: Yes -- user feedback creates evolving learning signal; performance fluctuates across training
- **Key insight**: Explicitly models procedural learning (how to do tasks better) not just declarative memory (what happened). Closest to testing adaptation in memory context.

---

#### InfiniteBench
- **Paper**: Chen et al., 2024. "InfiniteBench: Extending Long Context Evaluation Beyond 100K Tokens." ACL 2024. arXiv:2402.13718.
- **Core capability**: Processing 100K+ token contexts
- **Task structure**: 12 tasks across different aspects of long-context understanding
- **Action space**: Text answers
- **Horizon**: 100K+ tokens of input context
- **Scoring**: Task-specific automated metrics
- **Consequential actions?**: No
- **Dynamic?**: No
- **Key insight**: Tests raw context window capacity, not memory systems or agent behavior. Becoming less relevant as context windows expand to 1M+.

---

#### AMA-Bench
- **Paper**: 2026. "AMA-Bench: Evaluating Long-Horizon Memory for Agentic Applications." arXiv:2602.22769.
- **Core capability**: Memory over real agentic trajectories (not just conversations)
- **Task structure**: Two components: (1) Real trajectories from 6 domains (ALFWorld, BALROG/2048, WebArena, SWE-bench, GAIA, Spider) with expert-curated QA. (2) Synthetic trajectories (BabyAI, TextWorld) scaling to 128K tokens with rule-based QA. 12 QA pairs per trajectory testing Recall, Causal Inference, State Updating, State Abstraction.
- **Action space**: Memory queries -- agent must answer questions about past trajectories (doesn't take new actions)
- **Horizon**: Real: 6K-997K tokens (avg 57.5K). Synthetic: 8K-128K tokens.
- **Scoring**: LLM-as-judge (Qwen3-32B), accuracy + F1
- **Consequential actions?**: No -- evaluates memory *about* past actions, not taking new ones
- **Dynamic?**: No -- trajectories are pre-recorded
- **Key insight**: First benchmark testing memory of agentic (not conversational) trajectories. Found that existing memory systems fail because they lack causality graphs and are constrained by similarity-based retrieval. Best system: 57.2% accuracy. Frontier models (GPT-5.2): 72.3%.

---

#### MEMTRACK
- **Paper**: Patronus AI, 2025. "MEMTRACK: Evaluating Long-Term Memory and State Tracking in Multi-Platform Dynamic Agent Environments." NeurIPS 2025 Workshop. arXiv:2510.01353.
- **Core capability**: Cross-platform state tracking in noisy organizational workflows
- **Task structure**: Software development scenarios with interleaved timelines across Slack, Linear, and Git. Contains conflicting, cross-referencing, noisy information. Agent must track state, resolve conflicts, and complete related tasks using platform tools.
- **Action space**: Tool calls to Slack/Linear/Git servers for information gathering and task completion
- **Horizon**: Long; chronologically interleaved cross-platform timelines
- **Scoring**: Three metrics: Correctness, Efficiency, Redundancy (captures wasted effort). Mix of deterministic checks and evaluation.
- **Consequential actions?**: Partially -- tool calls but primarily information gathering and state tracking
- **Dynamic?**: Yes -- conflicting and evolving information across platforms
- **Multi-agent?**: No
- **Key insight**: Tests exactly what production agent memory needs: cross-platform conflict resolution. Zep and Mem0 showed no significant improvement over baselines. GPT-5 only 60% correctness.

---

#### AMemGym
- **Paper**: 2026. "AMemGym: Interactive Memory Benchmarking for Assistants in Long-Horizon Conversations." ICLR 2026. arXiv:2603.01966.
- **Core capability**: On-policy interactive memory evaluation
- **Task structure**: Schema-based approach generates structured user profiles, state-dependent questions, and state evolution trajectories. Decomposes memory failures into write/read/utilization stages.
- **Action space**: Conversational responses using memory-retrieved information
- **Horizon**: Long; multi-session conversations with evolving user state
- **Scoring**: Automated stage-wise evaluation (write accuracy, read accuracy, utilization accuracy)
- **Consequential actions?**: Partially -- memory write decisions affect future retrieval
- **Dynamic?**: Yes -- user state evolves according to pre-defined trajectories
- **Key insight**: First on-policy interactive memory benchmark. Revealed "Reuse Bias" in static evaluations -- models that look good on pre-recorded data fail on live interaction.

---

### 2.3 Dynamic/Adaptive Environment Benchmarks

---

#### PARTNR
- **Paper**: Meta AI, 2024. "PARTNR: A Benchmark for Planning and Reasoning in Embodied Multi-agent Tasks." ICLR 2025. arXiv:2411.00081.
- **Core capability**: Human-robot collaboration in household tasks
- **Task structure**: 100,000 natural language tasks in 60 simulated houses with 5,819 objects. Four task types: constraint-free, spatial (placement constraints), temporal (ordered execution), heterogeneous (role-restricted actions).
- **Action space**: Embodied actions in Habitat simulator -- navigate, pick, place, open/close, interact with objects
- **Horizon**: Medium; household tasks require 20-100+ actions
- **Scoring**: Semi-automated with simulation-in-the-loop verification. Task success + step efficiency.
- **Consequential actions?**: Yes -- physical state changes (objects moved, doors opened)
- **Dynamic?**: Yes -- second agent (human or AI partner) acts simultaneously
- **Multi-agent?**: Yes -- 2 agents collaborate. LLMs paired with real humans need 1.5x more steps than human-human pairs.
- **Key insight**: Tests coordination and error recovery. Smaller fine-tuned models match 9x larger models while being 8.6x faster.

---

#### CRADLE
- **Paper**: BAAI-Agents, 2024. "Cradle: Empowering Foundation Agents Towards General Computer Control." NeurIPS 2024. arXiv:2403.03186.
- **Core capability**: General computer control via screenshots + keyboard/mouse
- **Task structure**: Agent operates commercial video games (Red Dead Redemption 2, Cities: Skylines, Stardew Valley, Dealer's Life 2) and software (Chrome, Outlook, Meitu, CapCut) + OSWorld benchmark.
- **Action space**: Raw keyboard and mouse actions generated as executable code from screenshot observations. Six modules: Information Gathering, Self-Reflection, Task Inference, Skill Curation, Action Planning, Memory.
- **Horizon**: Long; RDR2 missions take ~1 hour of real-time gameplay
- **Scoring**: Task-specific (mission completion, build quality, etc.)
- **Consequential actions?**: Yes -- full game/application state modification
- **Dynamic?**: Yes -- game worlds have NPCs, physics, time cycles
- **Multi-agent?**: No
- **Key insight**: Framework, not benchmark. Demonstrates feasibility of hour-long autonomous game sessions but no standardized evaluation protocol.

---

#### MineDojo
- **Paper**: Fan et al., 2022. "MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge." NeurIPS 2022. arXiv:2206.08853.
- **Core capability**: Open-ended embodied task completion grounded in internet knowledge
- **Task structure**: 1,581 programmatic tasks (Survival, Harvest, Tech Tree, Creative) + creative tasks with free-form goals. Backed by 730K+ YouTube videos, 6K+ wiki pages, 340K+ Reddit posts.
- **Action space**: Minecraft control actions (movement, camera, inventory management, crafting)
- **Horizon**: Varies; Survival tasks can be indefinitely long
- **Scoring**: Programmatic task completion (deterministic) + learned video-language reward for creative tasks
- **Consequential actions?**: Yes
- **Dynamic?**: Yes (Minecraft world dynamics)
- **Multi-agent?**: Not natively
- **Key insight**: Pairing environment with internet-scale knowledge base is novel. Learned reward functions from video-language models enable evaluation of open-ended goals.

---

#### NetHack Learning Environment (NLE)
- **Paper**: Kuttler et al., 2020. "The NetHack Learning Environment." NeurIPS 2020.
- **Core capability**: Extreme long-horizon decision-making in complex roguelike
- **Task structure**: Full NetHack game -- 40+ dungeon levels, ~70 action types, hundreds of item/monster types, complex interaction mechanics
- **Action space**: ~70 discrete keyboard actions (movement, combat, inventory, spellcasting, etc.)
- **Horizon**: Ultra-long; full game takes 100K+ steps. Winning takes human experts months-years.
- **Scoring**: In-game score + custom progression metrics. Deterministic given actions.
- **Consequential actions?**: Yes -- every action is consequential; death is permanent
- **Dynamic?**: Yes -- procedurally generated levels, monster AI, hunger/health mechanics
- **Multi-agent?**: No
- **Key insight**: The hardest game benchmark. No LLM or RL agent has come close to winning. Tests extreme long-term planning, resource management, and adaptation to novel situations.

---

#### Crafter / Craftax
- **Paper**: Hafner, 2022. "Benchmarking the Spectrum of Agent Capabilities." ICLR 2022. / Matthews et al., 2024. Craftax. NeurIPS 2024.
- **Core capability**: Broad spectrum of agent abilities in 2D survival game
- **Task structure**: Procedurally generated 2D world. Agent must survive (health, hunger, thirst), gather resources, craft tools, explore. 22 achievements spanning resource discovery to tool crafting.
- **Action space**: 17 discrete actions (move 4 directions, do, sleep, place_stone/table/furnace/plant, make_wood/stone/iron/diamond pickaxe/sword, rest)
- **Horizon**: Long; 1M environment steps budget for training. Episodes run until death.
- **Scoring**: Deterministic. Success rate per achievement + geometric mean score across all 22 achievements.
- **Consequential actions?**: Yes
- **Dynamic?**: Yes -- procedurally generated, enemies, day/night, resource depletion
- **Multi-agent?**: No
- **Key insight**: Geometric mean scoring means improving hard achievements matters more than easy ones. Craftax (JAX version) has 65+ achievements and enables massive parallelism for open-ended RL research.

---

#### CLEMBENCH
- **Paper**: Chalamalasetti et al., 2023. "clembench: Using Game Play to Evaluate Chat-Optimized Language Models." arXiv:2305.13455. Updated 2024.
- **Core capability**: Conversational game-playing ability
- **Task structure**: Dialogue games (Wordle, Taboo, drawing instruction games, reference games). Two players interact via chat -- one gives instructions, other follows.
- **Action space**: Natural language dialogue turns; must follow game rules
- **Horizon**: Short-to-medium; game-length conversations
- **Scoring**: Game-specific metrics measuring both rule compliance and play quality. Automated.
- **Consequential actions?**: Yes within game context -- game state changes on each move
- **Dynamic?**: Mildly -- opponent's actions create dynamic context. Dynamic benchmark (games rotated to prevent contamination).
- **Multi-agent?**: Yes -- two LLM players
- **Key insight**: Correlates with Chatbot Arena more than traditional benchmarks. Tests instruction following + strategic communication. Dynamic benchmark design prevents data contamination.

---

#### ARC-AGI / ARC-AGI-2
- **Paper**: Chollet, 2019. "On the Measure of Intelligence." / ARC Prize, 2024-2025.
- **Core capability**: Fluid intelligence / novel pattern generalization
- **Task structure**: Grid puzzles with 2-5 demonstration pairs (input -> output). Must infer transformation rule and apply to test input. ARC-AGI-2 (2025) adds harder tasks + efficiency metrics.
- **Action space**: Produce colored grid (exact pixel values). 2 guesses allowed per test input.
- **Horizon**: Very short per task (seconds of reasoning); no temporal extent
- **Scoring**: Fully deterministic exact-match. Cost-per-task efficiency metric in ARC-AGI-2.
- **Consequential actions?**: No -- single output
- **Dynamic?**: No
- **Multi-agent?**: No
- **Key insight**: Tests generalization ability specifically. o3 reached 88% on ARC-AGI-1 but only 24% SOTA on ARC-AGI-2. Not relevant to long-horizon or dynamic evaluation but important as a measure of fundamental reasoning.

---

### 2.4 Consequential-Action Environment Benchmarks

---

#### TheAgentCompany
- **Paper**: Xu et al., 2024. "TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks." arXiv:2412.14161.
- **Core capability**: Autonomous office work with real tools and simulated colleagues
- **Task structure**: Tasks in a simulated software company with GitLab (code repos), OwnCloud (documents), Plane (project management), RocketChat (team chat with LLM-powered NPCs). Tasks span HR, engineering, product, finance workflows.
- **Action space**: Bash terminal + Python runtime + web browser (Playwright: navigate, click, type, scroll). Actions modify persistent state across all platforms.
- **Horizon**: Medium-to-long; average ~27 steps per task, $4+ API cost per task
- **Scoring**: Checkpoint-based: 0.5 * (checkpoints_passed/total) + 0.5 * full_completion. Checkpoints use deterministic Python evaluators OR LLM-based evaluation for subjective deliverables.
- **Consequential actions?**: Yes -- fully consequential. Creates repos, modifies documents, updates projects, sends messages.
- **Dynamic?**: Reactive -- platforms respond to actions; LLM-powered colleagues respond to messages
- **Multi-agent?**: Pseudo -- agent interacts with NPC colleagues, not true multi-agent
- **Key insight**: Best model (Gemini 2.5 Pro) only 30.3% success. Social interaction tasks are hardest. Partial credit scoring via checkpoints is a good model for us.

---

#### AppWorld
- **Paper**: Trivedi et al., 2024. "AppWorld: A Controllable World of Apps and People." ACL 2024 Best Resource Paper. arXiv:2407.18901.
- **Core capability**: Interactive coding with real app APIs and collateral damage detection
- **Task structure**: 750 tasks across 9 apps (Gmail, Venmo, Amazon, file system, etc.) with 457 APIs. ~100 simulated users with realistic digital lives. Tasks range from "normal" to "challenge" difficulty.
- **Action space**: Python code that calls app APIs. Agent writes code, executes it, observes results, iterates.
- **Horizon**: Medium; multi-step API orchestration
- **Scoring**: Deterministic state-based unit tests. Crucially checks for **collateral damage** -- unintended state changes beyond task scope.
- **Consequential actions?**: Yes -- API calls modify shared state (send money, delete emails, modify orders)
- **Dynamic?**: Reactive -- state changes persist and affect subsequent actions
- **Multi-agent?**: No
- **Key insight**: Collateral damage detection is novel and critical. An agent that completes a task but breaks something else scores lower. GPT-4o: ~49% normal, ~30% challenge.

---

#### Gaia2 / ARE (Agent Reasoning Environments)
- **Paper**: Meta, 2025-2026. "ARE: Scaling Up Agent Environments and Evaluations." arXiv:2509.17158. / "Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments." arXiv:2602.11964.
- **Core capability**: Agent operation in asynchronous, temporally-aware, dynamic environments
- **Task structure**: 1,120 annotated scenarios in smartphone-like environment with 10 app "universes" and 101 tools. Environment evolves independently of agent -- messages arrive, calendar events fire, conditions change.
- **Action space**: Tool calls classified as Read (query-only: search_emails) or Write (state-modifying: send_email). Role-scoped (agent/user/env). Asynchronous notification queue as primary observation channel.
- **Horizon**: Variable; scenarios that would take real-world hours compressed to minutes via event-queue acceleration. Heaviest: 200K+ tokens, 400+ tool calls.
- **Scoring**: Write-action verification: Hard (exact-match) for concrete values + Soft (LLM judge) for nuanced arguments. Only write actions scored. RL-compatible verifiable rewards.
- **Consequential actions?**: Yes -- write actions modify shared state
- **Dynamic?**: **Yes -- this is the key differentiator.** Environment evolves asynchronously. Five event types: Agent, User, Environment, Conditional (trigger on state), Scheduled (time-based). Notifications at configurable verbosity (low/medium/high).
- **Multi-agent?**: Yes -- supports multiple simultaneous agents with shared notification queues
- **Key insight**: Most architecturally relevant benchmark for our purposes. GPT-5 reaches 42% pass@1. No model dominates across capability, cost, and latency. Asynchronous execution surfaces failure modes invisible in synchronous benchmarks. Open-source ARE platform enables custom scenario creation.

---

### 2.5 Adaptation-Specific Benchmarks

---

#### AgentChangeBench
- **Paper**: 2025. "AgentChangeBench: A Multi-Dimensional Evaluation Framework for Goal-Shift Robustness in Conversational AI." NeurIPS 2025 Workshop. arXiv:2510.18170.
- **Core capability**: Adaptation to mid-dialogue goal shifts
- **Task structure**: 2,835 task sequences across banking, retail, airline domains with 5 user personas. Each sequence includes explicit goal-shift points where user changes their objective mid-conversation.
- **Action space**: Tool calls + conversational responses
- **Horizon**: Medium; multi-turn conversations with shift points
- **Scoring**: Four metrics: Task Success Rate (TSR), Tool Use Efficiency (TUE), Tool Call Redundancy Rate (TCRR), Goal-Shift Recovery Time (GSRT). Primarily deterministic.
- **Consequential actions?**: Yes -- tool calls execute transactions
- **Dynamic?**: Yes -- explicit regime changes via goal shifts
- **Multi-agent?**: Two-party (agent + simulated user)
- **Key insight**: GSRT metric quantifies adaptation latency -- how many turns before agent recovers from a goal shift. Claude-3.7 recovers fastest. GPT-4o 92.2% on airline shifts but Gemini collapses to 48.6%.

---

#### AbideGym
- **Paper**: 2025. "AbideGym: Turning Static RL Worlds into Adaptive Challenges." arXiv:2509.21234.
- **Core capability**: Intra-episode adaptation when rules change
- **Task structure**: Modified MiniGrid environments where rules change mid-episode based on agent behavior. If agent is inactive/stuck, task rules modify (e.g., key no longer opens door; alternative trigger tile appears).
- **Action space**: Standard MiniGrid actions (movement, interaction)
- **Horizon**: Variable; episodes extend until goal or timeout
- **Scoring**: Episode success + adaptation capability (implicit, not formally defined)
- **Consequential actions?**: Yes within grid world
- **Dynamic?**: **Yes -- rules change during episode** based on agent behavior
- **Multi-agent?**: No
- **Key insight**: Only benchmark testing intra-episode rule changes. Exposes catastrophic failure of static policies. But very narrow domain (grid world only).

---

### 2.6 Multi-Agent Benchmarks

---

#### Sotopia / Sotopia-S4
- **Paper**: Zhou et al., 2024. "SOTOPIA: Interactive Evaluation for Social Intelligence in Language Agents." ICLR 2024. arXiv:2310.11667.
- **Core capability**: Social intelligence in complex interpersonal scenarios
- **Task structure**: 90 social scenarios (cooperative, competitive, mixed) with 40 characters having unique personalities, secrets, and relationships. Agents role-play and pursue social goals.
- **Action space**: Natural language dialogue + social actions (persuade, negotiate, reveal/conceal information)
- **Horizon**: Medium; multi-turn social interactions
- **Scoring**: Multi-dimensional LLM judge (SOTOPIA-Eval) assessing goal completion, believability, relationship maintenance, knowledge handling, social rules compliance, financial/resource outcomes, and overall quality.
- **Consequential actions?**: Socially consequential -- affect relationships and goal outcomes
- **Dynamic?**: Yes -- other agent creates dynamic, unpredictable context
- **Multi-agent?**: Yes -- primarily dyadic, Sotopia-S4 extends to multiparty
- **Key insight**: GPT-4 significantly below human goal completion, especially in scenarios requiring strategic communication and theory-of-mind. Multi-dimensional evaluation with 7+ axes is expensive but captures nuances binary metrics miss.

---

#### MultiAgentBench (MARBLE)
- **Paper**: Zhang et al., 2025. "MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents." ACL 2025. arXiv:2503.01935.
- **Core capability**: Multi-agent collaboration and competition
- **Task structure**: 6 domains: Research proposal generation, Minecraft building, database error analysis, coding challenges (collaborative) + Werewolf, Bargaining (competitive). 100 test cases per domain.
- **Action space**: Function-calling interface with domain-specific tools (code editors, search engines, game moves)
- **Horizon**: Short-to-medium; 5-20 iterations max
- **Scoring**: Milestone-based KPIs (individual contribution / total milestones) + task-based scores (LLM rubrics or rule-based). Communication Score (5-point scale). Planning Score.
- **Consequential actions?**: Yes -- actions modify shared state
- **Dynamic?**: Yes -- other agents create dynamic context; state evolves based on all agents' actions
- **Multi-agent?**: Yes -- multiple coordination topologies tested (star, chain, tree, graph)
- **Key insight**: Graph topology performs best. Cognitive planning improves milestone rates by only 3%. Performance degrades beyond 7 iterations due to coordination overhead. Short horizons limit relevance for long-term coherence testing.

---

#### ChatArena
- **Paper**: Farama Foundation. Multi-agent language game environments.
- **Core capability**: Communication and collaboration in structured games
- **Task structure**: Multi-agent language games where agents interact via structured dialogue
- **Action space**: Natural language messages within game rules
- **Horizon**: Game-length (short)
- **Scoring**: Game-specific metrics
- **Multi-agent?**: Yes
- **Key insight**: More of a framework than a benchmark. Lightweight and extensible but limited in scope.

---

#### CAMEL
- **Paper**: Li et al., 2023. "CAMEL: Communicative Agents for 'Mind' Exploration of Large Language Model Society." arXiv:2303.17760.
- **Core capability**: Autonomous agent cooperation via inception prompting
- **Task structure**: Two agents (AI User + AI Assistant) collaborate on tasks through role-play dialogue guided by task prompts
- **Action space**: Natural language dialogue
- **Horizon**: Medium; multi-turn conversations until task completion
- **Scoring**: Task completion assessment
- **Multi-agent?**: Yes -- dyadic cooperation
- **Key insight**: Framework for studying emergent cooperation, not a standardized benchmark with scores. The inception prompting technique is interesting for controlling agent behavior.

---

### 2.7 Recently Emerging Benchmarks (2025-2026)

---

#### UltraHorizon
- **Paper**: 2025. "UltraHorizon: Benchmarking Agent Capabilities in Ultra Long-Horizon Scenarios." arXiv:2509.21766.
- **Core capability**: Sustained reasoning, planning, memory across ultra-long trajectories
- **Task structure**: Exploration tasks across 3 environments. Standard: 35K+ tokens, 60+ tool calls. Heavy: 200K+ tokens, 400+ tool calls.
- **Action space**: Tool calls (environment-specific)
- **Scoring**: Deterministic task completion metrics
- **Key insight**: Identified "in-context locking" -- models get stuck in reasoning ruts at long horizons. Simple scaling (more tokens, more calls) fails.

---

#### TRIP-Bench
- **Paper**: Shen et al., 2026. "TRIP-Bench: A Benchmark for Long-Horizon Interactive Agents in Real-World Scenarios." arXiv:2602.01675.
- **Core capability**: Complex travel planning with iterative revision
- **Task structure**: Real-world travel planning with 18 tools and 40+ requirements. Hard split: 15 user turns, 150+ tool calls, 200K+ tokens. Features style shifts, feasibility changes, version revision.
- **Action space**: Tool calls + conversational responses to simulated user
- **Scoring**: Constraint satisfaction (deterministic verification of requirements met)
- **Key insight**: Best models achieve <50% on easy, <10% on hard. GTPO (online multi-turn RL) significantly improves performance. Tests iterative revision under changing requirements.

---

#### DeepPlanning
- **Paper**: 2026. "DeepPlanning: Benchmarking Long-Horizon Agentic Planning with Verifiable Constraints." arXiv:2601.18137.
- **Core capability**: Multi-day travel planning + multi-product shopping with verifiable constraints
- **Task structure**: Planning tasks requiring proactive information acquisition, local constrained reasoning, global constrained optimization
- **Action space**: Tool calls for search and booking
- **Scoring**: Deterministic constraint verification
- **Key insight**: Verifiable constraints enable deterministic scoring of complex planning tasks.

---

#### OdysseyBench
- **Paper**: 2025. "OdysseyBench: Evaluating LLM Agents on Long-Horizon Complex Office Application Workflows." arXiv:2508.09124.
- **Core capability**: Multi-step office workflows across Word, Excel, PDF, Email, Calendar
- **Task structure**: 602 tasks (300 real-world derived + 302 synthesized complex). Multi-application workflows.
- **Action space**: Application-specific operations
- **Scoring**: Task completion metrics
- **Key insight**: Tests cross-application workflow coherence in office settings.

---

#### HeroBench
- **Paper**: Anokhin & Khalikov, 2025. "HeroBench: A Benchmark for Long-Horizon Planning and Structured Reasoning in Virtual Worlds." arXiv:2508.12782.
- **Core capability**: Long-horizon planning with resource/crafting/combat dependencies
- **Task structure**: RPG-inspired virtual world with resource gathering (17 types), crafting dependency graphs, turn-based combat (4 elemental types). Difficulty ratings 2-97.
- **Action space**: Python code with loops for continuous actions (gather, craft, equip, fight). Discrete game actions.
- **Horizon**: Long; complex chains of dependent actions
- **Scoring**: Binary success + Progress Score (partial credit for valid intermediate actions). Deterministic.
- **Consequential actions?**: Yes
- **Dynamic?**: Deterministic by default; optional stochastic mode
- **Multi-agent?**: Yes -- tested with decomposer/critic pairs and hierarchical subagent architectures
- **Key insight**: Massive performance disparities between models -- rarely seen in standard benchmarks. Tests structured reasoning about dependencies, not just sequential execution.

---

## 3. Comparative Analysis

### 3.1 Feature Matrix

| Benchmark | Consequential Actions | Dynamic Environment | Long Horizon (>100K tok) | Deterministic Scoring | Multi-Agent | Adaptation/Regime Change | Memory Required |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gaia2/ARE** | Yes | **Async** | Yes | Hybrid | Yes | Partial | Yes |
| **TheAgentCompany** | Yes | Reactive | No | Hybrid | Pseudo | No | Mild |
| **BALROG** | Yes | Procedural | Extreme (NetHack) | **Yes** | No | Implicit | Yes |
| **tau-bench** | Yes | Dynamic (user) | No | **Yes** (DB state) | 2-party | Partial (user shifts) | Mild |
| **AppWorld** | Yes | Reactive | No | **Yes** (state tests) | No | No | Mild |
| **WebArena** | Yes | Reactive | No | **Yes** | No | No | Mild |
| **PARTNR** | Yes | Dynamic (partner) | No | Semi-auto | Yes | Error recovery | Mild |
| **HeroBench** | Yes | Optional | Medium | **Yes** | Yes | No | Yes |
| **StreamBench** | Partial | Yes (feedback) | Yes | **Yes** | No | **Yes** (drift) | **Yes** |
| **AgentChangeBench** | Yes | **Yes (goal shifts)** | No | Mostly | 2-party | **Yes** | Mild |
| **AbideGym** | Yes | **Yes (rule changes)** | Short | Implicit | No | **Yes** | Mild |
| **Crafter** | Yes | Procedural | Long | **Yes** | No | Implicit | Yes |
| **MultiAgentBench** | Yes | Dynamic (agents) | Short | Mixed | **Yes** | No | Mild |
| **Sotopia** | Social | Dynamic (agents) | Medium | LLM judge | **Yes** | Implicit | Mild |
| **Generative Agents** | Social | **Emergent** | Ultra-long | Human judge | **Yes (25)** | Implicit | **Yes** |
| **MemoryBench** | Partial | Yes (feedback) | Medium | Hybrid | No | **Yes** (learning) | **Yes** |
| **AMA-Bench** | No | No | Yes | LLM judge | No | No | **Yes** |
| **MEMTRACK** | Partial | Yes (conflicts) | Long | Mixed | No | Implicit | **Yes** |
| **SWE-bench** | Yes | No | Medium | **Yes** | No | No | No |

### 3.2 Scoring Method Comparison

| Method | Benchmarks Using It | Pros | Cons |
|--------|-------------------|------|------|
| **Exact match / pass-fail** | SWE-bench, ARC-AGI, GAIA | Perfectly reproducible, cheap | Binary -- no partial credit; misses nuance |
| **State comparison** | tau-bench, AppWorld, Gaia2 | Deterministic, captures real outcomes | Requires careful oracle annotation |
| **Checkpoint/milestone** | TheAgentCompany, MultiAgentBench, HeroBench | Partial credit, tracks progress | Checkpoint design is subjective |
| **Achievement scores** | Crafter, BALROG, Voyager | Captures breadth of capability | May not weight difficulty appropriately |
| **LLM-as-judge** | Sotopia, AMA-Bench, MemoryBench | Handles subjective/open-ended | Non-deterministic, prompt-sensitive, expensive |
| **Human evaluation** | Generative Agents, LoCoMo | Gold standard for nuance | Expensive, doesn't scale, low reproducibility |
| **Hybrid (deterministic + LLM)** | TheAgentCompany, Gaia2, MemoryBench | Deterministic where possible, flexible elsewhere | Complex evaluation pipeline |

---

## 4. Gap Analysis: What's Missing

### What we want that no single benchmark provides:

| Desired Property | Closest Benchmark | What's Missing |
|-----------------|-------------------|----------------|
| Agent takes consequential scored actions | Gaia2, TheAgentCompany, AppWorld | None test actions over truly long horizons with memory requirements |
| Environment changes dynamically over time | Gaia2 (async events), AbideGym (rule changes) | No benchmark has structured regime changes that test adaptation specifically |
| Long horizon exceeding context window | UltraHorizon, BALROG/NetHack, AMA-Bench | Game benchmarks test RL, not LLM agents with memory. AMA-Bench tests memory passively |
| Deterministic/near-deterministic scoring | SWE-bench, tau-bench, Crafter | Deterministic benchmarks tend to be short-horizon or static |
| Multi-agent support | PARTNR, MultiAgentBench, Sotopia | Multi-agent benchmarks are short-horizon and use LLM judges |
| Tests adaptation to regime changes | AgentChangeBench, AbideGym, StreamBench | AgentChangeBench is conversational only. AbideGym is grid-world only. StreamBench tests task improvement, not world-state adaptation |
| Tests coherence under noise | MEMTRACK, LENS (ours) | MEMTRACK is closest but is QA-based, not action-based |

### The fundamental gap:
**No existing benchmark combines consequential actions + dynamic environment + long horizon + deterministic scoring + memory requirement + adaptation testing.** The closest are Gaia2/ARE (actions + dynamic + scoring) and BALROG (actions + long horizon + scoring), but they don't intersect.

---

## 5. Top 8 Most Relevant Benchmarks -- Detailed Design Lessons

### 1. Gaia2/ARE (Meta) -- The Architecture Template

**Why it matters**: Most complete dynamic-environment agent benchmark. The asynchronous event system is exactly what we need.

**Design lessons to adopt**:
- **Event-driven architecture**: Everything is an event (agent actions, environment changes, user messages, scheduled triggers). Events have creation -> scheduling -> execution -> logging lifecycle.
- **Read/Write action separation**: Only write actions are scored. Cleanly separates information gathering from consequential decisions.
- **Configurable notification verbosity**: Controls how much the agent observes (low/medium/high). Can test information overload vs scarcity.
- **Time acceleration**: Queue-based event-to-event jumping means hour-long scenarios run in minutes.
- **Hybrid scoring**: Hard (exact match) for objective values + Soft (LLM judge) for nuanced arguments. Practical compromise.

**What to improve on**:
- Scenarios are smartphone-centric (email, calendar, shopping). We need domain-agnostic environments.
- No explicit regime changes -- environment evolves but doesn't have structured phase transitions.
- Scenarios are one-shot; no long-term state accumulation across sessions.

---

### 2. BALROG -- The Horizon Benchmark

**Why it matters**: Spans 6 orders of magnitude in horizon length. Deterministic scoring. Procedurally generated prevents memorization.

**Design lessons to adopt**:
- **Graduated difficulty across environments**: BabyAI (10 steps) -> NetHack (100K steps) lets you test at multiple scales.
- **Progression metric**: Not just pass/fail but continuous 0-100 progress score. Custom data-informed scoring for hardest games.
- **Procedural generation**: Eliminates data contamination without dynamic generation at test time.
- **Observation history management**: Configurable history window (default 16 observations).
- **Text vs multimodal**: Testing both reveals fundamental capability differences.

**What to improve on**:
- Single-agent only. No collaboration or competition.
- Games don't have explicit regime changes (though procedural generation creates implicit novelty).
- No memory system evaluation -- tests raw in-context processing.

---

### 3. tau-bench -- The Reliability Benchmark

**Why it matters**: The pass^k metric is brilliant for measuring reliability, not just peak performance. Database state comparison is the cleanest deterministic scoring for consequential actions.

**Design lessons to adopt**:
- **pass^k metric**: Run the same task k times, check if ALL k attempts succeed. Measures consistency. pass^8 for GPT-4o is <25% even when pass^1 is ~50%.
- **Database state comparison**: Final DB state vs oracle state. No ambiguity about whether the right thing happened.
- **Policy constraints**: Agent must follow specific rules (return policies, booking rules). Tests rule-adherence under pressure.
- **Dynamic user simulation**: LLM-simulated user introduces realistic ambiguity and mid-conversation changes.

**What to improve on**:
- Short horizon (5-15 turns). Needs scaling to 100+ turn sessions.
- User changes are minor (clarifications, preference shifts), not regime-level changes.
- No persistent state across sessions.

---

### 4. AgentChangeBench -- The Adaptation Benchmark

**Why it matters**: Only benchmark explicitly measuring adaptation *latency* -- not just whether the agent adapts, but how quickly.

**Design lessons to adopt**:
- **Goal-Shift Recovery Time (GSRT)**: Measures turns until agent acknowledges change, adjusts tool use, and achieves new goal. Three-dimensional recovery metric.
- **Explicit shift points**: Tasks have annotated moments where objectives change. Enables controlled testing of adaptation.
- **Multi-dimensional evaluation**: TSR (effectiveness), TUE (efficiency), TCRR (wasted effort), GSRT (adaptation speed).
- **Persona-driven shifts**: Different user personas create different shift patterns.

**What to improve on**:
- Conversational only (no environment state beyond tool calls).
- Short horizon (single conversation).
- Goal shifts are user-initiated, not environment-initiated. Need to test adaptation to world changes, not just user preference changes.

---

### 5. TheAgentCompany -- The Consequential Actions Benchmark

**Why it matters**: Most realistic workplace simulation with fully consequential actions across multiple platforms.

**Design lessons to adopt**:
- **Checkpoint-based partial credit**: 0.5 * (checkpoints/total) + 0.5 * full_completion. Recognizes progress without rewarding incomplete work too much.
- **Multi-platform integration**: GitLab + OwnCloud + Plane + RocketChat. Tests cross-platform coordination.
- **NPC colleagues**: LLM-powered NPCs create social dynamics (asking for info, responding to queries).
- **Deterministic + LLM hybrid evaluation**: Python evaluators for objective checks, LLM judges for subjective deliverables.

**What to improve on**:
- Each task is independent. No long-term state accumulation.
- No regime changes or adaptation testing.
- Medium horizon only (~27 steps average).

---

### 6. StreamBench -- The Continuous Improvement Benchmark

**Why it matters**: Only benchmark explicitly testing whether agents get *better over time* from feedback.

**Design lessons to adopt**:
- **Streaming evaluation**: Performance measured across a sequence of tasks, not on isolated instances.
- **Feedback integration**: Agent receives feedback after each attempt and must incorporate it.
- **Multiple domains**: Text-to-SQL, coding, tool use, medical diagnosis, QA.
- **Online learning framing**: Directly tests whether memory/learning systems enable improvement.

**What to improve on**:
- Tasks are independent (no shared world state).
- Feedback is explicit (gold labels). Real environments give implicit, noisy feedback.
- No adversarial elements (noise, contradictions, distractors).

---

### 7. MEMTRACK -- The Cross-Platform Memory Benchmark

**Why it matters**: Tests exactly the kind of noisy, conflicting, cross-source information that production agents face.

**Design lessons to adopt**:
- **Cross-platform interleaving**: Information scattered across Slack, Linear, Git. Agent must correlate.
- **Conflicting information**: Different platforms may have inconsistent data. Agent must resolve.
- **Three-axis evaluation**: Correctness + Efficiency + Redundancy. Penalizes wasted work.
- **Realistic software dev scenarios**: Grounded in actual engineering workflows.

**What to improve on**:
- Primarily QA-based (state tracking), not action-based.
- Memory backends (Zep, Mem0) showed no improvement -- suggests the benchmark may be too focused on raw reasoning vs memory retrieval.
- Limited scale (workshop paper, not full benchmark release).

---

### 8. AppWorld -- The Collateral Damage Benchmark

**Why it matters**: Uniquely tests whether agents cause unintended side effects beyond their primary task.

**Design lessons to adopt**:
- **Collateral damage detection**: State-based unit tests check not just "did you do the task" but "did you break anything else." Critical for consequential action evaluation.
- **457 APIs across 9 apps**: Rich, realistic action space.
- **Simulated user population**: ~100 users with realistic digital lives create complex state.
- **Programmatic evaluation**: Fully deterministic state-based testing.

**What to improve on**:
- Short horizon (single tasks).
- Static environment (no async events).
- No memory requirement (context sufficient).

---

## 6. Synthesis: Design Principles for a Next-Generation Benchmark

Based on this survey, a benchmark that tests long-term coherence and adaptation through scored actions in dynamic environments should combine:

### From Gaia2/ARE:
- Event-driven asynchronous architecture where environment evolves independently
- Read/write action classification with only write actions scored
- Configurable observation verbosity
- Time acceleration for practical execution

### From BALROG/Crafter:
- Graduated difficulty tiers spanning short to ultra-long horizons
- Deterministic progression metrics (not just pass/fail)
- Procedural generation to prevent memorization

### From tau-bench:
- pass^k reliability metric
- Database/world-state comparison for scoring (deterministic when possible)
- Policy constraints that agents must follow

### From AgentChangeBench:
- Explicit, annotated regime change points
- Recovery time measurement (GSRT-style)
- Multi-dimensional evaluation (effectiveness + efficiency + adaptation speed)

### From TheAgentCompany:
- Checkpoint-based partial credit
- Multi-platform integration
- NPC interactions

### From StreamBench:
- Continuous evaluation over time (improvement curves)
- Feedback loops that test learning

### From MEMTRACK:
- Cross-source information with conflicts and noise
- Efficiency and redundancy penalties

### From AppWorld:
- Collateral damage detection
- State-based deterministic evaluation

### Novel elements not found in any existing benchmark:
1. **Structured regime changes**: Environment rules change at defined points (not just user goal shifts). Tests whether agent detects the change and adapts strategy.
2. **Noise resistance scoring**: Distractors and irrelevant events that should NOT change agent behavior. Tests signal-to-noise discrimination.
3. **Cross-session state accumulation**: Persistent world state across episodes where context window is insufficient. Forces real memory use.
4. **Deterministic scoring of adaptation**: Not just "did you eventually adapt" but "how many wasted actions before adaptation" and "did you correctly identify what changed."
5. **Both individual and multi-agent modes**: Same environment, same scoring, but testing coordination overhead.

---

## Sources

### Agent/LLM Long-Horizon Benchmarks
- [GAIA: A Benchmark for General AI Assistants](https://arxiv.org/abs/2311.12983)
- [WebArena: A Realistic Web Environment for Building Autonomous Agents](https://arxiv.org/abs/2307.13854)
- [VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks](https://arxiv.org/abs/2401.13649)
- [OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks](https://arxiv.org/abs/2404.07972)
- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770)
- [SWE-bench Verified (OpenAI)](https://openai.com/index/introducing-swe-bench-verified/)
- [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688)
- [tau-bench: Tool-Agent-User Interaction](https://arxiv.org/abs/2406.12045)
- [tau2-bench (Sierra Research)](https://github.com/sierra-research/tau2-bench)
- [MLAgentBench](https://arxiv.org/abs/2310.03302)
- [MLE-bench (OpenAI)](https://arxiv.org/abs/2410.07095)
- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)
- [BALROG: Benchmarking Agentic LLM and VLM Reasoning On Games](https://arxiv.org/abs/2411.13543)
- [Voyager: An Open-Ended Embodied Agent with LLMs](https://arxiv.org/abs/2305.16291)
- [SPRING: Studying the Paper and Reasoning to Play Games](https://arxiv.org/abs/2305.15486)

### Memory-Specific Benchmarks
- [LoCoMo: Evaluating Very Long-Term Conversational Memory](https://arxiv.org/abs/2402.17753)
- [LongMemEval: Benchmarking Chat Assistants on Long-Term Memory](https://arxiv.org/abs/2410.10813)
- [StreamBench: Benchmarking Continuous Improvement](https://arxiv.org/abs/2406.08747)
- [MemoryBench: Memory and Continual Learning in LLM Systems](https://arxiv.org/abs/2510.17281)
- [InfiniteBench: Extending Long Context Beyond 100K Tokens](https://arxiv.org/abs/2402.13718)
- [AMA-Bench: Evaluating Long-Horizon Memory for Agentic Applications](https://arxiv.org/abs/2602.22769)
- [MEMTRACK: Long-Term Memory and State Tracking](https://arxiv.org/abs/2510.01353)
- [AMemGym: Interactive Memory Benchmarking (ICLR 2026)](https://arxiv.org/abs/2603.01966)
- [Agent Memory Benchmark Manifesto (Hindsight/Vectorize)](https://hindsight.vectorize.io/blog/2026/03/23/agent-memory-benchmark)

### Dynamic/Adaptive Environment Benchmarks
- [PARTNR: Planning and Reasoning in Embodied Multi-agent Tasks](https://arxiv.org/abs/2411.00081)
- [CRADLE: Empowering Foundation Agents Towards General Computer Control](https://arxiv.org/abs/2403.03186)
- [MineDojo: Building Open-Ended Embodied Agents](https://arxiv.org/abs/2206.08853)
- [The NetHack Learning Environment](https://arxiv.org/abs/2006.13760)
- [Crafter: Benchmarking the Spectrum of Agent Capabilities](https://arxiv.org/abs/2109.06780)
- [CLEMBENCH: Using Game Play to Evaluate Chat-Optimized LLMs](https://arxiv.org/abs/2305.13455)
- [ARC-AGI / ARC Prize](https://arcprize.org/)
- [AbideGym: Turning Static RL Worlds into Adaptive Challenges](https://arxiv.org/abs/2509.21234)
- [AgentChangeBench: Goal-Shift Robustness](https://arxiv.org/abs/2510.18170)

### Consequential Action Benchmarks
- [TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks](https://arxiv.org/abs/2412.14161)
- [AppWorld: A Controllable World of Apps and People](https://arxiv.org/abs/2407.18901)
- [Gaia2/ARE: Dynamic and Asynchronous Environments](https://arxiv.org/abs/2602.11964)
- [ARE: Scaling Up Agent Environments and Evaluations](https://arxiv.org/abs/2509.17158)

### Multi-Agent Benchmarks
- [Sotopia: Interactive Evaluation for Social Intelligence](https://arxiv.org/abs/2310.11667)
- [MultiAgentBench (MARBLE): Collaboration and Competition](https://arxiv.org/abs/2503.01935)
- [ChatArena (Farama Foundation)](https://github.com/Farama-Foundation/chatarena)
- [CAMEL: Communicative Agents Framework](https://arxiv.org/abs/2303.17760)

### Long-Horizon Planning Benchmarks (2025-2026)
- [UltraHorizon: Ultra Long-Horizon Scenarios](https://arxiv.org/abs/2509.21766)
- [TRIP-Bench: Long-Horizon Interactive Agents](https://arxiv.org/abs/2602.01675)
- [DeepPlanning: Long-Horizon Agentic Planning](https://arxiv.org/abs/2601.18137)
- [OdysseyBench: Long-Horizon Office Application Workflows](https://arxiv.org/abs/2508.09124)
- [HeroBench: Long-Horizon Planning in Virtual Worlds](https://arxiv.org/abs/2508.12782)

### Surveys
- [Evaluation and Benchmarking of LLM Agents: A Survey (KDD 2025)](https://arxiv.org/abs/2507.21504)
- [Memory in the Age of AI Agents](https://arxiv.org/abs/2512.13564)
