<h1 align="center">Awesome Efficient Agents: A Survey of Memory, Tool Learning, and Planning</h1>
<div align="center">
  <a href="https://arxiv.org/abs/2601.14192"><img src="https://img.shields.io/badge/arXiv-2601.14192-b31b1b.svg?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://efficient-agents.github.io/"><img src="https://img.shields.io/badge/Project%20Page-efficient--agents.github.io-2ea44f.svg?logo=googlehome&logoColor=white" alt="Project Page"></a>
</div>


🤝 **Contributions welcome!** Open an issue or submit a pull request to add papers, fix links, or improve categorization.

## ⚡Introduction

Recent years have seen growing interest in extending large language models into agentic systems. While agent capabilities have advanced rapidly, efficiency has received comparatively less attention despite being crucial for real-world deployment. This repository studies efficiency-guided agent design from three core components: memory, tool learning, and planning.

We provide a curated paper list to help readers quickly locate representative work, along with lightweight notes on how each topic connects to efficiency.

- **Efficient Memory.** We organize memory-related papers into three processes: construction, management, and access.
- **Efficient Tool Learning.** We group papers into tool selection, tool calling, and tool-integrated reasoning.
- **Efficient Planning.** We collect work on planning that improves overall agent efficiency by reducing unnecessary actions and shortening trajectories.

![introduction-picture](assets/introduction-picture.png)

## 🧾Paper List

<details>
  <summary>📂 Table of Contents<em>(click to expand/collapse)</em></summary>
  <ul>
    <li><a href="#memory">🧠Memory</a>
      <ul>
        <li><a href="#working-memory">Working Memory</a>
          <ul>
            <li><a href="#textual-memory">Textual Memory</a></li>
            <li><a href="#latent-memory">Latent Memory</a></li>
          </ul>
        </li>
        <li><a href="#external-memory">External Memory</a>
          <ul>
            <li><a href="#item-based-memory">Item-based Memory</a></li>
            <li><a href="#graph-based-memory">Graph-based Memory</a></li>
            <li><a href="#hierarchical-memory">Hierarchical Memory</a></li>
          </ul>
        </li>
        <li><a href="#multi-agent-memory">Multi-Agent Memory</a>
          <ul>
            <li><a href="#shared-memory">Shared Memory</a></li>
            <li><a href="#local-memory">Local Memory</a></li>
            <li><a href="#mixed-memory">Mixed Memory</a></li>
          </ul>
        </li>
      </ul>
    </li>
    <li><a href="#tool-learning">🛠️Tool Learning</a>
      <ul>
        <li><a href="#tool-selection">Tool Selection</a>
          <ul>
            <li><a href="#external-retriever">External Retriever</a></li>
            <li><a href="#multi-label-classification-mlc">Multi-Label Classification (MLC)</a></li>
            <li><a href="#vocabulary-based-retrieval">Vocabulary-based Retrieval</a></li>
          </ul>
        </li>
        <li><a href="#tool-calling">Tool Calling</a>
          <ul>
            <li><a href="#in-place-parameter-filling">In-Place Parameter Filling</a></li>
            <li><a href="#parallel-tool-calling">Parallel Tool Calling</a></li>
            <li><a href="#cost-aware-tool-calling">Cost-Aware Tool Calling</a></li>
            <li><a href="#efficient-test-time-scaling">Efficient Test-Time Scaling</a></li>
            <li><a href="#efficient-tool-calling-with-post-training">Efficient Tool Calling with Post-training</a></li>
          </ul>
        </li>
        <li><a href="#tool-integrated-reasoning-tir">Tool-Integrated Reasoning (TIR)</a>
          <ul>
            <li><a href="#selective-invocation">Selective Invocation</a></li>
            <li><a href="#cost-aware-policy-optimization">Cost-Aware Policy Optimization</a></li>
          </ul>
        </li>
      </ul>
    </li>
    <li><a href="#planning">🧩Planning</a>
      <ul>
        <li><a href="#single-agent-planning-efficiency">Single-Agent Planning Efficiency</a>
          <ul>
            <li><a href="#adaptive-budgeting-and-control">Adaptive Budgeting and Control</a></li>
            <li><a href="#structured-search-strategies">Structured Search</a></li>
            <li><a href="#task-decomposition-and-routing">Task Decomposition</a></li>
            <li><a href="#policy-optimization">Policy Optimization</a></li>
            <li><a href="#architectural-synergy-with-memory-and-tools">Memory and Skill Acquisition</a></li>
          </ul>
        </li>
        <li><a href="#multi-agent-collaborative-efficiency">Multi-Agent Collaborative Efficiency</a>
          <ul>
            <li><a href="#topological-efficiency">Topological Efficiency and Sparsification</a></li>
            <li><a href="#protocol-and-context-optimization">Protocol and Context Optimization</a></li>
            <li><a href="#distilling-coordination-into-planning">Distilling Coordination into Planning</a></li>
          </ul>
        </li>
      </ul>
    </li>
  </ul>
</details>

<a name="memory"></a>

### 🧠Memory

![paper-memory-picture](assets/paper-memory-picture.png)

In the paper, we organize memory into construction, management, and access. Since many papers overlap across these stages, this README is primarily organized around memory construction to avoid redundancy.

<a name="working-memory"></a>
#### Working Memory

<a name="textual-memory"></a>

#####  Textual Memory

* (2025-10) [AgentFold: Long-Horizon Web Agents with Proactive Context Management](https://arxiv.org/abs/2510.24699) [![Star](https://img.shields.io/github/stars/Alibaba-NLP/DeepResearch.svg?style=social&label=Star)](https://github.com/Alibaba-NLP/DeepResearch)
* (2025-07) [MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent](https://arxiv.org/abs/2507.02259) [![Website](https://img.shields.io/badge/Website-Project-green)](https://memagent-sialab.github.io/) [![Star](https://img.shields.io/github/stars/BytedTsinghua-SIA/MemAgent.svg?style=social&label=Star)](https://github.com/BytedTsinghua-SIA/MemAgent)
* (2025-06) [MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents](https://arxiv.org/abs/2506.15841) ![NeurIPS WS 2025](https://img.shields.io/badge/NeurIPS%20WS%202025-blue) ![COLM WS 2025](https://img.shields.io/badge/COLM%20WS%202025-blue) [![Website](https://img.shields.io/badge/Website-Project-green)](https://mit-mi.github.io/mem1-site/) [![Star](https://img.shields.io/github/stars/MIT-MI/MEM1.svg?style=social&label=Star)](https://github.com/MIT-MI/MEM1)

* (2025-04) [Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory](https://arxiv.org/abs/2504.07952) [![Star](https://img.shields.io/github/stars/suzgunmirac/dynamic-cheatsheet.svg?style=social&label=Star)](https://github.com/suzgunmirac/dynamic-cheatsheet)

* (2024-02) [Compress to Impress: Unleashing the Potential of Compressive Memory in Real-World Long-Term Conversations](https://arxiv.org/abs/2402.11975)  ![COLING 2025](https://img.shields.io/badge/COLING%202025-blue) [![Star](https://img.shields.io/github/stars/nuochenpku/COMEDY.svg?style=social&label=Star)](https://github.com/nuochenpku/COMEDY)

<a name="latent-memory"></a>

#####  Latent Memory

* (2025-09) [MemGen: Weaving Generative Latent Memory for Self-Evolving Agents](https://arxiv.org/abs/2509.24704) [![Star](https://img.shields.io/github/stars/KANABOON1/MemGen.svg?style=social&label=Star)](https://github.com/KANABOON1/MemGen)
* (2025-02) [M+: Extending MemoryLLM with Scalable Long-Term Memory](https://arxiv.org/abs/2502.00592) [![ICML 2025](https://img.shields.io/badge/ICML%202025-blue)](https://openreview.net/forum?id=OcqbkROe8J) [![Star](https://img.shields.io/github/stars/wangyu-ustc/MemoryLLM.svg?style=social&label=Star)](https://github.com/wangyu-ustc/MemoryLLM)
* (2025-01) [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)
* (2024-09) [MemoRAG: Boosting Long Context Processing with Global Memory-Enhanced Retrieval Augmentation](https://arxiv.org/abs/2409.05591) [![TheWebConf 2025](https://img.shields.io/badge/TheWebConf%202025-blue)](https://dl.acm.org/doi/abs/10.1145/3696410.3714805) [![Star](https://img.shields.io/github/stars/qhjqhj00/MemoRAG.svg?style=social&label=Star)](https://github.com/qhjqhj00/MemoRAG)
* (2024-07) [$\text{Memory}^3$: Language Modeling with Explicit Memory](https://arxiv.org/abs/2407.01178)
* (2024-02) [MEMORYLLM: Towards Self-Updatable Large Language Models](https://arxiv.org/abs/2402.04624) [![ICML 2024](https://img.shields.io/badge/ICML%202024-blue)](https://proceedings.mlr.press/v235/wang24s.html) [![Star](https://img.shields.io/github/stars/wangyu-ustc/MemoryLLM.svg?style=social&label=Star)](https://github.com/wangyu-ustc/MemoryLLM)
* (2024-01) [Long Context Compression with Activation Beacon](https://arxiv.org/abs/2401.03462) [![ICLR 2025](https://img.shields.io/badge/ICLR%202025-blue)](https://openreview.net/forum?id=1eQT9OzfNQ) [![Star](https://img.shields.io/github/stars/FlagOpen/FlagEmbedding.svg?style=social&label=Star)](https://github.com/FlagOpen/FlagEmbedding)

<a name="external-memory"></a>

#### External Memory

<a name="item-based-memory"></a>

#####  Item-based Memory

* (2025-10) [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/abs/2510.04618) [![Star](https://img.shields.io/github/stars/ace-agent/ace.svg?style=social&label=Star)](https://github.com/ace-agent/ace) 
* (2025-09) [ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory](https://arxiv.org/abs/2509.25140) 
* (2025-08) [Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning](https://arxiv.org/abs/2508.19828) 
* (2025-08) [Memento: Fine-tuning LLM Agents without Fine-tuning LLMs](https://arxiv.org/abs/2508.16153) [![Star](https://img.shields.io/github/stars/Agent-on-the-Fly/Memento.svg?style=social&label=Star)](https://github.com/Agent-on-the-Fly/Memento)
* (2025-07) [Agent KB: Leveraging Cross-Domain Experience for Agentic Problem Solving](https://arxiv.org/abs/2507.06229) ![ICML 2025 Workshop](https://img.shields.io/badge/ICML%202025%20Workshop-blue) [![Star](https://img.shields.io/github/stars/OPPO-PersonalAI/Agent-KB.svg?style=social&label=Star)](https://github.com/OPPO-PersonalAI/Agent-KB)
* (2025-06) [Cost-Efficient Serving of LLM Agents via Test-Time Plan Caching(Agentic Plan Caching: Test-Time Memory for Fast and Cost-Efficient LLM Agents)](https://arxiv.org/abs/2506.14852) ![NeurIPS](https://img.shields.io/badge/NeurIPS%202025-blue) 
* (2025-04) [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413) [![Star](https://img.shields.io/github/stars/mem0ai/mem0.svg?style=social&label=Star)](https://github.com/mem0ai/mem0)
* (2025-03) [MemInsight: Autonomous Memory Augmentation for LLM Agents](https://arxiv.org/abs/2503.21760) ![EMNLP](https://img.shields.io/badge/EMNLP%202025-blue) [![Star](https://img.shields.io/github/stars/amazon-science/MemInsight.svg?style=social&label=Star)](https://github.com/amazon-science/MemInsight)
* (2025-03) [In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents](https://arxiv.org/abs/2503.08026) ![ACL](https://img.shields.io/badge/ACL%202025-blue)
* (2025-02) [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110) ![NeurIPS](https://img.shields.io/badge/NeurIPS%202025-blue) [![Star](https://img.shields.io/github/stars/agiresearch/A-mem.svg?style=social&label=Star)](https://github.com/agiresearch/A-mem)
* (2025-02) [On Memory Construction and Retrieval for Personalized Conversational Agents](https://arxiv.org/abs/2502.05589) ![ICLR](https://img.shields.io/badge/ICLR%202025-blue)  [![Website](https://img.shields.io/badge/Website-Project-green)](https://llmlingua.com/secom.html) [![Star](https://img.shields.io/github/stars/microsoft/SeCom.svg?style=social&label=Star)](https://github.com/microsoft/SeCom)
* (2024-06) [Hello Again! LLM-powered Personalized Agent for Long-term Dialogue](https://arxiv.org/abs/2406.05925)  ![NAACL 2025](https://img.shields.io/badge/NAACL%202025-blue) [![Star](https://img.shields.io/github/stars/leolee99/LD-Agent.svg?style=social&label=Star)](https://github.com/leolee99/LD-Agent)
* (2024-04) ["My agent understands me better": Integrating Dynamic Human-like Memory Recall and Consolidation in LLM-Based Agents](https://arxiv.org/abs/2404.00573) ![CHI EA 2024](https://img.shields.io/badge/CHI%20EA%202024-blue)
* (2023-10) [RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation](https://arxiv.org/abs/2310.04408) ![ICLR 2024](https://img.shields.io/badge/ICLR%202024-blue)  [![Star](https://img.shields.io/github/stars/carriex/recomp.svg?style=social&label=Star)](https://github.com/carriex/recomp)
* (2023-08) [ExpeL: LLM Agents Are Experiential Learners](https://arxiv.org/abs/2308.10144) ![AAAI](https://img.shields.io/badge/AAAI%202024-blue) [![Website](https://img.shields.io/badge/Website-Project-green)](https://andrewzh112.github.io/expel/) [![Star](https://img.shields.io/github/stars/LeapLabTHU/ExpeL.svg?style=social&label=Star)](https://github.com/LeapLabTHU/ExpeL)
* (2023-08) [MemoChat: Tuning LLMs to Use Memos for Consistent Long-Range Open-Domain Conversation](https://arxiv.org/abs/2308.08239) [![Star](https://img.shields.io/github/stars/LuJunru/MemoChat.svg?style=social&label=Star)](https://github.com/LuJunru/MemoChat)
* (2023-05) [MemoryBank: Enhancing Large Language Models with Long-Term Memory](https://arxiv.org/abs/2305.10250)  ![AAAI](https://img.shields.io/badge/AAAI%202024-blue) [![Star](https://img.shields.io/github/stars/zhongwanjun/MemoryBank-SiliconFriend.svg?style=social&label=Star)](https://github.com/zhongwanjun/MemoryBank-SiliconFriend)
<a name="graph-based-memory"></a>

#####  Graph-based Memory

* (2025-10) [D-SMART: Enhancing LLM Dialogue Consistency via Dynamic Structured Memory And Reasoning Tree](https://arxiv.org/abs/2510.13363) 
* (2025-04) [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413) [![Star](https://img.shields.io/github/stars/mem0ai/mem0.svg?style=social&label=Star)](https://github.com/mem0ai/mem0)
* (2025-01) [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956) [![Star](https://img.shields.io/github/stars/getzep/zep.svg?style=social&label=Star)](https://github.com/getzep/zep) 
* (2024-07) [AriGraph: Learning Knowledge Graph World Models with Episodic Memory for LLM Agents](https://arxiv.org/abs/2407.04363)  ![IJCAI 2025](https://img.shields.io/badge/IJCAI%202025-blue) [![Star](https://img.shields.io/github/stars/AIRI-Institute/AriGraph.svg?style=social&label=Star)](https://github.com/AIRI-Institute/AriGraph) 
* (2024-06) [GraphReader: Building Graph-based Agent to Enhance Long-Context Abilities of Large Language Models](https://arxiv.org/abs/2406.14550) ![EMNLP 2024 Findings](https://img.shields.io/badge/EMNLP%202024%20Findings-blue) 
* (2024-02) [KG-Agent: An Efficient Autonomous Agent Framework for Complex Reasoning over Knowledge Graph](https://arxiv.org/abs/2402.11163) ![ACL](https://img.shields.io/badge/ACL%202025-blue) 

<a name="hierarchical-memory"></a>
#####  Hierarchical Memory

* (2025-10) [LightMem: Lightweight and Efficient Memory-Augmented Generation](https://www.arxiv.org/abs/2510.18866) [![Star](https://img.shields.io/github/stars/zjunlp/LightMem.svg?style=social&label=Star)](https://github.com/zjunlp/LightMem)
* (2025-07) [Hierarchical Memory for High-Efficiency Long-Term Reasoning in LLM Agents](https://arxiv.org/abs/2507.22925)
* (2025-07) [MemOS: A Memory OS for AI System](https://arxiv.org/abs/2507.03724) [![Star](https://img.shields.io/github/stars/MemTensor/MemOS.svg?style=social&label=Star)](https://github.com/MemTensor/MemOS)
* (2025-06) [Memory OS of AI Agent](https://arxiv.org/abs/2506.06326) ![EMNLP 2025](https://img.shields.io/badge/EMNLP%202025-blue) [![Website](https://img.shields.io/badge/Website-Project-green)](https://baijia.online/memoryos/) [![Star](https://img.shields.io/github/stars/BAI-LAB/MemoryOS.svg?style=social&label=Star)](https://github.com/BAI-LAB/MemoryOS)
* (2024-08) [HiAgent: Hierarchical Working Memory Management for Solving Long-Horizon Agent Tasks with Large Language Model](https://arxiv.org/abs/2408.09559) ![ACL](https://img.shields.io/badge/ACL%202025-blue) [![Star](https://img.shields.io/github/stars/HiAgent2024/HiAgent.svg?style=social&label=Star)](https://github.com/HiAgent2024/HiAgent)
* (2024-02) [A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts](https://arxiv.org/abs/2402.09727) ![ICML 2024](https://img.shields.io/badge/ICML%202024-blue) [![Website](https://img.shields.io/badge/Website-Project-green)](https://read-agent.github.io/) [![Star](https://img.shields.io/github/stars/read-agent/read-agent.github.io.svg?style=social&label=Star)](https://github.com/read-agent/read-agent.github.io/blob/main/assets/read_agent_demo.ipynb)
* (2023-10) [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)  [![Star](https://img.shields.io/github/stars/letta-ai/letta.svg?style=social&label=Star)](https://github.com/letta-ai/letta) 

<a name="multi-agent-memory"></a>

#### Multi-Agent Memory

<a name="shared-memory"></a>
#####  Shared Memory

* (2025-11) [MemIndex: Agentic Event-based Distributed Memory Management for Multi-agent Systems](https://dl.acm.org/doi/10.1145/3774946) ![ACM TAAS](https://img.shields.io/badge/ACM%20TAAS%202025-blue)
* (2025-08) [RCR-Router: Efficient Role-Aware Context Routing for Multi-Agent LLM Systems with Structured Memory](https://arxiv.org/abs/2508.04903) 
* (2025-07) [MIRIX: Multi-Agent Memory System for LLM-Based Agents](https://arxiv.org/abs/2507.07957) [![Website](https://img.shields.io/badge/Website-Project-green)](https://mirix.io/)  [![Star](https://img.shields.io/github/stars/Mirix-AI/MIRIX.svg?style=social&label=Star)](https://github.com/Mirix-AI/MIRIX)
* (2025-06) [G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems](https://arxiv.org/abs/2506.07398) ![NeurIPS](https://img.shields.io/badge/NeurIPS%202025%20spotlight-blue) [![Star](https://img.shields.io/github/stars/bingreeky/GMemory.svg?style=social&label=Star)](https://github.com/bingreeky/GMemory)
* (2024-04) [Memory Sharing for Large Language Model based Agents](https://arxiv.org/abs/2404.09982) [![Star](https://img.shields.io/github/stars/GHupppp/InteractiveMemorySharingLLM.svg?style=social&label=Star)](https://github.com/GHupppp/InteractiveMemorySharingLLM)

<a name="local-memory"></a>
#####  Local Memory

* (2025-08) [Intrinsic Memory Agents: Heterogeneous Multi-Agent LLM Systems through Structured Contextual Memory](https://arxiv.org/abs/2508.08997) 
* (2025-04) [AgentNet: Decentralized Evolutionary Coordination for LLM-based Multi-Agent Systems](https://arxiv.org/abs/2504.00587) ![NeurIPS](https://img.shields.io/badge/NeurIPS%202025-blue)  [![Star](https://img.shields.io/github/stars/zoe-yyx/AgentNet.svg?style=social&label=Star)](https://github.com/zoe-yyx/AgentNet)
* (2025-02) [LLM-Powered Decentralized Generative Agents with Adaptive Hierarchical Knowledge Graph for Cooperative Planning](https://arxiv.org/abs/2502.05453)  [![Website](https://img.shields.io/badge/Website-Project-green)](https://happyeureka.github.io/damcs/) [![Star](https://img.shields.io/github/stars/13RENDA/Mcrafter_LLM_Agent.svg?style=social&label=Star)](https://github.com/13RENDA/Mcrafter_LLM_Agent)

<a name="mixed-memory"></a>

#####  Mixed Memory

* (2025-10) [LEGOMem: Modular Procedural Memory for Multi-agent LLM Systems for Workflow Automation](https://arxiv.org/abs/2510.04851) ![AAMAS 2026](https://img.shields.io/badge/AAMAS%202026%20Extended%20Abstract-blue)
* (2025-05) [Collaborative Memory: Multi-User Memory Sharing in LLM Agents with Dynamic Access Control](https://arxiv.org/abs/2505.18279)
* (2025-01) [SRMT: Shared Memory for Multi-agent Lifelong Pathfinding](https://arxiv.org/abs/2501.13200)  [![Star](https://img.shields.io/github/stars/Aloriosa/srmt.svg?style=social&label=Star)](https://github.com/Aloriosa/srmt)

<a name="tool-learning"></a>

### 🛠️Tool Learning

![tool-learning](assets/tool-learning.png)

<a name="tool-selection"></a>
#### Tool Selection

<a name="external-retriever"></a>

#####  External Retriever

* (2025-10) [ToolScope: Enhancing LLM Agent Tool Use through Tool Merging and Context-Aware Filtering](https://arxiv.org/abs/2510.20036) 

* (2024-10) [Toolshed: Scale Tool-Equipped Agents with Advanced RAG-Tool Fusion and Tool Knowledge Bases](https://arxiv.org/abs/2410.14594) ![ICAART](https://img.shields.io/badge/ICAART%202025-blue)  [![Star](https://img.shields.io/github/stars/EliasLumer/Toolshed-Scale-Tool-Equipped-Agents-with-Advanced-RAG-Tool-Fusion-and-Tool-Knowledge-Bases.svg?style=social&label=Star)](https://github.com/EliasLumer/Toolshed-Scale-Tool-Equipped-Agents-with-Advanced-RAG-Tool-Fusion-and-Tool-Knowledge-Bases)
* (2024-10) [From Exploration to Mastery: Enabling LLMs to Master Tools via Self-Driven Interactions](https://arxiv.org/abs/2410.08197) ![ICLR](https://img.shields.io/badge/ICLR%202025%20oral-blue)  [![Star](https://img.shields.io/github/stars/quchangle1/DRAFT.svg?style=social&label=Star)](https://github.com/quchangle1/DRAFT)
* (2024-02) [AnyTool: Self-Reflective, Hierarchical Agents for Large-Scale API Calls](https://arxiv.org/abs/2402.04253) ![ICML](https://img.shields.io/badge/ICML%202024-blue)  [![Star](https://img.shields.io/github/stars/dyabel/AnyTool.svg?style=social&label=Star)](https://github.com/dyabel/AnyTool)
* (2023-12) [ProTIP: Progressive Tool Retrieval Improves Planning](https://arxiv.org/abs/2312.10332) ![EACL 2024 Workshop](https://img.shields.io/badge/EACL%202024%20Workshop-blue)

<a name="multi-label-classification-mlc"></a>
#####  Multi-Label Classification (MLC)

* (2024-09) [Efficient and Scalable Estimation of Tool Representations in Vector Space](https://arxiv.org/abs/2409.02141) [![Star](https://img.shields.io/github/stars/SqueezeAILab/Tool2Vec.svg?style=social&label=Star)](https://github.com/SqueezeAILab/Tool2Vec)
* (2024-09) [TinyAgent: Function Calling at the Edge](https://arxiv.org/abs/2409.00608) ![EMNLP 2024 Demo](https://img.shields.io/badge/EMNLP%202024%20Demo-blue)  [![Star](https://img.shields.io/github/stars/SqueezeAILab/TinyAgent.svg?style=social&label=Star)](https://github.com/SqueezeAILab/TinyAgent)

<a name="vocabulary-based-retrieval"></a>
#####  Vocabulary-based Retrieval

* (2025-03) [Chain-of-Tools: Utilizing Massive Unseen Tools in the CoT Reasoning of Frozen Language Models](https://arxiv.org/abs/2503.16779)  [![Star](https://img.shields.io/github/stars/fairyshine/Chain-of-Tools.svg?style=social&label=Star)](https://github.com/fairyshine/Chain-of-Tools)
* (2024-10) [Toolken+: Improving LLM Tool Usage with Reranking and a Reject Option](https://arxiv.org/abs/2410.12004) ![EMNLP 2024 Findings](https://img.shields.io/badge/EMNLP%202024%20Findings-blue)
* (2024-10) [ToolGen: Unified Tool Retrieval and Calling via Generation](https://arxiv.org/abs/2410.03439) ![ICLR](https://img.shields.io/badge/ICLR%202025-blue) [![Star](https://img.shields.io/github/stars/Reason-Wang/ToolGen.svg?style=social&label=Star)](https://github.com/Reason-Wang/ToolGen)
* (2024-07) [Concise and Precise Context Compression for Tool-Using Language Models](https://arxiv.org/abs/2407.02043) ![ACL 2024 Findings](https://img.shields.io/badge/ACL%202024%20Findings-blue) 
* (2023-05) [ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings](https://arxiv.org/abs/2305.11554) ![NeurIPS](https://img.shields.io/badge/NeurIPS%202023%20oral-blue) [![Star](https://img.shields.io/github/stars/Ber666/ToolkenGPT.svg?style=social&label=Star)](https://github.com/Ber666/ToolkenGPT)

<a name="tool-calling"></a>

#### Tool Calling

<a name="in-place-parameter-filling"></a>
#####  In-Place Parameter Filling

* (2024-01) [Efficient Tool Use with Chain-of-Abstraction Reasoning](https://arxiv.org/abs/2401.17464) ![COLING 2025](https://img.shields.io/badge/COLING%202025-blue)
* (2023-02) [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) ![NeurIPS](https://img.shields.io/badge/NeurIPS%202023%20oral-blue)

<a name="parallel-tool-calling"></a>

#####  Parallel Tool Calling

* (2024-11) [CATP-LLM: Empowering Large Language Models for Cost-Aware Tool Planning](https://arxiv.org/abs/2411.16313) ![ICCV 2025](https://img.shields.io/badge/ICCV%202025-blue) [![Star](https://img.shields.io/github/stars/duowuyms/OpenCATP-LLM.svg?style=social&label=Star)](https://github.com/duowuyms/OpenCATP-LLM)
* (2024-05) [An LLM-Tool Compiler for Fused Parallel Function Calling](https://arxiv.org/abs/2405.17438)
* (2023-12) [An LLM Compiler for Parallel Function Calling](https://arxiv.org/abs/2312.04511) ![ICML 2024](https://img.shields.io/badge/ICML%202024-blue) [![Star](https://img.shields.io/github/stars/SqueezeAILab/LLMCompiler.svg?style=social&label=Star)](https://github.com/SqueezeAILab/LLMCompiler)

<a name="cost-aware-tool-calling"></a>

#####  Cost-Aware Tool Calling

* (2025-07) [A Joint Optimization Framework for Enhancing Efficiency of Tool Utilization in LLM Agents](https://aclanthology.org/2025.findings-acl.1149/) ![ACL 2025 Findings](https://img.shields.io/badge/ACL%202025%20Findings-blue) [![Star](https://img.shields.io/github/stars/Bingo-W/ToolOptimization.svg?style=social&label=Star)](https://github.com/Bingo-W/ToolOptimization)
* (2025-05) [Distilling LLM Agent into Small Models with Retrieval and Code Tools](https://arxiv.org/abs/2505.17612) [![Star](https://img.shields.io/github/stars/Nardien/agent-distillation.svg?style=social&label=Star)](https://github.com/Nardien/agent-distillation)
* (2025-03) [Alignment for Efficient Tool Calling of Large Language Models](https://arxiv.org/abs/2503.06708) ![EMNLP 2025](https://img.shields.io/badge/EMNLP%202025-blue)
* (2025-02) [ToolCoder: A Systematic Code-Empowered Tool Learning Framework for Large Language Models](https://arxiv.org/abs/2502.11404) ![ACL 2025](https://img.shields.io/badge/ACL%202025-blue) [![Star](https://img.shields.io/github/stars/dhx20150812/toolcoder.svg?style=social&label=Star)](https://github.com/dhx20150812/toolcoder)
* (2024-02) [Budget-Constrained Tool Learning with Planning](https://arxiv.org/abs/2402.15960) ![ACL 2024 Findings](https://img.shields.io/badge/ACL%202024%20Findings-blue) [![Star](https://img.shields.io/github/stars/THUNLP-MT/BTP.svg?style=social&label=Star)](https://github.com/THUNLP-MT/BTP)
* (2024-01) [TroVE: Inducing Verifiable and Efficient Toolboxes for Solving Programmatic Tasks](https://arxiv.org/abs/2401.12869) ![ICML 2024](https://img.shields.io/badge/ICML%202024-blue) [![Star](https://img.shields.io/github/stars/zorazrw/trove.svg?style=social&label=Star)](https://github.com/zorazrw/trove)

<a name="efficient-test-time-scaling"></a>

#####  Efficient Test-Time Scaling

* (2024-09) [ToolPlanner: A Tool Augmented LLM for Multi Granularity Instructions with Path Planning and Feedback](https://arxiv.org/abs/2409.14826) ![EMNLP 2024](https://img.shields.io/badge/EMNLP%202024-blue) [![Star](https://img.shields.io/github/stars/XiaoMi/toolplanner.svg?style=social&label=Star)](https://github.com/XiaoMi/toolplanner)
* (2023-10) [ToolChain\*: Efficient Action Space Navigation in Large Language Models with A\* Search](https://arxiv.org/abs/2310.13227) ![ICLR](https://img.shields.io/badge/ICLR%202024-blue) 

<a name="efficient-tool-calling-with-post-training"></a>

#####  Efficient Tool Calling with Post-training

* (2025-11) [ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration](https://arxiv.org/abs/2511.21689)  [![Website](https://img.shields.io/badge/Website-Project-green)](https://research.nvidia.com/labs/lpr/ToolOrchestra/) [![Star](https://img.shields.io/github/stars/NVlabs/ToolOrchestra.svg?style=social&label=Star)](https://github.com/NVlabs/ToolOrchestra)
* (2025-09) [ToolRM: Outcome Reward Models for Tool-Calling Large Language Models](https://arxiv.org/abs/2509.11963)
* (2025-04) [Acting Less is Reasoning More! Teaching Model to Act Efficiently](https://arxiv.org/abs/2504.14870)

<a name="tool-integrated-reasoning-tir"></a>
#### Tool-Integrated Reasoning (TIR)

<a name="selective-invocation"></a>

#####  Selective Invocation

* (2025-09) [TableMind: An Autonomous Programmatic Agent for Tool-Augmented Table Reasoning](https://arxiv.org/abs/2509.06278) ![WSDM 2026](https://img.shields.io/badge/WSDM%202026-blue) [![Star](https://img.shields.io/github/stars/lennendd/TableMind.svg?style=social&label=Star)](https://github.com/lennendd/TableMind)
* (2025-02) [SMART: Self-Aware Agent for Tool Overuse Mitigation](https://arxiv.org/abs/2502.11435) ![ACL 2025 Findings](https://img.shields.io/badge/ACL%202025%20Findings-blue) [![Star](https://img.shields.io/github/stars/qiancheng0/Open-SMARTAgent.svg?style=social&label=Star)](https://github.com/qiancheng0/Open-SMARTAgent)
* (2024-03) [Agent-FLAN: Designing Data and Methods of Effective Agent Tuning for Large Language Models](https://arxiv.org/abs/2403.12881) ![ACL 2024 Findings](https://img.shields.io/badge/ACL%202024%20Findings-blue) [![Website](https://img.shields.io/badge/Website-Project-green)](https://internlm.github.io/Agent-FLAN/)  [![Star](https://img.shields.io/github/stars/InternLM/Agent-FLAN.svg?style=social&label=Star)](https://github.com/InternLM/Agent-FLAN)

<a name="cost-aware-policy-optimization"></a>

#####  Cost-Aware Policy Optimization

* (2025-10) [PORTool: Tool-Use LLM Training with Rewarded Tree](https://arxiv.org/abs/2510.26020)
* (2025-10) [A$^2$FM: An Adaptive Agent Foundation Model for Tool-Aware Hybrid Reasoning](https://arxiv.org/abs/2510.12838) [![Website](https://img.shields.io/badge/Website-Project-green)](https://chain-of-agents-afm.github.io/) [![Star](https://img.shields.io/github/stars/OPPO-PersonalAI/Adaptive_Agent_Foundation_Models.svg?style=social&label=Star)](https://github.com/OPPO-PersonalAI/Adaptive_Agent_Foundation_Models)
* (2025-09) [TableMind: An Autonomous Programmatic Agent for Tool-Augmented Table Reasoning](https://arxiv.org/abs/2509.06278) ![WSDM 2026](https://img.shields.io/badge/WSDM%202026-blue) [![Star](https://img.shields.io/github/stars/lennendd/TableMind.svg?style=social&label=Star)](https://github.com/lennendd/TableMind)
* (2025-07) [AutoTIR: Autonomous Tools Integrated Reasoning via Reinforcement Learning](https://arxiv.org/abs/2507.21836) [![Star](https://img.shields.io/github/stars/weiyifan1023/AutoTIR.svg?style=social&label=Star)](https://github.com/weiyifan1023/AutoTIR)
* (2025-05) [Reinforced Internal-External Knowledge Synergistic Reasoning for Efficient Adaptive Search Agent](https://arxiv.org/abs/2505.07596) [![Star](https://img.shields.io/github/stars/hzy312/knowledge-r1.svg?style=social&label=Star)](https://github.com/hzy312/knowledge-r1)
* (2025-05) [Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning](https://arxiv.org/abs/2505.01441)
* (2025-04) [ToolRL: Reward is All Tool Learning Needs](https://arxiv.org/abs/2504.13958) ![NeurIPS 2025](https://img.shields.io/badge/NeurIPS%202025-blue) [![Star](https://img.shields.io/github/stars/qiancheng0/ToolRL.svg?style=social&label=Star)](https://github.com/qiancheng0/ToolRL)
* (2025-04) [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536) [![Website](https://img.shields.io/badge/Website-Project-green)](https://retool-rl.github.io/) [![Star](https://img.shields.io/github/stars/ReTool-RL/ReTool.svg?style=social&label=Star)](https://github.com/ReTool-RL/ReTool)
* (2025-04) [Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use](https://arxiv.org/abs/2504.04736) ![COLM 2025](https://img.shields.io/badge/COLM%202025-blue)
* (2025-04) [Acting Less is Reasoning More! Teaching Model to Act Efficiently](https://arxiv.org/abs/2504.14870)


<a name="planning"></a>
### 🧩Planning

![planning](assets/planning.png)

<a name="single-agent-planning-efficiency"></a>
#### Single-Agent Planning Efficiency

<a name="adaptive-budgeting-and-control"></a>
#####  Adaptive Budgeting and Control

* (2025-11) [Budget-Aware Tool-Use Enables Effective Agent Scaling](https://arxiv.org/abs/2511.17006)
* (2025-09) [Learning When to Plan: Efficiently Allocating Test-Time Compute for LLM Agents](https://arxiv.org/abs/2509.03581) 
* (2023-12) [ReST meets ReAct: Self-Improvement for Multi-Step Reasoning LLM Agent](https://arxiv.org/abs/2312.10003) ![ICLR 2024 Workshop](https://img.shields.io/badge/ICLR%202024%20Workshop-blue) 
* (2023-05) [SwiftSage: A Generative Agent with Fast and Slow Thinking for Complex Interactive Tasks](https://arxiv.org/abs/2305.17390) ![NeurIPS](https://img.shields.io/badge/NeurIPS%202023%20spotlight-blue) [![Website](https://img.shields.io/badge/Website-Project-green)](https://swiftsage.github.io/)  [![Star](https://img.shields.io/github/stars/SwiftSage/SwiftSage.svg?style=social&label=Star)](https://github.com/SwiftSage/SwiftSage)
* (2023-03) [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) ![NeurIPS](https://img.shields.io/badge/NeurIPS%202023-blue) [![Star](https://img.shields.io/github/stars/noahshinn/reflexion.svg?style=social&label=Star)](https://github.com/noahshinn/reflexion)


<a name="structured-search-strategies"></a>

#####  Structured Search

* (2025-05) [Cost-Augmented Monte Carlo Tree Search for LLM-Assisted Planning](https://arxiv.org/abs/2505.14656)
* (2023-12) [ProTIP: Progressive Tool Retrieval Improves Planning](https://arxiv.org/abs/2312.10332)
* (2023-10) [ToolChain\*: Efficient Action Space Navigation in Large Language Models with A\* Search](https://arxiv.org/abs/2310.13227) ![ICLR 2024](https://img.shields.io/badge/ICLR%202024%20poster-blue)
* (2023-10) [Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models](https://arxiv.org/abs/2310.04406) ![ICML 2024](https://img.shields.io/badge/ICML%202024-blue) [![Star](https://img.shields.io/github/stars/lapisrocks/LanguageAgentTreeSearch.svg?style=social&label=Star)](https://github.com/lapisrocks/LanguageAgentTreeSearch)

<a name="task-decomposition-and-routing"></a>

#####  Task Decomposition

* (2025-05) [Alita: Generalist Agent Enabling Scalable Agentic Reasoning with Minimal Predefinition and Maximal Self-Evolution](https://arxiv.org/abs/2505.20286)  [![Star](https://img.shields.io/github/stars/CharlesQ9/Alita.svg?style=social&label=Star)](https://github.com/CharlesQ9/Alita)
* (2025-03) [ReSo: A Reward-driven Self-organizing LLM-based Multi-Agent System for Reasoning Tasks](https://arxiv.org/abs/2503.02390) ![EMNLP 2025](https://img.shields.io/badge/EMNLP%202025-blue) [![Star](https://img.shields.io/github/stars/hengzzzhou/ReSo.svg?style=social&label=Star)](https://github.com/hengzzzhou/ReSo)
* (2024-11) [BudgetMLAgent: A Cost-Effective LLM Multi-Agent system for Automating Machine Learning Tasks](https://arxiv.org/abs/2411.07464) ![AIMLSystems 2024](https://img.shields.io/badge/AIMLSystems%202024-blue)
* (2024-02) [AutoGPT+P: Affordance-based Task Planning with Large Language Models](https://arxiv.org/abs/2402.10778) [![Star](https://img.shields.io/gitlab/stars/874?gitlab_url=https%3A%2F%2Fgit.h2t.iar.kit.edu&style=social&label=Star)](https://git.h2t.iar.kit.edu/birr/autogpt-p-standalone)
* (2023-05) [ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models](https://arxiv.org/abs/2305.18323)  [![Star](https://img.shields.io/github/stars/billxbf/ReWOO.svg?style=social&label=Star)](https://github.com/billxbf/ReWOO)
* (2023-03) [HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face](https://arxiv.org/abs/2303.17580) ![NeurIPS 2023](https://img.shields.io/badge/NeurIPS%202023-blue) [![Star](https://img.shields.io/github/stars/microsoft/JARVIS.svg?style=social&label=Star)](https://github.com/microsoft/JARVIS)

<a name="policy-optimization"></a>

#####  Policy Optimization

* (2025-09) [Planner-R1: Reward Shaping Enables Efficient Agentic RL with Smaller LLMs](https://arxiv.org/abs/2509.25779)
* (2025-08) [Encouraging Good Processes Without the Need for Good Answers: Reinforcement Learning for LLM Agent Planning](https://arxiv.org/abs/2508.19598) ![EMNLP 2025 Industry](https://img.shields.io/badge/EMNLP%202025%20Industry-blue)
* (2025-05) [Planning without Search: Refining Frontier LLMs with Offline Goal-Conditioned RL](https://arxiv.org/abs/2505.18098) ![NeurIPS 2025](https://img.shields.io/badge/NeurIPS%202025-blue) [![Website](https://img.shields.io/badge/Website-Project-green)](https://jxihong.github.io/pnlc_website/)
* (2025-02) [QLASS: Boosting Language Agent Inference via Q-Guided Stepwise Search](https://arxiv.org/abs/2502.02584) ![ICML 2025](https://img.shields.io/badge/ICML%202025-blue) [![Star](https://img.shields.io/github/stars/Rafa-zy/QLASS.svg?style=social&label=Star)](https://github.com/Rafa-zy/QLASS)
* (2024-03) [Trial and Error: Exploration-Based Trajectory Optimization for LLM Agents](https://arxiv.org/abs/2403.02502) ![ACL 2024](https://img.shields.io/badge/ACL%202024-blue) [![Star](https://img.shields.io/github/stars/Yifan-Song793/ETO.svg?style=social&label=Star)](https://github.com/Yifan-Song793/ETO)


<a name="architectural-synergy-with-memory-and-tools"></a>

#####   Memory and Skill Acquisition

* (2025-10) [GAP: Graph-Based Agent Planning with Parallel Tool Use and Reinforcement Learning](https://arxiv.org/abs/2510.25320)  [![Star](https://img.shields.io/github/stars/WJQ7777/Graph-Agent-Planning.svg?style=social&label=Star)](https://github.com/WJQ7777/Graph-Agent-Planning)
* (2024-07) [Sibyl: Simple yet Effective Agent Framework for Complex Real-world Reasoning](https://arxiv.org/abs/2407.10718) [![Star](https://img.shields.io/github/stars/Ag2S1/Sibyl-System.svg?style=social&label=Star)](https://github.com/Ag2S1/Sibyl-System)
* (2024-06) [GraphReader: Building Graph-based Agent to Enhance Long-Context Abilities of Large Language Models](https://arxiv.org/abs/2406.14550) ![EMNLP 2024 Findings](https://img.shields.io/badge/EMNLP%202024%20Findings-blue)
* (2024-02) [Graph-enhanced Large Language Models in Asynchronous Plan Reasoning](https://arxiv.org/abs/2402.02805) ![ICML 2024](https://img.shields.io/badge/ICML%202024-blue) [![Star](https://img.shields.io/github/stars/fangru-lin/graph-llm-asynchow-plan.svg?style=social&label=Star)](https://github.com/fangru-lin/graph-llm-asynchow-plan)
* (2023-05) [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291) ![TMLR 2024](https://img.shields.io/badge/TMLR%202024-blue) [![Website](https://img.shields.io/badge/Website-Project-green)](https://voyager.minedojo.org/) [![Star](https://img.shields.io/github/stars/MineDojo/Voyager.svg?style=social&label=Star)](https://github.com/MineDojo/Voyager)


<a name="multi-agent-collaborative-efficiency"></a>
#### Multi-Agent Collaborative Efficiency

<a name="topological-efficiency"></a>

#####  Topological Efficiency and Sparsification

* (2025-09) [MARS: toward more efficient multi-agent collaboration for LLM reasoning](https://arxiv.org/abs/2509.20502)  [![Star](https://img.shields.io/github/stars/xwang97/MARS.svg?style=social&label=Star)](https://github.com/xwang97/MARS)
* (2025-08) [SafeSieve: From Heuristics to Experience in Progressive Pruning for LLM-based Multi-Agent Communication](https://arxiv.org/abs/2508.11733) ![AAAI 2026](https://img.shields.io/badge/AAAI%202026-blue) [![Star](https://img.shields.io/github/stars/csgen/SafeSieve.svg?style=social&label=Star)](https://github.com/csgen/SafeSieve)
* (2025-03) [AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration](https://arxiv.org/abs/2503.18891) ![ACL 2025](https://img.shields.io/badge/ACL%202025-blue) [![Star](https://img.shields.io/github/stars/wangzx1219/AgentDropout.svg?style=social&label=Star)](https://github.com/wangzx1219/AgentDropout)
* (2025-02) [S$^2$-MAD: Breaking the Token Barrier to Enhance Multi-Agent Debate Efficiency](https://arxiv.org/abs/2502.04790) ![NAACL 2025](https://img.shields.io/badge/NAACL%202025-blue)
* (2024-10) [Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems](https://arxiv.org/abs/2410.02506) ![ICLR 2025](https://img.shields.io/badge/ICLR%202025-blue) [![Star](https://img.shields.io/github/stars/yanweiyue/AgentPrune.svg?style=social&label=Star)](https://github.com/yanweiyue/AgentPrune)
* (2024-09) [GroupDebate: Enhancing the Efficiency of Multi-Agent Debate Using Group Discussion](https://arxiv.org/abs/2409.14051) 
* (2024-06) [Scaling Large Language Model-based Multi-Agent Collaboration](https://arxiv.org/abs/2406.07155) ![ICLR 2025](https://img.shields.io/badge/ICLR%202025-blue) [![Star](https://img.shields.io/github/stars/OpenBMB/ChatDev.svg?style=social&label=Star)](https://github.com/OpenBMB/ChatDev) 
* (2024-06) [Chain of Agents: Large Language Models Collaborating on Long-Context Tasks](https://arxiv.org/abs/2406.02818) ![NeurIPS 2024](https://img.shields.io/badge/NeurIPS%202024-blue)


<a name="protocol-and-context-optimization"></a>

#####  Protocol and Context Optimization

* (2025-10) [Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems](https://arxiv.org/abs/2510.26585) 
* (2025-09) [Free-MAD: Consensus-Free Multi-Agent Debate](https://arxiv.org/abs/2509.11035)
* (2025-07) [CONSENSAGENT: Towards Efficient and Effective Consensus in Multi-Agent LLM Interactions Through Sycophancy Mitigation](https://aclanthology.org/2025.findings-acl.1141/) ![ACL 2025 Findings](https://img.shields.io/badge/ACL%202025%20Findings-blue)
* (2025-07) [CodeAgents: A Token-Efficient Framework for Codified Multi-Agent Reasoning in LLMs](https://arxiv.org/abs/2507.03254)  [![Code](https://img.shields.io/badge/Code-Anonymous-lightgrey?style=social&logo=github&logoColor=black)](https://anonymous.4open.science/r/CodifyingAgent-5A86)
* (2024-05) [Smurfs: Multi-Agent System using Context-Efficient DFSDT for Tool Planning](https://arxiv.org/abs/2405.05955) ![NAACL 2025](https://img.shields.io/badge/NAACL%202025-blue) [![Star](https://img.shields.io/github/stars/FreedomIntelligence/Smurfs.svg?style=social&label=Star)](https://github.com/FreedomIntelligence/Smurfs)


<a name="distilling-coordination-into-planning"></a>

#####  Distilling Coordination into Planning

* (2025-11) [SMAGDi: Socratic Multi Agent Interaction Graph Distillation for Efficient High Accuracy Reasoning](https://arxiv.org/abs/2511.05528) ![MTI-LLM @ NeurIPS 2025 Workshop](https://img.shields.io/badge/NeurIPS%202025%20Workshop-blue)
* (2025-06) [Debate, Reflect, and Distill: Multi-Agent Feedback with Tree-Structured Preference Optimization for Efficient Language Model Enhancement](https://arxiv.org/abs/2506.03541) ![ACL 2025 Findings](https://img.shields.io/badge/ACL%202025%20Findings-blue) [![Star](https://img.shields.io/github/stars/zhouxiaofengshelf/D-R.svg?style=social&label=Star)](https://github.com/zhouxiaofengshelf/D-R)
* (2024-02) [MAGDi: Structured Distillation of Multi-Agent Interaction Graphs Improves Reasoning in Smaller Language Models](https://arxiv.org/abs/2402.01620) ![ICML 2024](https://img.shields.io/badge/ICML%202024-blue) [![Star](https://img.shields.io/github/stars/dinobby/MAGDi.svg?style=social&label=Star)](https://github.com/dinobby/MAGDi)

## 📌Citation
If you find this survey useful, please cite:
```bibtex
@misc{yang2026efficientagentsmemorytool,
      title={Toward Efficient Agents: Memory, Tool learning, and Planning}, 
      author={Xiaofang Yang and Lijun Li and Heng Zhou and Tong Zhu and Xiaoye Qu and Yuchen Fan and Qianshan Wei and Rui Ye and Li Kang and Yiran Qin and Zhiqiang Kou and Daizong Liu and Qi Li and Ning Ding and Siheng Chen and Jing Shao},
      year={2026},
      eprint={2601.14192},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.14192}, 
}
```

