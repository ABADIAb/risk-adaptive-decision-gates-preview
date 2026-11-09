# MultiAgentON: LLM-Assisted Risk-Adaptive Neurosymbolic Intent Planning for Optical Networks

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Orchestration](https://img.shields.io/badge/Framework-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Tests](https://img.shields.io/badge/Tests-364%20passed%20(Strict%20TDD)-brightgreen.svg)](#experimental-validation)
[![Safety Invariant](https://img.shields.io/badge/Safety%20Invariant-UAR%20%3D%200.0%25-success.svg)](#experimental-validation)
[![Topology](https://img.shields.io/badge/Topology-17--Node%20Nobel--Germany-informational.svg)](#network-environment)
[![Status](https://img.shields.io/badge/Status-Codebase%20Under%20Academic%20Embargo-red.svg)](#academic-embargo--codebase-access)

> **Master's Thesis Research Project**  
> *Politecnico di Milano — Software-Defined Optical Networks (SDON) Research Laboratory*  
> **Topic:** A Pre-Deployment Decision Mechanism with Joint Semantic and Quality of Transmission (QoT) Assessment for Intent-Based Optical Networks (IBON).

---

> [!IMPORTANT]
> ## Academic Embargo & Codebase Access
> 
> This repository is the public project overview and architectural showcase for the Master's Thesis: **"LLM-Assisted Risk-Adaptive Neurosymbolic Intent Planning for Optical Networks: A Pre-Deployment Decision Mechanism with Joint Semantic and QoT Assessment"** developed at **Politecnico di Milano (DEIB)**.
> 
> **The production implementation codebase is currently hosted in a private repository under academic embargo** pending journal publication and thesis defense confidentiality agreements.
> 
> ### Requesting Full Codebase & Replication Package Access
> If you are an academic researcher, peer reviewer, or industry practitioner interested in inspecting the source code, replicating the benchmarks, or reviewing technical artifacts under NDA:
> 
> - **Author:** Felipe Abadía
> - **Email:** [felipeabadia01@gmail.com](mailto:felipeabadia01@gmail.com)
> - **Institution:** Politecnico di Milano — Dipartimento di Elettronica, Informazione e Bioingegneria (DEIB)
> - **GitHub Profile:** [@ABADIAb](https://github.com/ABADIAb)
> - **Location:** Milan, Italy

---

## Executive Summary

Translating high-level operator intents into physical optical network configurations using Large Language Models (LLMs) suffers from three fatal vulnerabilities:

1. **Physical Hallucinations:** LLMs cannot calculate nonlinear fiber impairments (e.g., Generalized Signal-to-Noise Ratio [GSNR], Non-Linear Interference [NLI], and Stimulated Raman Scattering). Allowing an LLM to heuristically assign optical lightpaths produces physically unfeasible deployments.
2. **Context Window Saturation:** Injecting raw optical network state (massive RESTConf JSON payloads) causes severe attention degradation (*"lost-in-the-middle"*), inflating inference costs and token latency.
3. **Suboptimal Human-in-the-Loop (HITL) Interventions:** Existing paradigms oscillate between *always-on HITL* (operator fatigue, operational bottlenecks) and *blind autonomous execution* (high risk of unfeasible physical deployments).

**MultiAgentON** introduces a **pre-deployment, fail-fast neurosymbolic pipeline** governed by **Risk-Adaptive Decision Gates (RADGs)**. Enforcing the strict principle **"LLMs reason, deterministic tools calculate"**, the system evaluates semantic uncertainty ($U_{sem}$) and physical feasibility ($QoT_{valid}$) sequentially *before* any configuration touches the optical network controller.

---

## Architecture Overview

The system decouples natural language reasoning from deterministic physical simulation through a 7-phase LangGraph pipeline:

![Conceptual Framework](docs/figures/conceptual_framework.png)

### The 7-Phase Fail-Fast Pipeline

| Phase | Pipeline Stage | Description | Core Technology |
|---|---|---|---|
| **Phase 1** | **Intent Ingestion & Optical RAG** | Ingests natural language operator intent and dynamically extracts localized $k$-hop subtopologies via Mock GraphRAG to bound LLM context. | NetworkX, LangChain |
| **Phase 2** | **PDDL Translation & Grammar Check** | Translates natural language into formal Planning Domain Definition Language (PDDL) constraints, validated by a deterministic Context-Free Grammar (CFG) parser ($v_{struct} \in \{0, 1\}$). | Custom CFG AST Validator |
| **Phase 3** | **Semantic RADG ($U_{sem}$)** | Evaluates semantic uncertainty before any heavy routing or physics calculations. Automatically reconstructs intent via Reverse Prompting without human intervention unless uncertainty exceeds tolerance ($\tau_{sem}$). | Automated Reverse Prompting |
| **Phase 3b** | **HITL Clarification** | Interactive operator disambiguation via LangGraph `interrupt()`, triggered strictly when semantic ambiguity is detected. | LangGraph HITL State |
| **Phase 4** | **Symbolic Solver** | Computes 3–5 candidate lightpaths satisfying topological and path-exclusion constraints. | Yen's K-Shortest Paths (K-SP) |
| **Phase 5** | **QoT Physics Engine** | Evaluates candidate routes using a deterministic physical model, calculating exact GSNR accumulation and received optical power. | C-Band Coherent GN-Model |
| **Phase 6** | **Physical RADG** | Pre-deployment safety gate enforcing binary feasibility ($\text{GSNR} \ge \text{GSNR}_{threshold}$). Auto-approves feasible lightpaths or suggests constraint relaxation replanning. | Mathematical Decision Gate |
| **Phase 7** | **Plan Synthesis & Provisioning** | Generates auditable planning reports with full decision traces and provisions lightpaths via RESTConf testbed APIs. | RESTConf, Rich Report Generator |

---

## The Risk-Adaptive Decision Gates (RADGs)

The RADG mechanism formalizes pre-deployment safety as a piecewise decision function $D(U_{sem}, \text{QoT}_{valid})$ mapping to a ternary action space $\mathcal{A} = \{\text{approve}, \text{clarify}, \text{replan}\}$:

![RADG Decision Space](docs/figures/radg_decision_space.png)

### 1. Semantic Uncertainty ($U_{sem}$)
Evaluated sequentially before physical path computation:
$$U_{sem} = \begin{cases} 1 & \text{if } v_{struct} = 0 \text{ (Grammar/AST Failure)} \\ d_{sem} & \text{if } v_{struct} = 1 \text{ (Semantic Divergence)} \end{cases}$$

Where $d_{sem} \in [0, 1]$ measures the semantic divergence between the original operator intent and the automated reverse-prompted reconstruction.

### 2. Physical Viability ($QoT_{valid}$)
Evaluated using the coherent Gaussian Noise (GN) physical model:
$$QoT_{valid} = \mathbb{I}(\text{GSNR}_{computed} \ge \text{GSNR}_{threshold} \land P_{rx} \ge P_{rx, min})$$

### 3. Pre-Deployment Decision Rule
$$D(U_{sem}, \text{QoT}_{valid}) = \begin{cases} \text{clarify} & \text{if } U_{sem} > \tau_{sem} \\ \text{replan} & \text{if } U_{sem} \le \tau_{sem} \land \text{QoT}_{valid} = 0 \\ \text{approve} & \text{if } U_{sem} \le \tau_{sem} \land \text{QoT}_{valid} = 1 \end{cases}$$

---

## Network Environment

Evaluations are conducted on the standardized **17-Node Nobel-Germany Optical Backbone Network**:

![17-Node Nobel-Germany Topology](docs/figures/17_node_german.png)

- **Topology Specifications:** 17 ROADM nodes, 26 bidirectional fiber links.
- **Physical Channel Model:** Standard Single-Mode Fiber (SSMF, $\alpha = 0.2\text{ dB/km}$, $\gamma = 1.3\text{ W}^{-1}\text{km}^{-1}$, $D = 16.7\text{ ps/(nm}\cdot\text{km)}$).
- **Amplification:** Inline EDFAs with $6.0\text{ dB}$ noise figure and total gain compensation.
- **Spectrum Grid:** C-Band 96 channels, $100\text{ GHz}$ channel spacing, 100G/400G coherent modulation.

---

## Experimental Validation

The architecture was evaluated against **three operational baselines** across a 4-class balanced evaluation corpus (Nominal, Semantically Ambiguous, Physically Infeasible, and Adversarial intents):

1. **Proposed RADG (V5):** Two-gate fail-fast neurosymbolic pipeline.
2. **Always-On HITL:** Paranoid baseline requiring human review on every single intent turn.
3. **LLM-Only Baseline:** Blind LLM execution without semantic validation gates (simulating reactive post-deployment controller failure).

### Four Core Validation Pillars

![Comparative Efficiency Pillars](docs/figures/comparative_efficiency_pillars.png)

![Cross-Model Gate Accuracy](docs/figures/cross_model_gate_accuracy.png)

### Key Empirical Findings

- **Zero Unsafe Approvals ($UAR = 0.0\%$):** Strict mathematical safety invariant. Infeasible physical configurations never reach the optical network controller.
- **100% Gate Decision Accuracy (GDA):** Both Semantic and Physical RADGs correctly classify 100% of nominal, ambiguous, and physically infeasible intents across evaluated Small Language Models (SLMs) and Frontier LLMs.
- **80% Reduction in Operator Fatigue (HIC):** By autonomously approving nominal intents and filtering trivial passes, RADG cuts human interventions from 100% (Always-On) to 20% without compromising safety.
- **Token Economy:** Context Bounding via Mock GraphRAG reduces prompt token consumption by $>90\%$ compared to full RESTConf JSON topology injection.

---

## Technology Stack

- **Pipeline Orchestration:** LangGraph (StateGraph, checkpointer persistence, `interrupt()` HITL primitives).
- **Language Models Supported:** Local Open-Weights (Ollama: `qwen2.5:3b`, `phi4-mini:latest`, `qwen3:4b`), OpenAI (`gpt-6-luna`), Moonshot / Kimi, OpenRouter.
- **Formal Verification:** Python Context-Free Grammar AST validator for PDDL subsets.
- **Routing & Physics:** Yen's K-Shortest Paths (NetworkX) + C-Band Coherent Gaussian Noise (GN) Model.
- **Software Engineering Standards:** Strict Test-Driven Development (TDD) with 364 automated tests, full Ruff compliance, and modular domain architecture.

---

## Academic Citation

If you reference this architecture or research methodology in your work, please cite:

```bibtex
@mastersthesis{abadia2026riskadaptive,
  author       = {Abadía, Felipe},
  title        = {LLM-Assisted Risk-Adaptive Neurosymbolic Intent Planning for Optical Networks: A Pre-Deployment Decision Mechanism with Joint Semantic and QoT Assessment},
  school       = {Politecnico di Milano},
  year         = {2026},
  address      = {Milan, Italy},
  department   = {Dipartimento di Elettronica, Informazione e Bioingegneria (DEIB)},
  note         = {Software-Defined Optical Networks (SDON) Research Laboratory}
}
```

---

## License

This project documentation and public assets are licensed under the [Apache 2.0 License](LICENSE).
