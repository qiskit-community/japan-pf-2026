# 量子プログラミング実践講座 | Japan Practitioners Forum 2026

## Table of Content

- [プログラム | Program](#プログラム--program-)
    - [Day 1](#day-1-feb-2-2026-mon)
    - [Day 2](#day-2-feb-3-2026-tue)
- [Optimization Functions Challenges](#optimization-functions-challenges)
    - [Setup](#setup)
    - [Challenges](#challenges)

## プログラム | Program ：
### Day 1: Feb 2, 2026 (Mon)

| Time        | Description                                                                                      | Speaker                                       | Materials |
|-------------|--------------------------------------------------------------------------------------------------|-----------------------------------------------|-----------|
| 9:00–9:30   | Registration / 受付                                                                                |                                               |           |
| 9:30–9:45   | Opening — "Scaling for Quantum Advantage and Beyond"                                             | Hiroshi Horii; Antonio Corcoles (IBM Quantum) |           |
| 9:45–10:45  | Qiskit Updates and Live Demo                                                           | Matthew Treinish (IBM Quantum) | [presentation](./qiskit_updates/Qiskit_Updates.pdf) [notebook](./qiskit_updates/C_api_interactive.ipynb) |
|             | Qiskit FTQC Updates                                                           | Julien Gacon (IBM Quantum)  | [presentation](./qiskit_updates/qiskit_ftqc.pdf)        |
| 10:45–11:00 | Break                                                                                            |                                               |           |
| 11:00–12:30 | Dynamic Circuits and Hands-on Demo: Simulation of Kicked Ising Hamiltonian with Dynamic Circuits | Mirko Amico (IBM Quantum)                     | [presentation](./dynamic-circuits/utility-scale-dynamic-circuits.pdf) [notebook](./dynamic-circuits/dc_hex_ising.ipynb)      |
| 12:30–13:30 | Lunch Break (A Bento Box Will Be Served / お弁当を用意しております)                                |                                               |           |
| 13:30–14:30 | Areas of Advantage I: Optimization Landscape and Capabilities                                    | Elena Peña Tapia (IBM Quantum)                | [notebooks](./opt_landscape_capabilities/)          |
| 14:30–15:30 | Areas of Advantage II: Hamiltonian Simulation                                                    | Yuri Kobayashi (IBM Quantum)                  | [presentation](./hamiltonian_simulation/hamiltonian_simulation_overview.pdf) [notebook](./hamiltonian_simulation/hamiltonian-simulation.ipynb)         |
| 15:30–16:00 | Coffee Break                                                                                     |                                               |           |
| 16:00–17:00 | Areas of Advantage III: Machine Learning                                                         | Hiroshi Yamauchi (Softbank)                   |           |
| 17:00–17:30 | Qiskit Functions Challenge Setup                                                                 | Junye Huang; Elena Peña Tapia (IBM Quantum)                                              | [notebook](./challenges/save_account&check_functions_access.ipynb)          |
| 18:00–20:00 | Banquet / 懇親会                                                                                   |                                               |           |

### Day 2: Feb 3, 2026 (Tue)

| Time        | Description                                                                                                                      | Speaker                                     | Materials |
|-------------|----------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------|-----------|
| 9:00–9:30   | Registration / 受付                                                                                                                |                                             |           |
| 9:30–10:30  | Track A: Solving Optimization Problems with Qiskit Functions                                                                     | Junye Huang; Elena Peña Tapia (IBM Quantum) | [notebooks](/challenges/)          |
|             | Track B: Exploring Samplomatic and Advanced Usage (Layering in Advanced Classical Mitigation)                                    | Ian Hincks (IBM Quantum)                    | [presentation](./hello-samplomatic/slides.pdf) [notebook](./hello-samplomatic/hello-samplomatic.ipynb)          |
| 10:30–12:00 | Track A: Solving Optimization Problems with Qiskit Functions (continued)                                                         | Junye Huang; Elena Peña Tapia (IBM Quantum) | [notebooks](/challenges/)          |
|             | Track B: Advanced Error Mitigation Techniques using Samplomatic, Samplex, and Executor Frameworks (SLC (Evolution of PEC), PNA, ZNE)     | Nate Earnest-Noble (IBM Quantum)            | [presentation](./Error%20Mitigation%20Techniques/PFJapan_EM_SLC_PNA.pdf) [notebooks](./Error%20Mitigation%20Techniques/)       |
| 12:00–13:00 | Lunch Break (A Bento Box Will Be Served / お弁当を用意しております)                                                                |                                             |           |
| 13:00–15:00 | Quantum Clinic / 量子コン・クリニック(I) — Consult with IBM Experts to Help Solve Specific Challenges in Current Research Projects | IBM Experts                                 |           |
| 15:00–15:30 | Coffee Break                                                                                                                     |                                             |           |
| 15:30–17:30 | Quantum Clinic / 量子コン・クリニック(II) — Consult with IBM Experts to Help Solve Specific Challenges in Current Research Projects         | IBM Experts                                 |           |
| 17:30       | Closing                                                                                                                          |                                             |           |

#### Day 2 午後の量子コン・クリニックについて

2/3(火) Day 2 午後は「量子コン・クリニック」と題しまして、現在取り組まれている研究課題をIBM Quantumの量子コンピューターを使って解くにあたってのお困りごとやご相談を受け付けます。

## Optimization Functions Challenges

### Setup

- Installation instructions [here](./challenges/INSTALL.md)
- Save account and check Qiskit functions access [here](./challenges/save_account&check_functions_access.ipynb).
    - NEDO Challenge participants can request for Qiskit functions access following the instructions [here](./for_nedo_participants/[NEDO]Qiskit%20Functions%20Activate申請方法.pdf).

### Challenges

| Challenge | Qiskit Function Used | Notebooks |
|---|---|---|
| Market Split | [Kipu Iskay Quantum Optimizer](https://quantum.cloud.ibm.com/functions?id=kipu-quantum-iskay-quantum-optimizer) | [EN](./challenges/market-split/market_split_problem.ipynb), [日本語](./challenges/market-split/market_split_problem-ja.ipynb) |
| Maximum Independent Set | [Q-CTRL Optimization Solver](https://quantum.cloud.ibm.com/functions?id=q-ctrl-optimization-solver) | [EN](./challenges/maximum-independent-set/maximum-independent-set.ipynb), [日本語](./challenges/maximum-independent-set/maximum-independent-set-ja.ipynb) |
| Grid Stability Classification | [Multiverse Singularity](https://quantum.cloud.ibm.com/functions?id=multiverse-singularity) | [EN](./challenges/multiverse-grid-stability-classification/singularity_qiskit_exercises.ipynb), [日本語](./challenges/multiverse-grid-stability-classification/singularity_qiskit_exercises_ja.ipynb) |
| Quantum Multiverse Objective Optimization | No Qiskit function required | [EN](./challenges/qmoo/qmoo_qdc25.ipynb), [日本語](./challenges/qmoo/qmoo_qdc25_ja.ipynb) |

You can find some tips for solving the challenges in the [cheatsheet](./challenges/challenges_cheat_sheet.md).
