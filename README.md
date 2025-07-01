# Reasoning by Superposition  
Official code for [**Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought**](https://arxiv.org/abs/2505.12514)


## Background

Large Language Models (LLMs) have demonstrated remarkable performance in many applications, including challenging reasoning problems via chain-of-thoughts (CoTs) techniques that generate *"thinking tokens"* before answering the questions. While existing theoretical works demonstrate that CoTs with discrete tokens boost the capability of LLMs, recent work on **continuous CoTs** ([Coconut](https://arxiv.org/abs/2412.06769)) lacks a theoretical understanding of why it outperforms discrete counterparts in various reasoning tasks such as **directed graph reachability**, a fundamental graph reasoning problem that includes many practical domain applications as special cases.

In this paper, we prove that a **two-layer transformer with D steps of continuous CoTs** can solve the directed graph reachability problem, where *D* is the diameter of the graph, while the best known result of constant-depth transformers with **discrete CoTs** requires *O(n²)* decoding steps where *n* is the number of vertices (*D < n*). 

In our construction, each continuous thought vector is a **superposition state** that encodes multiple search frontiers simultaneously (i.e., parallel **breadth-first search (BFS)**), while discrete CoTs must choose a single path sampled from the superposition state, which leads to **sequential search** that requires many more steps and may be trapped into local solutions.

We also performed extensive experiments to verify that our theoretical construction aligns well with the empirical solution obtained via training dynamics. Notably, **encoding of multiple search frontiers as a superposition state automatically emerges** in training continuous CoTs, without explicit supervision to guide the model to explore multiple paths simultaneously.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Ber666/reasoning-by-superposition.git
cd reasoning-by-superposition

# 2. Set up environment (conda example)
conda create -n superposition python=3.12
conda activate superposition
pip install -r requirements.txt

# 3. Play with the demo
jupyter lab notebook.ipynb
```

The notebook walks through how Coconut solves a graph reachability problem ([ProsQA](https://arxiv.org/abs/2412.06769)), shows latent-space "search frontiers", and reproduces key figures from the paper—all on one CPU in a few seconds.

## Training the Model Yourself

We provide scripts to reproduce the main setting in this paper, which is to train a 2-layer, 8-head, 768-dim Transformer with [Coconut](https://arxiv.org/abs/2412.06769).

```bash
# Training
torchrun --nnodes 1 --nproc_per_node 2 run.py args/prosqa_coconut_2l_8h_768d.yaml
```

The training will be logged to wandb automatically. When the training is done, please select the checkpoint with the best validation accuracy and run the following evaluation script:

```bash
# Evaluation
torchrun --nnodes 1 --nproc_per_node 2 run.py args/eval_prosqa_coconut_2l_8h_768d.yaml
```

## Citation

```bibtex
@misc{zhu2025reasoning,
  title     = {Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought},
  author    = {Hanlin Zhu and Shibo Hao and Zhiting Hu and Jiantao Jiao and Stuart Russell and Yuandong Tian},
  year      = {2025},
  eprint    = {2505.12514},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

## License

This project is released under the MIT License. See LICENSE for details.