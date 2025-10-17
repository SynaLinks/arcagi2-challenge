# Synalinks ARCAGI2 challenge

## An evolutionary neuro-symbolic approach with dominated novelty search

This repository present 🧠🔗 [Synalinks](https://github.com/SynaLinks/synalinks), a LM framework for neuro-symbolic applications and an evolutionary optimizer (OMEGA) based on dominated novelty search (DNS).

For this competition we focused on providing a **production-ready system** in order to solve ARC-AGI2 with the aim to be able to transfer the techniques to real-world problems like in biology, algorithmic discovery and more broadly optimization problems and neuro-symbolic architectures.

## What is Synalinks?

Synalinks is a framework to build directed acyclic graphs (DAG) of modules that have trainable variables and operates on JSON data. It is based on Keras 3 architecture and follow the progressinve disclosure of complexity, allowing to build simple pipelines using the functional API while allowing to build more complex ones using a subclassing strategy.

For people already familiar with Keras, here is a mapping from Keras concepts to Synalinks ones:

- Layer -> Module
- Model -> Program
- Tensor -> Data Model (JSON object)
- Tensor shape -> JSON schema
- Trainable Tensor -> Trainable Variable (JSON object)

In this repository, we present a novel approach to LM-based evolutionary optimization inspired by Dominated Novelty Search.

Our approach consist in 2 parts:

- **Pretraining**: This first stage is used to generate as many python programs possible to solve ARC-AGI using the trainset, for this we used the same strategy than Test-Time Adaptation, by creating 1 dataset per task using the one-leave-out strategy to generate multiple trainset samples. Then we train our system on each task individually until solved, the training here involve an evolutionary strategy throught our optimizer (OMEGA) where programs has a self-evolving python script (their variable) to solve the task, for each batch a new variable is generated and evaluated against the validation set, at the end of an epoch, the best candidates are selected (based on Dominated Novelty Search approach) and the training continue until the task is solved or no progression is done (using an EarlyStopping mechanism). During the pre-training, a library of programs is constructed (similar to AlphaEvolve), this library is used during pre-training to find the best seeds per task, allowing to reduce the computation needs, this can be described as a form of neuro-symbolic continuous/transfer learning.

- **Solving**: At this stage, we use the program library to find the best seeds to solve the testset samples and use the same technique than during the pretraining to solve the task.

## What is Dominated Novelty Search?

Dominated Novelty search is a SOTA Quality-Diversity algorithm that introduce a competition function to select solutions candidates based on a distance function. This algorithm allows the system to focus on promising solutions or solutions that explore different approaches. In our work, we use an embedding model to compute the pairwise distance between solutions according to DNS algorithm. This approach allows us to avoid rigid grid structures like in MAP-Elites (which DNS outperform) while avoiding local minima that could arize by just selecting the best solutions over time.

# Install

```
uv venv
source .venv/bin/activate
uv pip install synalinks
```

Add your API keys to `.env.template` and rename it `.env`

# Run

To launch the pretraining, use this command:

```
python arcagi.py pretrain --epochs 10 --patience 5 --concurrency 20
```

To launch the solver, use this command:

```
python arcagi.py solve --epochs 20 --patience 5 --concurrency 20
```

## Tips and warnings

- Start with concurrency 1 to check that everything is fine, then augment it.
- The cost of pretraining is more or less around 1k$ so beware of your LM costs
- You can change the embedding model or language model as you wish (refer to Synalinks documentation)

## References
- [Synalinks: Keras based Neuro-Symbolic LM framework](https://github.com/SynaLinks/synalinks)
- [Dominated Novelty Search: Rethinking Local Competition in Quality-Diversity](https://arxiv.org/pdf/2502.00593)
- [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/pdf/2506.13131)
