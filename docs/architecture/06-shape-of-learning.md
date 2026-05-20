# Shape-of-Learning Reports

**STATUS:** PROPOSED
**Phase:** v0.1
**Spec source:** TopoGeomML Specification Part II §8

---

## What it is

A shape-of-learning report is a per-epoch or per-step record of the topology of a model's representations during training. The simplest form: a trajectory of β₀ and β₁ of the model's latent space, sampled at fixed intervals, plotted against training step.

Richer forms include persistence-image evolution, descriptor-moment trajectories, and cohort-stratified shape trajectories.

## Why it matters

Loss curves are 1D summaries of a high-dimensional process. They miss everything about *what the model is learning*. Two models with identical loss curves can have radically different shape-of-learning trajectories: one might collapse representations into a low-dim manifold by epoch 5; another might preserve full topology throughout. The downstream behaviour will differ even if the final loss is identical.

For research, the shape-of-learning report is a debugging tool: when does collapse start? Did the regularizer prevent it? For governance, it is part of the evidence bundle: the deployment gate may require that no collapse signal appears within the last 20% of training.

## Core constructs

- `TrainingHook` — adapter that pulls representations from a training loop at sampled steps
- `ShapeReport` — the per-step record: step number, invariants, descriptor stats, cohort splits
- `Trajectory` — sequence of ShapeReports across training
- `Renderer` — produces HTML / JSON / matplotlib / plotly views

## Open questions

- [ ] **Sampling cadence**: every step is too much; every epoch may be too sparse. Adaptive cadence based on detected change?
- [ ] **Memory budget**: storing full activations for PH at every sampled step is expensive. Subsample? Stream?
- [ ] **PyTorch hook surface**: forward hooks, gradient hooks, `LightningCallback`. Pick one and adapt others.
- [ ] **Streaming vs offline**: should reports be queryable mid-training, or only post-hoc?

## Implementation plan

| Phase | Deliverable |
|---|---|
| v0.0.1 | None |
| v0.1 | `topogeoml.training.ShapeOfLearningHook` for PyTorch; offline rendering to HTML |
| v0.2 | TF / JAX adapters; live streaming view |
| v1.0 | Integrated with Benchmark Foundry as a standard report type |
