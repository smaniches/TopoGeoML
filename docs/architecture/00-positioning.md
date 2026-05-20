# Strategic Positioning

**STATUS:** DRAFT
**Phase:** binding from v0.0.1 onward; no implementation, but every implementation decision must be checked against this document
**Spec source:** TopoGeomML Specification Part II §1, §2, §24

---

## The question

> *"TensorFlow and PyTorch answer: how do I compute and train?
>  TopoGeomML should answer: what did training do to the structure represented by the model, and what evidence supports deploying it?"*

This is the single sentence that determines every scope decision.

## What TopoGeomML is *not*

TopoGeomML is **not** a tensor runtime. It is not a competitor to TensorFlow, PyTorch, or JAX. It does not provide:

- Autograd engines
- Distributed training schedulers
- Tensor kernels (matmul, convolution, attention)
- Optimizers, schedulers, mixed-precision support
- Production deployment runtimes for model inference
- A replacement for `nn.Module` or `tf.keras.Model`

If TopoGeomML attempts to compete on any of those axes, it loses by definition. TF, PyTorch, and JAX are mature, well-funded, well-staffed projects with deep ecosystem integration. The asymmetric move is not to build another tensor framework — it is to build the layer **above** them that no one currently owns.

## What TopoGeomML *is*

TopoGeomML is the **epistemic operating layer above ordinary ML frameworks**. Its job is to convert the artifacts that TF / PyTorch / JAX produce — trained models, intermediate activations, latent embeddings, prediction sets — into structured, verifiable, topology-aware evidence about what those artifacts actually represent and whether they should be deployed.

Concretely, TopoGeomML provides:

1. **Topology-native operators** (layer 3) on geometric data — persistence, complexes, Hodge structure, descriptors. These are the primitives the rest of the stack builds on. v0.0.1 lives here.
2. **Epistemic training and evaluation** (layer 4) — shape-of-learning reports, cohort topology audits, latent regression tests, autotopology search.
3. **Claims and governance** (layer 5) — claim graphs, evidence bundles, invariant ledgers, model cards generated from evidence, deployment gates that block model promotion when invariants regress.

Layers 1 and 2 (backend runtime, geometric data ingestion) are **borrowed from existing libraries**: numpy, scipy, ripser, GUDHI, NetworkX, PyG, TopoNetX, FAISS, TF, PyTorch, JAX. TopoGeomML composes them through narrow adapter interfaces; it does not re-implement them.

## Why this is the right positioning

Three distinct reasons:

### 1. Strategic — there is no incumbent

The tensor-framework war is over. TensorFlow, PyTorch, and JAX have stable APIs, large user bases, and clear differentiation. The "another framework" move loses to all three.

But the question of **what evidence justifies deploying a model** is unanswered at the framework level. Every ML team currently answers it in an ad-hoc way: a notebook of metrics, a slack message saying "looks good", a model card that lists hyperparameters. There is no Pandas-or-NumPy of evidence governance. That's the slot.

### 2. Technical — topology gives an honest substrate for evidence

Most "model evidence" today is a number: accuracy, F1, ROC-AUC. Numbers compress structure into a single dimension and lose almost all information about what the model represents. The complaint "the model overfit" is a structural complaint with no structural language.

Topology gives that language. β₀ of a latent space tells you how many clusters the model formed. β₁ tells you what loops it preserves. Persistence diagrams characterize the *shape* of representations, not just their numeric quality. A drift in the H₁ structure between epochs is observable; a drift in "accuracy" without H₁ drift is qualitatively different from one with H₁ drift. Topology makes these distinctions reportable.

### 3. Commercial — every regulated industry needs this

Healthcare, finance, defense, autonomous systems, and increasingly public-sector ML all need pre-deployment evidence that is more than metric reports. Model cards (Mitchell et al. 2018) are now mandatory for many deployments; they're currently filled in by hand. A system that **generates** model cards from a structured evidence bundle — with cryptographically signed attestations from training runs — is a commercially defensible product.

## The asymmetric move

The asymmetric move is to **let everyone else compete at the tensor level**. TopoGeomML cedes that battleground completely. It does not run training jobs. It does not produce inference servers. It is the **inspector**, **auditor**, and **certificate authority** for what other systems produce.

This means the most important thing TopoGeomML can ship in any release is not a faster filter or a new vectorizer. It is a sharper, more rigorous evidence contract. An invariant ledger that lets a deployment gate refuse to merge a PR because the H₁ structure of the recommender's user-embedding regressed by more than 2σ is worth a thousand persistence-image speedups.

## What this implies for v0.0.1

v0.0.1 is the operator-layer foundation. None of the layer-4 or layer-5 product surface ships in v0.0.1. That is **intentional**: the operators must be solid before the governance layer that depends on them is meaningful. A claim graph signed by a buggy persistence backend is worse than no claim graph.

But every v0.0.1 design choice has been made with the governance layer in mind:

- `DiagramProvenance` exists because evidence requires origin metadata.
- `FitProvenance` exists because reproducibility requires fit-time records.
- The JSON output schema captures environment, timestamp, and config because that's the seed of an evidence bundle.
- The `audits/` subpackage exists because audit is a first-class concept, not a notebook activity.

These are not arbitrary. They are the lowest-cost seeds of the layer-5 product.

## What this rules out

- **Implementing autograd**, even a tiny one. TF / PyTorch / JAX do this. We compose with them.
- **Replacing `nn.Module`**. PyTorch already has it.
- **Writing a new optimizer**. Adam, AdamW, Lion exist.
- **Replacing scipy.sparse**. We use it.
- **Replacing GUDHI / ripser**. We use them.
- **Writing a "TopologyTensor" class** that wraps `torch.Tensor`. The tensor IS the tensor. Topology is the structure we report about tensors, not a new container.

Any future PR that proposes to add capability in the layer-1 or layer-2 space is on by-default rejected unless it can show concretely that the existing primitives are insufficient for layer-3+ work.

## Open questions

- [ ] Where does **multi-backend support** end and **framework duplication** begin? Concrete: do we ship our own boundary-operator construction (as we do today over scipy.sparse) or do we eventually delegate to TopoNetX? Likely answer: keep our boundaries for the verification gate to depend on; let TopoNetX own the higher-order combinatorial complex types.
- [ ] What is the **first product touchpoint** for layers 4 and 5? Candidates: a `topogeoml audit` CLI that emits a model card JSON; a `topogeoml.gate` decorator for CI gates; a FastAPI service that ingests model artifacts and returns evidence bundles. Decision deferred to v0.1.
- [ ] Is **EMLIR** (the proposed intermediate representation, see `02-emlir.md`) actually needed in v0.1, or can layers 4 and 5 be built directly on Python dataclasses + JSON schemas? Strong prior that EMLIR is over-engineering at first, but the question stays open.

## Cross-references

- [01-five-layer-architecture.md](01-five-layer-architecture.md) — the layered model in detail
- [12-governance.md](12-governance.md) — what the governance layer must produce
- [13-moat.md](13-moat.md) — what makes this positioning defensible
- [11-adapters.md](11-adapters.md) — how TF / PyTorch / JAX integrate without TopoGeomML competing with them
