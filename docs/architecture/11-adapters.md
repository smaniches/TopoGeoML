# TF / PyTorch / JAX Adapter Strategy

**STATUS:** DRAFT
**Phase:** PyTorch in v0.0.1 (Hodge layer); TF and JAX in v0.1+
**Spec source:** TopoGeomML Specification Part II §21
**Depends on:** [00-positioning.md](00-positioning.md), [01-five-layer-architecture.md](01-five-layer-architecture.md)

---

## What it is

The adapter strategy defines how TopoGeomML interoperates with the three major tensor frameworks without competing with any of them. Each framework gets a *narrow*, *one-way* adapter: TopoGeomML extracts the geometric data it needs from framework-native tensors and returns canonical types defined at layer 2.

There is no `TopologyTensor` class wrapping `torch.Tensor`. There is no `topogeoml.nn` re-implementation of PyTorch primitives. There is no "PyTorch mode" vs "TF mode" — operators are framework-agnostic and operate on canonical types.

## Why this is non-negotiable

If TopoGeomML introduces framework-specific code paths, it becomes a worse version of each framework. The adapter discipline keeps the core code framework-agnostic.

The Hodge MP layer is the *exception that proves the rule*: it lives in `topogeoml.nn` because it is a differentiable layer, and differentiability is a framework concept. But it requires PyTorch (gated import); the rest of the package does not. If TF / JAX users want differentiable Hodge layers, they implement them in their framework, possibly using a shared core math module.

## Adapter pattern (general form)

For each framework, two pure functions:

```python
# Ingress: framework tensor → canonical type
def embeddings_from_torch(tensor: torch.Tensor) -> NDArray[np.float64]: ...
def embeddings_from_tf(tensor: tf.Tensor) -> NDArray[np.float64]: ...
def embeddings_from_jax(tensor: jax.Array) -> NDArray[np.float64]: ...

# Egress: canonical type → framework tensor
def embeddings_to_torch(arr: NDArray[np.float64]) -> torch.Tensor: ...
def embeddings_to_tf(arr: NDArray[np.float64]) -> tf.Tensor: ...
def embeddings_to_jax(arr: NDArray[np.float64]) -> jax.Array: ...
```

Both are stateless. Both prefer zero-copy where possible. Neither hides the framework — calling code knows whether it's in framework world or canonical world.

## Differentiable layers

When a TopoGeomML primitive needs to be differentiable (e.g. the Hodge MP layer), it is implemented **per framework**. Shared mathematical core lives as pure functions on numpy / scipy; framework-specific wrappers register them as differentiable ops.

For v0.0.1, only PyTorch is supported. v0.1 adds JAX (priority over TF because JAX's functional style is closer to our pure-function discipline). TF support is best-effort.

## Open questions

- [ ] **JAX vs TF priority**: leaning JAX for v0.1 because of cleaner functional API. Confirm with first user demand.
- [ ] **Zero-copy via dlpack**: implement now or v0.2? PyTorch and JAX both support dlpack; TF support is partial.
- [ ] **Lightning / Keras / Flax integration**: at the framework-extension level, do we ship `topogeoml-lightning`, `topogeoml-keras`, `topogeoml-flax` callbacks? Leaning toward yes, as separate packages, post-v0.1.
- [ ] **GPU adapters**: when an embedding is on GPU, do we move it to CPU for layer-3 work, or do we eventually push layer-3 to GPU? v0.2 question.

## Implementation plan

| Phase | Deliverable |
|---|---|
| v0.0.1 | PyTorch only, via the Hodge MP layer; no general adapter module yet |
| v0.1 | `topogeoml.adapters.torch`, `topogeoml.adapters.jax`; embedding ingress + egress |
| v0.2 | `topogeoml.adapters.tf`; zero-copy via dlpack where supported |
| v1.0 | Framework-extension packages (lightning, keras, flax) as separate repos |
