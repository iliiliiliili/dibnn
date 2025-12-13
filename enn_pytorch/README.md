# ENN PyTorch - JAX to PyTorch Translation

This directory contains a PyTorch translation of the Epistemic Neural Networks (ENN) library, originally implemented in JAX/Haiku.

## Overview

The ENN library provides tools for implementing epistemic neural networks - neural networks designed to represent uncertainty through epistemic indices. This PyTorch version maintains the same conceptual structure while adapting to PyTorch's programming model.

## Translation Summary

### Core Components Translated

#### 1. **Base Module** (`base.py`)
- **Key Changes:**
  - Replaced JAX arrays (`jnp.DeviceArray`) with PyTorch tensors
  - Changed from Haiku's `hk.Module` to PyTorch's `nn.Module`
  - Modified `EpistemicModule` to use `forward()` instead of `__call__()`
  - Updated parameter handling: Haiku's `hk.Params` → PyTorch model instances
  - Random key handling: JAX's PRNG keys → PyTorch generators or integer seeds

#### 2. **Utils Module** (`utils.py`)
- **Key Changes:**
  - Replaced `hk.transform()` pattern with direct PyTorch model instantiation
  - Adapted batch iterator to use PyTorch's `DataLoader`
  - Modified indexer batching to work with PyTorch tensors
  - Updated data handling for PyTorch's eager execution model

#### 3. **Networks Module** (`networks/`)

**Indexers** (`indexers.py`):
- Translated all indexer types to PyTorch:
  - `EnsembleIndexer`: Random integer selection for ensemble members
  - `LayerEnsembleIndexer`: Multi-layer ensemble indexing
  - `ScaledGaussianIndexer`: Gaussian noise indexing
  - `PrngIndexer`: Pass-through indexer for random keys
  - `DirichletIndexer`: Dirichlet distribution sampling

**Ensembles** (`ensembles.py`):
- `Ensemble`: Discrete ensemble using `nn.ModuleList`
- `MLPEnsembleEnn`: Ensemble of MLPs using `nn.LazyLinear` for flexible input sizes
- Factory methods for creating ensembles with various prior functions

**Layer Ensembles** (`layer_ensembles.py`):
- `LayerEnsembleNetwork`: Per-layer ensemble architecture
- `LayerEnsembleNetworkWithPriors`: Layer ensemble with matched priors
- Factory methods for creating layer-wise ensembles

**Priors** (`priors.py`):
- `EnnWithAdditivePrior`: Add prior functions to existing ENNs
- `make_random_feat_gp()`: Random feature GP approximation using random kitchen sinks
- `get_random_mlp_with_index()`: Random MLP prior generation

**Dropout** (`dropout.py`):
- `MLPDropoutENN`: Dropout as uncertainty approximation
- Uses deterministic dropout via manual seed control

#### 4. **Losses Module** (`losses/`)

**Single Index Losses** (`single_index.py`):
- `L2Loss`: Mean squared error for regression
- `XentLoss`: Cross-entropy for classification
- `AccuracyErrorLoss`: Classification accuracy metric
- `ElboLoss`: Variational inference loss (ELBO)
- Loss averaging utilities for multiple index samples

#### 5. **Data Noise Module** (`data_noise/`)
- Base protocols for data augmentation and noise
- Bootstrap and reweighting utilities (base structure)

#### 6. **Supervised Module** (`supervised/`)
- `BaseExperiment`: Abstract interface for supervised learning experiments
- Foundation for training loops and evaluation

#### 7. **Experiments Module** (`experiments/`)

**NeurIPS 2021 Experiments** (`experiments/neurips_2021/`):
- `base.py`: Base classes for testbed problems and agents
  - `TestbedProblem`: Abstract problem interface
  - `TestbedAgent`: Agent training protocol
  - `EpistemicSampler`: Posterior sampling interface
- `enn_losses.py`: Loss function factories
  - `default_enn_loss()`: Standard L2 regression
  - `gaussian_regression_loss()`: Gaussian regression
  - `regularized_dropout_loss()`: Dropout uncertainty
- `agents.py`: Agent implementations
  - `VanillaEnnAgent`: Main agent class
  - `VanillaEnnConfig`: Configuration dataclass
- `agent_factories.py`: Pre-configured agent creation
  - `make_ensemble_agent()`: Ensemble-based agent
  - `make_dropout_agent()`: Dropout-based agent
  - `make_layer_ensemble_agent()`: Layer-ensemble agent
- `testbed.py`: Problem implementations
  - `SimpleRegression`: Regression testbed
  - `make_synthetic_regression()`: Synthetic data generation
- `example.py`: Complete example script
- `README.md`: Experiments documentation

## Key Translation Patterns

### 1. Functional to Object-Oriented
**JAX/Haiku (Functional):**
```python
transformed = hk.transform(network_fn)
params = transformed.init(rng, x, z)
output = transformed.apply(params, x, z)
```

**PyTorch (Object-Oriented):**
```python
model = Network()
model.eval()  # or model.train()
output = model(x, z)
```

### 2. Parameter Management
**JAX/Haiku:**
- Parameters stored as nested dictionaries
- Explicitly passed to apply functions

**PyTorch:**
- Parameters stored in model state
- Automatically managed by `nn.Module`

### 3. Random Number Generation
**JAX:**
```python
key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key)
value = jax.random.normal(subkey, shape)
```

**PyTorch:**
```python
torch.manual_seed(seed)
generator = torch.Generator().manual_seed(seed)
value = torch.randn(shape, generator=generator)
```

### 4. Vectorization
**JAX:** Uses `jax.vmap()` for automatic vectorization

**PyTorch:** Manual batching or list comprehensions, native batch operations

### 5. Gradient Stopping
**JAX:** `jax.lax.stop_gradient()`

**PyTorch:** `tensor.detach()` or `with torch.no_grad():`

## Usage Example

```python
import torch
from enn_pytorch import base
from enn_pytorch.networks import MLPEnsembleEnn, EnsembleIndexer
from enn_pytorch.losses import L2Loss, average_single_index_loss

# Create an ensemble ENN
enn = MLPEnsembleEnn(
    output_sizes=[50, 50, 1],
    num_ensemble=10
)

# Initialize the model
model = enn.init()

# Generate data
x = torch.randn(32, 10)  # batch of 32, 10 features
y = torch.randn(32, 1)   # targets

# Create batch
batch = base.Batch(x=x, y=y)

# Sample an index
key = 42
index = enn.indexer(key)

# Forward pass
output = enn.apply(model, x, index)

# Compute loss
loss_fn = average_single_index_loss(L2Loss(), num_index_samples=5)
loss, metrics = loss_fn(enn, model, batch, key)
```

## Differences and Limitations

### Features Preserved
✅ Core ENN abstractions (epistemic indices, ensemble architectures)
✅ Network architectures (ensembles, dropout, layer ensembles)
✅ Loss functions (L2, cross-entropy, ELBO)
✅ Indexing mechanisms (ensemble, Gaussian, layer-wise)
✅ Prior function integration

### Notable Differences
⚠️ **Eager vs. JIT**: PyTorch uses eager execution by default (can use `torch.jit` for compilation)
⚠️ **Initialization**: PyTorch uses lazy initialization (`nn.LazyLinear`) vs. Haiku's explicit init
⚠️ **Gradient Computation**: PyTorch's autograd vs. JAX's functional gradients
⚠️ **Device Management**: Explicit `.to(device)` calls needed in PyTorch
⚠️ **Immutability**: PyTorch parameters are mutable vs. JAX's immutable arrays

### Not Fully Translated
❌ Advanced experiment runners (neurips_2021 experiments)
❌ Some specialized hypermodel implementations
❌ Test files
❌ Some data noise implementations (bootstrapping details)
❌ Variational inference specific utilities

## Dependencies

**Original (JAX):**
- jax
- haiku
- chex
- optax (for optimization)

**PyTorch Version:**
- torch
- numpy
- typing-extensions
- sklearn (for test data)
- absl-py (for logging)

## Installation

To use the PyTorch version:

```bash
pip install torch numpy scikit-learn typing-extensions absl-py
```

## File Structure

```
enn_pytorch/
├── __init__.py           # Main module exports
├── base.py               # Core abstractions
├── utils.py              # Utility functions
├── _metadata.py          # Version info
├── networks/             # Neural network implementations
│   ├── __init__.py
│   ├── indexers.py       # Index sampling strategies
│   ├── ensembles.py      # Ensemble networks
│   ├── layer_ensembles.py # Layer-wise ensembles
│   ├── priors.py         # Prior function utilities
│   └── dropout.py        # Dropout-based uncertainty
├── losses/               # Loss functions
│   ├── __init__.py
│   └── single_index.py   # Single-index loss functions
├── data_noise/           # Data augmentation
│   ├── __init__.py
│   └── base.py
├── supervised/           # Supervised learning utilities
│   ├── __init__.py
│   └── base.py
└── experiments/          # Experiment frameworks
    ├── __init__.py
    └── neurips_2021/
        └── __init__.py
```

## Contributing

When extending this translation:

1. **Maintain API compatibility** where possible with the original JAX version
2. **Use PyTorch idioms**: Prefer `nn.Module`, `forward()`, etc.
3. **Document differences**: Note where PyTorch version differs from JAX
4. **Test thoroughly**: Ensure numerical equivalence where expected
5. **Handle devices**: Consider GPU/CPU placement in implementations

## References

- Original ENN JAX repository: https://github.com/deepmind/enn
- Paper: "Epistemic Neural Networks" (Osband et al., 2021)
- PyTorch documentation: https://pytorch.org/docs/

## License

Maintains the original Apache 2.0 license from DeepMind's ENN library.
