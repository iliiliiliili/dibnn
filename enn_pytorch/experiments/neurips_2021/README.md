# NeurIPS 2021 Experiments - PyTorch Version

This directory contains PyTorch implementations of the epistemic neural network (ENN) experiments from the NeurIPS 2021 paper.

## Overview

The experiments module provides a framework for evaluating ENNs on regression and classification tasks. It includes:

- **Testbed Base Classes**: Abstract interfaces for problems, agents, and samplers
- **Loss Functions**: Standard and specialized loss functions for ENN training
- **Agents**: Wrapper classes for training ENNs
- **Agent Factories**: Convenience functions for creating common agent types
- **Example Problems**: Simple regression testbeds

## Key Components

### Base Classes (`base.py`)

- **`Data`**: Container for input/output pairs
- **`PriorKnowledge`**: Problem metadata (dimensionality, noise level, etc.)
- **`EpistemicSampler`**: Protocol for generating posterior samples
- **`TestbedAgent`**: Protocol for training agents on problems
- **`TestbedProblem`**: Abstract base for problem implementations
- **`ENNQuality`**: Quality metrics for evaluating samplers

### Loss Functions (`enn_losses.py`)

- **`default_enn_loss()`**: Standard L2 regression loss
- **`gaussian_regression_loss()`**: Gaussian regression with noise
- **`regularized_dropout_loss()`**: Dropout-based uncertainty with regularization

### Agents (`agents.py`)

- **`VanillaEnnConfig`**: Configuration dataclass for agents
- **`VanillaEnnAgent`**: Basic ENN agent implementing the TestbedAgent protocol

### Agent Factories (`agent_factories.py`)

Convenience functions for creating pre-configured agents:

- **`make_ensemble_agent()`**: Creates an ensemble-based agent
- **`make_dropout_agent()`**: Creates a dropout-based agent
- **`make_layer_ensemble_agent()`**: Creates a layer-ensemble agent

### Testbeds (`testbed.py`)

- **`SimpleRegression`**: Generic regression problem for testing
- **`make_synthetic_regression()`**: Factory for creating synthetic regression tasks

## Usage Example

```python
from enn_pytorch.experiments import neurips_2021

# Create a regression problem
problem = neurips_2021.make_synthetic_regression(
    input_dim=10,
    num_train=100,
    num_test=50,
    noise_std=0.1,
    seed=42
)

# Create an agent
agent = neurips_2021.make_ensemble_agent(
    num_ensemble=5,
    hidden_sizes=(50, 50),
    num_batches=1000,
    batch_size=32,
    learning_rate=1e-3
)

# Train the agent
train_data = problem.train_data
prior = problem.prior_knowledge
sampler = agent(train_data, prior)

# Evaluate
quality = problem.evaluate_quality(sampler)
print(f"KL Estimate: {quality.kl_estimate:.4f}")
```

## Running the Example

To run the included example script:

```bash
python enn_pytorch/experiments/neurips_2021/example.py
```

## Architecture Differences from JAX

### Key Translation Changes

1. **Parameter Management**
   - JAX: Functional parameters passed to apply functions
   - PyTorch: Parameters stored in model state, accessed via `.parameters()`

2. **Training Loop**
   - JAX: Custom training functions with gradient computation
   - PyTorch: Standard PyTorch training loop with `optimizer.step()`

3. **Randomness**
   - JAX: Explicit PRNG keys and splitting
   - PyTorch: Implicit state via `torch.manual_seed()` and `torch.Generator`

4. **JIT Compilation**
   - JAX: Uses `jax.jit()` by default
   - PyTorch: Optional via `torch.jit` or TorchScript

### Simplified Implementation

This PyTorch version provides a simplified but complete implementation:

- Core ENN training loop preserved
- Simplified testbed (no neural tangent kernels)
- Focus on main agent types (ensemble, dropout, layer-ensemble)
- PyTorch-idiomatic code using standard patterns

## Extending the Experiments

### Adding New Problems

Subclass `TestbedProblem`:

```python
class MyProblem(testbed_base.TestbedProblem):
    @property
    def train_data(self):
        return testbed_base.Data(x_train, y_train)
    
    @property
    def prior_knowledge(self):
        return testbed_base.PriorKnowledge(
            input_dim=...,
            num_train=...
        )
    
    def evaluate_quality(self, enn_sampler):
        # Compute quality metrics
        return testbed_base.ENNQuality(...)
```

### Adding New Agent Types

Use the agent factory pattern:

```python
def make_my_agent(...) -> testbed_base.TestbedAgent:
    def enn_ctor(prior):
        # Create your ENN
        return my_enn
    
    config = agents.VanillaEnnConfig(
        enn_ctor=enn_ctor,
        ...
    )
    return agents.VanillaEnnAgent(config)
```

## Dependencies

- torch
- numpy
- sklearn (for synthetic data generation)

## Notes

- This is a simplified PyTorch adaptation of the original JAX experiments
- Some advanced features (distillation, advanced sampling) are not included
- Focus is on core ENN training and evaluation functionality
- Suitable for research and prototyping
