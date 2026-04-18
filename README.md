# Time-Embedded Annealed Flow Transport

This repository contains the implementation of Time-Embedded Annealed Flow Transport algorithms developed as part of my undergraduate Honours Project in March 2024.

The code adapts and extends the [Annealed Flow Transport repository](https://github.com/google-deepmind/annealed_flow_transport) from Google DeepMind, introducing novel time-embedded variants of flow-based sampling algorithms for Bayesian inference.

## Introduction

### Background

Sampling from complex, unnormalized probability distributions is a fundamental challenge in Bayesian inference and probabilistic modeling. Sequential Monte Carlo (SMC) methods address this by progressively transforming samples from a simple initial distribution to a complex target distribution through a sequence of intermediate "bridging" distributions.

**Annealed Flow Transport (AFT)** and **Continual Repeated AFT (CRAFT)** are recent advances that incorporate normalizing flows into the SMC framework. These algorithms use neural network-based diffeomorphisms to transport particles between intermediate distributions, improving sample quality and providing more accurate estimates of normalizing constants.

### Thesis Contributions

This repository implements three novel algorithms introduced in the accompanying thesis:

1. **Time-Embedded AFT (TE-AFT)**: Replaces the multiple normalizing flows used in standard AFT with a single time-embedded flow. The flow takes information about the current and previous iterations' intermediate distributions as input, reducing the number of parameters that need to be learned.

2. **Time-Embedded CRAFT (TE-CRAFT)**: Extends the time-embedding approach to the CRAFT algorithm, which performs repeated annealing with trained flows for improved efficiency.

3. **Adaptive TE-AFT**: An adaptive variant that dynamically selects intermediate distributions based on the current iteration's weighted samples and flow parameters. Introduces a modified conditional effective sample size metric, CESS(f), to account for flow transport when adaptively choosing temperature schedules.

### Key Features

- **JAX-based implementation** for GPU acceleration and automatic differentiation
- **Multiple normalizing flow architectures**: Real-NVP, Affine IAF, Diagonal Affine, with time-embedded variants
- **Baseline algorithms**: SMC, AFT, CRAFT, and Adaptive CRAFT implementations
- **Target distributions**: Neal's funnel, Log-Gaussian Cox Process, multivariate Gaussians, and mixture models
- **Hamiltonian Monte Carlo (HMC)** kernels for MCMC rejuvenation steps
- **Comprehensive configuration system** using ML Collections for reproducible experiments
- **Extensive benchmarking results** comparing fixed and adaptive temperature schedules

## Installation

### Requirements

This project requires Python 3.8+ and the following core dependencies:

- **JAX** (with GPU support recommended)
- **Flax** - Neural network library built on JAX
- **Optax** - Gradient-based optimization
- **NumPy** - Numerical computing
- **Matplotlib** - Visualization
- **ML Collections** - Configuration management
- **ABSL-py** - Command-line flags and app utilities

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/stevyseah/time-embedded-annealed-flow-transport.git
   cd time-embedded-annealed-flow-transport
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   # Install JAX (CPU version)
   pip install jax jaxlib
   
   # For GPU support, follow JAX installation guide:
   # https://github.com/google/jax#installation
   
   # Install other dependencies
   pip install flax optax numpy matplotlib ml-collections absl-py
   ```

4. **Verify installation**:
   ```bash
   python -c "import jax; print(jax.devices())"
   ```

## Usage

The repository provides four main entry points for different use cases:

### 1. Single Sampling Experiment

Run a single sampling experiment with a specified configuration:

```bash
python src/main.py --config=configs/simple_normal.py
```

This will execute the sampling algorithm defined in the configuration file and return:
- Log evidence estimate
- Sampling time
- Weighted particle samples
- Miscellaneous algorithm-specific outputs

**Example configurations**:
- `configs/simple_normal.py` - Simple 2D Gaussian example
- `configs/funnel.py` - 10-dimensional Neal's funnel
- `configs/lgcp_pines.py` - Log-Gaussian Cox Process (Finnish Pines spatial data)

### 2. Repeated Experiments with Logging

Run multiple repetitions of an experiment and log results to CSV:

```bash
python src/experiment.py --config=configs/funnel/te_aft4.py
```

Results are saved to the `results/` directory with columns including:
- Log evidence estimate
- Sampling time
- Number of temperatures used
- Algorithm-specific metrics

**Experiment configurations** (50+ variants available):
- `configs/funnel/` - Neal's funnel experiments
  - `te_aft4.py`, `te_aft8.py` - TE-AFT with 4, 8 temperatures
  - `te_craft5.py`, `te_craft10.py` - TE-CRAFT variants
  - `aft_adaptive.py` - Adaptive temperature selection
- `configs/pines/` - Log-Gaussian Cox Process experiments
  - Similar naming convention for different algorithms and settings

### 3. Time-Embedded Flow Visualization

Demonstrate time-embedding with a 2D example and visualization:

```bash
python src/time_embedding_example.py
```

This script:
- Generates samples from intermediate distributions
- Trains a time-embedded flow
- Visualizes how the flow transports particles over time
- Saves plots to the `images/` directory

### 4. Maximum Likelihood Training

Train normalizing flows via MLE on the Two Moons toy dataset:

```bash
python src/mle.py
```

This demonstrates:
- Training Real-NVP on the Two Moons dataset (Figure 1 in thesis)
- Maximum likelihood optimization using Optax
- Visualization of learned flow transformations

### Configuration System

All experiments are defined using ML Collections `ConfigDict` objects. Key configuration parameters include:

```python
config.algo = 'aft'  # Algorithm: 'smc', 'aft', 'craft', 'vi'
config.seed = 0  # Random seed for reproducibility
config.num_particles = 1000  # Number of particles
config.particle_dim = 10  # Dimension of the problem
config.num_temps = 10  # Number of intermediate distributions

# Flow configuration
config.flow_config.type = 'realnvp'  # Flow architecture
config.flow_config.use_time_embedding = True  # Enable time-embedding

# AFT training parameters
config.aft_config.num_iters = 10000  # Training iterations per temperature
config.aft_config.learning_rate = 1e-3  # Learning rate
```

### Algorithm Selection

Choose the algorithm by setting `config.algo`:

- `'smc'` - Sequential Monte Carlo (baseline)
- `'aft'` - Annealed Flow Transport
- `'craft'` - Continual Repeated Annealed Flow Transport
- `'vi'` - Variational Inference

Time-embedding is enabled via `config.flow_config.use_time_embedding = True`.

Adaptive temperature selection is enabled via `config.smc_config.use_adaptive_temps = True`.

### Example: Running TE-AFT on Neal's Funnel

```bash
# Fixed temperature schedule with 8 intermediate distributions
python src/experiment.py --config=configs/funnel/te_aft8.py

# Adaptive temperature schedule
python src/experiment.py --config=configs/funnel/te_aft_adaptive.py
```

### Example: Comparing Algorithms

Run experiments comparing different algorithms on the same problem:

```bash
# Baseline SMC
python src/experiment.py --config=configs/funnel/smc.py

# Standard AFT
python src/experiment.py --config=configs/funnel/aft4.py

# Time-Embedded AFT (proposed)
python src/experiment.py --config=configs/funnel/te_aft4.py

# Time-Embedded CRAFT (proposed)
python src/experiment.py --config=configs/funnel/te_craft4.py
```

Results can be compared by examining the CSV files in the `results/` directory.

### Custom Configurations

Create custom experiments by copying and modifying existing configuration files:

1. Copy a template configuration:
   ```bash
   cp configs/funnel/te_aft4.py configs/my_experiment.py
   ```

2. Edit parameters in `configs/my_experiment.py`

3. Run your experiment:
   ```bash
   python src/experiment.py --config=configs/my_experiment.py
   ```

### Testing

Run unit tests to verify core functionality:

```bash
python -m pytest tests/
```

## Project Structure

```
.
├── src/
│   ├── main.py                                # Single experiment runner
│   ├── experiment.py                          # Repeated experiments with logging
│   ├── mle.py                                 # MLE training for flows
│   ├── time_embedding_example.py              # Time-embedding demonstration
│   └── annealed_flow_transport/
│       ├── aft.py                             # AFT algorithm implementation
│       ├── craft.py                           # CRAFT algorithm
│       ├── adaptive_craft.py                  # Adaptive CRAFT
│       ├── smc.py                             # Sequential Monte Carlo
│       ├── flows.py                           # Normalizing flow models
│       ├── densities.py                       # Target distributions
│       ├── samplers.py                        # Initial samplers
│       ├── train.py                           # Training orchestration
│       └── utils/
│           ├── smc_utils.py                   # SMC utilities
│           ├── hmc.py                         # HMC kernel
│           └── aft_types.py                   # Type definitions
├── configs/                                   # Experiment configurations
│   ├── funnel/                                # Neal's funnel experiments
│   ├── pines/                                 # Log-Gaussian Cox Process
│   └── special/                               # Special cases (two_moons, etc.)
├── data/                                      # Datasets
│   ├── two_moons.csv
│   └── finpines.csv
├── results/                                   # Experimental results (CSV)
├── images/                                    # Visualization outputs
├── tests/                                     # Unit tests
├── time-embedded-annealed-flow-transport.pdf  # Full thesis document
└── README.md
```

## References

The thesis builds upon several key works in the literature:
- Sequential Monte Carlo (SMC) samplers
- Annealed Flow Transport (AFT) - Arbel, Matthews & Doucet (2021)
- Continual Repeated AFT (CRAFT) - Matthews et al. (2022)
- Real-NVP normalizing flows - Dinh, Sohl-Dickstein & Bengio (2016)
- Adaptive SMC - Zhou, Johansen & Aston (2016)

## License

This project is based on the Google DeepMind Annealed Flow Transport repository. Please refer to the original repository for licensing information.

---

**Note**: This repository contains the complete implementation used for the experiments reported in the thesis. The full thesis document (`time-embedded-annealed-flow-transport.pdf`) is included in the root directory for reference.
