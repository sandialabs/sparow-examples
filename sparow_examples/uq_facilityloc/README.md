# `mrp_facilityloc`

This directory contains small facility-location examples for demonstrating different multifidelity workflows and uncertainty quantification capabilities in SPAROW.

## Files

- `bigM.txt`  
  Stores the big-$M$ constant used in several facility-location formulations.  
  This is read by the Python model files at runtime.

- `uq_facilityloc.py`  
  Contains a small stochastic capacitated facility-location example with one high-fidelity model and two low-fidelity approximations Problem data adapted from https://ampl.com/colab/notebooks/ampl-development-tutorial-26-stochastic-capacitated-facility-location-problem.html#problem-description
 

- `uq_discrete_facilityloc.py`  
  Contains the scenario info and facility-location examples that are compatible with the confidence-interval code, including the `get_sp_model_for_uq(...)` and `get_model_ensemble_for_uq(...)` factory functions.  
  It supports both standard MRP and multifidelity workflows such as ACV-MRP and PyApprox integration.

- `demo_uq_discrete_facilityloc.ipynb`  
  Notebook demo for running the confidence-interval workflows on the discrete facility-location example.  
  Shows how to use the model wrappers with the UQ code.

- `demo_pyapprox_discrete_facilityloc.ipynb`  
  Notebook demo for using the PyApprox integration on the discrete facility-location example.  
  Shows how to estimate model costs, compute pilot correlations, and allocate samples to achieve greatest possible variance reduction within fixed computational budget.

- `discrete_facilityloc_scenarios.npy`  
  Saved finite scenario population for the discrete facility-location example.  
  This can be used as a reusable scenario file instead of regenerating scenarios each time. If you need to regenerate this file, you can run `mrp_discrete_facilityloc.py` as a script from the command line.

---

## `uq_facilityloc.py` model formulations

### Variables and constants

- $i \in \{1,\dots,n\}$: facility index  
- $j \in \{1,\dots,t\}$: customer index  
- $f_i$: fixed cost of opening facility $i$  
- $c_{ij}$: per-unit servicing cost from facility $i$ to customer $j$  
- $k_i$: capacity of facility $i$  
- $d_j$: demand of customer $j$ in a scenario  
- $M$: big-$M$ constant  
- $x_i \in \{0,1\}$: 1 if facility $i$ is opened  
- $z_{ij} \ge 0$: amount of customer $j$'s demand served by facility $i$

### High-fidelity model (`HF_builder`)

The high-fidelity model is a capacitated facility-location formulation:

$$
\min \sum_{i=1}^{n}\sum_{j=1}^{t} c_{ij} z_{ij} + \sum_{i=1}^{n} f_i x_i
$$

subject to

$$
\sum_{i=1}^{n} z_{ij} \ge d_j, \qquad j=1,\dots,t
$$

$$
\sum_{i=1}^{n} k_i x_i \ge \sum_{j=1}^{t} d_j
$$

$$
\sum_{j=1}^{t} z_{ij} \le k_i x_i, \qquad i=1,\dots,n
$$

$$
x_i \in \{0,1\}, \qquad i=1,\dots,n
$$

$$
z_{ij} \ge 0, \qquad i=1,\dots,n,\; j=1,\dots,t
$$

### Low-fidelity model 1 (`LF1_builder`)

This low-fidelity model replaces the facility capacity constraint with a big-$M$ logic constraint:

$$
\min \sum_{i=1}^{n}\sum_{j=1}^{t} c_{ij} z_{ij} + \sum_{i=1}^{n} f_i x_i
$$

subject to

$$
\sum_{i=1}^{n} z_{ij} \ge d_j, \qquad j=1,\dots,t
$$

$$
\sum_{i=1}^{n} k_i x_i \ge \sum_{j=1}^{t} d_j
$$

$$
\sum_{j=1}^{t} z_{ij} \le M x_i, \qquad i=1,\dots,n
$$

$$
x_i \in \{0,1\}, \qquad i=1,\dots,n
$$

$$
z_{ij} \ge 0, \qquad i=1,\dots,n,\; j=1,\dots,t
$$

### Low-fidelity model 2 (`LF2_builder`)

This low-fidelity model aggregates the customer demand constraints into one total-demand constraint:

$$
\min \sum_{i=1}^{n}\sum_{j=1}^{t} c_{ij} z_{ij} + \sum_{i=1}^{n} f_i x_i
$$

subject to

$$
\sum_{j=1}^{t}\sum_{i=1}^{n} z_{ij} \ge \sum_{j=1}^{t} d_j
$$

$$
\sum_{i=1}^{n} k_i x_i \ge \sum_{j=1}^{t} d_j
$$

$$
\sum_{j=1}^{t} z_{ij} \le k_i x_i, \qquad i=1,\dots,n
$$

$$
x_i \in \{0,1\}, \qquad i=1,\dots,n
$$

$$
z_{ij} \ge 0, \qquad i=1,\dots,n,\; j=1,\dots,t
$$

---

## `uq_discrete_facilityloc.py` model formulations

### Variables and constants

- $i \in \{1,\dots,n\}$: facility index  
- $j \in \{1,\dots,t\}$: customer index  
- $f_i$: fixed cost of opening facility $i$  
- $c_{ij}$: per-unit servicing cost from facility $i$ to customer $j$  
- $a_{ij}$: transportation / assignment cost for serving customer $j$ from facility $i$  
- $k_i$: capacity of facility $i$  
- $s_i$: maximum number of customers that facility $i$ may serve  
- $d_j$: demand of customer $j$ in a scenario  
- $M$: big-$M$ constant  
- $x_i \in \{0,1\}$: 1 if facility $i$ is opened  
- $y_{ij}$: assignment indicator for customer $j$ and facility $i$  
- $z_{ij} \ge 0$: amount of customer $j$'s demand served by facility $i$

### High-fidelity model (`HF_builder`)

The high-fidelity model includes binary assignment variables:

$$
\min \sum_{i=1}^{n}\sum_{j=1}^{t} c_{ij} z_{ij}
+ \sum_{i=1}^{n}\sum_{j=1}^{t} a_{ij} y_{ij}
+ \sum_{i=1}^{n} f_i x_i
$$

subject to

$$
\sum_{i=1}^{n} z_{ij} \ge d_j, \qquad j=1,\dots,t
$$

$$
\sum_{i=1}^{n} k_i x_i \ge \sum_{j=1}^{t} d_j
$$

$$
\sum_{j=1}^{t} z_{ij} \le k_i x_i, \qquad i=1,\dots,n
$$

$$
y_{ij} \le x_i, \qquad i=1,\dots,n,\; j=1,\dots,t
$$

$$
z_{ij} \le M y_{ij}, \qquad i=1,\dots,n,\; j=1,\dots,t
$$

$$
\sum_{j=1}^{t} y_{ij} \le s_i, \qquad i=1,\dots,n
$$

$$
x_i \in \{0,1\}, \qquad i=1,\dots,n
$$

$$
y_{ij} \in \{0,1\}, \qquad i=1,\dots,n,\; j=1,\dots,t
$$

$$
z_{ij} \ge 0, \qquad i=1,\dots,n,\; j=1,\dots,t
$$

### Low-fidelity model (`LF_builder`)

The low-fidelity model relaxes the assignment variables from binary to continuous and removes the big-$M$ linking constraint:

$$
\min \sum_{i=1}^{n}\sum_{j=1}^{t} c_{ij} z_{ij}
+ \sum_{i=1}^{n}\sum_{j=1}^{t} a_{ij} y_{ij}
+ \sum_{i=1}^{n} f_i x_i
$$

subject to

$$
\sum_{i=1}^{n} z_{ij} \ge d_j, \qquad j=1,\dots,t
$$

$$
\sum_{i=1}^{n} k_i x_i \ge \sum_{j=1}^{t} d_j
$$

$$
\sum_{j=1}^{t} z_{ij} \le k_i x_i, \qquad i=1,\dots,n
$$

$$
y_{ij} \le x_i, \qquad i=1,\dots,n,\; j=1,\dots,t
$$

$$
\sum_{j=1}^{t} y_{ij} \le s_i, \qquad i=1,\dots,n
$$

$$
x_i \in \{0,1\}, \qquad i=1,\dots,n
$$

$$
0 \le y_{ij} \le 1, \qquad i=1,\dots,n,\; j=1,\dots,t
$$

$$
z_{ij} \ge 0, \qquad i=1,\dots,n,\; j=1,\dots,t
$$