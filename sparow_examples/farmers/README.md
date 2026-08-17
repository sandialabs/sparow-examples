# `farmers`

This directory contains farmer problem examples for demonstrating different stochastic programming workflows and uncertainty quantification capabilities in SPAROW.

## Files

- `MFfarmers.py`  
  Contains several versions of the farmer problem for stochastic programming experiments, including basic, low-fidelity, high-fidelity, and multifidelity formulations.

- `MRPfarmers.py`  
  Contains the Basic Farmers and Advanced Farmers examples. Note that the Advanced Farmers example is compatible with the UQ code, including the `get_sp_model_for_uq(...)` factory function.  
  It supports the single-fidelity MRP workflow for confidence interval estimation.

- `demo_uq_farmers.ipynb`  
  Notebook demo for running the confidence-interval workflows on the farmer examples.  
  Shows how to use the single-fidelity model wrapper with the UQ code.

- `advanced_farmers_1000_scenarios.npy`  
  Saved finite scenario population for the Advanced Farmers example.  
  This can be used as a reusable scenario file instead of regenerating scenarios each time. If you need to regenerate this file, you can run `MRPfarmers.py` as a script from the command line.
  
---

## `MRPfarmers.py` model formulations

### Variables and constants

- $c \in \{\text{WHEAT}, \text{CORN}, \text{SUGAR\_BEETS}\}$: crop index  
- $A$: total available acreage  
- $f_c$: planting cost per acre for crop $c$  
- $q_c$: sales quota for crop $c$  
- $p_c^{\text{sub}}$: selling price below quota for crop $c$  
- $p_c^{\text{sup}}$: selling price above quota for crop $c$  
- $r_c$: cattle feed requirement for crop $c$  
- $u_c$: purchase price for crop $c$  
- $y_c$: random yield for crop $c$ in a scenario  
- $x_c \ge 0$: acreage devoted to crop $c$  
- $s_c^{\text{sub}} \ge 0$: quantity of crop $c$ sold below quota  
- $s_c^{\text{sup}} \ge 0$: quantity of crop $c$ sold above quota  
- $b_c \ge 0$: quantity of crop $c$ purchased  

### Common model (`model_builder`)

Both the Basic Farmers and Advanced Farmers examples use the same optimization model; only the scenario set for the random yields differs:

$$
\min \sum_c f_c x_c + \sum_c u_c b_c - \sum_c p_c^{\text{sub}} s_c^{\text{sub}} - \sum_c p_c^{\text{sup}} s_c^{\text{sup}}
$$

subject to

$$
\sum_c x_c \le A
$$

$$
r_c \le y_c x_c + b_c - s_c^{\text{sub}} - s_c^{\text{sup}}, \qquad \forall c
$$

$$
s_c^{\text{sub}} + s_c^{\text{sup}} \le y_c x_c, \qquad \forall c
$$

$$
0 \le s_c^{\text{sub}} \le q_c, \qquad \forall c
$$

$$
x_c \ge 0,\qquad b_c \ge 0,\qquad s_c^{\text{sup}} \ge 0,\qquad \forall c
$$

### Basic Farmers scenarios

The Basic Farmers example uses the classic three-scenario yield set from Birge and Louveaux (2011):

- below-average yield scenario  
- average yield scenario  
- above-average yield scenario  

Each scenario specifies yields for wheat, corn, and sugar beets, with equal probability.

### Advanced Farmers scenarios

The Advanced Farmers example constructs a larger finite scenario population by linearly interpolating between the below-average and above-average yields for each crop. If each crop has $K$ interpolated support points, then the total number of scenarios is $K^3$, with equal probability assigned to each scenario vector in the Cartesian product of the crop-wise supports.

## Reference

Birge, J. R., & Louveaux, F. (2011). *Introduction to Stochastic Programming* (2nd ed.). New York, NY: Springer New York.