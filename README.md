# # Associated data and code

The files in this code repository are updated and commented versions of the code used for the following work:

_Graph Partitioning in Connected Components with Minimum Size Constraints via Integer Programming Techniques_ by Mishelle Cordero, Andrés Miniguano-Trujillo, Diego Recalde, Ramiro Torres, Polo Vaca.

## Main files in this repository

* [`ReadMe.md`](ReadMe.md): This file.
* [`InstancesGenerator.py`](InstancesGenerator.py): Instance generator containing parameters for creating each of the instances stored in [`Instances`](Instances). Note that the generated instances are not deterministic, so it is preferred to load the already built instances.

Jupyter notebooks with code for graph partitioning in connected components with minimum size constraints. The ```Gurobi``` package is needed for the linear models. 

* [`Flux Formulation 1 - Base-Lazy.ipynb`](Flux%20Formulation%201%20-%20Base-Lazy.ipynb): First MIP model (M<sub>1</sub>).
* [`Flux Formulation 1 - Base-Lazy Valid Inequalities.ipynb`](Flux%20Formulation%201%20-%20Base-Lazy%20Valid%20Inequalities.ipynb): First MIP model (M<sub>1</sub>) with Cuts (Theorems 2 to 9).
* [`Flux Formulation 2 - Base-Lazy.ipynb`](Flux%20Formulation%202%20-%20Base-Lazy.ipynb): Second MIP model (M<sub>2</sub>).
* [`Flux Formulation 2 - Base-Lazy Valid Inequalities.ipynb`](Flux%20Formulation%202%20-%20Base-Lazy%20Valid%20Inequalities.ipynb): Second MIP model (M<sub>2</sub>) with Cuts (Theorems 2 to 9).
* [`Column Generation Approach [Spectral].ipynb`](Column%20Generation%20Approach%20%5BSpectral%5D.ipynb): Column Generation model (M<sub>3</sub>) with Columns Heuristic (Algorithm 1) and spectral complementary partitioning.

* [`Column Generation Approach [Spectral] - Valid Inequalities.ipynb`](Column%20Generation%20Approach%20%5BSpectral%5D%20-%20Valid%20Inequalities.ipynb): Column Generation model (M<sub>3</sub>) with Columns Heuristic (Algorithm 1) and spectral complementary partitioning.

The [`Instances`](Instances) subfolder contains each instance with a name identifying its generator, number of nodes, number of edges, expected \(k\), and expected \(\alpha\) (note, these last two can be overwritten).

The [`Tests - Table 1`](Tests%20-%20Table%201) folder contains the code and results for Table 1.

The [`Tests - Table 2`](Tests%20-%20Table%202) folder contains the code and results for Table 2.