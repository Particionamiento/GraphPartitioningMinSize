# Associated data and code

In this folder, the code and results for Table 2 are included.

## Files in this folder

* [`ReadMe.md`](ReadMe.md): This file.

Python scripts for the Graph Partitioning Problem in Connected Components with Minimum Size Constraints. The Gurobi package is needed to test the code for the linear models. The scripts are presented _as it is_.

* [`M1_VI.py`](M1_VI.py): First MIP model (M<sub>1</sub>) with valid inequalities (Theorems 2 to 9).
 Second MIP model (M<sub>2</sub>).
* [`M2_VI.py`](M2_VI.py): Second MIP model (M<sub>2</sub>) with valid inequalities (Theorems 2 to 9).
* [`CG_VIs_30.py`](CG_VIs_30.py) and [`CG_VIs_100.py`](CG_VIs_100.py): Column Generation model (M<sub>3</sub>) with Columns Heuristic (Algorithm 1) and valid inequalities (Theorems 12 & 13).
* [`Instancias.ipynb`](Instancias.ipynb): Instance generator.

The best integer solutions found, if any, are in the folders [`MIP_1`](MIP_1), [`MIP_2`](MIP_2), and [`CG`](CG), respectively. The Excel files starting with `Summary` compile all the results for different configurations of the solution approaches and solvers. 