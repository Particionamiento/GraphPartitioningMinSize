#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Model 1 with valid inequalities
#----------------------------------------------------------------------------
# Created by: AMT
# Created Date: 10/02/2023
# version ='1.0'
# ---------------------------------------------------------------------------

# **** Packages **** #
from   gurobipy   import *

import pandas as pd    # https://pandas.pydata.org/
import numpy  as np    # Numpy
import igraph as ig    # iGraph
import time

#from InstancesGenerator import Graph_Instance      # Not needed
from warnings     import warn
from os           import listdir
from re           import findall

from collections  import deque, Counter
from itertools    import chain 
from numpy.random import default_rng

from scipy.sparse        import csr_matrix, spdiags
from scipy.sparse.linalg import eigs
from k_means_constrained import KMeansConstrained

# Aliases
from numpy import argsort, delete, unique, floor, ceil, argsort, around



# **** Folder with instances **** #
Instance_path = 'Instances/'
Instances = [f for f in listdir(Instance_path) if f[0] != '.']

# **** Configuration of Gurobi **** 
Config = True              # Do we really want to use nonstandard parameters?



# **** Dictionary for storing metrics **** #
Out = {
    'Name' : [],
    'Instance': [],
    'z_R': [],
    'Obj': [],
    'gap': [],
    'nodes': [], 
    'time': [],
    'status': []
}






# **** RUN **** #
for ins in Instances:
    
    instance = [f[:-4] for f in Instances if f.startswith(ins)][0]
    
    '''
    if instance == 'Name_of_instance_to_skip':
        continue
    '''
    
    k, α = [int(i) for i in findall(r'\((.*?)\)',instance)[-1].split(',')]
    #file = 'Instances/{0},({1},{2}).pkl'.format(ins,k,α)
    file = 'Instances/{0}.pkl'.format(instance)
    print('\n*** {0} *** \n'.format(instance))
    
    # Read instance
    G = ig.Graph.Read_Pickle(file)
    # Get info from graph
    A = G.get_edgelist() # edges
    V = G.vs.indices     # nodes
    n = G.vcount()
    m = G.ecount()
    print('(n,m) =', (n,m),'\nCost:',sum(G.es['w']))
    
    # Collect costs
    d = {a:G.es[G.get_eid(a[0],a[1])]['w'] for a in G.get_edgelist()}
    
    # Create complementary graph
    E = sorted(set(A))                                                   # Original edges in directed graph
    Aᶜ = G.complementer(loops=False).get_edgelist()
    Eᶜ = sorted(set(Aᶜ))                                                 # Edges not in original graph
    TE = E + [(j,i) for (i,j) in E]                                      # All edges from complete graph
    
    # Determine feasibility
    K = tuplelist(range(k))               # Index of connected components
    β = n - (k-1)*α                       # Maximum number of nodes per component
    print('Created instance with (k,α) = ({0},{1}).'.format(
        k,α)) if α <= floor(n/k) else warn('\n\n*** α is too big! ***\n\n')
    
    # Scale costs
    D = 0.5 * sum(G.es['w']) / k
    
    # Compute L  and check feasibility
    Shortest_Paths = pd.DataFrame(G.shortest_paths(weights=None))
    Largest_SP = {i:j for (i,j) in Shortest_Paths.max(0).to_dict().items() if j >= β}
    𝖫 = Largest_SP.keys()
    q = len(𝖫)

    if q > k:
        warn('\n\n*** Infeasibility check! Corollary 1. ***\n\n')
    elif q > 0:
        print('There are',q,'nodes separated with a path with length greater than {0}.\n\n'.format(β))
    else:
        print('No nodes are separated with a path with length greater than {0}.\n\n'.format(β))

    β_Shortest_Paths = {i:Shortest_Paths.where(Shortest_Paths>=β)[i].dropna().to_dict() for i in 𝖫}
    
    # Build model using Gurobi
    if 'mo' in globals():        mo.dispose();    disposeDefaultEnv();    del mo
    
    mo = Model()
    x, x̄, y, f = tupledict(), tupledict(), tupledict(), tupledict()    # Dictionaries will contain variables

    # *** Variables ***
    # Link inside connected component #|x| = m * k
    x = mo.addVars(E, K, vtype = 'B', name = 'x')
    deque( (x[i,j,c].setAttr('obj', d[i,j]/D) for (i,j) in E for c in K), maxlen=0);    # Objective costs

    # Indicator of connected component #|y| = n * k
    y = mo.addVars(V, K, vtype = 'B', name = 'y')

    # Connected component without direct link #|x̄| = (n*(n-1)//2 - m) * k
    x̄ = mo.addVars(Eᶜ, K, vtype = 'B', name = 'x_c')

    # Flow on induced graph #|f| = n*n*(n-1) * k
    f = mo.addVars(TE, V, K, vtype = 'C', name = 'f')

    mo.update()

    # *** Constraints ***
    # Each node must belong exactly to one cluster
    mo.addConstrs((quicksum(y[i,c] for c in K) == 1 for i in V), name = 'R-1b-');
    # If two nodes i,j ∈ V are assigned to Vₗ, then the edge {i,j} ∈ E belongs to the induced subgraph (Vₗ,E(Vₗ))
    mo.addConstrs( (y[i,c]+y[j,c]-  x[i,j,c] <= 1.0 for (i,j) in E for c in K), name = 'R-1c-');
    mo.addConstrs( (y[i,c]+y[j,c]-2*x[i,j,c] >= 0.0 for (i,j) in E for c in K), name = 'R-1d-');
    # Connectivity between non-adjacent nodes within a connected component
    mo.addConstrs( (y[i,c]+y[j,c]-  x̄[i,j,c] <= 1.0 for (i,j) in Eᶜ for c in K), name = 'R-1e-');
    mo.addConstrs( (y[i,c]+y[j,c]-2*x̄[i,j,c] >= 0.0 for (i,j) in Eᶜ for c in K), name = 'R-1f-');
    # Capacity for the antiparallel arcs associated to edge {i, j} ∈ E
    mo.addConstrs((quicksum(f[i,j,l,c] + f[j,i,l,c] for l in V) <= (2*n*k) * x[i,j,c] for (i,j) in E for c in K), 
                  name ='R-1g-');
    # Flow conservation is ensured for all the nodes within the same connected component. One unit of flow is 
    # sent through a path linking two nodes l, j ∈ V in the same connected component whenever {l,j} ∉ E. 
    mo.addConstrs((quicksum(f[i,j,j,c] - f[j,i,j,c] for i in G.neighbors(j)) == quicksum(
                           -x̄[j,i,c] for i in V if (j,i) in Eᶜ)  for j in V for c in K), name = 'R-1h-a-');
    '''mo.addConstrs((quicksum(f[i,j,l,c] - f[j,i,l,c] for i in G.neighbors(j)) == 0.0
                   for j in V for l in V if ((l,j) in E and l<j)  for c in K), name = 'R-1h-b-');'''
    mo.addConstrs((quicksum(f[i,j,l,c] - f[j,i,l,c] for i in G.neighbors(j)) == 0.0
                   for (l,j) in E if l<j  for c in K), name = 'R-1h-b-');
    '''FC = mo.addConstrs((quicksum(f[i,j,l,c] - f[j,i,l,c] for i in G.neighbors(j)) == x̄[l,j,c] 
             for j in V for l in V if (l != j and ((l,j) in Eᶜ)) for c in K), name = 'R-1h-c-');'''
    FC = mo.addConstrs((quicksum(f[i,j,l,c] - f[j,i,l,c] for i in G.neighbors(j)) == x̄[l,j,c] 
                   for (l,j) in Eᶜ if l<j for c in K), name = 'R-1h-b-');
    # Lower bounds for the number of nodes present at each connected component
    mo.addConstrs((quicksum(y[i,c] for i in V) >= α for c in K), name = 'R-1i-');
    
    
    # Valid Inequalities
    
    # LC [Th 2]
    Leafs    = [i for i in V if G.degree(i) == 1];           print(len(Leafs),'leaf(s) detected.')
    Leafs_N  = [(i,G.neighbors(i)[0]) for i in Leafs]
    Leafs_NE = [e if e in E else e[::-1] for e in Leafs_N]

    mo.addConstrs( (quicksum(x[i,j,c] for c in K) == 1.0 for (i,j) in Leafs_NE), name = 'VI-LC-a');
    # Alternative ineq that is propagated from the previous
    #mo.addConstrs( (quicksum(y[i,c] + y[j,c] for c in K) == 2.0 for (i,j) in Leafs), name = 'VI-LC-c'); 
    mo.addConstrs( (quicksum(f[i,j,l,c] for l in V if l!=i for c in K) == 0.0 for (i,j) in Leafs_N), name = 'VI-LC-c');
    # This one also works but removes less variables
    #mo.addConstrs((quicksum(f[j,i,l,c] for l in V if l not in Leafs for c in K)==0.0 for (i,j) in Leafs_NE))
    
    # SCP [Th 3]
    if len(β_Shortest_Paths) > 0:
        # We avoid adding the constraint twice
        Covered = []
        for (i,r) in β_Shortest_Paths.items():
            for j in r.keys():
                '''if ([i,j] in Covered) or ([j,i] in Covered):
                    continue
                Covered.append([i,j])'''
                if {i,j} in Covered:
                    continue
                else:
                    Covered.append({i,j})
                    mo.addConstrs( (y[i,c] + y[j,c] <= 1.0 for c in K), name = 'VI-SCP-'+str({i,j})+'-')
                    
    # LBC [Th 9]
    mo.addConstr(quicksum(x[i,j,c] for (i,j) in E for c in K) >= n-k, name = 'VI-LBC');
    
    
    # Find integer solutions
    if Config:
        mo.Params.PreCrush = 1      # This one does not seem to be significantly different, though
        mo.Params.Cuts = 0
        mo.Params.Presolve = 0
    mo.Params.TimeLimit = 360.0 #60*60.0
    mo.optimize()
    
    Solution_Available = (mo.SolCount > 0)
    if Solution_Available:        print('\nIntegral solutions found!')

    print('\nCurrent objective: {0} ({1})'.format( around(mo.ObjVal,3), around(D*mo.ObjVal,3) ))
    
    
    # Display solution
    if Solution_Available:
        Vₖ = {c:[i for i in V if y[i,c].x>0.0] for c in K}
        print('Displaying partition\n',Vₖ)

    # Verify partition
    if Solution_Available:
        print('\nEffective partition [{0}] into components [{1}].'.format( 
            V == list(set().union(*Vₖ.values())), all([G.induced_subgraph(Vₖ[c]).is_connected() for c in K])) )
    
    # Store metrics
    Out['Name'].append(instance)
    Out['Instance'].append((n,m,k,α))
    Out['z_R'].append(mo.ObjBound * D)
    Out['Obj'].append(mo.objval * D)
    Out['gap'].append(mo.MIPgap)
    Out['nodes'].append(int(mo.nodecount))
    Out['time'].append(mo.RunTime)
    Out['status'].append(Solution_Available)
    
    # Store solution
    Data = pd.DataFrame.from_dict(
        {'Instance':[(n,m,k,α)],
         'z_R':[mo.ObjBound * D],'Obj': mo.objval * D,'gap': mo.MIPgap,'nodes':int(mo.nodecount), 'time':mo.RunTime})
    
    name_out = 'Results/Out[{0},{1},{2}]-M1-VI({3}).xlsx'.format(ins,k,α,Config)
    with pd.ExcelWriter(name_out) as writer:  
        Data.to_excel(writer, sheet_name='Sheet_1')
        if Solution_Available:
            pd.DataFrame.from_dict(Vₖ, orient='index').fillna('').to_excel(writer, sheet_name='Sheet_2')
    
    
    # Clean a little bit of memory
    del x, G
    
    
    # Tester
    #if instance == 'I(55)-1,(8,2)':        break













# **** Store consolidated summary **** #
pd.DataFrame.from_dict(Out).to_excel("Summary_M1-VI.xlsx") 

