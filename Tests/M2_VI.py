#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Model 2 with valid inequalities
#----------------------------------------------------------------------------
# Created by: AMT
# Created Date: 10/02/2023
# version = '1.0'
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
    
    # Determine feasibility
    K = tuplelist(range(k))               # Index of connected components
    β = n - (k-1)*α                       # Maximum number of nodes per component
    print('Created instance with (k,α) = ({0},{1}).'.format(
        k,α)) if α <= floor(n/k) else warn('\n\n*** α is too big! ***\n\n')
    
    # Scale costs
    D = 0.5 * sum(G.es['w']) / k
    
    
    # Collect costs
    d = {a:G.es[G.get_eid(a[0],a[1])]['w'] for a in G.get_edgelist()}
    
    # Create extended graph
    𝔸ᵏ = [n+i for i in K]
    Vᵏ = V + 𝔸ᵏ
    Eᵏ = [(i,j) for i in 𝔸ᵏ for j in V]
    d.update({e: 0.0 for e in Eᵏ})                                       # Extend costs

    Eₒ = sorted(set(A))
    E  = sorted( set(A).union(set(Eᵏ)) )                                 # Augmented graph
    Aᶜ = G.complementer(loops=False).get_edgelist()
    Eᶜ = sorted(set(Aᶜ))                                                 # Edges not in original graph
    TE = E + [(j,i) for (i,j) in sorted(set(A))]                         # All edges from complete graph
    
    
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
    # Link inside connected component #|x| = m + n*k
    x = mo.addVars(E, vtype = 'B', name = 'x')
    deque( (x[i,j].setAttr('obj', d[i,j]/D) for (i,j) in E), maxlen=0);    # Objective costs

    # Indicator of connected component #|y| = n * k
    y = mo.addVars(V, K, vtype = 'B', name = 'y')

    # Flow on induced graph #|f| = 2*m + n*k
    f = mo.addVars(TE, vtype = 'C', name = 'f')

    mo.update()

    # *** Constraints ***
    # Each node must belong exactly to one cluster
    mo.addConstrs((quicksum(y[i,c] for c in K) == 1 for i in V), name = 'R-2b-');
    # If two nodes i,j ∈ V are assigned to Vₖ, then the edge {i,j} ∈ E belongs to the induced subgraph (Vₖ,E(Vₖ))
    mo.addConstrs((y[i,c]+y[j,c]-x[i,j] <= 1.0  for (i,j) in Eₒ for c in K), name = 'R-2c-');
    # Edges with end nodes in different connected components must be equal to zero
    mo.addConstrs((y[i,c]+y[j,l]+x[i,j] <= 2.0 for (i,j) in Eₒ for c in K for l in K if c!=l), name = 'R-2d-');
    # Only one artificial edge is active
    mo.addConstrs((quicksum(x[i,j] for j in V) == 1.0 for i in 𝔸ᵏ), name = 'R-2e-');
    # Flow will be sent through the active xᵢⱼ
    mo.addConstrs((quicksum(f[i,j] - y[j,i-n] for j in V) == 0.0 for i in 𝔸ᵏ), name = 'R-2f-');
    # Flow conservation
    '''In_V  = {i:[(j,i) for j in G.neighbors(i)]+[(j,i) for j in Aᵏ] for i in V}
    Out_V = {i:[(i,j) for j in G.neighbors(i)] for i in V}
    mo.addConstrs((quicksum(f[i,j] for (i,j) in In_V[j]) - 
                   quicksum(f[j,i] for (j,i) in Out_V[j]) == 1.0  for j in V), name = 'R-2i-g-');'''
    FC = mo.addConstrs((quicksum(f[i,j] - f[j,i] for i in G.neighbors(j)) + 
                        quicksum(f[i,j] for i in Aᵏ) == 1.0  for j in V), name = 'R-2i-g-');

    # Capacities for the antiparallel arcs associated to edge {i,j} ∈ E
    mo.addConstrs((f[i,j] + f[j,i] <= β*x[i,j] for (i,j) in Eₒ), name = 'R-2h-');
    # Min and max flow
    mo.addConstrs((α * x[i,j] <= f[i,j] for (i,j) in Eᵏ), name = 'R-2i-a-');
    mo.addConstrs((β * x[i,j] >= f[i,j] for (i,j) in Eᵏ), name = 'R-2i-b-');
    
    
    # Valid Inequalities
    
    # LC [Th 2]
    Leafs = [(i,G.neighbors(i)[0]) for i in V if G.degree(i) == 1]
    Leafs = [e if e in E else e[::-1] for e in Leafs]
    print(len([i for i in V if G.degree(i) == 1]),'leaf(s) detected.')
    #mo.addConstrs((x[i,j] == 1.0 for (i,j) in Leafs), name = 'VI-LC-a');
    deque( (x[i,j].setAttr('lb', 1.0) for (i,j) in Leafs), maxlen=0);
    deque( (f[i,j].setAttr('ub', 1.0) for (i,j) in Leafs), maxlen=0);
    deque( (f[i,j].setAttr('lb', 1.0) for (i,j) in Leafs), maxlen=0);
    deque( (f[j,i].setAttr('lb', 0.0) for (i,j) in Leafs), maxlen=0);
    deque( (f[j,i].setAttr('ub', 0.0) for (i,j) in Leafs), maxlen=0);
    #mo.addConstrs( (quicksum(y[i,c] + y[j,c] for c in K) == 2.0 for (i,j) in Leafs), name = 'VI-LC-c');
    
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
    mo.addConstr(quicksum(x[i,j] for (i,j) in E) >= n-k, name = 'VI-LBC');
    
    
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
    
    name_out = 'Results/Out[{0},{1},{2}]-M2-VI({3}).xlsx'.format(ins,k,α,Config)
    with pd.ExcelWriter(name_out) as writer:  
        Data.to_excel(writer, sheet_name='Sheet_1')
        if Solution_Available:
            pd.DataFrame.from_dict(Vₖ, orient='index').fillna('').to_excel(writer, sheet_name='Sheet_2')
    
    
    # Clean a little bit of memory
    del x, G
    
    
    # Tester
    #if instance == 'I(55)-1,(8,2)':        break













# **** Store consolidated summary **** #
pd.DataFrame.from_dict(Out).to_excel("Summary_M2-VI.xlsx") 

