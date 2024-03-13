#!/usr/bin/env python
# -*- coding: utf-8 -*-

# CG-VIs with 100 rounds of CG
#----------------------------------------------------------------------------
# Created by: AMT
# Created Date: 4/11/2022
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

# **** Configuration of CG **** 
CG_Iterations = 100          # Number of column generation iterations
δ = 30                       # Max number of columns to be added each iteration


# **** Problem-independent functions **** #
def C_Cut(G, C):
    '''
        Return cut edges of a given node set C
        Input:
            G: Undirected graph
            C: subset of nodes in G
    '''
    Set_Incidence = [set(G.incident(i)) for i in C]        # List of incident edges for each node in C
    # Compute symmetric difference of all sets in Set_Incidence:
    freq = Counter(chain.from_iterable(Set_Incidence) )    # Obtain frequencies
    res  = {idx for idx in freq if freq[idx] == 1}         # Retrieve elements with frequency 1
    return res

def Initial_Partition(G, k, α):
    '''
        Return a initial partition F of V that induces connected components in G
        Input:
            G: Undirected graph
            k: Number of aimed connected components
            α: Minimum size of partitions
    '''
    
    F = {};    Q = G.vs.indices.copy()
    Ĝ = G.copy()
    # Create a dictionary to identify each node with its index in G
    Ids = {Ĝ.vs['_nx_name'][i]: Ĝ.vs.indices[i] for i in range(len(Ĝ.vs.indices)) }
    
    # Build k-1 connected components
    for c in range(k-1):
        C = {Q.pop(0)}
        # Add nodes to C
        while (len(C) < α):
            # Identify node in subgraph
            Id_C = { Ids[c] for c in C }
            # Find cut of C
            Cut_of_C = C_Cut(Ĝ, Id_C)
            # If cut is empty, this method is not suitable for finding a partition
            if len(Cut_of_C) == 0:
                Q = []    # This will send the message of a disconnected subgraph
                break
            # Retrieve weights from cut
            weights = {e: Ĝ.es[e]['w'] for e in Cut_of_C}
            # Smallest weight in cut
            Idₑ, cₑ = min(weights.items(), key = lambda x: x[1])
            # Identify edge in the original graph
            e = ( Ĝ.vs[Ĝ.es[Idₑ].source]['_nx_name'], Ĝ.vs[Ĝ.es[Idₑ].target]['_nx_name'])
            # Select node that will be added to C
            i = e[1] if e[0] in C else e[0]
            # Remove it from Q and add node
            Q.remove(i)
            C |= {i} 
        
        # Add nodes to component
        F[c] = sorted(C)
        # Reduce graph
        Ĝ   = Ĝ.induced_subgraph({ Ids[c] for c in Q })
        Ids = {Ĝ.vs['_nx_name'][i]: Ĝ.vs.indices[i] for i in range(len(Ĝ.vs.indices)) }
    
    # Last component is made up of the unused nodes.
    if G.induced_subgraph(Q).is_connected():
        F[k-1] = sorted(set(Q))
        print('Feasible partition found.')
        return F
    else:
        warn('No feasible partition found. Use another method to generate an initial partition to proceed.')
        return None
    
def C_neighbours(G,C):
    '''
        Return neighbours of a given set C
        Input:
            G: Undirected graph
            C: Set containing nodes in G 
    '''
    return set().union(*G.neighborhood(C, mindist = 1)) - C

# Check viability of neighborhood selection
def Heuristic_Pricing(G, F, α, β, π, γ, Infeasible,   D, S):
    '''
        Return set of connected components from G
        Input:
            G:          Undirected graph
            F:          Set of feasible connected components
            (α,β):      Lower and upper bounds on size of components
            (π,γ):      Shadow prices associated with nodes and adding a column
            Infeasible: Set with forbidden components
    '''
    
    P  = []                         # To store generated connected components, reduced cost, and real cost
    NC = []                         # To store generated connected components
    for v in G.vs.indices:
        C = {v}
        while (len(C) <= β - 1):
            N = C_neighbours(G,C)   # Obtain the neighborhood of C
            N = {i for i in N if abs(π[i]) > 0.0 } # Only add nodes that contribute something
            
            if N == set():                break
            
            # We evaluate the reduced costs for expanding C with elements of the neighborhood
            r_ŝ = np.inf            # Smallest reduced cost
            ŝ   = -1                # Label of node associated with r_ŝ
            c_f = 0                 # Cost of C ∪ {ŝ}
            
            # Evaluate reduced cost for each member of the neighborhood
            for s in N:
                # Expand C
                Cₛ = list(C | {s})
                # Check if resulting component is not forbidden
                if sorted(Cₛ) in Infeasible:
                    continue                
                # Compute cumulative costs of C ∪ {s}
                cf = sum(G.subgraph(Cₛ).es['w'])/D if (len(C) <= 55) else (0.5 * S[np.ix_(Cₛ, Cₛ)].A.sum())
                
                # Compute reduced cost of C ∪ {i}
                reduced = cf - sum(π[e] for e in Cₛ) - γ
                if reduced < r_ŝ:
                    ŝ   = s
                    r_ŝ = reduced
                    c_f = cf
            
            # Check if all possible expansions weren't feasible
            if ŝ < 0:
                break
            # Add node with smallest reduced cost
            C |= {ŝ}
            Cₛ = sorted(C)
            
            # Preserve column only if it hasn't been added already
            if Cₛ not in F.values():
                if (len(Cₛ) >= α and Cₛ not in NC and r_ŝ < 0):
                    NC.append(Cₛ)
                    P.append([Cₛ, r_ŝ, c_f])
    return P




# **** Dictionary for storing metrics **** #
Out = {
    'Name' : [],
    'Instance': [],
    'z_R': [],
    'Obj': [],
    'gap': [],
    'nodes': [], 
    'time': [],
    'CG iterations': []
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
    
    # Determine feasibility
    K = tuplelist(range(k))               # Index of connected components
    β = n - (k-1)*α                       # Maximum number of nodes per component
    print('Created instance with (k,α) = ({0},{1}).'.format(
        k,α)) if α <= floor(n/k) else warn('\n\n*** α is too big! ***\n\n')
    
    # Scale costs
    D = 0.5 * sum(G.es['w']) / k
    
    # Find an initial partition
    F = Initial_Partition(G, k, α)
    print(F)
    
    # If an initial partition is not found, then create a spectral partitioning
    smart_init = False
    if F is None:
        smart_init = True

    # Build cost matrix using adjacency data
    S  = csr_matrix(( list(d.values()), ([i for (i,j) in d.keys()], 
                                         [j for (i,j) in d.keys()])), shape=(n, n), dtype='float' )
    S += S.T
    S /= D
    
    
    # *** Create base spectral embedding ***
    dₛ = S.sum(axis=1).A1                                 # Weighted degree
    P = spdiags(1/dₛ, 0, n, n, format='csr').dot(S)       # Transition matrix
    λₖ, eₖ = eigs(P, k, which='LR') # ojo
    # Order obtained eigenvalues and retrieve encoding
    ordered = argsort(λₖ.real)
    eₖ = eₖ[:,ordered].real[:,:-1]
    λₖ = λₖ[ordered].real[:-1]
    
    
    # Create spectral partition
    if smart_init:
        km = KMeansConstrained(n_clusters = k, size_min = α, size_max = β,
                       random_state=0, n_init=20, tol = 1e-10, verbose=0, n_jobs=-1).fit_predict(eₖ)
        #print(km)
        F = {c: sorted([i for i in V if km[i]==c]) for c in K}
    
    # S allows for fast computation of costs associated to components
    c = {i: 0.5 * S[np.ix_(F[i],F[i])].A.sum() for i in F.keys()}
    
    
    
    # Build model using Gurobi
    if 'mo' in globals():        mo.dispose();    disposeDefaultEnv();    del mo
    
    mo = Model()
    x  = tupledict()    # Dictionaries will contain variables
    
    # *** Variables ***
    x = mo.addVars(F.keys(), vtype = 'B', name = 'x', obj = c)        # Indicators of each component
    # *** Constraints ***
    # Each node is part of just one cluster (As there is only one partition, we only iterate over its elements)
    for i,f in F.items():
        mo.addConstrs( (x[i] == 1 for j in f), name = 'Rb')
        #for i in f:mo.addConstr(x[i]==1, name='R[{0}]'.format(i))
    # k connected components have to be selected
    mo.addConstr( x.sum() == k, name = 'Rc')
    mo.update()
    
    #mo.write('Problem[{0},{1},{2}].lp'.format(ins,k,α))
    print('')
    mo.optimize()
    print('\nCurrent objective is: ', int(D * mo.ObjVal), '.', sep='')
    
    
    # Initialise CG
    RelObj = []
    κ = len(F)                  # Number of connected components
    
    # Problem-dependent functions
    def Complementary_Partions(G, Fixed, S, eₖ, k, α, β):
        '''
            Return set of connected components from G
            Input:
                G:     Undirected graph
                Fixed: Initial connected component
                S:     Weight matrix from adjacency data of G
                eₖ:    Spectral embedding for the nodes of G
                k:     Number of connected components in the partition
                (α,β): Lower and upper bounds on size of components
        '''
        # Complementary nodes
        UnFixed = delete(G.vs.indices, Fixed)
        # Algorithm may fail for recyling, in that case we can compute eigeninformation from scratch
        retry = 1

        # Check if induced subgraph is connected:
        if G.induced_subgraph(UnFixed).is_connected():
            # Recyle eigen-information
            eF = delete(eₖ,Fixed,axis=0)
            km = KMeansConstrained(n_clusters = k-1, size_min = α, size_max = (G.vcount()-len(Fixed)) - (k-2)*α,
                           random_state=0, n_init=20, tol = 1e-10, verbose=0, n_jobs=-1).fit_predict(eF)
            Vₖ = [ [UnFixed[i] for i in range(len(km)) if km[i]==c] for c in range(k-1)]
            cₖ = [0.5 * S[np.ix_(Vₖ[c], Vₖ[c])].A.sum() for c in range(k-1)]

            retry = all([G.induced_subgraph(Vₖ[c]).is_connected() for c in range(k-1) ]) * 1

        # For disconnected graphs, it might be needed to compute new eigenvectors
        if (not G.induced_subgraph(UnFixed).is_connected()) or (retry == 0):
            # Identify connected component
            Members = G.induced_subgraph(UnFixed).clusters().membership
            # Check if number of components is at most k-1:
            p = unique(Members).size
            # If there are more than p components, the fixed component cannot be in a feasible solution [FCNF]
            if p > k-1:
                return {}, {}, False
            # Build each component, evaluate its cost and size
            Components = [ [UnFixed[i] for i in range(UnFixed.size) if Members[i]==c] for c in range(p)]
            Comp_Costs = [ S[np.ix_(Components[c], Components[c])].A.sum() for c in range(p)]
            Comp_Sizes = [ len(Components[c]) for c in range(p)]
            # If one component is smaller than α: [FCNF]
            if all([ α <= n̂ for n̂ in Comp_Sizes]) == False:
                return {}, {}, False
            # If there are k-1 components and all are smaller than β: Partition already found
            if p == k-1:
                if all([ β >= n̂ for n̂ in Comp_Sizes]) == True:
                    return Components, Comp_Costs, True
                # If at least one component is too big: [FCNF] # Nunca se va a cumplir, ala Ramiro
                else:
                    return {}, {}, False
            # Evaluate whether the components are partitionable
            Feasible_parts = [ [k̂ for k̂ in range(1,k-p+1) if α <= floor(Comp_Sizes[i]/k̂) ] for i in range(p) ]
            Feasible_parts = [max(k̂) for k̂ in Feasible_parts] # Max number of possible partitions in each component
            # If the sum of all these possible partitions is less than k-1: [FCNF]
            if sum(Feasible_parts) < k-1:
                return {}, {}, False

            '''If we have got here, it means we can actually find a complementary partition to Fixed'''
            # Determine the number of partitions for each component
            if p == 1:
                # No need to use linear programming tool
                Parts = [Feasible_parts[0]]
            else:
                m̂ = Model();            m̂.ModelSense = -1;            m̂.Params.OutputFlag = False;
                y = tupledict()
                y = m̂.addVars(range(p), vtype = 'I', name = 'y')
                # Objective, lower, and upper bounds
                deque( ( ( y[i].setAttr('obj', Comp_Costs[i]), 
                           y[i].setAttr('ub', Feasible_parts[i]), y[i].setAttr('lb', 1.0) )
                         for i in range(p)), maxlen=0);
                m̂.addConstr(y.sum() == k-1)
                m̂.optimize()
                Parts = [ int(around(y[i].x,0)) for i in range(p)]

            # Partition each connected component on the given number of partitions
            Vₖ, cₖ = [], []
            for i in range(p):
                # Assemble projected transition matrix
                Sᵢ = S[np.ix_(Components[i], Components[i])]
                # Verify if partition is not trivial
                if Parts[i] > 1:
                    dᵢ = Sᵢ.sum(axis=1).A1
                    Pᵢ = spdiags(1/dᵢ, 0, len(Components[i]), len(Components[i]), format='csr').dot(Sᵢ)
                    # Compute eigen-info again
                    kᵢ = Parts[i] if (Parts[i] > 1) else (Parts[i] + 1)
                    λᵢ, eᵢ = eigs(Pᵢ, kᵢ, which='LR')
                    # Encode nodes in component            
                    ordered = argsort(λᵢ.real)
                    eᵢ = eᵢ[:,ordered].real[:,:-1]
                    λᵢ = λᵢ[ordered].real[:-1]
                    # Run constrained kᵢ-means
                    km = KMeansConstrained(n_clusters = kᵢ, size_min = α, size_max = Comp_Sizes[i] - (kᵢ-1)*α,
                               random_state=0, n_init=20, tol = 1e-10, verbose=0, n_jobs=-1).fit_predict(eᵢ)
                    #print(km)
                    Out = [ [Components[i][j] for j in range(len(km)) if km[j]==c] for c in unique(km)]
                    #print([G.induced_subgraph(Out[c]).is_connected() for c in range(kᵢ)])

                # The trivial case can avoid the spectral step
                else:
                    Out = [ Components[i] ]

                Vₖ += Out
                cₖ += [S[np.ix_(c, c)].A.sum() for c in Out]

        return Vₖ, cₖ, True
    def Update_Model(mo, κ, k, s, c, Parts_Id):
        '''
            Update gurobi model with additional connected components in G
            Input:
                mo:       Gurobi model
                κ:        Current number of connected components (and variables)
                k:        Size of partition
                s:        Number of components to add
                c:        Costs associated with each component
                Parts_Id: Nodes from G associated with the connected component they belong to
        '''
        # Add new variables with their cost
        x.update(mo.addVars( range(κ,κ+s), vtype = 'B', name='x', obj = c))
        mo.update()
        # Update constraints
        sense, rhs = '=', 1.0
        # Add each component with respect to its constituent nodes
        for i,j in Parts_Id.items():
            nom  = 'Rb[{0}]'.format(i)                # Name the constraint
            co   = mo.getConstrByName(nom)            # Retrieve constraint
            lhs  = mo.getRow(co)                      # Get info
            mo.remove(co)                             # Remove constraint
            lhs += x[j]                               # Add new information
            mo.addLConstr(lhs, sense, rhs, nom)        # Add new constraint

        # Number of components is fixed to k
        co   = mo.getConstrByName('Rc')
        lhs  = mo.getRow(co)
        mo.remove(co)                                  # Remove constraint
        lhs += quicksum(x[i] for i in range(κ,κ+s))    # Add new information
        mo.addLConstr(lhs, sense, k, 'Rc')              # Add new constraint
        mo.update()

        return None
    
    
    # Relax model and optimize
    r = mo.relax();    r.Params.OutputFlag = False;   r.update();    r.optimize()
    # Retrieve shadow prices
    π  = {i: r.getConstrByName('Rb[{0}]'.format(i)).Pi for i in V}
    γ  = r.getConstrByName('Rc').Pi
    # Collect current objective
    RelObj.append(r.ObjVal)
    # Infeasible columns cannot be tested again
    Infeasible = []
    
    
    start = time.time()

    # Run algorithm
    Effective_Iterations = CG_Iterations
    for ı in range(CG_Iterations):

        # Obtain list of components and sort it w.r.t. reduced cost
        P̂ = Heuristic_Pricing(G, F, α, β, π, γ, Infeasible,   D, S)
        P̂ = sorted(P̂, key = lambda x: x[1])[:δ]
        if len(P̂) == 0:
            Effective_Iterations = ı
            break

        # Generate spectral complementary partition
        for ȷ in range( min(δ, len(P̂)) ):
            # Fix a connected component
            Fixed = P̂[ȷ][0]
            # Run algorithm 3
            Vₖ, cₖ, Fixed_Feasibility = Complementary_Partions(G, Fixed, S, eₖ, k, α, β)
            # If the fixed componnent is not feasible, we add it to the forbidden list
            if not Fixed_Feasibility:
                Infeasible.append(Fixed)
                continue
            else:
                # Remove components that have already been created
                Newies = {a:b for a,b in enumerate(Vₖ) if b not in F.values()}
                cₖ = [cₖ[i] for i in Newies.keys()]
                Vₖ = list(Newies.values())
                # Add fixed component to partition
                Vₖ.append(Fixed)
                cₖ.append( P̂[ȷ][2] )
                # Add components to F and store their costs
                F.update({ a+κ: b for a,b in enumerate(Vₖ) })
                c.update({ a+κ: b for a,b in enumerate(cₖ) })

                # Identify each connected component with a number
                Parts_Id = dict()
                for part in range( len(Vₖ) ):
                    Parts_Id.update({node: part+κ for node in Vₖ[part]})
                # Update model with partition    
                Update_Model(mo, κ, k, len(Vₖ), c, Parts_Id)
                # Update counter of available columns (at most k)
                κ += len(Vₖ)

        # Relax model and optimize
        r = mo.relax();    r.Params.OutputFlag = False;   r.update();    r.optimize()
        # Retrieve shadow prices
        π  = {i: r.getConstrByName('Rb[{0}]'.format(i)).Pi for i in V}
        γ  = r.getConstrByName('Rc').Pi
        # Collect current objective
        RelObj.append(r.ObjVal)

    end = time.time()
    print('CG :: Time taken for {0} iterations: {1}'.format(Effective_Iterations, end-start))
    
    
    # Valid inequalities
    # CGC [Th 12]

    # Lower and upper bounds
    Lₗ  = int(ceil(n/k)) if ceil(n/k) > α else int(ceil(n/k) + 1)
    # Add constraints
    for ℓ in range(Lₗ, β+1):
        # Select connected components of size at most ℓ
        H_ℓ = [a for a,b in F.items() if len(b) >= ℓ ]
        # If the length of H_ℓ is less than 2, we are not adding any meaningful information to the model
        if len(H_ℓ) <= 1:
            break
        max_feas = floor( (n-k*α)/(ℓ-α) )
        # Similarly if the length of H_ℓ is less or equal than max_feas:
        if len(H_ℓ) <= max_feas:
            break
        else:
            mo.addConstr( quicksum(x[f] for f in H_ℓ) <= floor((n-k*α)/(ℓ-α)), name = 'VI-CGC-[{0}]'.format(ℓ - Lₗ))
            #print( len(H_ℓ), max_feas)
    print('Added {0} additional VIs.'.format(ℓ - Lₗ))
    
    
    # Find integer solutions
    mo.Params.PreCrush = 1      # This one does not seem to be significantly different, though
    mo.Params.Cuts = 0
    mo.Params.Presolve = 0
    mo.Params.TimeLimit = 3600.0 #60*60.0
    mo.optimize()
    
    print('\nNº of columns:     {0}\nTime spent in CG:  {1}'.format(len(F), end-start ))
    print('CG iterations:     {0}'.format(Effective_Iterations))
    print('Relaxed objective: {0} ({1})\nInteger objective: {2} ({3})'.format(
        around(RelObj[-1],3), around(D*RelObj[-1],3), around(mo.ObjVal,3), around(D*mo.ObjVal,3) ))
    
    
    
    # Retrieve solution
    X = [i for i in x.keys() if x[i].x > 1e-8]
    #print(X)
    
    # Display solution
    Vₖ = {i: F[j] for i,j in enumerate(X)}
    print('Displaying partition\n',Vₖ)
    
    # Verify partition
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
    Out['CG iterations'].append(Effective_Iterations)

    
    # Store solution
    Data = pd.DataFrame.from_dict(
        {'Instance':[(n,m,k,α)],
         'z_R':[mo.ObjBound * D],'Obj': mo.objval * D,'gap': mo.MIPgap,'nodes':int(mo.nodecount), 'time':mo.RunTime})
    
    name_out = 'Results/Out[{0},{1},{2}]-CG-VI({3}).xlsx'.format(ins,k,α,Effective_Iterations)
    with pd.ExcelWriter(name_out) as writer:  
        Data.to_excel(writer, sheet_name='Sheet_1')
        pd.DataFrame.from_dict(Vₖ, orient='index').fillna('').to_excel(writer, sheet_name='Sheet_2')
    
    
    # Clean a little bit of memory
    del x, F, G, S
    
    # Tester
    #if instance == 'I(55)-1,(8,2)':        break













# **** Store consolidated summary **** #
pd.DataFrame.from_dict(Out).to_excel("Summary_CG-VI-100.xlsx") 

