# Import required functions
from numpy        import argwhere, triu, asarray, concatenate, zeros, eye
from igraph       import Graph
from numpy.random import default_rng, seed
from scipy.stats  import bernoulli
from networkx     import dense_gnm_random_graph, watts_strogatz_graph, erdos_renyi_graph

rng = default_rng(0)
def weight_generator(m): 
    rng = default_rng(0)
    return rng.integers(1, 10, m, endpoint=True)
def iGnx_gnm(n,m,s): return Graph.from_networkx(dense_gnm_random_graph(n,m,s))
def Gen_Instance_Uniform(n,p,s):
    rng  = default_rng(s)
    Prob = rng.uniform(0,1,(n,n))
    Prov = argwhere(triu(Prob, 2) >= p)
    Tup  = [(i,i+1) for i in range(0,n-1)];    Tup += [ (a[0],a[1]) for a in Prov ];    Tup = sorted(Tup)
    G = Graph.TupleList(Tup)
    return G
def Erdos_Graph(n,p,s):
    rng  = default_rng(s)
    G = Graph()
    G.add_vertices(n)
    G.add_edges([ (i,j) for i in range(n) for j in range(n) if i < j and bernoulli.rvs(p) ])
    return G

def Graph_Instance(name):
    rng = default_rng(0)
    seed(0)
    '''
        Other graphs
    '''
    G = Graph()
    if name == 'Atlas-G200':        G = Graph.Atlas(200)
    if name == 'Ford-Fulkerson':
        G = Graph.TupleList([(0,1,7),(0,2,3),(1,2,1),(1,3,6),(2,3,3),(2,4,8),(3,4,2),(3,5,2),
                             (4,5,8),(0,6,5),(2,6,10),(6,7,4),(6,8,3),(4,7,2),(5,7,6),(7,8,2),
                             (1,9,7),(3,9,7)], edge_attrs = 'w')
    if name == 'WSG(60)':
        G = Graph.from_networkx(watts_strogatz_graph(60, 3, 0.5, 38))
    if name == 'ERG(60)-1':
        G = erdos_renyi_graph(60, 0.05, 39)
        G.add_edges_from([(2,16),(2,45), (16,17),(22,33)])
        G = Graph.from_networkx(G)
    if name == 'ERG(60)-2':
        G = Graph.from_networkx(erdos_renyi_graph(60, 0.1, 40))
        
    if name == 'Tree(33)':
        G = Graph.Tree(33,3)
    
    '''
        Graphs generated using an uniform distribution
    '''
    if name == 'IU(50)-1':
        G = Gen_Instance_Uniform(50,1-0.05,31)
    if name == 'IU(50)-2':
        G = Gen_Instance_Uniform(50,1-0.15,32)
    if name == 'IU(50)-3':
        G = Gen_Instance_Uniform(50,1-0.20,33)
    if name == 'IU(50)-4':
        G = Gen_Instance_Uniform(50,1-0.30,34)
    if name == 'IU(50)-5':
        G = Gen_Instance_Uniform(50,1-0.40,35)
    if name == 'IU(60)-1':
        G = Gen_Instance_Uniform(60,1-0.05,36)
    if name == 'IU(58)-2':
        G = Gen_Instance_Uniform(58,1-0.15,37)
    
    '''
        General dense instances
    '''
    if name == 'I(20)-1':
        G = Graph.TupleList([(0, 1, 1), (0, 6, 1), (0, 8, 2), (0, 12, 5), (0, 19, 7), (1, 2, 6), (1, 3, 7), 
                     (1, 6, 7), (1, 13, 6), (1, 14, 3), (1, 15, 8), (1, 17, 5), (1, 19, 4), (2, 3, 7), 
                     (2, 7, 5), (2, 8, 2), (2, 12, 6), (2, 16, 10), (2, 17, 1), (2, 19, 7), (3, 4, 4), 
                     (3, 6, 3), (3, 9, 10), (3, 10, 4), (3, 12, 4), (3, 16, 4), (3, 18, 3), (4, 5, 1),
                     (4, 11, 4), (4, 13, 3), (5, 6, 1), (5, 11, 4), (5, 12, 1), (5, 17, 1), (6, 7, 8),
                     (6, 8, 4), (6, 10, 9), (6, 11, 8), (6, 18, 5), (7, 8, 5), (7, 17, 1), (8, 9, 1), 
                     (8, 12, 4), (8, 13, 4), (8, 14, 2), (8, 15, 10), (9, 10, 5), (9, 15, 6), (9, 19, 8),
                     (10, 11, 1), (10, 12, 4), (10, 14, 6), (10, 18, 7), (11, 12, 2), (11, 17, 5), 
                     (11, 18, 5), (12, 13, 5), (12, 14, 6), (12, 15, 5), (13, 14, 7), (13, 17, 4), 
                     (14, 15, 1), (15, 16, 6), (15, 17, 9), (15, 18, 4), (16, 17, 7), (16, 18, 3), 
                     (17, 18, 9), (18, 19, 9)], edge_attrs = 'w')
    if name == 'I(20)-2':
        G = iGnx_gnm(20, 83, 0)
    if name == 'I(20)-3':
        G = iGnx_gnm(20, 95, 1)
    if name == 'I(20)-4':
        G = iGnx_gnm(20, 150, 2)
    if name == 'I(20)-5':
        G = iGnx_gnm(20, 108, 3)
    if name == 'I(25)-1':
        G = iGnx_gnm(25, 120, 4)
    if name == 'I(25)-2':
        G = iGnx_gnm(25, 180, 5)
    if name == 'I(25)-3':
        G = iGnx_gnm(25, 210, 6)
    if name == 'I(25)-4':
        G = iGnx_gnm(25, 135, 7)
    if name == 'I(25)-5':
        G = iGnx_gnm(25, 245, 8)
    if name == 'I(30)-1':
        G = iGnx_gnm(30, 218, 9)
    if name == 'I(30)-2':
        G = iGnx_gnm(30, 145, 10)
    if name == 'I(30)-3':
        G = iGnx_gnm(30, 300, 11)
    if name == 'I(30)-4':
        G = iGnx_gnm(30, 325, 12)
    if name == 'I(30)-5':
        G = iGnx_gnm(30, 261, 13)
    if name == 'I(35)-1':
        G = iGnx_gnm(35, 250, 14)
    if name == 'I(35)-2':
        G = iGnx_gnm(35, 300, 15)
    if name == 'I(35)-3':
        G = iGnx_gnm(35, 238, 16)
    if name == 'I(35)-4':
        G = iGnx_gnm(35, 400, 17)
    if name == 'I(35)-5':
        G = iGnx_gnm(35, 445, 18)
        
    if name == 'I(40)-1':
        G = iGnx_gnm(40, 312, 19)
    if name == 'I(40)-2':
        G = iGnx_gnm(40, 390, 20)
    if name == 'I(40)-3':
        G = iGnx_gnm(40, 468, 21)
    if name == 'I(40)-4':
        G = iGnx_gnm(40, 507, 22)
    if name == 'I(40)-5':
        G = iGnx_gnm(40, 550, 23)
        
    if name == 'I(45)-1':
        G = iGnx_gnm(45, 495, 24)
    if name == 'I(45)-2':
        G = iGnx_gnm(45, 400, 25)
    if name == 'I(45)-3':
        G = iGnx_gnm(45, 643, 26)
    if name == 'I(45)-4':
        G = iGnx_gnm(45, 600, 27)
    if name == 'I(45)-5':
        G = iGnx_gnm(45, 700, 28)
        
    if name == 'I(50)-1':
        #G = iGnx_gnm(50, 105, 29) # gets disconnected graph
        G = iGnx_gnm(50, 105, 0) # gets disconnected graph
    if name == 'I(50)-2':
        G = iGnx_gnm(50, 220, 39)
    if name == 'I(50)-3':
        G = iGnx_gnm(50, 298, 49)
    if name == 'I(50)-4':
        G = iGnx_gnm(50, 408, 59)
    if name == 'I(50)-5':
        G = iGnx_gnm(50, 529, 69)
        
    if name == 'I(55)-1':
        G = iGnx_gnm(55, 800, 51)
    if name == 'I(55)-2':
        G = iGnx_gnm(55, 594, 52)
    if name == 'I(55)-3':
        G = iGnx_gnm(55, 891, 53)
    if name == 'I(55)-4':
        G = iGnx_gnm(55, 350, 54)
    if name == 'I(55)-5':
        G = iGnx_gnm(55, 550, 55)
    if name == 'I(55)-6':
        G = iGnx_gnm(55, 445, 30)
        
    if name == 'I(60)-1':
        G = iGnx_gnm(60, 150, 31)
    if name == 'I(60)-2':
        G = iGnx_gnm(60, 291, 32)
    if name == 'I(60)-3':
        G = iGnx_gnm(60, 60, 33) # gets disconnected graph by a quite obvious argument
        G = iGnx_gnm(60, 500, 33)
    if name == 'I(60)-4':
        #G = iGnx_gnm(60, 102, 34) # gets disconnected graph
        G = iGnx_gnm(60, 102, 4)
    if name == 'I(60)-5':
        G = iGnx_gnm(60, 182, 35)
        
    if name == 'I(65)-1':
        G = iGnx_gnm(65, 832, 56)
    if name == 'I(65)-2':
        G = iGnx_gnm(65, 1248, 57)
    if name == 'I(65)-3':
        G = iGnx_gnm(65, 728, 58)
    if name == 'I(65)-4':
        G = iGnx_gnm(65, 1352, 59)
    if name == 'I(65)-5':
        G = iGnx_gnm(65, 1000, 60)
    if name == 'I(70)-1':
        G = iGnx_gnm(70, 497, 61)
    if name == 'I(70)-2':
        G = iGnx_gnm(70, 1000, 62)
    if name == 'I(70)-3':
        G = iGnx_gnm(70, 800, 63)
    if name == 'I(70)-4':
        G = iGnx_gnm(70, 248, 64)
    if name == 'I(70)-5':
        G = iGnx_gnm(70, 1500, 65)
    if name == 'I(80)-1':
        G = iGnx_gnm(80, 500, 41)
    if name == 'I(80)-2':
        G = iGnx_gnm(80, 600, 42)    
    if name == 'I(80)-3':
        G = iGnx_gnm(80, 800, 43)    
    if name == 'I(100)-1':
        G = iGnx_gnm(100, 800, 44)
    if name == 'I(100)-2':
        G = iGnx_gnm(100, 900, 45)
    if name == 'I(150)':
        G = iGnx_gnm(150, 900, 46)
    
    '''
        Erdős graphs
    '''
    if name == 'IEG(35)':
        G = Erdos_Graph(30,0.1,47)
        G.add_vertices(5)
        G.add_edges([(5,30),(30,31),(31,32),(32,33),(33,34)])
    if name == 'IEG(30)':
        G = Erdos_Graph(24,0.3,48)
        G.add_vertices(6)
        G.add_edges([(0,24),(24,25),(25,26),(26,27),(27,28),(28,29)])
    if name == 'IEG(40)':
        G = Erdos_Graph(30,0.3,48)
        G.add_vertices(10)
        G.add_edges([(0,30),(30,31),(31,32),(32,33),(33,34),(20,35),(35,36),(36,37),(21,38),(21,39)])
    if name == 'IEG(45)':
        G = Erdos_Graph(40,0.3,49)
        G.add_vertices(5)
        G.add_edges([(0,30),(30,31),(31,32),(32,33),(33,34),(34,35),(20,36),(36,37),
                     (37,38),(38,39),(21,40),(40,41),(41,42),(42,43),(43,44)])
    if name == 'IEG(55)':
        G = Erdos_Graph(40,0.1,50)
        G.add_vertices(15)
        G.add_edges([(13,39),(39,40),(40,41),(41,42),(42,43),(22,44),(44,45),(45,46),
                     (46,47),(47,48),(48,49),(16,50),(50,51),(51,52),(52,53),(53,54)])
        
    '''
        Additional graphs for further tests
    '''
    if name == 'Tailed_Tree':
        G = Graph.Tree(33,3)
        tailₐ = 20
        G.add_vertices(2*tailₐ)
        G.add_edges([(i,i+1) for i in range(32,32+tailₐ)])
        G.add_edges([(30,32+tailₐ+1)] + [(i,i+1) for i in range(32+tailₐ+1,32+2*tailₐ)])
        
    if name == 'Glued_Mitosis':
        G = Graph.Erdos_Renyi(20, 0.2)
        A = asarray(G.get_adjacency().data);        s = A.shape[0]
        B = concatenate([A, zeros((s,2*s))], axis=1)
        B = concatenate([B, concatenate([zeros((s,s)), 
                                               eye(s)+eye(s,k=-1)+eye(s,k=1), zeros((s,s))], axis=1)], axis=0)
        B = concatenate( [B, concatenate([zeros((s,2*s)), A], axis=1)], axis=0)
        B[s,s-7] = B[s-7,s] = 1.0
        B[2*s-1,(2*s)+7] = B[(2*s)+7,2*s-1] = 1.0
        #plt.imshow(B);    plt.axis('off');    plt.show()
        G = Graph.Adjacency((abs(B) > 0).tolist())
    
    if name == '4_Tails':
        G.add_vertices(41)
        G.add_edges(
                    [(i,i+1) for i in range(20)] + [(10,21)] 
                        + [(21+i,21+i+1) for i in range(9)] + [(10,31)] + [(31+i,31+i+1) for i in range(9)]
                   )
    
    '''
        Default
    '''
    if len(G.vs.indices) == 0:
        G = Graph.Famous('Zachary')
    
    if len(G.es.attribute_names()) == 0:
        rng = default_rng(0)
        seed(0)
        G.es['w'] = weight_generator(G.ecount())        # Generate random weights
    
    return G