# LayoutOptimization - Enrico Stragiotti - Feb 2021
# mm, N, MPa

## Basic imports
import bcs as BCS
import trussplot
import time
import ipopt_routines
import os
import shelve
import copy
#import trussfem
import cvxpy as cvx
import numpy as np
import scipy.sparse as sp
import cyipopt
#import mosek

#################################
## Generic methods
#################################

def calcB(myBCs: BCS.MBB2D_Symm):
    m, nA, nB = len(myBCs.ground_structure), myBCs.ground_structure[:,0].astype(int), myBCs.ground_structure[:,1].astype(int)
    l, X, Y = myBCs.ground_structure_length, myBCs.nodes[nB,0]-myBCs.nodes[nA,0], myBCs.nodes[nB,1]-myBCs.nodes[nA,1]
    d0, d1, d2, d3 = myBCs.dofs[nA*2], myBCs.dofs[nA*2+1], myBCs.dofs[nB*2], myBCs.dofs[nB*2+1] # Defined on boundary conditions. One line for each candidate
    s = np.concatenate((-X/l * d0, -Y/l * d1, X/l * d2, Y/l * d3))
    r = np.concatenate((nA*2, nA*2+1, nB*2, nB*2+1)) # dofs
    c = np.concatenate((np.arange(m), np.arange(m), np.arange(m), np.arange(m)))
    B = sp.coo_matrix((s, (r, c)), shape = (len(myBCs.dofs), m))
    sp.coo_matrix.eliminate_zeros(B) # Eliminate explicit zeros
    return B.tocsc()

def calcB_3D(myBCs: BCS.CantileverBeam_3D):
    m, nA, nB = len(myBCs.ground_structure), myBCs.ground_structure[:,0].astype(int), myBCs.ground_structure[:,1].astype(int)
    l, X, Y, Z = myBCs.ground_structure_length, myBCs.nodes[nB,0]-myBCs.nodes[nA,0], myBCs.nodes[nB,1]-myBCs.nodes[nA,1], myBCs.nodes[nB,2]-myBCs.nodes[nA,2]
    d0, d1, d2, d3, d4, d5 = myBCs.dofs[nA*3], myBCs.dofs[nA*3+1], myBCs.dofs[nA*3+2], myBCs.dofs[nB*3], myBCs.dofs[nB*3+1], myBCs.dofs[nB*3+2] # Defined on boundary conditions. One line for each candidate
    s = np.concatenate((-X/l * d0, -Y/l * d1, -Z/l * d2, X/l * d3, Y/l * d4, Z/l * d5 ))
    r = np.concatenate((nA*3, nA*3+1, nA*3+2, nB*3, nB*3+1, nB*3+2)) # dofs
    c = np.tile(np.arange(m), 6)
    B = sp.coo_matrix((s, (r, c)), shape = (len(myBCs.dofs), m))
    sp.coo_matrix.eliminate_zeros(B) # Eliminate explicit zeros
    return B.tocsc() 

def calcK(myBCs, B, E, A):
    K = B @ sp.diags(E*A/myBCs.ground_structure_length) @ B.T
    return K

############################
# Utilities
############################
def reduce_BCs(BCs: BCS.MBB2D_Symm, tol, a, q=False, a_cell=False, cell_mapping_vector=False, delete_chain=False):
    """ Use this method to reduce the size of the Bcs 
    keep.size == BCs.ground_structure.size """
    
    reduced_BCs = copy.deepcopy(BCs)
    keep_value = np.max(a)*tol
    reduced_BCs.candidates = a>=keep_value#(q>0 & a>keep_value_t) + (q<0 & a>keep_value_c)
    reduced_BCs.candidates_post_unstable_node_suppression = np.where(a>=keep_value)[0]#(q>0 & a>keep_value_t) + (q<0 & a>keep_value_c)
    a_reduced = a[reduced_BCs.candidates]
    
    # Ground structure (step 1)
    reduced_BCs.ground_structure_not_reduced = BCs.ground_structure.copy() # Used for plotting
    reduced_BCs.nodes_not_reduced = BCs.nodes.copy()
    reduced_BCs.ground_structure = BCs.ground_structure[reduced_BCs.candidates].copy()
    reduced_BCs.ground_structure_length = BCs.ground_structure_length[reduced_BCs.candidates].copy()
    
    if not np.any(a_cell!=False):
        ### Eliminates buckling chain
        if delete_chain:
            if BCs.is3D:
                q_reduced = q[reduced_BCs.candidates]
                # Identify if there is some chain buckling
                unique, counts = np.unique(reduced_BCs.ground_structure, return_counts=True)
                possible_chain_nodes = np.array(unique[np.where(counts==2)]).ravel()
                chain_nodes = []
                for k in range(possible_chain_nodes.shape[0]): # needed to avoid adding BC nodes
                    candidates_id = (reduced_BCs.ground_structure[:,0]==possible_chain_nodes[k]) | (reduced_BCs.ground_structure[:,1]==possible_chain_nodes[k])
                    candidates = reduced_BCs.ground_structure[candidates_id,:]
                    diff = np.setxor1d(candidates[0,:],candidates[1,:])
                    start, end = np.min(diff), np.max(diff)
                    are_bars_in_compression = False
                    if np.all(q_reduced[candidates_id]<0):
                        are_bars_in_compression = True
                    # Evaluate cross product
                    n1 = reduced_BCs.nodes[candidates[0,0],:]
                    n2 = reduced_BCs.nodes[candidates[0,1],:]
                    n3 = reduced_BCs.nodes[candidates[1,0],:]
                    n4 = reduced_BCs.nodes[candidates[1,1],:]
                    v1 = n2-n1
                    v2 = n4-n3
                    if np.allclose(np.linalg.norm(np.cross(v2,v1),2),0,rtol=1e-2): # cross prod = 0
                        if (possible_chain_nodes[k]>start) and (possible_chain_nodes[k]<end):#If not some BCs can become candidates
                            if are_bars_in_compression: # We charge only nodes in a compression chain
                                chain_nodes.append(possible_chain_nodes[k])
                
                merged_candidates = []
                reduced_BCs.merged_candidates=[]
                # Merge chain nodes
                for i, node in enumerate(chain_nodes):# need to use loops and not a vectorized routine
                    leng = 0
                    candidates_to_merge = reduced_BCs.ground_structure[(reduced_BCs.ground_structure[:,0]==node) | (reduced_BCs.ground_structure[:,1]==node),:] 
                    merged_candidates.append(sorted([candidates_to_merge[candidates_to_merge!=node]]))
                    # Drop candidates_to_merge and add merged_candidates
                    delll = []
                    for k in range(candidates_to_merge.shape[0]):
                        dell = int(np.where((reduced_BCs.ground_structure == candidates_to_merge[k,:]).all(axis=1))[0])
                        delll.append(dell)
                    reduced_BCs.merged_candidates.append(delll)
                    for k in range(candidates_to_merge.shape[0]):
                        dell = np.where((reduced_BCs.ground_structure == candidates_to_merge[k,:]).all(axis=1))
                        reduced_BCs.ground_structure = np.delete(reduced_BCs.ground_structure, dell, axis=0)
                        add_a = a_reduced[dell]
                        add_q = q_reduced[dell]
                        a_reduced = np.delete(a_reduced, dell, axis=0)
                        id_cand = np.nonzero(reduced_BCs.candidates_post_unstable_node_suppression)[0]
                        id_remove = id_cand[dell[0]]
                        reduced_BCs.candidates_post_unstable_node_suppression = np.delete(reduced_BCs.candidates_post_unstable_node_suppression, id_remove)
                        q_reduced = np.delete(q_reduced, dell, axis=0)
                        leng += reduced_BCs.ground_structure_length[dell]
                        reduced_BCs.ground_structure_length = np.delete(reduced_BCs.ground_structure_length, dell, axis=0)
                    add = np.array(merged_candidates[i]).reshape(1,2)
                    if not np.equal(add, reduced_BCs.ground_structure).all(axis=1).any(): # check before adding duplicates 
                        reduced_BCs.ground_structure = np.vstack([reduced_BCs.ground_structure, add])
                        a_reduced = np.append(a_reduced, add_a)
                        reduced_BCs.candidates_post_unstable_node_suppression = np.append(reduced_BCs.candidates_post_unstable_node_suppression, True)
                        try:
                            q_reduced = np.vstack([q_reduced, add_q])
                        except:
                            q_reduced = np.append(q_reduced, add_q)
                            
                        reduced_BCs.ground_structure_length = np.append(reduced_BCs.ground_structure_length, leng)
                reduced_BCs.merged_candidates = np.array(reduced_BCs.merged_candidates)
            else:
                q_reduced = q[reduced_BCs.candidates]
                # Identify if there is some chain buckling
                unique, counts = np.unique(reduced_BCs.ground_structure, return_counts=True)
                possible_chain_nodes = np.array(unique[np.where(counts==2)]).ravel()
                chain_nodes = []
                for k in range(possible_chain_nodes.shape[0]): # needed to avoid adding BC nodes
                    cand = reduced_BCs.ground_structure[(reduced_BCs.ground_structure[:,0]==possible_chain_nodes[k]) | (reduced_BCs.ground_structure[:,1]==possible_chain_nodes[k]),:]
                    diff = np.setxor1d(cand[0,:],cand[1,:])
                    start, end = np.min(diff), np.max(diff)
                    x0,y0 = reduced_BCs.nodes[cand[0,0],:]
                    x1,y1 = reduced_BCs.nodes[cand[0,1],:]
                    x2,y2 = reduced_BCs.nodes[cand[1,0],:]
                    x3,y3 = reduced_BCs.nodes[cand[1,1],:]
                    angle1 = np.arctan2(y1-y0, x1-x0)
                    angle2 = np.arctan2(y3-y2, x3-x2)
                    if np.allclose(np.abs(angle1),np.abs(angle2),rtol=1e-2):
                        if (possible_chain_nodes[k]>start) and (possible_chain_nodes[k]<end):#If not some BCs can become candidates
                            chain_nodes.append(possible_chain_nodes[k])
                
                merged_candidates = []
                reduced_BCs.merged_candidates=[]
                # Merge chain nodes
                for i, node in enumerate(chain_nodes):# need to use loops and not a vectorized routine
                    leng = 0
                    candidates_to_merge = reduced_BCs.ground_structure[(reduced_BCs.ground_structure[:,0]==node) | (reduced_BCs.ground_structure[:,1]==node),:] 
                    merged_candidates.append(sorted([candidates_to_merge[candidates_to_merge!=node]]))
                    # Drop candidates_to_merge and add merged_candidates
                    delll = []
                    for k in range(candidates_to_merge.shape[0]):
                        dell = int(np.where((reduced_BCs.ground_structure == candidates_to_merge[k,:]).all(axis=1))[0])
                        delll.append(dell)
                    reduced_BCs.merged_candidates.append(delll)
                    for k in range(candidates_to_merge.shape[0]):
                        dell = int(np.where((reduced_BCs.ground_structure == candidates_to_merge[k,:]).all(axis=1))[0])
                        reduced_BCs.ground_structure = np.delete(reduced_BCs.ground_structure, dell, axis=0)
                        add_a = a_reduced[dell]
                        add_q = q_reduced[dell]
                        a_reduced = np.delete(a_reduced, dell, axis=0)
                        try:
                            eq = int(np.where((BCs.ground_structure == candidates_to_merge[k,:]).all(axis=1))[0])
                            r = int(np.where(reduced_BCs.candidates_post_unstable_node_suppression==eq)[0])
                            reduced_BCs.candidates_post_unstable_node_suppression = np.delete(reduced_BCs.candidates_post_unstable_node_suppression, r)
                            delete=True
                        except:
                            delete=False
                        q_reduced = np.delete(q_reduced, dell, axis=0)
                        leng += reduced_BCs.ground_structure_length[dell]
                        reduced_BCs.ground_structure_length = np.delete(reduced_BCs.ground_structure_length, dell, axis=0)
                    add = np.array(merged_candidates[i]).reshape(1,2)
                    if not np.equal(add, reduced_BCs.ground_structure).all(axis=1).any(): # check before adding duplicates 
                        reduced_BCs.ground_structure = np.vstack([reduced_BCs.ground_structure, add])
                        a_reduced = np.append(a_reduced, add_a)
                        if delete:
                            reduced_BCs.candidates_post_unstable_node_suppression = np.append(reduced_BCs.candidates_post_unstable_node_suppression, eq)
                        try:
                            q_reduced = np.vstack([q_reduced, add_q])
                        except:
                            q_reduced = np.append(q_reduced, add_q)
                        reduced_BCs.ground_structure_length = np.append(reduced_BCs.ground_structure_length, leng)
                reduced_BCs.merged_candidates = np.array(reduced_BCs.merged_candidates)
    # Nodes
    # Identify the nodes to be removed from the array
    reduced_BCs.nodes_candidates = reduced_BCs.ground_structure[:,[0,1]].copy()
    reduced_BCs.nodes_candidates = reduced_BCs.nodes_candidates.ravel().tolist()
    # Sort and cancel duplicates
    reduced_BCs.nodes_candidates = np.array(sorted(set(reduced_BCs.nodes_candidates)))
    reduced_BCs.nodes = BCs.nodes[reduced_BCs.nodes_candidates,:].copy()
    
    # Fixed DOFs
    if BCs.is3D:
        reduced_BCs.dofs_candidates = reduced_BCs.nodes_candidates.copy()
        reduced_BCs.dofs_candidates = np.append(reduced_BCs.dofs_candidates*3, reduced_BCs.nodes_candidates*3 + 1)
        reduced_BCs.dofs_candidates = np.sort(np.append(reduced_BCs.dofs_candidates, reduced_BCs.nodes_candidates*3 + 2))
    else:
        reduced_BCs.dofs_candidates = reduced_BCs.nodes_candidates.copy()
        reduced_BCs.dofs_candidates = np.sort(np.append(reduced_BCs.dofs_candidates*2, reduced_BCs.nodes_candidates*2 + 1))
    # We need to translate the ID of the full BC to the reduced
    dofs_list = np.arange(reduced_BCs.dofs_candidates.size)
    reduced_BCs.dofs = BCs.dofs[reduced_BCs.dofs_candidates].copy()
    reduced_BCs.fixed_dofs = np.array(np.where(reduced_BCs.dofs == 0)).ravel()
    reduced_BCs.free_dofs = np.setdiff1d(dofs_list,reduced_BCs.fixed_dofs)
    if BCs.is3D:
        reduced_BCs.dofs_nodes_connectivity = reduced_BCs.dofs.reshape((-1,3))
    else:
        reduced_BCs.dofs_nodes_connectivity = reduced_BCs.dofs.reshape((-1,2))
    
    # Force
    tmp = np.zeros(BCs.dofs.shape[0], dtype='int')
    tmp[BCs.force_dofs] = 1
    tmp = tmp[reduced_BCs.dofs_candidates]
    reduced_BCs.force_dofs = np.array(np.where(tmp == 1)).ravel()
    reduced_BCs.R = BCs.R[reduced_BCs.dofs_candidates].copy()
    reduced_BCs.R_free = reduced_BCs.R[reduced_BCs.free_dofs].copy()
    
    # Ground structure (step 2)
    # The node ID must be replaced as the nodes matrix has changed
    replace_matrix = np.vstack([reduced_BCs.nodes_candidates,np.arange(reduced_BCs.nodes_candidates.size)]).T # find (left col) and replace (right col) matrix used for the select conditions
    condlist = replace_matrix[:,0][:,None,None] == reduced_BCs.ground_structure 
    reduced_BCs.ground_structure_sel = np.select(condlist, replace_matrix[:,1])
    reduced_BCs.ground_structure[:,:2] = reduced_BCs.ground_structure_sel[:,:2]
    
    if np.any(cell_mapping_vector) != False:
        n_topologies_cell = len(set(cell_mapping_vector.ravel().tolist()))
    else:
        n_topologies_cell = 1        
    
    if np.any(a_cell!=False):
        a_cell_reduced, reduced_BCs.keep_cell = reduce_A(a_cell, keep_value)
        reduced_BCs.N_cell = np.zeros(n_topologies_cell, dtype='int')
        for i in range(n_topologies_cell):
                reduced_BCs.N_cell[i] = a_cell_reduced[i].size
        reduced_BCs.N_cell_max = np.max(reduced_BCs.N_cell)
        
        # add -1 values to the sections of the cells. This permits the Kronecher product
        a_cell_reduced_pad = a_cell_reduced.copy()
        for i in range(a_cell_reduced.size):
            a_cell_reduced_pad[i] = np.pad(a_cell_reduced_pad[i],pad_width=(0,reduced_BCs.N_cell_max-a_cell_reduced_pad[i].size), constant_values=-1)
        
        # Create a bool matrix used to understand which sections are active
        reduced_BCs.reduction_pattern = (a_cell_reduced_pad[0]>=0).reshape(1,-1) # this array should be at least 2D
        for i in range(a_cell_reduced.size-1):
            reduced_BCs.reduction_pattern = np.vstack((reduced_BCs.reduction_pattern,a_cell_reduced_pad[i+1]>=0))
        
        # Create a bool matrix to calculate the lenght of every section      
        reduced_BCs.reduction_pattern_len =  (a_cell[0]>keep_value).reshape(1,-1)
        for i in range(a_cell_reduced.size-1):
            reduced_BCs.reduction_pattern_len = np.vstack((reduced_BCs.reduction_pattern_len,a_cell[i+1]>keep_value))
            
        # Create a matrix with the unreduced legnht of the members of every cell topology
        reduced_BCs.l_cell_full = np.tile(BCs.ground_structure_length[:a_cell[0].shape[0]],a_cell.shape[0]).reshape((-1,a_cell[0].shape[0]))
        
        # Update the list of the ground structure of the reduced problem
        reduced_BCs.ground_stucture_list_cell = []
        for i in range(n_topologies_cell):
            reduced_BCs.ground_stucture_list_cell.append(np.array(BCs.ground_stucture_list_cell)[a_cell[i]>keep_value].tolist())
        
    reduced_BCs.ground_stucture_list_not_reduced = BCs.ground_structure[:,[0,1]].tolist()
    reduced_BCs.ground_stucture_list = reduced_BCs.ground_structure.tolist()
    
    reduced_BCs.isReduced = True
    
    if reduced_BCs.isFreeForm:
        p = np.array(np.where(reduced_BCs.ground_structure[:,2]==1)).ravel()
        a = np.array(np.where(reduced_BCs.ground_structure[:,2]!=1)).ravel()
        reduced_BCs.ground_structure_cellular = reduced_BCs.ground_structure[p,:].copy()
        reduced_BCs.ground_structure_aperiodic = reduced_BCs.ground_structure[a,:].copy()
        
        reduced_BCs.N = len(reduced_BCs.ground_structure) # Number of member of the Ground Structure
        reduced_BCs.M = len(reduced_BCs.dofs) # Number of DOFs of the Groud Structure
        reduced_BCs.N_cellular = len(reduced_BCs.ground_structure_cellular)
        reduced_BCs.N_aperiodic = len(reduced_BCs.ground_structure_aperiodic)
        reduced_BCs.N_cell = int(reduced_BCs.N_cell)
        if np.any(a_cell!=False):
            reduced_BCs.ground_structure_length_cell = BCs.ground_structure_length_cell[keep_cell].copy()
        return reduced_BCs, a_reduced 
    
    if np.any(a_cell!=False):
        return reduced_BCs, a_cell_reduced_pad
    else:
        return reduced_BCs, a_reduced

# def reduce_BCs_candidates(BCs: BCS.MBB2D_Symm, candidates, a, q=False, a_cell=False, cell_mapping_vector=False, delete_chain=False):
    
#     reduced_BCs = copy.deepcopy(BCs)
#     reduced_BCs.candidates = candidates
#     a_reduced = a[reduced_BCs.candidates]
    
#     # Ground structure (step 1)
#     reduced_BCs.ground_structure_not_reduced = BCs.ground_structure.copy() # Used for plotting
#     reduced_BCs.nodes_not_reduced = BCs.nodes.copy()
#     reduced_BCs.ground_structure = BCs.ground_structure[reduced_BCs.candidates].copy()
#     reduced_BCs.ground_structure_length = BCs.ground_structure_length[reduced_BCs.candidates].copy()
    
#     if not np.any(a_cell!=False):
#         ### Eliminates buckling chain
#         if delete_chain:
#             q_reduced = q[reduced_BCs.candidates]
#             # Identify if there is some chain buckling
#             unique, counts = np.unique(reduced_BCs.ground_structure, return_counts=True)
#             possible_chain_nodes = np.array(unique[np.where(counts==2)]).ravel()
#             chain_nodes = []
#             for k in range(possible_chain_nodes.shape[0]): # needed to avoid adding BC nodes
#                 cand = reduced_BCs.ground_structure[(reduced_BCs.ground_structure[:,0]==possible_chain_nodes[k]) | (reduced_BCs.ground_structure[:,1]==possible_chain_nodes[k]),:]
#                 diff = np.setxor1d(cand[0,:],cand[1,:])
#                 start, end = np.min(diff), np.max(diff)
#                 x0,y0 = reduced_BCs.nodes[cand[0,0],:]
#                 x1,y1 = reduced_BCs.nodes[cand[0,1],:]
#                 x2,y2 = reduced_BCs.nodes[cand[1,0],:]
#                 x3,y3 = reduced_BCs.nodes[cand[1,1],:]
#                 angle1 = np.arctan2(y1-y0, x1-x0)
#                 angle2 = np.arctan2(y3-y2, x3-x2)
#                 if np.allclose(np.abs(angle1),np.abs(angle2),rtol=1e-2):
#                     if (possible_chain_nodes[k]>start) and (possible_chain_nodes[k]<end):#If not some BCs can become candidates
#                         chain_nodes.append(possible_chain_nodes[k])
            
#             merged_candidates = []
#             # Merge chain nodes
#             for i, node in enumerate(chain_nodes):# need to use loops and not a vectorized routine
#                 leng = 0
#                 candidates_to_merge = reduced_BCs.ground_structure[(reduced_BCs.ground_structure[:,0]==node) | (reduced_BCs.ground_structure[:,1]==node),:] 
#                 merged_candidates.append(sorted([candidates_to_merge[candidates_to_merge!=node]]))
#                 # Drop candidates_to_merge and add merged_candidates
#                 for k in range(candidates_to_merge.shape[0]):
#                     dell = np.where((reduced_BCs.ground_structure == candidates_to_merge[k,:]).all(axis=1))
#                     reduced_BCs.ground_structure = np.delete(reduced_BCs.ground_structure, dell, axis=0)
#                     add_a = a_reduced[dell]
#                     add_q = q_reduced[dell]
#                     a_reduced = np.delete(a_reduced, dell, axis=0)
#                     q_reduced = np.delete(q_reduced, dell, axis=0)
#                     leng += reduced_BCs.ground_structure_length[dell]
#                     reduced_BCs.ground_structure_length = np.delete(reduced_BCs.ground_structure_length, dell, axis=0)
#                 add = np.array(merged_candidates[i]).reshape(1,2)
#                 if not np.equal(add, reduced_BCs.ground_structure).all(axis=1).any(): # check before adding duplicates 
#                     reduced_BCs.ground_structure = np.vstack([reduced_BCs.ground_structure, add])
#                     a_reduced = np.append(a_reduced, add_a)
#                     q_reduced = np.append(q_reduced, add_q)
#                     reduced_BCs.ground_structure_length = np.append(reduced_BCs.ground_structure_length, leng)
    
#     # Nodes
#     # Identify the nodes to be removed from the array
#     reduced_BCs.nodes_candidates = reduced_BCs.ground_structure[:,[0,1]].copy()
#     reduced_BCs.nodes_candidates = reduced_BCs.nodes_candidates.ravel().tolist()
#     # Sort and cancel duplicates
#     reduced_BCs.nodes_candidates = np.array(sorted(set(reduced_BCs.nodes_candidates)))
#     reduced_BCs.nodes = BCs.nodes[reduced_BCs.nodes_candidates,:].copy()
    
#     # Fixed DOFs
#     if BCs.is3D:
#         reduced_BCs.dofs_candidates = reduced_BCs.nodes_candidates.copy()
#         reduced_BCs.dofs_candidates = np.append(reduced_BCs.dofs_candidates*3, reduced_BCs.nodes_candidates*3 + 1)
#         reduced_BCs.dofs_candidates = np.sort(np.append(reduced_BCs.dofs_candidates, reduced_BCs.nodes_candidates*3 + 2))
#     else:
#         reduced_BCs.dofs_candidates = reduced_BCs.nodes_candidates.copy()
#         reduced_BCs.dofs_candidates = np.sort(np.append(reduced_BCs.dofs_candidates*2, reduced_BCs.nodes_candidates*2 + 1))
#     # We need to translate the ID of the full BC to the reduced
#     dofs_list = np.arange(reduced_BCs.dofs_candidates.size)
#     reduced_BCs.dofs = BCs.dofs[reduced_BCs.dofs_candidates].copy()
#     reduced_BCs.fixed_dofs = np.array(np.where(reduced_BCs.dofs == 0)).ravel()
#     reduced_BCs.free_dofs = np.setdiff1d(dofs_list,reduced_BCs.fixed_dofs)
#     if BCs.is3D:
#         reduced_BCs.dofs_nodes_connectivity = reduced_BCs.dofs.reshape((-1,3))
#     else:
#         reduced_BCs.dofs_nodes_connectivity = reduced_BCs.dofs.reshape((-1,2))
    
#     # Force
#     tmp = np.zeros(BCs.dofs.shape[0], dtype='int')
#     tmp[BCs.force_dofs] = 1
#     tmp = tmp[reduced_BCs.dofs_candidates]
#     reduced_BCs.force_dofs = np.array(np.where(tmp == 1)).ravel()
#     reduced_BCs.R = BCs.R[reduced_BCs.dofs_candidates].copy()
#     reduced_BCs.R_free = reduced_BCs.R[reduced_BCs.free_dofs].copy()
    
#     # Ground structure (step 2)
#     # The node ID must be replaced as the nodes matrix has changed
#     replace_matrix = np.vstack([reduced_BCs.nodes_candidates,np.arange(reduced_BCs.nodes_candidates.size)]).T # find (left col) and replace (right col) matrix used for the select conditions
#     condlist = replace_matrix[:,0][:,None,None] == reduced_BCs.ground_structure 
#     reduced_BCs.ground_structure_sel = np.select(condlist, replace_matrix[:,1])
#     reduced_BCs.ground_structure[:,:2] = reduced_BCs.ground_structure_sel[:,:2]
    
#     n_topologies_cell = 1        
       
#     reduced_BCs.ground_stucture_list_not_reduced = BCs.ground_structure[:,[0,1]].tolist()
#     reduced_BCs.ground_stucture_list = BCs.ground_structure[:,[0,1]][reduced_BCs.candidates].tolist()
    
#     reduced_BCs.isReduced = True
    
#     p = np.array(np.where(reduced_BCs.ground_structure[:,2]==1)).ravel()
#     a = np.array(np.where(reduced_BCs.ground_structure[:,2]!=1)).ravel()
#     reduced_BCs.ground_structure_cellular = reduced_BCs.ground_structure[p,:].copy()
#     reduced_BCs.ground_structure_aperiodic = reduced_BCs.ground_structure[a,:].copy()
    
#     reduced_BCs.N = len(reduced_BCs.ground_structure) # Number of member of the Ground Structure
#     reduced_BCs.M = len(reduced_BCs.dofs) # Number of DOFs of the Groud Structure
#     reduced_BCs.N_cellular = len(reduced_BCs.ground_structure_cellular)
#     reduced_BCs.N_aperiodic = len(reduced_BCs.ground_structure_aperiodic)
#     reduced_BCs.N_cell = int(reduced_BCs.N_cell)
#     return reduced_BCs, a_reduced 

# def reduce_BCs_section(BCs: BCS.MBB2D_Symm, tol, a, a_cell=False, cell_mapping_vector=False, delete_chain=False):
#     """ Use this method to reduce the size of the Bcs based on the bool vector keep.
#     keep.size == BCs.ground_structure.size 
#     A better coding alternative would be to use a dataclass of a BC instead of a deepcopy 
#     TO CLEAN"""
    
#     import copy
#     reduced_BCs = copy.deepcopy(BCs)
    
#     keep_value = np.max(a)*tol
#     reduced_BCs.candidates = a>keep_value
#     a_reduced = a[reduced_BCs.candidates]
    
#     # Ground structure (step 1)
#     reduced_BCs.ground_structure_not_reduced = BCs.ground_structure.copy() # Used for plotting
#     reduced_BCs.nodes_not_reduced = BCs.nodes.copy()
#     reduced_BCs.ground_structure = BCs.ground_structure[reduced_BCs.candidates].copy()
#     reduced_BCs.ground_structure_length = BCs.ground_structure_length[reduced_BCs.candidates].copy()
    
#     if not np.any(a_cell!=False):
#         ### Eliminates buckling chain
#         if delete_chain:
#             # Identify chain nodes
#             """ if BCs.is3D:
#                 B = calcB_3D(BCs)
#             else:
#                 B = calcB(BCs)
                
#             force_dofs =  np.einsum('j,ij->ij', q, B.todense())[:,reduced_BCs.candidates] # bar forces for every dof
#             n_bar_per_dof = np.count_nonzero(force_dofs, axis=1)
#             n_bar_per_node = n_bar_per_dof.reshape(-1,2).sum(axis=1) # they are'nt exactly the number of bars per nodes as diagonal bars are counted twice. But it permits to know the nodes that have only 2 contributes
#             chain_nodes = np.array(np.where(n_bar_per_node==2)).ravel() """
#             # Identify if there is some chain buckling
#             unique, counts = np.unique(reduced_BCs.ground_structure, return_counts=True)
#             possible_chain_nodes = np.array(unique[np.where(counts==2)]).ravel()
#             chain_nodes = []
#             for k in range(possible_chain_nodes.shape[0]): # needed to avoid adding BC nodes
#                 cand = reduced_BCs.ground_structure[(reduced_BCs.ground_structure[:,0]==possible_chain_nodes[k]) | (reduced_BCs.ground_structure[:,1]==possible_chain_nodes[k]),:]
#                 diff = np.setxor1d(cand[0,:],cand[1,:])
#                 start, end = np.min(diff), np.max(diff)
#                 x0,y0 = myBCs.nodes[cand[0,0],:]
#                 x1,y1 = myBCs.nodes[cand[0,1],:]
#                 x2,y2 = myBCs.nodes[cand[1,0],:]
#                 x3,y3 = myBCs.nodes[cand[1,1],:]
#                 angle1 = np.arctan2(y1-y0, x1-x0)
#                 angle2 = np.arctan2(y3-y2, x3-x2)
#                 if np.allclose(np.abs(angle1),np.abs(angle2),rtol=1e-2):
#                     if (possible_chain_nodes[k]>start) and (possible_chain_nodes[k]<end):#If not some BCs can become candidates
#                         chain_nodes.append(possible_chain_nodes[k])
            
#             merged_candidates = []
#             # Merge chain nodes
#             for i, node in enumerate(chain_nodes):# need to use loops and not a vectorized routine
#                 leng = 0
#                 candidates_to_merge = reduced_BCs.ground_structure[(reduced_BCs.ground_structure[:,0]==node) | (reduced_BCs.ground_structure[:,1]==node),:] 
#                 merged_candidates.append(sorted([candidates_to_merge[candidates_to_merge!=node]]))
#                 # Drop candidates_to_merge and add merged_candidates
#                 for k in range(candidates_to_merge.shape[0]):
#                     dell = np.where((reduced_BCs.ground_structure == candidates_to_merge[k,:]).all(axis=1))
#                     reduced_BCs.ground_structure = np.delete(reduced_BCs.ground_structure, dell, axis=0)
#                     add_a = a_reduced[dell]
#                     a_reduced = np.delete(a_reduced, dell, axis=0)
#                     leng += reduced_BCs.ground_structure_length[dell]
#                     reduced_BCs.ground_structure_length = np.delete(reduced_BCs.ground_structure_length, dell, axis=0)
#                 add = np.array(merged_candidates[i]).reshape(1,2)
#                 if not np.equal(add, reduced_BCs.ground_structure).all(axis=1).any(): # check before adding duplicates 
#                     reduced_BCs.ground_structure = np.vstack([reduced_BCs.ground_structure, add])
#                     a_reduced = np.append(a_reduced, add_a)
#                     reduced_BCs.ground_structure_length = np.append(reduced_BCs.ground_structure_length, leng)
    
#     # Nodes
#     # Identify the nodes to be removed from the array
#     reduced_BCs.nodes_candidates = reduced_BCs.ground_structure[:,[0,1]].copy()
#     reduced_BCs.nodes_candidates = reduced_BCs.nodes_candidates.ravel().tolist()
#     # Sort and cancel duplicates
#     reduced_BCs.nodes_candidates = np.array(sorted(set(reduced_BCs.nodes_candidates)))
#     reduced_BCs.nodes = BCs.nodes[reduced_BCs.nodes_candidates,:].copy()
    
#     # Fixed DOFs
#     if BCs.is3D:
#         reduced_BCs.dofs_candidates = reduced_BCs.nodes_candidates.copy()
#         reduced_BCs.dofs_candidates = np.append(reduced_BCs.dofs_candidates*3, reduced_BCs.nodes_candidates*3 + 1)
#         reduced_BCs.dofs_candidates = np.sort(np.append(reduced_BCs.dofs_candidates, reduced_BCs.nodes_candidates*3 + 2))
#     else:
#         reduced_BCs.dofs_candidates = reduced_BCs.nodes_candidates.copy()
#         reduced_BCs.dofs_candidates = np.sort(np.append(reduced_BCs.dofs_candidates*2, reduced_BCs.nodes_candidates*2 + 1))
#     # We need to translate the ID of the full BC to the reduced
#     dofs_list = np.arange(reduced_BCs.dofs_candidates.size)
#     reduced_BCs.dofs = BCs.dofs[reduced_BCs.dofs_candidates].copy()
#     reduced_BCs.fixed_dofs = np.array(np.where(reduced_BCs.dofs == 0)).ravel()
#     reduced_BCs.free_dofs = np.setdiff1d(dofs_list,reduced_BCs.fixed_dofs)
#     if BCs.is3D:
#         reduced_BCs.dofs_nodes_connectivity = reduced_BCs.dofs.reshape((-1,3))
#     else:
#         reduced_BCs.dofs_nodes_connectivity = reduced_BCs.dofs.reshape((-1,2))
    
#     # Force
#     tmp = np.zeros(BCs.dofs.shape[0], dtype='int')
#     tmp[BCs.force_dofs] = 1
#     tmp = tmp[reduced_BCs.dofs_candidates]
#     reduced_BCs.force_dofs = np.array(np.where(tmp == 1)).ravel()
#     reduced_BCs.R = BCs.R[reduced_BCs.dofs_candidates].copy()
#     reduced_BCs.R_free = reduced_BCs.R[reduced_BCs.free_dofs].copy()
    
#     # Ground structure (step 2)
#     # The node ID must be replaced as the nodes matrix has changed
#     replace_matrix = np.vstack([reduced_BCs.nodes_candidates,np.arange(reduced_BCs.nodes_candidates.size)]).T # find (left col) and replace (right col) matrix used for the select conditions
#     condlist = replace_matrix[:,0][:,None,None] == reduced_BCs.ground_structure 
#     reduced_BCs.ground_structure_sel = np.select(condlist, replace_matrix[:,1])
#     reduced_BCs.ground_structure[:,:2] = reduced_BCs.ground_structure_sel[:,:2]
    
#     if np.any(cell_mapping_vector) != False:
#         n_topologies_cell = len(set(cell_mapping_vector.ravel().tolist()))
#     else:
#         n_topologies_cell = 1        
    
#     if np.any(a_cell!=False):
#         a_cell_reduced, keep_cell = reduce_A(a_cell, keep_value)
#         reduced_BCs.N_cell = np.count_nonzero(keep_cell[0])
#         reduced_BCs.N_periodic_section = np.count_nonzero(keep_cell[1])
        
#         """ # add -1 values to the sections of the cells. This permits the Kronecher product
#         a_cell_reduced_pad = a_cell_reduced.copy()
#         for i in range(a_cell_reduced.size):
#             a_cell_reduced_pad[i] = np.pad(a_cell_reduced_pad[i],pad_width=(0,reduced_BCs.N_cell_max-a_cell_reduced_pad[i].size), constant_values=-1)
        
#         # Create a bool matrix used to understand which sections are active
#         reduced_BCs.reduction_pattern = (a_cell_reduced_pad[0]>=0).reshape(1,-1) # this array should be at least 2D
#         for i in range(a_cell_reduced.size-1):
#             reduced_BCs.reduction_pattern = np.vstack((reduced_BCs.reduction_pattern,a_cell_reduced_pad[i+1]>=0))
        
#         # Create a bool matrix to calculate the lenght of every section      
#         reduced_BCs.reduction_pattern_len =  (a_cell[0]>keep_value).reshape(1,-1)
#         for i in range(a_cell_reduced.size-1):
#             reduced_BCs.reduction_pattern_len = np.vstack((reduced_BCs.reduction_pattern_len,a_cell[i+1]>keep_value)) """
            
#         # Create a matrix with the unreduced legnht of the members of every cell topology
#         reduced_BCs.l_cell_full = np.tile(BCs.ground_structure_length[:a_cell[0].shape[0]],a_cell.shape[0]).reshape((-1,a_cell[0].shape[0]))
        
#         # Update the list of the ground structure of the reduced problem
#         reduced_BCs.ground_stucture_list_cell = []
#         for i in range(n_topologies_cell):
#             reduced_BCs.ground_stucture_list_cell.append(np.array(BCs.ground_stucture_list_cell)[a_cell[i]>keep_value].tolist())
        
#     reduced_BCs.ground_stucture_list_not_reduced = BCs.ground_structure[:,[0,1]].tolist()
#     reduced_BCs.ground_stucture_list = BCs.ground_structure[:,[0,1]][reduced_BCs.candidates].tolist()
    
#     reduced_BCs.isReduced = True
    
#     p = np.array(np.where(reduced_BCs.ground_structure[:,2]==1)).ravel()
#     s = np.array(np.where(reduced_BCs.ground_structure[:,2]==2)).ravel()
#     reduced_BCs.ground_structure_cellular = reduced_BCs.ground_structure[p,:].copy()
#     reduced_BCs.ground_structure_section = reduced_BCs.ground_structure[s,:].copy()
    
#     reduced_BCs.N = len(reduced_BCs.ground_structure) # Number of member of the Ground Structure
#     reduced_BCs.M = len(reduced_BCs.dofs) # Number of DOFs of the Groud Structure
#     reduced_BCs.N_cellular = len(reduced_BCs.ground_structure_cellular)
#     reduced_BCs.N_aperiodic = len(reduced_BCs.ground_structure_aperiodic)
#     reduced_BCs.N_cell = int(reduced_BCs.N_cell)
#     reduced_BCs.N_periodic_section = int(reduced_BCs.N_periodic_section)
#     reduced_BCs.ground_structure_length_cell = BCs.ground_structure_length_cell[keep_cell[0]].copy()
#     reduced_BCs.ground_structure_length_section = BCs.ground_structure_length_section[keep_cell[1]].copy()
#     return reduced_BCs, a_reduced 
    
def reduce_A(a, thresold):
    a_reduced = a.copy()
    keep = np.zeros(a_reduced.size, dtype='object_')
    
    for i in range(a.size):
        keep[i] = a_reduced[i]>thresold
        a_reduced[i] = a_reduced[i][keep[i]]
        
    return a_reduced, keep

def mapping_matrix_init(cell_mapping_vector, BC):
    # VL parameter
    if np.any(cell_mapping_vector) == False:
        cell_mapping_vector = np.ones(BC.ncel_str, dtype='int').ravel()
    topologies_ID = set(cell_mapping_vector.ravel().tolist()) # unique
    n_topologies_cell = len(topologies_ID) # How many different topologies are present
    cell_mapping_matrix = np.zeros((BC.ncel_str,n_topologies_cell), dtype='int')
    
    for i, item in enumerate(topologies_ID): # starting from the lower left corner
        cell_mapping_matrix[np.where(cell_mapping_vector == item),i] += 1 # create matrix to correctly assign the different cell topologies according to cell_mapping_matrix
    
    if not BC.isReduced:
        if np.any(n_topologies_cell) != False: 
            BC.N_cell = np.ones(n_topologies_cell, dtype='int') * BC.N_cell
    
    NN = np.sum(BC.N_cell) # Number of elements in the cell times the number of different cells (area design variables)
            
    return NN, n_topologies_cell, cell_mapping_matrix
  
def calculate_starting_point_on_unreduced_BCs(myBCs, myBCs_reduced, N_design_var, area_id, force_id, U_id, E, a_init, a_fem):
    """ Method used to calculate the starting point of the design variables (x0) for a cellular case with or withour clusters and preinit """
    if myBCs.is3D:
        B_old = calcB_3D(myBCs) # equilibrium matrix
    else:
        B_old = calcB(myBCs) # equilibrium matrix
        
    M_old = myBCs.M # Number of DOFs of the Groud Structure
        
    ## Starting point
    x0 = np.zeros(N_design_var) # Define an initial guess for the optimization
    # Areas   
    x0[area_id] = a_init 
    area_phys = a_fem
    area_phys[area_phys<np.max(a_init)*1e-8] = np.max(a_init)*1e-8
        
    # Initial forces and displacements are calculated using FEM
    U = np.zeros(M_old)
    K = calcK(myBCs,B_old,E,area_phys)
    keep = myBCs.free_dofs
    K = K[keep, :][:, keep]
    U[keep] = sp.linalg.spsolve(K, myBCs.R_free) # FEM analysis linear sistem
    
    # Forces   
    F_old = area_phys*E/myBCs.ground_structure_length * (B_old.T @ U)   
    x0[force_id] = F_old[myBCs_reduced.candidates_post_unstable_node_suppression]
    # Displacements
    x0[U_id] = U[myBCs_reduced.dofs_candidates]  
    
    return x0

def calculate_starting_point_cell_on_unreduced_BCs(myBCs, myBCs_reduced, N_design_var, area_id, force_id, U_id, N, M, B, E, n_topologies_cell, N_cell, cell_mapping_matrix, a_cell):
    """ Method used to calculate the starting point of the design variables (x0) for a cellular case with or withour clusters and preinit """
    if myBCs.is3D:
        B_old = calcB_3D(myBCs) # equilibrium matrix
    else:
        B_old = calcB(myBCs) # equilibrium matrix
        
    M_old = myBCs.M # Number of DOFs of the Groud Structure
        
    ## Starting point
    x0 = np.zeros(N_design_var) # Define an initial guess for the optimization
    # Areas
    a_cell_list = []
    for i in range(n_topologies_cell):
        a_cell_list = np.append(a_cell_list, a_cell[i][a_cell[i]>-1e-5]) # Eliminate -1 values used for the Kronecher product, a little neg value is accepted for numeric errors
        a_cell
    x0[area_id] = a_cell_list
        
    area_phys = np.zeros(myBCs.ncel_str*myBCs.N_cell[0])
    for i in range(n_topologies_cell): # starting from the lower left corner
        a_cell_pad = np.ones(myBCs.N_cell[0]) * np.max(a_cell[0])*1e-8
        a_cell_pad[myBCs_reduced.keep_cell[i]] = a_cell[i][range(myBCs_reduced.N_cell[i])]
        area_phys += np.kron(cell_mapping_matrix[:,i], a_cell_pad)
        
    # Initial forces and displacements are calculated using FEM
    U = np.zeros(M_old)
    K = calcK(myBCs,B_old,E,area_phys)
    keep = myBCs.free_dofs
    K = K[keep, :][:, keep]
    U[keep] = sp.linalg.spsolve(K, myBCs.R_free) # FEM analysis linear sistem
    
    # Forces   
    F_old = area_phys*E/myBCs.ground_structure_length * (B_old.T @ U) 
    x0[force_id] = F_old[myBCs_reduced.candidates]
    # Displacements
    x0[U_id] = U[myBCs_reduced.dofs_candidates]  
    
    return x0

# def calculate_starting_point_free_form(myBCs, N_design_var, area_id, force_id, U_id, N, M, B, E, a_init):
#     """ Method used to calculate the starting point of the design variables (x0) for a cellular case with or withour clusters and preinit """
#     ## Starting point
#     x0 = np.zeros(N_design_var) # Define an initial guess for the optimization
#     # Areas
#     if np.any(a_init!=False):
#         # dividere in periodic e aperiodic
#         p_cell = np.array(np.where((myBCs.ground_structure[:,2]!=-1) & (myBCs.ground_structure[:,3]==0))).ravel()
#         a = np.array(np.where(myBCs.ground_structure[:,2]==-1)).ravel()        
#         a_init_design_var = np.concatenate([a_init[p_cell], a_init[a]])
#         x0[area_id] = a_init_design_var
#         area_phys = a_init
#     else: # This part doesen't work if a reduced problem is given
#         value = 1. # x0 area value
#         x0[area_id] = value
#         area_phys = np.ones(N) * value
        
#     # Initial forces and displacements are calculated using FEM
#     U = np.zeros(M)
#     K = calcK(myBCs,B,E,area_phys)
#     keep = myBCs.free_dofs
#     K = K[keep, :][:, keep]
#     U[keep] = sp.linalg.spsolve(K, myBCs.R_free) # FEM analysis linear sistem
#     # Forces
#     if np.any(np.isnan(U)): # Truss coming from the SLP can be a mechanism
#         U[np.isnan(U)] = 0
#     x0[force_id] = area_phys*E/myBCs.ground_structure_length * (B.T @ U)
#     # Displacements
#     x0[U_id] = U  
    
#     return x0

# def calculate_starting_point_free_form_on_unreduced_BCs(myBCs, myBCs_reduced, N_design_var, area_id, force_id, U_id, E, a_init, a_init_reduced):
#     """ Method used to calculate the starting point of the design variables (x0) for a cellular case with or withour clusters and preinit """
#     B_old = calcB_3D(myBCs) # equilibrium matrix
#     M_old = myBCs.M # Number of DOFs of the Groud Structure
        
#     ## Starting point
#     x0 = np.zeros(N_design_var) # Define an initial guess for the optimization
#     # Areas
#     if np.any(a_init!=False):
#         p_cell = np.array(np.where((myBCs_reduced.ground_structure[:,2]!=-1) & (myBCs_reduced.ground_structure[:,3]==0))).ravel()
#         a = np.array(np.where(myBCs_reduced.ground_structure[:,2]==-1)).ravel()        
#         a_init_design_var = np.concatenate([a_init_reduced[p_cell], a_init_reduced[a]])
#         x0[area_id] = a_init_design_var * 1
#         area_phys = a_init * 1 + np.max(a_init)*1e-8 # to avoid K singular matrix
#     else: # This part doesen't work if a reduced problem is given
#         value = 1. # x0 area value
#         x0[area_id] = value
#         area_phys = np.ones(myBCs_reduced.N) * value
        
#     # Initial forces and displacements are calculated using FEM
#     U = np.zeros(M_old)
#     K = calcK(myBCs,B_old,E,area_phys)
#     keep = myBCs.free_dofs
#     K = K[keep, :][:, keep]
#     U[keep] = sp.linalg.spsolve(K, myBCs.R_free) # FEM analysis linear sistem
    
#     # Forces   
#     F_old = area_phys*E/myBCs.ground_structure_length * (B_old.T @ U) 
#     stress = F_old/ area_phys   
#     x0[force_id] = F_old[myBCs_reduced.candidates]
#     # Displacements
#     x0[U_id] = U[myBCs_reduced.dofs_candidates]  
    
#     return x0

# def calculate_starting_point_free_form_on_unreduced_BCs_section(myBCs, myBCs_reduced, N_design_var, area_id, force_id, U_id, E, a_init, a_init_reduced):
#     """ Method used to calculate the starting point of the design variables (x0) for a cellular case with or withour clusters and preinit """
#     B_old = calcB_3D(myBCs) # equilibrium matrix
#     M_old = myBCs.M # Number of DOFs of the Groud Structure
        
#     ## Starting point
#     x0 = np.zeros(N_design_var) # Define an initial guess for the optimization
#     # Areas
#     if np.any(a_init!=False):
#         p_cell = np.array(np.where((myBCs_reduced.ground_structure[:,2]==1) & (myBCs_reduced.ground_structure[:,3]==0))).ravel()
#         p_sec = np.array(np.where((myBCs_reduced.ground_structure[:,2]==2) & (myBCs_reduced.ground_structure[:,3]==0))).ravel()      
#         a_init_design_var = np.concatenate([a_init_reduced[p_cell], a_init_reduced[p_sec]])
#         x0[area_id] = a_init_design_var * 1
#         area_phys = a_init * 1 + np.max(a_init)*1e-8 # to avoid K singular matrix
#     else: # This part doesen't work if a reduced problem is given
#         value = 1. # x0 area value
#         x0[area_id] = value
#         area_phys = np.ones(myBCs_reduced.N) * value
        
#     # Initial forces and displacements are calculated using FEM
#     U = np.zeros(M_old)
#     K = calcK(myBCs,B_old,E,area_phys)
#     keep = myBCs.free_dofs
#     K = K[keep, :][:, keep]
#     U[keep] = sp.linalg.spsolve(K, myBCs.R_free) # FEM analysis linear sistem
    
#     # Forces   
#     F_old = area_phys*E/myBCs.ground_structure_length * (B_old.T @ U) 
#     stress = F_old/ area_phys   
#     x0[force_id] = F_old[myBCs_reduced.candidates]
#     # Displacements
#     x0[U_id] = U[myBCs_reduced.dofs_candidates]  
    
#     return x0

# def calculate_starting_point_free_form_on_unreduced_BCs_multiload_correct_areas(myBCs, myBCs_reduced, N_design_var, area_id, force_id, U_id, s_t, s_c, E, s_buck, sf, candidates, a_init, a_init_reduced, foldername):
#     """ Method used to calculate the starting point of the design variables (x0)  """
#     if not os.path.isdir(foldername+'/corrected_areas'):
#         os.makedirs(foldername+'/corrected_areas')
#     if myBCs.is3D:
#         B_old = calcB_3D(myBCs) # equilibrium matrix
#     else:   
#         B_old = calcB(myBCs) # equilibrium matrix
#     M_old = myBCs.dofs.shape[0] # Number of DOFs of the Groud Structure
#     N_old = myBCs.N  # Number of DOFs of the Groud Structure
        
#     n_load_cases = myBCs.R.shape[-1]    
#     ## Starting point
#     x0 = np.zeros(N_design_var) # Define an initial guess for the optimization
#     # Areas

#     area_phys = np.ones(N_old)*1e-10 # to avoid K singular matrix
#     area_phys[candidates] = a_init[candidates]
    
#     M = myBCs.M
#     N = myBCs.N 
        
#     q_0 = np.zeros((N,1)) 
#     U_0 = np.zeros((M,1))[myBCs_reduced.dofs_candidates]
#     for p in range(n_load_cases):    
#     # Initial forces and displacements are calculated using FEM
#         U = np.zeros(M_old)
#         K = calcK(myBCs,B_old,E,area_phys)
#         keep = myBCs.free_dofs
#         K = K[keep, :][:, keep]
#         U[keep] = sp.linalg.spsolve(K, myBCs.R_free[:,p]) # FEM analysis linear sistem
#         U_0 = np.hstack([U_0, U[myBCs_reduced.dofs_candidates].reshape((-1,1))])
#         # Forces   
#         q = area_phys*E/myBCs.ground_structure_length * (B_old.T @ U) 
#         q_0 = np.hstack([q_0,q.reshape((-1,1))])
        
#     a_cand = a_init_reduced 
#     q_cand = np.array(q_0[candidates,1:])
    
#     violation = True
#     i=0
#     aug = 1.01
#     print("\n***************************\n*** SOLUTION REGULARIZATION ***\n***************************\n") 
#     while violation:
#         i += 1
#         # Evaluate stress and buckling constraints
#         stress_c = np.zeros((a_cand.shape[0],n_load_cases))
#         for p in range(n_load_cases):
#             stress_c[:,p] = q_cand[:,p] - s_c * a_cand / sf[p]
#         stress_t = np.zeros((a_cand.shape[0],n_load_cases))
#         for p in range(n_load_cases):
#             stress_t[:,p] = q_cand[:,p] - s_t * a_cand / sf[p]
#         ### Buckling (N eq)
#         buckling = np.zeros((a_cand.shape[0],n_load_cases))
#         for p in range(n_load_cases):
#             buckling[:,p] = q_cand[:,p] + (s_buck/myBCs.ground_structure_length[candidates]**2) * a_cand**2 / sf[p]
            
#         stress_c_viol = stress_c[stress_c<0]
#         stress_t_viol = stress_t[stress_t>0]
#         buck_viol = buckling[buckling<0]
#         n_violation = stress_c_viol.size + stress_t_viol.size + buck_viol.size
        
#         if i==1:
#             print('It {0:d} - Vol. = {1:.6f} - Viol. constraints {2:d}'.format(i, myBCs.ground_structure_length[candidates].T@a_cand, n_violation)) 
        
#         if i % 50==0:
#             print('It {0:d} - New vol. = {1:.6f} - Viol. constraints {2:d}'.format(i, myBCs.ground_structure_length[candidates].T@a_cand, n_violation))
#             a_plot = np.zeros(N_old)
#             q_plot = np.zeros((N_old,3))
#             U_plot = np.zeros((M_old,3))
#             a_plot[candidates] = a_cand
#             q_plot[candidates,:] = q_cand
#             U_plot[myBCs_reduced.dofs_candidates,:] = U_cand
#             trussplot.plot3D.plotTrussDeformation_FEM(myBCs, a_plot, q_plot, U_plot, candidates, 1, foldername+'/corrected_areas/', axis=False, title='It.{0:04d}-LC'.format(i))
#             aug += 0.005
            
#         if i==1:
#             with open(foldername+'/'+'correct_areas'+'.txt', 'w') as f:
#                 f.write('It {0:d} - Vol. = {1:.6f} - Viol. constraints {2:d}\n'.format(i, myBCs.ground_structure_length[candidates].T@a_cand, n_violation)) 
#         if i % 50==0:
#             with open(foldername+'/'+'correct_areas'+'.txt', 'a') as f:
#                 f.write('It {0:d} - New vol. = {1:.6f} - Viol. constraints {2:d}\n'.format(i, myBCs.ground_structure_length[candidates].T@a_cand, n_violation))
        
#         # reduce the violation
#         if n_violation == 0:
#             violation = False
#         else:
#             c = np.any((stress_c<0), axis=1)
#             t = np.any((stress_t>0), axis=1)
#             b = np.any((buckling<0), axis=1)
#             viol = np.logical_or(np.logical_or(c,t),b)
#             a_cand[viol] *= aug
#             area_phys[candidates] = a_cand
        
#         q_cand = np.zeros((N,1)) 
#         U_cand = np.zeros((M,1))[myBCs_reduced.dofs_candidates]
#         for p in range(n_load_cases):    
#         # Initial forces and displacements are calculated using FEM
#             U = np.zeros(M_old)
#             K = calcK(myBCs,B_old,E,area_phys)
#             keep = myBCs.free_dofs
#             K = K[keep, :][:, keep]
#             U[keep] = sp.linalg.spsolve(K, myBCs.R_free[:,p]) # FEM analysis linear sistem
#             U_cand = np.hstack([U_cand, U[myBCs_reduced.dofs_candidates].reshape((-1,1))])
#             if np.max(np.abs(U_cand))>np.max(np.abs(U_0)):
#                 print('Solution is diverging. New displacement max: {0:f}'.format(np.max(np.abs(U_cand))))
#             # Forces   
#             q = area_phys*E/myBCs.ground_structure_length * (B_old.T @ U) 
#             q_cand = np.hstack([q_cand,q.reshape((-1,1))])
#         q_cand = np.array(q_cand[candidates,1:])
#         U_cand = np.array(U_cand[:,1:])
    
#     q_fin=q_cand.T.ravel()
#     U_fin=U_cand.T.ravel()
#     print('Looped {0:d} times to respect constraints. New vol. = {1:.6f}'.format(i, myBCs.ground_structure_length[candidates].T@a_cand))
#     with open(foldername+'/'+'correct_areas'+'.txt', 'a') as f:
#         f.write('Looped {0:d} times to respect constraints. New vol. = {1:.6f}\n'.format(i, myBCs.ground_structure_length[candidates].T@a_cand))
#     x0[area_id] = a_cand
#     x0[force_id] = q_fin
#     x0[U_id] = U_fin 
    
#     np.savetxt(foldername+'/db/'+'a_corrected_x0.dat', a_cand)
#     np.savetxt(foldername+'/db/'+'q_corrected_x0.dat', q_cand)
#     np.savetxt(foldername+'/db/'+'U_corrected_x0.dat', U_cand)
    
#     a_plot = np.zeros(N_old)
#     q_plot = np.zeros((N_old,3))
#     a_plot[candidates] = a_cand
#     q_plot[candidates,:] = q_cand
#     thick = 3 / max(a_plot)
#     trussplot.plot3D.plotTruss_ML(myBCs, a_plot, q_plot, candidates, myBCs.ground_structure_length[candidates].T@a_cand, thick, False, foldername+'/', 'fig1-Topology_x0.pdf')
#     U_plot = np.zeros((M_old,3))
#     U_plot[myBCs_reduced.dofs_candidates,:] = U_cand
#     trussplot.plot3D.plotTrussDeformation_FEM(myBCs, a_plot, q_plot, U_plot, candidates, 1, foldername+'/', axis=False, title='fig5-Topology_x0-LC'.format(i))
    
#     return x0

# def calculate_starting_point_free_form_on_unreduced_BCs_multiload(myBCs, myBCs_reduced, N_design_var, area_id, force_id, U_id, E, a_init, a_init_reduced):
#     """ Method used to calculate the starting point of the design variables (x0) for a cellular case with or withour clusters and preinit """
#     if myBCs.is3D:
#         B_old = calcB_3D(myBCs) # equilibrium matrix
#     else:   
#         B_old = calcB(myBCs) # equilibrium matrix
#     M_old = myBCs.dofs.shape[0] # Number of DOFs of the Groud Structure
        
#     n_load_cases = myBCs.R.shape[-1]    
#     ## Starting point
#     x0 = np.zeros(N_design_var) # Define an initial guess for the optimization
#     # Areas

#     x0[area_id] = a_init_reduced
#     area_phys = a_init + np.max(a_init)*1e-12 # to avoid K singular matrix
        
#     U_list, F_list = [], []
#     for p in range(n_load_cases):    
#     # Initial forces and displacements are calculated using FEM
#         U = np.zeros(M_old)
#         K = calcK(myBCs,B_old,E,area_phys)
#         keep = myBCs.free_dofs
#         K = K[keep, :][:, keep]
#         U[keep] = sp.linalg.spsolve(K, myBCs.R_free[:,p]) # FEM analysis linear sistem
#         U_list.append(U[myBCs_reduced.dofs_candidates])
#         # Forces   
#         F = area_phys*E/myBCs.ground_structure_length * (B_old.T @ U) 
#         F_list.append(F[myBCs_reduced.candidates])
  
#     x0[force_id] = np.array(F_list).ravel()
#     # Displacements
#     x0[U_id] = np.array(U_list).ravel() 
    
#     return x0

# def compute_displacement_and_force_error(myBCs, myBCs_reduced, A_optim_unreduced, F_optim, U_optim, E, foldername):
    
#     B = calcB_3D(myBCs) # equilibrium matrix
#     M = myBCs.M # Number of DOFs of the Groud Structure
#     reduction_pattern_F = myBCs_reduced.candidates
#     reduction_pattern_U = myBCs_reduced.dofs_candidates

#     # Forces and displacements are calculated using FEM
#     U = np.zeros(M)
#     K = calcK(myBCs,B,E,A_optim_unreduced)
#     keep = myBCs.free_dofs
#     K = K[keep, :][:, keep]
#     if myBCs.R.ndim == 1:
#         U[keep] = sp.linalg.spsolve(K, myBCs.R_free) # FEM analysis linear sistem
    
#         # Forces   
#         F = A_optim_unreduced*E/myBCs.ground_structure_length * (B.T @ U)
        
#         F_comp = F
#         U_comp = U
        
#         F_comp[reduction_pattern_F] = F_optim
#         U_comp[reduction_pattern_U] = U_optim
        
#         err_F = np.linalg.norm(F_comp-F,np.inf)
#         err_disp = np.linalg.norm(U_comp-U,np.inf)
        
#         with open(foldername+'/'+'log'+'.txt', 'a') as f:
#             f.write('\n')
#             f.write('Relative Force Error (inf-norm): {0:.6f}'.format(err_F))
#             f.write('Relative Displacement Error (inf-norm): {0:.6f}'.format(err_disp))
            
#     else:
#         for i in range(myBCs.R.shape[-1]):
#             U[keep] = sp.linalg.spsolve(K, myBCs.R[keep,i]) # FEM analysis linear sistem
            
#             # Forces   
#             F = A_optim_unreduced*E/myBCs.ground_structure_length * (B.T @ U)
            
#             F_comp = F.copy()
#             U_comp = U.copy()
            
#             F_comp[reduction_pattern_F] = F_optim[:,i]
#             U_comp[reduction_pattern_U] = U_optim[:,i]
            
#             err_F = np.linalg.norm(F_comp-F,np.inf)
#             err_disp = np.linalg.norm(U_comp-U,np.inf)
            
#             print('Relative Force Error (inf-norm) LC{0:03d}: {1:.6f}'.format(i+1,err_F))
#             print('Relative Displacement Error (inf-norm) LC{0:03d}: {1:.6f}'.format(i+1,err_disp))

#             with open(foldername+'/'+'log'+'.txt', 'a') as f:
#                 f.write('\n')
#                 f.write('Relative Force Error (inf-norm) LC{0:03d}: {1:.8f}'.format(i+1,err_F))
#                 f.write('\n')
#                 f.write('Relative Displacement Error (inf-norm) LC{0:03d}: {1:.8f}'.format(i+1,err_disp))
 
# def simplify_GS(myBCs, a, tol, tol_q, tol_U, tol_brutal, E, nodes_loads, foldername):
#     candidates = np.ones(a.shape[0],dtype=bool)
#     n_load_cases = myBCs.R.shape[1]  
#     B = calcB_3D(myBCs)
#     # Brutal thresold
#     a[a<tol_brutal*np.max(a)] = 1e-16
#     candidates[a<tol_brutal*np.max(a)]=False
#     K = calcK(myBCs, B, E, a)
#     a_iter = a.copy()
#     n_bars_elim_U = 0
#     n_bars_elim_q = 0
    
#     M = myBCs.M
#     N = myBCs.N
    
#     # FEM
#     q_0 = np.zeros((N,1)) 
#     U_0 = np.zeros((M,1)) 
#     for p in range(n_load_cases):    
#     # Initial forces and displacements are calculated using FEM
#         U = np.zeros(M)
#         keep = myBCs.free_dofs
#         K_free = K[keep, :][:, keep]
#         U[keep] = sp.linalg.spsolve(K_free, myBCs.R_free[:,p]) # FEM analysis linear sistem
#         U_0 = np.hstack([U_0, U.reshape((-1,1))])
#         # Forces   
#         q = a*E/myBCs.ground_structure_length * (B.T @ U) 
#         q_0 = np.hstack([q_0,q.reshape((-1,1))])
    
#     U_0 = U_0[:,1:]
#     q_0 = q_0[:,1:]
#     if np.max(U_0)>50:
#         raise Exception("Brutal threshold too low")
    
#     # Sort a
#     indexes = np.argsort(a)
#     a_ord = a[indexes]
    
    
    
#     # Apply threshold (the heuristics is used only on these bars)
#     a_ord_threshold = a_ord[a_ord<tol*a_ord[-1]]
#     print("\n***************************\n*** STRUCTURE REDUCTION ***\n***************************\n")
#     # Heuristic on all the thresholded bars
#     for i in range(a_ord_threshold.size):
#         # Solve the nodes equilibrium using FEM
#         nodes_candidates = np.unique(myBCs.ground_structure[candidates,:2].ravel()) # Update candidates and dofs candidates
#         dofs_candidates = np.sort(np.array([nodes_candidates*3,nodes_candidates*3+1,nodes_candidates*3+2]).ravel())
        
#         # Update B and K
#         a_iter[indexes[i]] = 1e-16
#         K = calcK(myBCs, B, E, a_iter) # Original B but updated a
#         try:
#             # FEM
#             q_perturbed = np.zeros((N,1)) 
#             U_perturbed = np.zeros((M,1)) 
#             for p in range(n_load_cases):    
#                 # Initial forces and displacements are calculated using FEM
#                 U = np.zeros(M)
#                 dofs = np.arange(M)
#                 dofs[myBCs.fixed_dofs] = -1
#                 dofs = dofs[dofs_candidates]
#                 keep = dofs[dofs!=-1]
#                 K_free = K[keep, :][:, keep]
#                 U[keep] = sp.linalg.spsolve(K_free, myBCs.R[keep,p]) # FEM analysis linear sistem
#                 U_perturbed = np.hstack([U_perturbed, U.reshape((-1,1))])
#                 # Forces   
#                 q = a*E/myBCs.ground_structure_length * (B.T @ U) 
#                 q_perturbed = np.hstack([q_perturbed,q.reshape((-1,1))])
            
#             U_perturbed = U_perturbed[:,1:]
#             q_perturbed = q_perturbed[:,1:]
#             # Check if the equilibrium is too perturbd
#             # Check on number of bars, do not delete bars important for loads eq
#             # Check forces and displacements, no degenerate structures
#             norm_q = np.linalg.norm(q_perturbed[candidates]-q_0[candidates], np.inf)
#             norm_U = np.linalg.norm(U_perturbed[dofs_candidates]-U_0[dofs_candidates], np.inf)
#             # Delete bar
#             if (norm_q<tol_q) and (norm_U<tol_U):
#                 candidates[indexes[i]]=False
#                 q_0 = q_perturbed.copy()
#                 U_0 = U_perturbed.copy()
#             else:
#                 a_iter[indexes[i]] = a[indexes[i]] # Restore original value
#             if norm_U<tol_U:
#                 n_bars_elim_U +=1
#             if norm_q<tol_q:
#                 n_bars_elim_q +=1
#         except:
#             pass
        
#     # Now restore all the bars used for loads (all the bars that start and end in a XY coordinate of load points)
#     X = myBCs.nodes[myBCs.ground_structure[:,1],0] - myBCs.nodes[myBCs.ground_structure[:,0],0]
#     Y = myBCs.nodes[myBCs.ground_structure[:,1],0] - myBCs.nodes[myBCs.ground_structure[:,0],0]
    
#     restore = []
#     for i in range(len(nodes_loads)):
#         xx = X==0.0
#         yy = Y==0.0
#         zz = np.all((myBCs.nodes[myBCs.ground_structure[:,0]] == myBCs.nodes[nodes_loads[i]])[:,:2],axis=1)
#         restore.extend(np.where(xx & yy & zz)[0].tolist())  
#     restore = list(set(restore))
#     candidates[restore]=True
    
     
#     print('Original number of bars: {0:d}'.format(candidates.size))       
#     print('Updated number of bars: {0:d}'.format(np.count_nonzero(candidates)))       
#     print('Bars eliminated: {0:d}'.format(candidates.size-np.count_nonzero(candidates)))       
#     print('Number of bars under threshold: {0:d}'.format(a_ord_threshold.size))       
#     print('Number of bars that would be eliminated by force: {0:d}'.format(n_bars_elim_q))       
#     print('Number of bars that would be eliminated by displacement: {0:d}\n'.format(n_bars_elim_U))   
#     with open(foldername+'/'+'simplify'+'.txt', 'w') as f:
#         f.write('Area tol: {0:.2E}\n'.format(tol))       
#         f.write('Area tol (brutal): {0:.2E}\n'.format(tol_brutal))       
#         f.write('q tol: {0:.2E}\n'.format(tol_q))       
#         f.write('U tol: {0:.2E}\n'.format(tol_U))       
#         f.write('Original number of bars: {0:d}\n'.format(candidates.size))       
#         f.write('Updated number of bars: {0:d}\n'.format(np.count_nonzero(candidates)))       
#         f.write('Bars eliminated: {0:d}\n'.format(candidates.size-np.count_nonzero(candidates)))       
#         f.write('Number of bars under threshold: {0:d}\n'.format(a_ord_threshold.size))       
#         f.write('Number of bars that would be eliminated by force: {0:d}\n'.format(n_bars_elim_q))       
#         f.write('Number of bars that would be eliminated by displacement: {0:d}\n'.format(n_bars_elim_U))   
#     return candidates
 
# def buckling_stress_pp(myBCs,myBCs_reduced,a_opt,s_buck,E,a_reduced):
#     if myBCs.is3D:
#         B_old = calcB_3D(myBCs) # equilibrium matrix
#     else:   
#         B_old = calcB(myBCs) # equilibrium matrix
#     M_old = myBCs.dofs_list.size # Number of DOFs of the Groud Structure

#     # Areas
#     area_phys = a_opt + np.max(a_opt)*1e-12 # to avoid K singular matrix
        
#     # Initial forces and displacements are calculated using FEM
#     U = np.zeros(M_old)
#     K = calcK(myBCs,B_old,E,area_phys)
#     keep = myBCs.free_dofs
#     K = K[keep, :][:, keep]
#     U[keep] = sp.linalg.spsolve(K, myBCs.R_free) # FEM analysis linear sistem
#     U_free = U[myBCs_reduced.dofs_candidates]
#     # Forces   
#     F = area_phys*E/myBCs.ground_structure_length * (B_old.T @ U)
#     F_free = F[myBCs_reduced.candidates]
        
            
#     a_upd=a_opt.copy()
#     try:
#         buck = F_free + (s_buck/myBCs_reduced.ground_structure_length**2) * a_opt[myBCs_reduced.candidates]**2
#         a_upd[myBCs_reduced.candidates][buck<0]=np.sqrt(-F_free[buck<0]/(s_buck/myBCs_reduced.ground_structure_length[buck<0]**2))
#         vol = a_upd[myBCs_reduced.candidates].T @ myBCs_reduced.ground_structure_length
#     except: # Does not work really well, to improve
#         FF = F_free.copy()
#         for xx in range(myBCs_reduced.merged_candidates.shape[0]):
#             ind = myBCs_reduced.merged_candidates[xx]
#             val = FF[ind]
#             FF = np.delete(FF, ind, axis=0)
#             FF = np.append(FF,val[0])
#         buck = FF + (s_buck/myBCs_reduced.ground_structure_length**2) * a_reduced**2
#         a_reduced[buck<0]=np.sqrt(-FF[buck<0]/(s_buck/myBCs_reduced.ground_structure_length[buck<0]**2))
#         vol = a_reduced.T @ myBCs_reduced.ground_structure_length
            
#     U = np.zeros(M_old)
#     K = calcK(myBCs,B_old,E,a_upd)
#     keep = myBCs.free_dofs
#     K = K[keep, :][:, keep]
#     U[keep] = sp.linalg.spsolve(K, myBCs.R_free) # FEM analysis linear sistem
#     U_free = U[myBCs_reduced.dofs_candidates]
#     # Forces   
#     F = area_phys*E/myBCs.ground_structure_length * (B_old.T @ U) 
#     F_free = F[myBCs_reduced.candidates]
        
#     F_out = np.zeros((area_phys.size))
#     U_out = np.zeros((M_old))
    
#     F_out[myBCs_reduced.candidates] = F_free
#     U_out[myBCs_reduced.dofs_candidates] = U_free 
    
#     if np.abs(vol-vol_LP)>1:
#         pass
        
#     return vol, a_upd, F_out, U_out

# def buckling_stress_pp_multi(myBCs,myBCs_reduced,a_opt,s_buck,E,a_reduced):
#     if myBCs.is3D:
#         B_old = calcB_3D(myBCs) # equilibrium matrix
#     else:   
#         B_old = calcB(myBCs) # equilibrium matrix
#     M_old = myBCs.dofs_list.size # Number of DOFs of the Groud Structure
        
#     n_load_cases = myBCs.R.shape[-1]    

#     # Areas
#     area_phys = a_opt + np.max(a_opt)*1e-12 # to avoid K singular matrix
        
#     U_list, F_list = [], []
#     for p in range(n_load_cases):    
#     # Initial forces and displacements are calculated using FEM
#         U = np.zeros(M_old)
#         K = calcK(myBCs,B_old,E,area_phys)
#         keep = myBCs.free_dofs
#         K = K[keep, :][:, keep]
#         U[keep] = sp.linalg.spsolve(K, myBCs.R_free[:,p]) # FEM analysis linear sistem
#         U_list.append(U[myBCs_reduced.dofs_candidates])
#         # Forces   
#         F = area_phys*E/myBCs.ground_structure_length * (B_old.T @ U) 
#         F_list.append(F[myBCs_reduced.candidates])
            
#     a_upd=a_opt.copy()
#     for p in range(n_load_cases):
#         try:
#             buck = F_list[p] + (s_buck/myBCs_reduced.ground_structure_length**2) * a_opt[myBCs_reduced.candidates]**2
#             a_upd[myBCs_reduced.candidates][buck<0]=np.sqrt(-F_list[p][buck<0]/(s_buck/myBCs_reduced.ground_structure_length[buck<0]**2))
#             vol = a_upd[myBCs_reduced.candidates].T @ myBCs_reduced.ground_structure_length
#         except: # Does not work really well, to improve
#             FF = np.array(F_list).copy().T[:,p]
#             for xx in range(myBCs_reduced.merged_candidates.shape[0]):
#                 ind = myBCs_reduced.merged_candidates[xx]
#                 val = FF[ind]
#                 FF = np.delete(FF, ind, axis=0)
#                 FF = np.append(FF,val[0])
#             buck = FF + (s_buck/myBCs_reduced.ground_structure_length**2) * a_reduced**2
#             a_reduced[buck<0]=np.sqrt(-FF[buck<0]/(s_buck/myBCs_reduced.ground_structure_length[buck<0]**2))
#             vol = a_reduced.T @ myBCs_reduced.ground_structure_length
            
    
    
    
#     U_list, F_list = [], []
#     for p in range(n_load_cases):    
#         U = np.zeros(M_old)
#         K = calcK(myBCs,B_old,E,a_upd)
#         keep = myBCs.free_dofs
#         K = K[keep, :][:, keep]
#         U[keep] = sp.linalg.spsolve(K, myBCs.R_free[:,p]) # FEM analysis linear sistem
#         U_list.append(U[myBCs_reduced.dofs_candidates])
#         # Forces   
#         F = area_phys*E/myBCs.ground_structure_length * (B_old.T @ U) 
#         F_list.append(F[myBCs_reduced.candidates])
        
#     F_out = np.zeros((area_phys.size,n_load_cases))
#     U_out = np.zeros((M_old,n_load_cases))
    
#     F_out[myBCs_reduced.candidates,:] = np.array(F_list).T
#     U_out[myBCs_reduced.dofs_candidates,:] = np.array(U_list).T 
    
#     if np.abs(vol-vol_LP)>1:
#         pass
        
#     return vol, a_upd, F_out, U_out
 
# def symmetry_model(myBCs, sections, foldername):
#     """ Create the full model for the SimplySuppTruss_3D_symmetric test case """
#     # Nodes
#     nodes = myBCs.nodes.copy()
#     add = myBCs.nodes.copy()
#     add[:,0] = myBCs.L[0]*2-myBCs.nodes[:,0]
#     nodes = np.vstack((nodes, add))
#     add = myBCs.nodes.copy()
#     add[:,1] = myBCs.L[1]*2-myBCs.nodes[:,1]
#     nodes = np.vstack((nodes, add))
#     add = myBCs.nodes.copy()
#     add[:,0] = myBCs.L[0]*2-myBCs.nodes[:,0]
#     add[:,1] = myBCs.L[1]*2-myBCs.nodes[:,1]
#     nodes = np.vstack((nodes, add))
    
#     # Connectivity
#     M_old = myBCs.nodes.shape[0]
#     gs = myBCs.ground_structure.copy()
#     add = myBCs.ground_structure.copy()
#     add += int(M_old)
#     gs = np.vstack((gs, add))
#     add += int(M_old)
#     gs = np.vstack((gs, add))
#     add += int(M_old)
#     gs = np.vstack((gs, add))
    
#     # Area
#     a=np.tile(sections,4)
    
#     # Save
#     folder = foldername+'/'+'db'+'/'
#     np.savetxt(folder+'a.dat', a, fmt='%.5f')
#     np.savetxt(folder+'nodes.dat', nodes, fmt='%.5f')
#     np.savetxt(folder+'GS.dat', gs, fmt='%d')
  
def save_files(a,q,U,myBCs,vol,stress_tension_max, stress_compression_max, joint_cost, obj_hist, a_cell, E, s_buck, cellsEquals, foldername, LP, candidates,  
               L, rho, ff, time, is3D, isL, vol_LP = False, eq_hist= False, s_c_hist= False, s_t_hist= False, comp_hist= False, buck_hist= False, is_free_form=False, a_unreduced=0, system=0, a_in = False,
               cell_mapping_vector=np.array([0])):
    if not os.path.isdir(foldername):
        os.makedirs(foldername)
    folder = foldername+'/'+'db'+'/'
    if not os.path.isdir(folder):
        os.makedirs(folder)
    if LP:
        folder += 'LP/'
        if not os.path.isdir(folder):
            os.makedirs(folder)
        if np.any(a_in)!=False:
            np.savetxt(folder+'a_LP_x0.dat', a_in)
        np.save(folder+'a_LP', a)
        np.save(folder+'q_LP', q)
        np.save(folder+'U_LP', U)
        np.savetxt(folder+'a_LP.dat', a, fmt='%.5f')
        np.savetxt(folder+'q_LP.dat', q, fmt='%.5f')
        np.savetxt(folder+'U_LP.dat', U, fmt='%.5f')
        np.savetxt(folder+'GS_LP.dat', myBCs.ground_structure, fmt='%d')
        np.savetxt(folder+'cell_mapping_vector.dat', cell_mapping_vector, fmt='%d')
        np.savetxt(folder+'f_LP.dat', myBCs.R, fmt='%.5f')
        np.savetxt(folder+'fix_dofs_LP.dat', myBCs.fixed_dofs, fmt='%d')
        np.savetxt(folder+'nodes_LP.dat', myBCs.nodes, fmt='%.5f')
        np.savetxt(folder+'L_LP.dat', myBCs.ground_structure_length, fmt='%.5f')
        np.savetxt(folder+'a_LP_cand.dat', a[candidates], fmt='%.5f')
        np.savetxt(folder+'q_LP_cand.dat', q[candidates], fmt='%.5f')
        np.savetxt(folder+'GS_LP_cand.dat', myBCs.ground_structure[candidates], fmt='%d')
        
    else: 
        if np.any(a_in)!=False:
            np.savetxt(folder+'a_x0.dat', a_in) 
        np.save(folder+'a', a)
        np.save(folder+'a_unreduced', a_unreduced)
        np.save(folder+'q', q)
        np.save(folder+'U', U)
        np.save(folder+'GS', myBCs.ground_structure)
        # np.save(folder+'GS_unreduced', myBCs.ground_structure_not_reduced)
        np.save(folder+'nodes', myBCs.nodes)
        # np.save(folder+'nodes_unreduced', myBCs.nodes_not_reduced)
        np.savetxt(folder+'a.dat', a, fmt='%.5f')
        np.savetxt(folder+'q.dat', q, fmt='%.5f')
        np.savetxt(folder+'U.dat', U, fmt='%.5f')
        np.savetxt(folder+'GS.dat', myBCs.ground_structure, fmt='%d')
        np.savetxt(folder+'f.dat', myBCs.R, fmt='%.5f')
        np.savetxt(folder+'fix_dofs.dat', myBCs.fixed_dofs, fmt='%d')
        np.savetxt(folder+'nodes.dat', myBCs.nodes, fmt='%.5f')
        np.savetxt(folder+'L.dat', myBCs.ground_structure_length, fmt='%.5f')
        
    with shelve.open(folder+'saved_session','c') as my_shelf:
        my_shelf['E'] = E
        my_shelf['s_buck'] = s_buck
        my_shelf['cellsEquals'] = cellsEquals
        
        if LP:
            my_shelf['BCs_LP'] = myBCs
            my_shelf['a_LP'] = a
            my_shelf['q_LP'] = q
            my_shelf['U_LP'] = U
            my_shelf['vol_LP'] = vol
            my_shelf['stress_c_max_LP'] = stress_compression_max
            my_shelf['stress_t_max_LP'] = stress_tension_max
            my_shelf['JC_LP'] = joint_cost
            my_shelf['obj_hist_LP'] = obj_hist
            my_shelf['a_cell_LP'] = a_cell
        else:
            my_shelf['BCs'] = myBCs
            my_shelf['a'] = a
            my_shelf['q'] = q
            my_shelf['U'] = U
            my_shelf['vol'] = vol
            my_shelf['stress_c_max'] = stress_compression_max
            my_shelf['stress_t_max'] = stress_tension_max
            my_shelf['JC'] = joint_cost
            my_shelf['obj_hist'] = obj_hist
            my_shelf['a_cell'] = a_cell
            if np.any(eq_hist) != False:
                my_shelf['eq_hist'] = eq_hist
                my_shelf['s_c_hist'] = s_c_hist
                my_shelf['s_t_hist'] = s_t_hist
                my_shelf['comp_hist'] = comp_hist
                my_shelf['buck_hist'] = buck_hist
    
    my_shelf.close()
    if is3D:
        if LP:
            with open(foldername+'/'+'log_LP'+'.txt', 'w') as f:
                if system == 0:
                    f.write("Vol: {0:.2f} mm3, Weight: {1:.5f} kg\n".format(vol, vol*rho*1000))
                elif system == 1:
                    f.write("Vol: {0:.2f} m3, Weight: {1:.5f} kg\n".format(vol, vol*rho))
                #f.write("Vol_star: {0:.5f} PL/sigma\n".format(vol/(ff*max(L)/((stress_tension_max-stress_compression_max)/2))))
                f.write("Vol_fraction: {0:.6f}%\n".format(vol*100/(L[0]*L[1]*L[2])))
                if system == 0:
                    f.write("Max section: {0:.3f} mm2\n".format(np.max(a)))
                elif system == 1:
                    f.write("Max section: {0:.3f} m2\n".format(np.max(a)))
                f.write("Optimization SLP time: %.2f seconds\n" % time) 
                f.write("N candidate bars: %d \n" % myBCs.ground_structure.shape[0]) 
                f.write("N active bars: {0:d} \n".format(np.count_nonzero(candidates))) 
                f.write("Min slenderness ratio: {0:.3f}\n".format(np.min(myBCs.ground_structure_length[candidates]/np.sqrt((s_buck*a[candidates])/(np.pi**2*E))))) # ratio between length and Radius of gyration
                if is_free_form:
                    a_aper=a[myBCs.ground_structure[:,2]==-1]
                    vol_aper=a_aper.T@myBCs.ground_structure_length[myBCs.ground_structure[:,2]==-1]
                    if vol_aper.size == 0:
                        vol_aper=0
                    a_cell=a[myBCs.ground_structure[:,2]==1]
                    vol_cell=a_cell.T@myBCs.ground_structure_length[myBCs.ground_structure[:,2]==1]
                    if vol_cell.size == 0:
                        vol_cell=0
                    a_sect=a[myBCs.ground_structure[:,2]==2]
                    vol_sect=a_sect.T@myBCs.ground_structure_length[myBCs.ground_structure[:,2]==2]
                    if vol_sect.size == 0:
                        vol_sect=0
                    a_equal=a[myBCs.ground_structure[:,2]==3]
                    vol_equal=a_equal.T@myBCs.ground_structure_length[myBCs.ground_structure[:,2]==3]
                    if vol_equal.size == 0:
                        vol_equal=0
                    if system == 0:
                        f.write("Vol_aper: {0:.2f} mm3, Weight_aper: {1:.5f} kg\n".format(vol_aper, vol_aper*rho*1000))
                        f.write("Vol_cell: {0:.2f} mm3, Weight_cell: {1:.5f} kg\n".format(vol_cell, vol_cell*rho*1000))
                        f.write("Vol_sec: {0:.2f} mm3, Weight_sec: {1:.5f} kg\n".format(vol_sect, vol_sect*rho*1000))
                        f.write("Vol_equal: {0:.2f} mm3, Weight_equal: {1:.5f} kg\n".format(vol_equal, vol_equal*rho*1000))
                    elif system == 1:
                        f.write("Vol_aper: {0:.2f} m3, Weight_aper: {1:.5f} kg\n".format(vol_aper, vol_aper*rho))
                        f.write("Vol_cell: {0:.2f} m3, Weight_cell: {1:.5f} kg\n".format(vol_cell, vol_cell*rho))
                        f.write("Vol_sec: {0:.2f} m3, Weight_sec: {1:.5f} kg\n".format(vol_sect, vol_sect*rho))
                        f.write("Vol_equal: {0:.2f} m3, Weight_equal: {1:.5f} kg\n".format(vol_equal, vol_equal*rho))
        else:
            with open(foldername+'/'+'log'+'.txt', 'w') as f:
                if system == 0:
                    f.write("Vol: {0:.2f} mm3, Weight: {1:.5f} kg\n".format(vol, vol*rho*1000))
                elif system == 1:
                    f.write("Vol: {0:.2f} m3, Weight: {1:.5f} kg\n".format(vol, vol*rho))
                #f.write("Vol_star: {0:.5f} PL/sigma\n".format(vol/(ff*max(L)/((stress_tension_max-stress_compression_max)/2))))
                f.write("Vol_fraction: {0:.6f}%\n".format(vol*100/(L[0]*L[1]*L[2])))
                if myBCs.R.ndim == 2:
                    for i in range(myBCs.R.shape[-1]):
                        R = myBCs.R[:,i].copy()
                        U_c = U[:,i].copy()
                        if system == 0:
                            f.write("Compliance - LC{0:03d}: {1:.2f} mJ\n".format(i+1,R.T@U_c))
                        elif system == 1:
                            f.write("Compliance - LC{0:03d}: {1:.2f} MJ\n".format(i+1,R.T@U_c))
                else:  
                    if system == 0:
                        f.write("Compliance: {0:.2f} mJ\n".format(myBCs.R.T@U))
                    elif system == 1:
                        f.write("Compliance: {0:.2f} MJ\n".format(myBCs.R.T@U))
                if system == 0:
                    f.write("Max section: {0:.3f} mm2\n".format(np.max(a)))
                elif system == 1:
                    f.write("Max section: {0:.3f} m2\n".format(np.max(a)))
                f.write("Optimization time: %.2f seconds\n" % time)
                f.write("N candidate bars: %d \n" % myBCs.ground_structure.shape[0]) 
                f.write("N active bars: {0:d} \n".format(np.count_nonzero(candidates))) 
                f.write("Min slenderness ratio: {0:.3f}\n".format(np.min(myBCs.ground_structure_length[candidates]/np.sqrt((s_buck*np.abs(a[candidates]))/(np.pi**2*E))))) # ratio between length and Radius of gyration
                if is_free_form:
                    a_aper=a[myBCs.ground_structure[:,2]==-1]
                    vol_aper=a_aper.T@myBCs.ground_structure_length[myBCs.ground_structure[:,2]==-1]
                    if vol_aper.size == 0:
                        vol_aper=0
                    a_cell=a[myBCs.ground_structure[:,2]==1]
                    vol_cell=a_cell.T@myBCs.ground_structure_length[myBCs.ground_structure[:,2]==1]
                    if vol_cell.size == 0:
                        vol_cell=0
                    a_sect=a[myBCs.ground_structure[:,2]==2]
                    vol_sect=a_sect.T@myBCs.ground_structure_length[myBCs.ground_structure[:,2]==2]
                    if vol_sect.size == 0:
                        vol_sect=0
                    a_equal=a[myBCs.ground_structure[:,2]==3]
                    vol_equal=a_equal.T@myBCs.ground_structure_length[myBCs.ground_structure[:,2]==3]
                    if vol_equal.size == 0:
                        vol_equal=0
                    if system == 0:
                        f.write("Vol_aper: {0:.2f} mm3, Weight_aper: {1:.5f} kg\n".format(vol_aper, vol_aper*rho*1000))
                        f.write("Vol_cell: {0:.2f} mm3, Weight_cell: {1:.5f} kg\n".format(vol_cell, vol_cell*rho*1000))
                        f.write("Vol_sec: {0:.2f} mm3, Weight_sec: {1:.5f} kg\n".format(vol_sect, vol_sect*rho*1000))
                        f.write("Vol_equal: {0:.2f} mm3, Weight_equal: {1:.5f} kg\n".format(vol_equal, vol_equal*rho*1000))
                    elif system == 1:
                        f.write("Vol_aper: {0:.2f} m3, Weight_aper: {1:.5f} kg\n".format(vol_aper, vol_aper*rho))
                        f.write("Vol_cell: {0:.2f} m3, Weight_cell: {1:.5f} kg\n".format(vol_cell, vol_cell*rho))
                        f.write("Vol_sec: {0:.2f} m3, Weight_sec: {1:.5f} kg\n".format(vol_sect, vol_sect*rho))
                        f.write("Vol_equal: {0:.2f} m3, Weight_equal: {1:.5f} kg\n".format(vol_equal, vol_equal*rho))
                if vol_LP != False:
                    f.write("NLP volume add: {0:.6f} %".format(vol/vol_LP*100 - 100)) 
    else: 
        if LP:
            with open(foldername+'/'+'log_LP'+'.txt', 'w') as f:
                if system == 0:
                    f.write("Vol: {0:.2f} mm3, Weight: {1:.5f} kg\n".format(vol, vol*rho*1000))
                elif system == 1:
                    f.write("Vol: {0:.2f} m3, Weight: {1:.5f} kg\n".format(vol, vol*rho))
                f.write("Vol_star: {0:.5f} PL/sigma\n".format(vol/(ff*max(L)/((stress_tension_max-stress_compression_max)/2))))
                if system == 0:
                    f.write("Max section: {0:.3f} mm2\n".format(np.max(a)))
                elif system == 1:
                    f.write("Max section: {0:.3f} m2\n".format(np.max(a)))
                f.write("Optimization SLP time: %.2f seconds\n" % time) 
                f.write("N candidate bars: %d \n" % myBCs.ground_structure.shape[0]) 
                f.write("N active bars: {0:d} \n".format(np.count_nonzero(candidates))) 
                f.write("Min slenderness ratio: {0:.3f}\n".format(np.min(myBCs.ground_structure_length[candidates]/np.sqrt((s_buck*np.abs(a[candidates]))/(np.pi**2*E))))) # ratio between length and Radius of gyration
        else:
            with open(foldername+'/'+'log'+'.txt', 'w') as f:
                if system == 0:
                    f.write("Vol: {0:.2f} mm3, Weight: {1:.5f} kg\n".format(vol, vol*rho*1000))
                elif system == 1:
                    f.write("Vol: {0:.2f} m3, Weight: {1:.5f} kg\n".format(vol, vol*rho))
                f.write("Vol_star: {0:.5f} PL/sigma\n".format(vol/(ff*max(L)/((stress_tension_max-stress_compression_max)/2))))
                if system == 0:
                    f.write("Max section: {0:.3f} mm2\n".format(np.max(a)))
                elif system == 1:
                    f.write("Max section: {0:.3f} m2\n".format(np.max(a)))
                f.write("Optimization time: %.2f seconds\n" % time)
                f.write("N candidate bars: %d \n" % myBCs.ground_structure.shape[0]) 
                f.write("N active bars: {0:d} \n".format(np.count_nonzero(candidates))) 
                f.write("Min slenderness ratio: {0:.3f}\n".format(np.min(myBCs.ground_structure_length[candidates]/np.sqrt((s_buck*a[candidates])/(np.pi**2*E))))) # ratio between length and Radius of gyration
                if vol_LP != False:
                    f.write("NLP volume add: {0:.6f} %".format(vol/vol_LP*100 - 100)) 
            
# def load_files(folder):
    
    with shelve.open(folder,'c') as my_shelf:
        for key in my_shelf:
            globals()[key]=my_shelf[key]
    my_shelf.close()

#################################
## Optimization methods
#################################
# def solveLP_2D(myBCs: BCS.MBB2D_Symm, stress_T_max, stress_C_max, joint_cost = 0, cellsEquals: bool = False, cell_mapping_vector = False):
#     l = myBCs.ground_structure_length + joint_cost * np.max(myBCs.ground_structure_length) # indipendency from the number of cells
#     B = calcB(myBCs) # equilibrium matrix
#     # VL parameter
#     NN, n_topologies_cell, cell_mapping_matrix  = mapping_matrix_init(cell_mapping_vector, myBCs)
    
#     if cellsEquals:        
#         a_wholestruct = np.zeros(myBCs.ncel_str)
#         a_cell = np.zeros(n_topologies_cell, dtype='object_')
#         for i in range(n_topologies_cell): # starting from the lower left corner
#             a_cell[i] = cvx.Variable((int(myBCs.ground_structure.shape[0]/myBCs.ncel_str),1), nonneg=True)
#             a_wholestruct += cvx.kron(cell_mapping_matrix[:,i].reshape(-1,1), a_cell[i]) # initialize a_phys 
#     else:
#         a_cell = cvx.Variable((myBCs.ground_structure.shape[0],1), nonneg=True) # initialization of the design variables
#         a_wholestruct = a_cell
#     obj = cvx.Minimize(l.T @ a_wholestruct) # definition of the objective function
#     # constraints
#     q = cvx.Variable((myBCs.ground_structure.shape[0],1))
#     cons = [B @ q == (myBCs.R * myBCs.dofs).reshape(-1,1)]
#     cons2 = q <= stress_T_max * a_wholestruct
#     cons3 = q >= stress_C_max * a_wholestruct
#     cons.extend([cons2, cons3])

#     prob = cvx.Problem(obj, cons) # Creation of LP
#     vol = prob.solve(verbose=True, solver='ECOS') # Problem solving
#     q = np.array(q.value).ravel() # Optimized forces
#     a = np.array(a_wholestruct.value).ravel() # Optimized areas

#     # Eliminate the influence of the joint cost to the objective function
#     vol = myBCs.ground_structure_length.T @ a
#     U = np.zeros(myBCs.dofs.size)
#     obj_hist = False
#     if cellsEquals:
#         a_cell_out = np.zeros(n_topologies_cell, dtype='object_')
#         for i in range(a_cell_out.size):
#             a_cell_out[i] = np.array(a_cell[i].value).ravel()
#         return vol, a, q, U, obj_hist, a_cell_out
#     else:
#         return vol, a, q, U, obj_hist

def solveLP_2D_SLP_Buckling(myBCs: BCS.MBB2D_Symm, stress_T_max, stress_C_max, s_buck, joint_cost = 0, cellsEquals: bool = False, cell_mapping_vector = False, a_init = False, chain=True, foldername = False):
    l = myBCs.ground_structure_length + joint_cost * np.max(myBCs.ground_structure_length) # indipendency from the number of cells
    B = calcB(myBCs) # equilibrium matrix
    #
    # Design variable initialization
    #
    # VL parameter
    NN, n_topologies_cell, cell_mapping_matrix  = mapping_matrix_init(cell_mapping_vector, myBCs)
    
    if cellsEquals:        
        a_wholestruct = np.zeros(myBCs.ncel_str, dtype='object_')
        a_cell = np.zeros(n_topologies_cell, dtype='object_')
        for i in range(n_topologies_cell): # starting from the lower left corner
            a_cell[i] = cvx.Variable((int(myBCs.ground_structure.shape[0]/myBCs.ncel_str),1), nonneg=True)
            a_wholestruct = cvx.kron(cell_mapping_matrix[:,i].reshape(-1,1), a_cell[i]) # initialize a_phys 
    else:
        a_cell = cvx.Variable((myBCs.ground_structure.shape[0],1), nonneg=True) # initialization of the design variables
        a_wholestruct = a_cell
    obj = cvx.Minimize(l.T @ a_wholestruct) # definition of the objective function
    #
    # Start of the buckling loop
    #
    tol, i = 1, 0
    tol_old = 1
    vol = 1
    k_buck = 1
    obj_hist = []
    #a  = np.ones(a_wholestruct.size) * 50 # Initialization
    if np.any(a_init!=False):
        a = a_init # Initialization
    else:
        a  = np.ones(a_wholestruct.size) * 10 # Initialization
        
    # Buckling specific parameters
    
    buckling_length = myBCs.ground_structure_length.copy()
    
    debug = False
    reset = False
    chain_control = chain
    only_first = True
    best_case = True
    plot_intermediate_reset = False
    min_section_linearized = False
    
    best_vol = 1e19
    reset_val_old = 0
    
    while ((tol>1e-06 and i<300) or reset):
        # Buckling calculation based on the previuos iteration section
        only_first_activate_constraint = False # buckling constr only on the first bar
        a_0 = a.copy()
        a_0_buck = a.copy() # need for another variable. we cant use a_buc for the tolerance calc
        buckling_length = myBCs.ground_structure_length.copy()
            
        if i>400: 
            break
        if (tol<1e-4 and tol_old<1e-4 and reset) or (i == 75) or (i == 150):
            reset_val = 0.8**k_buck
            if plot_intermediate_reset:
                # save image before perturbation
                folder_case = foldername+'/LP/reset/'+'It_{0:0>4}-K_{2:0>3}-val_{1:.3f}/'.format(i,reset_val_old,k_buck)
                if not os.path.isdir(folder_case):
                    os.makedirs(folder_case)
                trussplot.plot2D.plotTruss(myBCs, a, q, max(a) * 1e-3, myBCs.ground_structure_length.T @ a, folder_case)
            reset_val_old = reset_val
            a_0_buck[(a<np.max(a_0_buck)*0.05)] = np.max(a_0_buck)*reset_val
            k_buck *= 2
            if best_case:
                candidate_best_vol = myBCs.ground_structure_length.T @ a
                if candidate_best_vol<=best_vol:
                    best_vol = candidate_best_vol
                    best_a = a.copy()
                    best_q = q.copy()
                    if cellsEquals:
                        best_a_cell = np.zeros(n_topologies_cell, dtype='object_')
                        for ii in range(n_topologies_cell):
                            best_a_cell[ii] = np.array(a_cell[ii].value).ravel()
                    
                    
            if k_buck>=2: # stopping condition
                reset = False
            print('REINIT')
        
        are_bars_in_compression = False    
        if chain_control and i!=0:
            # Identify if there is some chain buckling
            reduction_pattern = a_0>(1e-3*max(a_0))
            reduced_ground_structure = myBCs.ground_structure[reduction_pattern].copy()
            reduced_q = q[reduction_pattern].copy()
            unique, counts = np.unique(reduced_ground_structure, return_counts=True)
            possible_chain_nodes = np.array(unique[np.where(counts==2)]).ravel()
            chain_nodes = []
            for k in range(possible_chain_nodes.shape[0]): # needed to avoid adding BC nodes
                candidates_id = (reduced_ground_structure[:,0]==possible_chain_nodes[k]) | (reduced_ground_structure[:,1]==possible_chain_nodes[k])
                candidates = reduced_ground_structure[candidates_id,:]
                diff = np.setxor1d(candidates[0,:],candidates[1,:])
                start, end = np.min(diff), np.max(diff)
                if np.all(reduced_q[candidates_id]<0):
                    are_bars_in_compression = True
                x0,y0 = myBCs.nodes[candidates[0,0],:]
                x1,y1 = myBCs.nodes[candidates[0,1],:]
                x2,y2 = myBCs.nodes[candidates[1,0],:]
                x3,y3 = myBCs.nodes[candidates[1,1],:]
                angle1 = np.arctan2(y1-y0, x1-x0)
                angle2 = np.arctan2(y3-y2, x3-x2)
                if np.allclose(np.abs(angle1),np.abs(angle2),rtol=1e-2):
                    if (possible_chain_nodes[k]>start) and (possible_chain_nodes[k]<end):#If not some BCs can become candidates
                        if are_bars_in_compression: # We charge only nodes in a compression chain
                            chain_nodes.append(possible_chain_nodes[k])
                
            if len(chain_nodes)!=0: # augmented length approach
                buckling_length = myBCs.ground_structure_length.copy()
                total_leng = 0
                treated_nodes = []
                first_bar_chain = []
                member_to_add_id = []
                buckling_length_reduced = buckling_length[reduction_pattern]
                for k, node in enumerate(chain_nodes): # need to use loops and not a vectorized routine
                    candidates_bars_id = (reduced_ground_structure[:,0]==node) | (reduced_ground_structure[:,1]==node)
                    candidates_bars = reduced_ground_structure[candidates_bars_id,:] 
                    # check if the 2 members have a nodes in common with the last treated chain_node, if not leng = 0
                    if k!=0:
                        if node in treated_nodes[-1]:
                            treated_nodes[-1].update(candidates_bars.ravel().tolist())
                            last = candidates_bars_id & ~member_to_add # get only the new bar and update the length based on it
                            member_to_add += candidates_bars_id
                            total_leng += buckling_length_reduced[last]
                            member_to_add_id[-1].update(set(np.where(member_to_add)[0]))
                        else:
                            first_bar_chain.append(np.where(candidates_bars_id)[0][0])
                            treated_nodes.append(set(candidates_bars.ravel().tolist()))
                            total_leng = np.sum(buckling_length_reduced[candidates_bars_id])
                            member_to_add = candidates_bars_id 
                            member_to_add_id.append(set(np.where(member_to_add)[0]))
                    else:
                        first_bar_chain.append(np.where(candidates_bars_id)[0][0])
                        treated_nodes.append(set(candidates_bars.ravel().tolist()))
                        total_leng = np.sum(buckling_length_reduced[candidates_bars_id])
                        member_to_add = candidates_bars_id
                        member_to_add_id.append(set(np.where(member_to_add)[0])) 
                       
                    buckling_length_reduced[member_to_add] = total_leng
            
                if only_first:
                    only_first_activate_constraint = True
                    first_bar_chain_pattern = np.zeros(buckling_length_reduced.size, dtype='bool')
                    first_bar_chain_pattern[first_bar_chain] = True
                    buckling_length_reduced[~first_bar_chain_pattern] = myBCs.ground_structure_length[reduction_pattern][~first_bar_chain_pattern]
                
                buckling_length[reduction_pattern] = buckling_length_reduced
                    
                print('Chain buckling nodes elim.')
                  
        #
        # Constraints
        #
        q = cvx.Variable((myBCs.ground_structure.shape[0],1))
        cons = [B @ q == (myBCs.R * myBCs.dofs).reshape(-1,1)]
        cons2 = q <= stress_T_max * a_wholestruct
        cons3 = q >= stress_C_max * a_wholestruct
        cons.extend([cons2, cons3])
        
        if only_first_activate_constraint:
            for ii, leading in enumerate(first_bar_chain):
                for kk, other in enumerate(tuple(member_to_add_id[ii])):
                    if leading != other:
                        side_constr = a_wholestruct[reduction_pattern][leading] <= a_wholestruct[reduction_pattern][other]               
                        cons.extend([side_constr])
        
        if min_section_linearized:
            #sect_min = (myBCs.ground_structure_length**2)/(s_buck*slend_min**2)
            x0 = np.load('x0_val_Achtz.npy')
            candidates_x0 = x0>0.01
            sect_min = 0.1 * np.min(x0[candidates_x0])
            beta = 16
            #heavyside_lin = (1-(1+beta*a_0_buck)*np.exp(-beta*a_0_buck)) / (1-beta*np.exp(-beta*a_0_buck)+np.exp(-beta))
            heavyside_lin = 1-np.exp(-beta*a_0_buck) + a_0_buck*np.exp(-beta)
            cons_min_sect = a_wholestruct >= sect_min * heavyside_lin.reshape(-1,1)
            cons.extend([cons_min_sect])
        
        try:
            xx=(-s_buck*a_0_buck/(buckling_length)**2).reshape(-1,1)
            b=2*a_wholestruct - a_0_buck.reshape(-1,1)
            cons_buck = q >= cvx.atoms.affine.binary_operators.multiply(xx, b) # Linearized buckling constr
            cons.extend([cons_buck])
        except:
            pass
                 
        #
        # Problem solution
        #
        prob = cvx.Problem(obj, cons) # Creation of LP
        vol = prob.solve(verbose=False, solver='ECOS', max_iters = 4000) # Problem solving bfs=True for mosek if you want the basic solution (simplex)
        q = np.array(q.value).ravel() # Optimized forces     
        a_1 = np.array(a_wholestruct.value).ravel() # Optimized areas
        a = a_1.copy()
        tol_old = tol
        tol = np.linalg.norm(a-a_0, np.inf)
        # Eliminate the influence of the joint cost to the objective function
        vol = myBCs.ground_structure_length.T @ a
        obj_hist = np.append(obj_hist,vol)
        i = i+1 
        print('\n############### SLP ITERATION {0} ###############\n'.format(i))
        print('vol: {0:.4f} | tol: {1:.6f}\n'.format(vol,tol))
        if debug:
            trussplot.plot2D.plotTruss(myBCs, a, q, max(a) * 1e-3, vol, 1)
            #trussplot.plot2D.plotTrussBucklingCheck(myBCs, a, q, max(a) * 1e-3, s_buck)
            #trussplot.plot2D.plotTrussStress(myBCs, a, q, max(a) * 1e-3)  
            #input()
            
    if best_case:
        candidate_best_vol = myBCs.ground_structure_length.T @ a
        if candidate_best_vol<=best_vol:
            best_vol = candidate_best_vol
            best_a = a.copy()
            best_q = q.copy()
            if cellsEquals:
                best_a_cell = np.zeros(n_topologies_cell, dtype='object_')
                for ii in range(n_topologies_cell):
                    best_a_cell[ii] = np.array(a_cell[ii].value).ravel()
        try:
            a = best_a
            q = best_q
            vol = best_vol
        except:
            pass
    print('CONVERGED')
    print('looped {0} times, tol = {1:.8f}'.format(i,tol))
    print("vol SLP: {0:.5f} ".format(vol))
    
    U = np.zeros(myBCs.dofs.size)
    if cellsEquals:
        a_cell_out = best_a_cell
        return vol, a, q, U, obj_hist, a_cell_out
    else:
        return vol, a, q, U, obj_hist

# def solveNLP_2D_IPOPT(myBCs: BCS.MBB2D_Symm, stress_T_max, stress_C_max, E, folder, joint_cost = 0):
#     # SAND IPOPT optimization
#     B = calcB(myBCs) # equilibrium matrix
    
#     N = len(myBCs.ground_structure) # Number of member of the Ground Structure
#     M = len(myBCs.dofs) # Number of DOFs of the Groud Structure
    
#     # Create the indexing variables used for splitting the design variables vector x
#     init = np.zeros(2*N+M,dtype='bool')
#     area_id = init.copy()
#     area_id[0:N] = True
#     force_id = init.copy()
#     force_id[N:2*N] = True
#     U_id = init.copy()
#     U_id[2*N:] = True
    
#     ## Design variables bounds
#     lb = np.zeros(2*N+M)
#     ub = np.zeros(2*N+M)
#     # Areas
#     lb[area_id] = 0
#     ub[area_id] = 2.0e19 # inf
#     # Forces (no box constraints)
#     lb[force_id] = -2.0e19
#     ub[force_id] = 2.0e19
#     # Displacements (no box constraints)
#     lb[U_id] = -2.0e19
#     ub[U_id] = 2.0e19
    
#     ## Starting point
#     x0 = np.zeros(2*N+M) # Define an initial guess for the optimization
#     # Areas
#     x0[area_id] = 1.
#     # Initial forces and displacements are calculated using FEM
#     U = np.zeros(M)
#     K = calcK(myBCs,B,E,x0[area_id])
#     keep = myBCs.free_dofs
#     K = K[keep, :][:, keep]
#     U[keep] = sp.linalg.spsolve(K, myBCs.R_free) # FEM analysis linear sistem
#     # Forces
#     if np.any(np.isnan(U)): # Truss coming from the SLP can be a mechanism
#         U[np.isnan(U)] = 0
#     x0[force_id] = x0[area_id]*E/myBCs.ground_structure_length * (B.T @ U)
#     # Displacements
#     x0[U_id] = U
    
#     ## Constraints bounds
#     cl = np.zeros(3*N+M)
#     cu = np.zeros(3*N+M)
#     # Equilibrium (N eq)
#     cl[0:M] = 0 # Equality constr
#     cu[0:M] = 0
#     # Stress (2*N eq)
#     # Compression
#     cl[M:N+M] = 0
#     cu[M:N+M] = 2.0e19
#     # Tension
#     cl[N+M:2*N+M] = -2.0e19
#     cu[N+M:2*N+M] = 0
#     # Compatibility (N eq)
#     cl[2*N+M:] = 0 # Equality constr
#     cu[2*N+M:] = 0
    
#     nlp = cyipopt.Problem(
#         n=len(x0),
#         m=len(cl),
#         problem_obj=ipopt_routines.Layopt_IPOPT(N, M, myBCs.ground_structure_length, joint_cost, B, myBCs.R * myBCs.dofs, stress_C_max, stress_T_max, E),
#         lb=lb,
#         ub=ub,
#         cl=cl,
#         cu=cu,
#         )

#     nlp.add_option('max_iter', 2000)
#     nlp.add_option('tol', 1e-5)
#     nlp.add_option('mu_strategy', 'adaptive')
#     nlp.add_option('alpha_for_y', 'min-dual-infeas')
#     nlp.add_option('expect_infeasible_problem', 'yes')
#     nlp.add_option('recalc_y', 'yes')
#     nlp.add_option('bound_push', 0.0001)
#     nlp.add_option('grad_f_constant', 'yes')
#     nlp.add_option('print_timing_statistics', 'yes')
#     nlp.add_option('hessian_approximation', 'limited-memory')
#     nlp.add_option('limited_memory_max_history', 25)
#     nlp.add_option('print_info_string', 'yes')
#     nlp.add_option('output_file', folder+'/'+'IPOPT_out.log')
#     nlp.add_option('print_level', 5)
#     nlp.add_option('print_user_options', 'yes')
#     x, info = nlp.solve(x0)
    
#     obj_hist = nlp.__objective.__self__.obj_hist
    
#     a = x[area_id] # Optimized areas
#     q = x[force_id] # Optimized forces
#     U = x[U_id]

#     vol = myBCs.ground_structure_length.T @ a
#     return vol, a, q, U, obj_hist

def solveNLP_2D_IPOPT_Buckling(myBCs: BCS.MBB2D_Symm, myBCs_unred: BCS.MBB2D_Symm, stress_T_max, stress_C_max, E, s_buck, folder, joint_cost = 0, a_init = False, a_fem = False):
    # SAND IPOPT optimization
    B = calcB(myBCs) # equilibrium matrix
    
    N = len(myBCs.ground_structure) # Number of member of the Ground Structure
    M = len(myBCs.dofs) # Number of DOFs of the Groud Structure
    
    N_design_var = 2*N+M # 2* total element + DOFs
    
    # Create the indexing variables used for splitting the design variables vector x
    init = np.zeros(2*N+M,dtype='bool')
    area_id = init.copy()
    area_id[0:N] = True
    force_id = init.copy()
    force_id[N:2*N] = True
    U_id = init.copy()
    U_id[2*N:] = True
    
    ## Design variables bounds
    lb = np.zeros(N_design_var)
    ub = np.zeros(N_design_var)
    # Areas
    lb[area_id] = 0
    ub[area_id] = 2.0e19 # inf
    # Forces (no box constraints)
    lb[force_id] = -2.0e19
    ub[force_id] = 2.0e19
    # Displacements (no box constraints)
    lb[U_id] = -2.0e19
    ub[U_id] = 2.0e19
    
    ## Starting point
    x0 = calculate_starting_point_on_unreduced_BCs(myBCs_unred, myBCs, N_design_var, area_id, force_id, U_id, E, a_init, a_fem)
    
    ## Constraints bounds
    cl = np.zeros(4*N+M)
    cu = np.zeros(4*N+M)
    # Equilibrium (N eq)
    cl[0:M] = 0 # Equality constr
    cu[0:M] = 0
    # Stress (2*N eq)
    # Compression
    cl[M:N+M] = 0
    cu[M:N+M] = 2.0e19
    # Tension
    cl[N+M:2*N+M] = -2.0e19
    cu[N+M:2*N+M] = 0
    # Compatibility (N eq)
    cl[2*N+M:3*N+M] = -1.0e-04 # Equality constr
    cu[2*N+M:3*N+M] = 1.0e-04
    # Buckling (N eq)
    cl[3*N+M:] = 0
    cu[3*N+M:] = 2.0e19
    
    nlp = cyipopt.Problem(
        n=len(x0),
        m=len(cl),
        problem_obj=ipopt_routines.Layopt_IPOPT_Buck(N, M, myBCs.ground_structure_length, joint_cost, B, myBCs.R * myBCs.dofs, stress_C_max, stress_T_max, E, s_buck),
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
        )
    
    nlp.add_option('max_iter', 2000)
    nlp.add_option('tol', 1e-5)
    nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('alpha_for_y', 'min-dual-infeas')
    nlp.add_option('expect_infeasible_problem', 'yes')
    nlp.add_option('recalc_y', 'yes')
    nlp.add_option('bound_push', 0.0001)
    nlp.add_option('grad_f_constant', 'yes')
    nlp.add_option('print_timing_statistics', 'yes')
    nlp.add_option('hessian_approximation', 'limited-memory')
    nlp.add_option('limited_memory_max_history', 25)
    nlp.add_option('print_info_string', 'yes')
    nlp.add_option('output_file', folder+'/'+'IPOPT_out.log')
    nlp.add_option('print_level', 5)
    nlp.add_option('print_user_options', 'yes')

    x, info = nlp.solve(x0)
    
    obj_hist = nlp.__objective.__self__.obj_hist
    
    a = x[area_id] # Optimized areas
    q = x[force_id] # Optimized forces
    U = x[U_id]

    vol = myBCs.ground_structure_length.T @ a
    return vol, a, q, U, obj_hist

# def solveNLP_2D_IPOPT_VL(myBCs: BCS.MBB2D_Symm, stress_T_max, stress_C_max, E, folder, joint_cost = 0, cell_mapping_vector = False, a_cell=False):
#     # SAND IPOPT cellular layout optimization 
#     B = calcB(myBCs) # equilibrium matrix
#     N_cell = myBCs.N_cell # Number of member per different cell 
#     N = len(myBCs.ground_structure) # Number of member of the Ground Structure
#     M = len(myBCs.dofs) # Number of DOFs of the Groud Structure
    
#     NN, n_topologies_cell, cell_mapping_matrix  = mapping_matrix_init(cell_mapping_vector, myBCs)
    
#     N_design_var = NN+N+M # elements per cell + total element + DOFs
    
#     # Create the indexing variables used for splitting the design variables vector x
#     init = np.zeros(N_design_var,dtype='bool')
#     area_id = init.copy()
#     area_id[0:NN] = True
#     force_id = init.copy()
#     force_id[NN:NN+N] = True
#     U_id = init.copy()
#     U_id[NN+N:] = True
    
#     ## Design variables bounds
#     lb = np.zeros(N_design_var)
#     ub = np.zeros(N_design_var)
#     # Areas
#     lb[area_id] = 0
#     ub[area_id] = 2.0e19 # inf
#     # Forces (no box constraints)
#     lb[force_id] = -2.0e19
#     ub[force_id] = 2.0e19
#     # Displacements (no box constraints)
#     lb[U_id] = -2.0e19
#     ub[U_id] = 2.0e19
    
#     ## Starting point
#     x0 = calculate_starting_point_cell(myBCs, N_design_var, area_id, force_id, U_id, N, M, B, E, n_topologies_cell, N_cell, cell_mapping_matrix, a_cell)
    
#     ## Constraints bounds
#     cl = np.zeros(3*N+M)
#     cu = np.zeros(3*N+M)
#     # Equilibrium (N eq)
#     cl[0:M] = 0 # Equality constr
#     cu[0:M] = 0
#     # Stress (2*N eq)
#     # Compression
#     cl[M:N+M] = 0
#     cu[M:N+M] = 2.0e19
#     # Tension
#     cl[N+M:2*N+M] = -2.0e19
#     cu[N+M:2*N+M] = 0
#     # Compatibility (N eq)
#     cl[2*N+M:] = -1.0e-04 # Equality constr
#     cu[2*N+M:] = 1.0e-04
    
#     NLP_prob = ipopt_routines.Layopt_IPOPT_VL(NN, N_cell, N, M, myBCs, joint_cost, B, myBCs.R * myBCs.dofs, stress_C_max, stress_T_max, E, cell_mapping_matrix)
    
#     nlp = cyipopt.Problem(
#         n=len(x0),
#         m=len(cl),
#         problem_obj=NLP_prob,
#         lb=lb,
#         ub=ub,
#         cl=cl,
#         cu=cu,
#         )

#     nlp.add_option('max_iter', 2000)
#     nlp.add_option('tol', 1e-5)
#     nlp.add_option('mu_strategy', 'adaptive')
#     nlp.add_option('alpha_for_y', 'min-dual-infeas')
#     nlp.add_option('expect_infeasible_problem', 'yes')
#     nlp.add_option('recalc_y', 'yes')
#     nlp.add_option('bound_push', 0.0001)
#     nlp.add_option('grad_f_constant', 'yes')
#     nlp.add_option('print_timing_statistics', 'yes')
#     nlp.add_option('hessian_approximation', 'limited-memory')
#     nlp.add_option('limited_memory_max_history', 25)
#     nlp.add_option('print_info_string', 'yes')
#     nlp.add_option('output_file', folder+'/'+'IPOPT_out.log')
#     nlp.add_option('print_level', 5)
#     nlp.add_option('print_user_options', 'yes')
#     x, info = nlp.solve(x0)
    
#     # Calculate physical areas 
#     a, a_cell = NLP_prob.calc_area_phys(x[:NN])
      
#     obj_hist = nlp.__objective.__self__.obj_hist
    
#     q = x[force_id] # Optimized forces
#     U = x[U_id]

#     vol = myBCs.ground_structure_length.T @ a
#     return vol, a, q, U, obj_hist, a_cell

def solveNLP_2D_IPOPT_VL_Buckling(myBCs: BCS.MBB2D_Symm, myBCs_unred: BCS.MBB2D_Symm, stress_T_max, stress_C_max, E, s_buck, folder, joint_cost = 0, cell_mapping_vector = False, a_cell=False):
    # SAND IPOPT cellular layout optimization 
    B = calcB(myBCs) # equilibrium matrix
    N = len(myBCs.ground_structure) # Number of member of the Ground Structure
    N_cell = myBCs.N_cell # Number of member per different cell 
    M = len(myBCs.dofs) # Number of DOFs of the Groud Structure
    
    NN, n_topologies_cell, cell_mapping_matrix  = mapping_matrix_init(cell_mapping_vector, myBCs)
    
    N_design_var = NN+N+M # elements per cell + total element + DOFs
    
    # Create the indexing variables used for splitting the design variables vector x
    init = np.zeros(N_design_var,dtype='bool')
    area_id = init.copy()
    area_id[0:NN] = True
    force_id = init.copy()
    force_id[NN:NN+N] = True
    U_id = init.copy()
    U_id[NN+N:] = True
    
    ## Design variables bounds
    lb = np.zeros(N_design_var)
    ub = np.zeros(N_design_var)
    # Areas
    lb[area_id] = 0
    ub[area_id] = 2.0e19 # inf
    # Forces (no box constraints)
    lb[force_id] = -2.0e19
    ub[force_id] = 2.0e19
    # Displacements (no box constraints)
    lb[U_id] = -2.0e19
    ub[U_id] = 2.0e19
    
    ## Starting point
    x0 = calculate_starting_point_cell_on_unreduced_BCs(myBCs_unred, myBCs, N_design_var, area_id, force_id, U_id, N, M, B, E, n_topologies_cell, N_cell, cell_mapping_matrix, a_cell)
    
    ## Constraints bounds
    cl = np.zeros(4*N+M)
    cu = np.zeros(4*N+M)
    # Equilibrium (N eq)
    cl[0:M] = 0 # Equality constr
    cu[0:M] = 0
    # Stress (2*N eq)
    # Compression
    cl[M:N+M] = 0
    cu[M:N+M] = 2.0e19
    # Tension
    cl[N+M:2*N+M] = -2.0e19
    cu[N+M:2*N+M] = 0
    # Compatibility (N eq)
    cl[2*N+M:3*N+M] = -1.0e-04 # Equality constr
    cu[2*N+M:3*N+M] = 1.0e-04
    # Buckling (N eq)
    cl[3*N+M:] = 0
    cu[3*N+M:] = 2.0e19
    
    NLP_prob = ipopt_routines.Layopt_IPOPT_VL_Buck(NN, N_cell, N, M, myBCs, joint_cost, B, myBCs.R * myBCs.dofs, stress_C_max, stress_T_max, E, s_buck, cell_mapping_matrix)
    
    nlp = cyipopt.Problem(
        n=len(x0),
        m=len(cl),
        problem_obj=NLP_prob,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
        )

    nlp.add_option('max_iter', 2000)
    nlp.add_option('tol', 1e-5)
    nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('alpha_for_y', 'min-dual-infeas')
    nlp.add_option('expect_infeasible_problem', 'yes')
    nlp.add_option('recalc_y', 'yes')
    nlp.add_option('bound_push', 0.0001)
    nlp.add_option('grad_f_constant', 'yes')
    nlp.add_option('print_timing_statistics', 'yes')
    nlp.add_option('hessian_approximation', 'limited-memory')
    nlp.add_option('limited_memory_max_history', 25)
    nlp.add_option('print_info_string', 'yes')
    nlp.add_option('output_file', folder+'/'+'IPOPT_out.log')
    nlp.add_option('print_level', 5)
    nlp.add_option('print_user_options', 'yes')
    x, info = nlp.solve(x0)
    
    # Calculate physical areas 
    a, a_cell = NLP_prob.calc_area_phys(x[:NN])
    
    q = x[force_id] # Optimized forces
    U = x[U_id]  
    obj_hist = nlp.__objective.__self__.obj_hist
    
    vol = myBCs.ground_structure_length.T @ a
    return vol, a, q, U, obj_hist, a_cell

# def solveLP_3D(myBCs: BCS.Wing_3D, stress_T_max, stress_C_max, joint_cost: float = 0, cellsEquals: bool = False, cell_mapping_vector = False):
#     l = myBCs.ground_structure_length + joint_cost * np.max(myBCs.ground_structure_length) # indipendency from the number of cells
#     B = calcB_3D(myBCs) # equilibrium matrix
#     # VL parameter
#     NN, n_topologies_cell, cell_mapping_matrix  = mapping_matrix_init(cell_mapping_vector, myBCs)
     
#     if cellsEquals:        
#         a_wholestruct = np.zeros(myBCs.ncel_str)
#         a_cell = np.zeros(n_topologies_cell, dtype='object_')
#         for i in range(n_topologies_cell): # starting from the lower left corner
#             a_cell[i] = cvx.Variable((int(myBCs.ground_structure.shape[0]/myBCs.ncel_str),1))
#             a_wholestruct = a_wholestruct + cvx.kron(cell_mapping_matrix[:,i].reshape(-1,1), a_cell[i]) # initialize a_phys 
#     else:
#         a_cell = cvx.Variable((myBCs.ground_structure.shape[0],1)) # initialization of the design variables
#         a_wholestruct = a_cell
#     obj = cvx.Minimize(l.T @ a_wholestruct) # definition of the objective function  
#     #
#     # Constraints
#     #
#     q = cvx.Variable((myBCs.ground_structure.shape[0],1))
#     cons = [a_wholestruct>=0] # beam section cannot be negative
#     cons2 = (B @ q == (myBCs.R * myBCs.dofs).reshape(-1,1))
#     cons3 = q <= stress_T_max * a_wholestruct
#     cons4 = q >= stress_C_max * a_wholestruct
#     cons.extend([cons2, cons3, cons4])

#     prob = cvx.Problem(obj, cons) # Creation of LP
#     vol = prob.solve(verbose=True, solver='ECOS') # Problem solving
#     q = np.array(q.value).ravel() # Optimized forces
#     a = np.array(a_wholestruct.value).ravel() # Optimized areas

#     # Eliminate the influence of the joint cost to the objective function
#     vol = myBCs.ground_structure_length.T @ a
#     U = np.zeros(myBCs.dofs.size)  
#     obj_hist = False
    
#     if cellsEquals:
#         a_cell_out = np.zeros(n_topologies_cell, dtype='object_')
#         for i in range(a_cell_out.size):
#             a_cell_out[i] = np.array(a_cell[i].value).ravel()
#         # return vol, a, q, U, obj_hist
#         return vol, a, q, U, obj_hist, a_cell_out
#     else:
#         return vol, a, q, U, obj_hist

def solveLP_3D_SLP_Buckling(myBCs: BCS.Wing_3D, stress_T_max, stress_C_max, s_buck, joint_cost = 0, cellsEquals: bool = False, cell_mapping_vector = False):
    l = myBCs.ground_structure_length + joint_cost * np.max(myBCs.ground_structure_length) # indipendency from the number of cells
    B = calcB_3D(myBCs) # equilibrium matrix
    #
    # Design variable initialization
    #
    # VL parameter
    NN, n_topologies_cell, cell_mapping_matrix  = mapping_matrix_init(cell_mapping_vector, myBCs)
    
    if cellsEquals:        
        a_wholestruct = np.zeros(myBCs.ncel_str)
        a_cell = np.zeros(n_topologies_cell, dtype='object_')
        for i in range(n_topologies_cell): # starting from the lower left corner
            a_cell[i] = cvx.Variable((int(myBCs.ground_structure.shape[0]/myBCs.ncel_str),1))
            a_wholestruct = a_wholestruct + cvx.kron(cell_mapping_matrix[:,i].reshape(-1,1), a_cell[i]) # initialize a_phys 
    else:
        a_cell = cvx.Variable((myBCs.ground_structure.shape[0],1)) * 0.1 # initialization of the design variables
        a_wholestruct = a_cell
    obj = cvx.Minimize(l.T @ a_wholestruct) # definition of the objective function
    #
    # Start of the buckling loop
    #
    tol, i = 1, 0
    tol_old = 1
    vol = 1
    k_buck = 1
    obj_hist = []
    # a  = np.ones(a_wholestruct.size) * np.sum(np.abs(myBCs.R))/-stress_C_max # Initialization
    a  = np.ones(a_wholestruct.size)

    # Buckling specific parameters
    
    buckling_length = myBCs.ground_structure_length.copy()

    debug = False
    reset = False
    chain_control = True
    only_first = True
    best_case = True
    
    best_vol = 1e19
    
    while ((tol>1e-06 and i<300) or reset):
        only_first_activate_constraint = False # buckling constr only on the first bar
        # Buckling calculation based on the previuos iteration section
        a_0 = a.copy()
        a_0_buck = a.copy() # need for another variable. we cant use a_buck for the tolerance calc
        
        if i>400: 
            break
        if (tol<1e-4 and tol_old<1e-4 and reset) or (i == 150):
            reset_val = 0.8**k_buck
            a_0_buck[(a<np.max(a_0_buck)*0.05)] = np.max(a_0_buck)*reset_val
            k_buck *= 2
            if best_case:
                candidate_best_vol = myBCs.ground_structure_length.T @ a
                if candidate_best_vol<=best_vol:
                    best_vol = candidate_best_vol
                    best_a = a.copy()
                    best_q = q.copy()  
                    if cellsEquals:
                        best_a_cell = np.zeros(n_topologies_cell, dtype='object_')
                        for ii in range(n_topologies_cell):
                            best_a_cell[ii] = np.array(a_cell[ii].value).ravel()
            if k_buck>=32: # stopping condition
                reset = False
            print('REINIT')
            
        are_bars_in_compression = False  
        if not cellsEquals:  
            if chain_control and i!=0:
                # Identify if there is some chain buckling
                reduction_pattern = a_0>(1e-3*max(a_0))
                reduced_ground_structure = myBCs.ground_structure[reduction_pattern].copy()
                reduced_q = q[reduction_pattern].copy()
                unique, counts = np.unique(reduced_ground_structure, return_counts=True)
                possible_chain_nodes = np.array(unique[np.where(counts==2)]).ravel()
                chain_nodes = []
                for k in range(possible_chain_nodes.shape[0]): # needed to avoid adding BC nodes
                    candidates_id = (reduced_ground_structure[:,0]==possible_chain_nodes[k]) | (reduced_ground_structure[:,1]==possible_chain_nodes[k])
                    candidates = reduced_ground_structure[candidates_id,:]
                    diff = np.setxor1d(candidates[0,:],candidates[1,:])
                    start, end = np.min(diff), np.max(diff)
                    if np.all(reduced_q[candidates_id]<0):
                        are_bars_in_compression = True
                    # Evaluate cross product
                    n1 = myBCs.nodes[candidates[0,0],:]
                    n2 = myBCs.nodes[candidates[0,1],:]
                    n3 = myBCs.nodes[candidates[1,0],:]
                    n4 = myBCs.nodes[candidates[1,1],:]
                    v1 = n2-n1
                    v2 = n4-n3
                    if np.allclose(np.linalg.norm(np.cross(v2,v1),2),0,rtol=1e-2): # cross prod = 0
                        if (possible_chain_nodes[k]>start) and (possible_chain_nodes[k]<end):#If not some BCs can become candidates
                            if are_bars_in_compression: # We charge only nodes in a compression chain
                                chain_nodes.append(possible_chain_nodes[k])
                    
                if len(chain_nodes)!=0: # augmented length approach
                    buckling_length = myBCs.ground_structure_length.copy()
                    total_leng = 0
                    treated_nodes = []
                    first_bar_chain = []
                    member_to_add_id = []
                    buckling_length_reduced = buckling_length[reduction_pattern]
                    for k, node in enumerate(chain_nodes): # need to use loops and not a vectorized routine
                        candidates_bars_id = (reduced_ground_structure[:,0]==node) | (reduced_ground_structure[:,1]==node)
                        candidates_bars = reduced_ground_structure[candidates_bars_id,:] 
                        # check if the 2 members have a nodes in common with the last treated chain_node, if not leng = 0
                        if k!=0:
                            if node in treated_nodes[-1]:
                                treated_nodes[-1].update(candidates_bars.ravel().tolist())
                                last = candidates_bars_id & ~member_to_add # get only the new bar and update the length based on it
                                member_to_add += candidates_bars_id
                                total_leng += buckling_length_reduced[last]
                                member_to_add_id[-1].update(set(np.where(member_to_add)[0]))
                            else:
                                first_bar_chain.append(np.where(candidates_bars_id)[0][0])
                                treated_nodes.append(set(candidates_bars.ravel().tolist()))
                                total_leng = np.sum(buckling_length_reduced[candidates_bars_id])
                                member_to_add = candidates_bars_id 
                                member_to_add_id.append(set(np.where(member_to_add)[0]))
                        else:
                            first_bar_chain.append(np.where(candidates_bars_id)[0][0])
                            treated_nodes.append(set(candidates_bars.ravel().tolist()))
                            total_leng = np.sum(buckling_length_reduced[candidates_bars_id])
                            member_to_add = candidates_bars_id
                            member_to_add_id.append(set(np.where(member_to_add)[0])) 
                        
                        buckling_length_reduced[member_to_add] = total_leng
                
                    if only_first:
                        only_first_activate_constraint = True
                        first_bar_chain_pattern = np.zeros(buckling_length_reduced.size, dtype='bool')
                        first_bar_chain_pattern[first_bar_chain] = True
                        buckling_length_reduced[~first_bar_chain_pattern] = myBCs.ground_structure_length[reduction_pattern][~first_bar_chain_pattern]
                    
                    buckling_length[reduction_pattern] = buckling_length_reduced
                        
                    print('Chain buckling nodes elim.')
            
            
        #
        # Constraints
        #
        q = cvx.Variable((myBCs.ground_structure.shape[0],1))
        cons = [B @ q == (myBCs.R * myBCs.dofs).reshape(-1,1)]
        cons2 = q <= stress_T_max * a_wholestruct
        cons3 = q >= stress_C_max * a_wholestruct
        cons.extend([cons2, cons3])
        
        if only_first_activate_constraint:
            for ii, leading in enumerate(first_bar_chain):
                for kk, other in enumerate(tuple(member_to_add_id[ii])):
                    if leading != other:
                        side_constr = a_wholestruct[reduction_pattern][leading] <= a_wholestruct[reduction_pattern][other]               
                        cons.extend([side_constr])

        
        try:
            xx=(-s_buck*a_0_buck/(buckling_length)**2).reshape(-1,1)
            b=2*a_wholestruct - a_0_buck.reshape(-1,1)
            cons_buck = q >= cvx.atoms.affine.binary_operators.multiply(xx, b) # Linearized buckling constr
            if i!=-1:
                cons.extend([cons_buck])
        except:
            pass
                        
        #
        # Problem solution
        #
        prob = cvx.Problem(obj, cons) # Creation of LP 
        vol = prob.solve(verbose=False, solver='ECOS', max_iters = 1500, reltol = 1.0e-5, reltol_inacc = 1.0e-2)
        q = np.array(q.value).ravel() # Optimized forces      
        a_1 = np.array(a_wholestruct.value).ravel() # Optimized areas
        a = a_1.copy()
        tol_old=tol
        tol = np.linalg.norm(a-a_0, np.inf)
        # Eliminate the influence of the joint cost to the objective function
        vol = myBCs.ground_structure_length.T @ a
        obj_hist = np.append(obj_hist,vol)
        i = i+1 
        print('\n############### SLP ITERATION {0} ###############\n'.format(i))
        print('vol: {0:.4f} | tol: {1:.6f}\n'.format(vol,tol))
        if debug:
            trussplot.plot3D.plotTruss(myBCs, a, q, max(a) * 1e-3, vol)
            # trussplot.plot3D.plotTrussBucklingCheck(myBCs, a, q, max(a) * 1e-3, s_buck)
            # trussplot.plot3D.plotTrussStress(myBCs, a, q, max(a) * 1e-3)
            # input()  
            
    if best_case:
        candidate_best_vol = myBCs.ground_structure_length.T @ a
        if candidate_best_vol<=best_vol:
            best_vol = candidate_best_vol
            best_a = a.copy()
            best_q = q.copy()
            if cellsEquals:
                best_a_cell = np.zeros(n_topologies_cell, dtype='object_')
                for ii in range(n_topologies_cell):
                    best_a_cell[ii] = np.array(a_cell[ii].value).ravel()
        try:
            a = best_a.copy()
            q = best_q.copy()
            vol = best_vol
        except:
            pass
    print('CONVERGED')
    print('looped {0} times, tol = {1:.4f}'.format(i,tol))
    print("vol SLP: {0:.5f} ".format(vol))
    
    U = np.zeros(myBCs.dofs.size)
    if cellsEquals:
        a_cell_out = best_a_cell
        return vol, a, q, U, obj_hist, a_cell_out
    else:
        return vol, a, q, U, obj_hist

# def solveNLP_3D_IPOPT_Buckling(myBCs: BCS.Wing_3D, stress_T_max, stress_C_max, E, s_buck, folder, joint_cost = 0, a = False):
#     # SAND IPOPT optimization
#     B = calcB_3D(myBCs) # equilibrium matrix
    
#     N = len(myBCs.ground_structure) # Number of member of the Ground Structure
#     M = len(myBCs.dofs) # Number of DOFs of the Ground Structure
    
#     # Create the indexing variables used for splitting the design variables vector x
#     init = np.zeros(2*N+M,dtype='bool')
#     area_id = init.copy()
#     area_id[0:N] = True
#     force_id = init.copy()
#     force_id[N:2*N] = True
#     U_id = init.copy()
#     U_id[2*N:] = True
    
#     ## Design variables bounds
#     lb = np.zeros(2*N+M)
#     ub = np.zeros(2*N+M)
#     # Areas
#     lb[area_id] = 0
#     ub[area_id] = 2.0e19 # inf
#     # Forces (no box constraints)
#     lb[force_id] = -2.0e19
#     ub[force_id] = 2.0e19
#     # Displacements (no box constraints)
#     lb[U_id] = -2.0e19
#     ub[U_id] = 2.0e19
    
#     ## Starting point
#     x0 = np.zeros(2*N+M) # Define an initial guess for the optimization
#     # Areas
#     if np.any(a!=False):
#         a[a<np.max(a)*1e-8] = 1e-8 # K can't be singular
#         x0[area_id] = a
#     else: 
#         x0[area_id] = 1.
#     # Initial forces and displacements are calculated using FEM
#     U = np.zeros(M)
#     K = calcK(myBCs,B,E,x0[area_id])
#     keep = myBCs.free_dofs
#     K = K[keep, :][:, keep]
#     U[keep] = sp.linalg.spsolve(K, myBCs.R_free) # FEM analysis linear sistem
#     # Forces
#     if np.any(np.isnan(U)): # Truss coming from the SLP can be a mechanism
#         U[np.isnan(U)] = 0
#     x0[force_id] = x0[area_id]*E/myBCs.ground_structure_length * (B.T @ U)
#     # Displacements
#     x0[U_id] = U
    
#     ## Constraints bounds
#     cl = np.zeros(4*N+M)
#     cu = np.zeros(4*N+M)
#     # Equilibrium (N eq)
#     cl[0:M] = 0 # Equality constr
#     cu[0:M] = 0
#     # Stress (2*N eq)
#     # Compression
#     cl[M:N+M] = 0
#     cu[M:N+M] = 2.0e19
#     # Tension
#     cl[N+M:2*N+M] = -2.0e19
#     cu[N+M:2*N+M] = 0
#     # Compatibility (N eq)
#     cl[2*N+M:3*N+M] = -1.0e-04 # Equality constr
#     cu[2*N+M:3*N+M] = 1.0e-04
#     # Buckling (N eq)
#     cl[3*N+M:] = 0
#     cu[3*N+M:] = 2.0e19
    
#     nlp = cyipopt.Problem(
#         n=len(x0),
#         m=len(cl),
#         problem_obj=ipopt_routines.Layopt_IPOPT_Buck(N, M, myBCs.ground_structure_length, joint_cost, B, myBCs.R * myBCs.dofs, stress_C_max, stress_T_max, E, s_buck),
#         lb=lb,
#         ub=ub,
#         cl=cl,
#         cu=cu,
#         )
 
#     nlp.add_option('max_iter', 2000)
#     nlp.add_option('tol', 1e-5)
#     nlp.add_option('mu_strategy', 'adaptive')
#     nlp.add_option('alpha_for_y', 'min-dual-infeas')
#     nlp.add_option('expect_infeasible_problem', 'yes')
#     nlp.add_option('recalc_y', 'yes')
#     nlp.add_option('bound_push', 0.0001)
#     nlp.add_option('grad_f_constant', 'yes')
#     nlp.add_option('print_timing_statistics', 'yes')
#     nlp.add_option('hessian_approximation', 'limited-memory')
#     nlp.add_option('limited_memory_max_history', 25)
#     nlp.add_option('print_info_string', 'yes')
#     nlp.add_option('output_file', folder+'/'+'IPOPT_out.log')
#     nlp.add_option('print_level', 5)
#     nlp.add_option('print_user_options', 'yes') 
    
#     x, info = nlp.solve(x0)

#     obj_hist = nlp.__objective.__self__.obj_hist
    
#     a = x[area_id] # Optimized areas
#     q = x[force_id] # Optimized forces
#     U = x[U_id]

#     vol = myBCs.ground_structure_length.T @ a
#     return vol, a, q, U, obj_hist

def solveNLP_3D_IPOPT_VL_Buckling(myBCs: BCS.Wing_3D, myBCs_unred: BCS.Wing_3D, stress_T_max, stress_C_max, E, s_buck, folder, joint_cost = 0, a_cell=False, cell_mapping_vector = False):
    # SAND IPOPT cellular layout optimization 
    B = calcB_3D(myBCs) # equilibrium matrix
    N = len(myBCs.ground_structure) # Number of member of the Ground Structure
    N_cell = myBCs.N_cell # Number of member per different cell 
    M = len(myBCs.dofs) # Number of DOFs of the Groud Structure
    
    NN, n_topologies_cell, cell_mapping_matrix  = mapping_matrix_init(cell_mapping_vector, myBCs)
    
    N_design_var = NN+N+M # elements per cell + total element + DOFs
    
    # Create the indexing variables used for splitting the design variables vector x
    init = np.zeros(N_design_var, dtype='bool')
    area_id = init.copy()
    area_id[0:NN] = True
    force_id = init.copy()
    force_id[NN:NN+N] = True
    U_id = init.copy()
    U_id[NN+N:] = True
    
    ## Design variables bounds
    lb = np.zeros(N_design_var)
    ub = np.zeros(N_design_var)
    # Areas
    lb[area_id] = 0
    ub[area_id] = 2.0e19 # inf
    # Forces (no box constraints)
    lb[force_id] = -2.0e19
    ub[force_id] = 2.0e19
    # Displacements (no box constraints)
    lb[U_id] = -2.0e19
    ub[U_id] = 2.0e19
    
    ## Starting point
    x0 = calculate_starting_point_cell_on_unreduced_BCs(myBCs_unred, myBCs, N_design_var, area_id, force_id, U_id, N, M, B, E, n_topologies_cell, N_cell, cell_mapping_matrix, a_cell)
    
    ## Constraints bounds
    cl = np.zeros(4*N+M)
    cu = np.zeros(4*N+M)
    # Equilibrium (N eq)
    cl[0:M] = 0 # Equality constr
    cu[0:M] = 0
    # Stress (2*N eq)
    # Compression
    cl[M:N+M] = 0
    cu[M:N+M] = 2.0e19
    # Tension
    cl[N+M:2*N+M] = -2.0e19
    cu[N+M:2*N+M] = 0
    # Compatibility (N eq)
    cl[2*N+M:3*N+M] = -1.0e-04 # Equality constr
    cu[2*N+M:3*N+M] = 1.0e-04
    # Buckling (N eq)
    cl[3*N+M:] = 0
    cu[3*N+M:] = 2.0e19
    
    NLP_prob = ipopt_routines.Layopt_IPOPT_VL_Buck(NN, N_cell, N, M, myBCs, joint_cost, B, myBCs.R * myBCs.dofs, stress_C_max, stress_T_max, E, s_buck, cell_mapping_matrix)
    
    nlp = cyipopt.Problem(
        n=len(x0),
        m=len(cl),
        problem_obj=NLP_prob,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
        )

    nlp.add_option('max_iter', 2000)
    nlp.add_option('tol', 1e-5)
    nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('alpha_for_y', 'min-dual-infeas')
    nlp.add_option('expect_infeasible_problem', 'yes')
    nlp.add_option('recalc_y', 'yes')
    nlp.add_option('bound_push', 0.0001)
    nlp.add_option('grad_f_constant', 'yes')
    nlp.add_option('print_timing_statistics', 'yes')
    nlp.add_option('hessian_approximation', 'limited-memory')
    nlp.add_option('limited_memory_max_history', 25)
    nlp.add_option('print_info_string', 'yes')
    nlp.add_option('output_file', folder+'/'+'IPOPT_out.log')
    nlp.add_option('print_level', 5)
    nlp.add_option('print_user_options', 'yes') 
    
    x, info = nlp.solve(x0)
    
    # Calculate physical areas 
    a, a_cell = NLP_prob.calc_area_phys(x[:NN])
    
    q = x[force_id] # Optimized forces
    U = x[U_id]  
    obj_hist = nlp.__objective.__self__.obj_hist
    
    vol = myBCs.ground_structure_length.T @ a
    return vol, a, q, U, obj_hist, a_cell

# ##################################################################################################

# def solveLP_3D_Free_Form(myBCs: BCS.Free_form, stress_T_max, stress_C_max, joint_cost: float = 0): #Eliminate completely the multiple cells
#     l = myBCs.ground_structure_length + joint_cost * np.max(myBCs.ground_structure_length) # indipendency from the number of cells
#     B = calcB_3D(myBCs) # equilibrium matrix
#     # VL parameter
#     r = np.array(np.where(myBCs.ground_structure[:,2]!=-1)).ravel()
#     repetitive = myBCs.ground_structure[r,:]
#     cell_mapping_vector = repetitive[::myBCs.N_cell,2].copy()
    
#     #Cellular        
#     a_cell = cvx.Variable((myBCs.N_cell,1), nonneg=True) # beam section cannot be negative
#     a_periodic = cvx.kron(cell_mapping_vector.reshape(-1,1), a_cell) # initialize a_periodic 
    
#     # Aperiodic
#     a_aperiodic = cvx.Variable((myBCs.ground_structure_aperiodic.shape[0],1), nonneg=True) # initialization of the design variables
    
    
#     a_wholestruct = cvx.vstack([a_periodic, a_aperiodic])
    
#     obj = cvx.Minimize(l.T @ a_wholestruct) # definition of the objective function
#     #
#     # Constraints
#     #
#     q = cvx.Variable((myBCs.ground_structure.shape[0],1))
#     cons = [B @ q == (myBCs.R * myBCs.dofs).reshape(-1,1)]
#     cons2 = q <= stress_T_max * a_wholestruct
#     cons3 = q >= stress_C_max * a_wholestruct
#     cons.extend([cons2, cons3])

#     prob = cvx.Problem(obj, cons) # Creation of LP
#     vol = prob.solve(verbose=True, solver='ECOS') # Problem solving
#     q = np.array(q.value).ravel() # Optimized forces
#     a = np.array(a_wholestruct.value).ravel() # Optimized areas

#     # Eliminate the influence of the joint cost to the objective function
#     vol = myBCs.ground_structure_length.T @ a
#     U = np.zeros(myBCs.dofs.size)  
#     obj_hist = False
    
#     n_topologies_cell = np.max(myBCs.ground_structure[:,2])
#     a_cell_out = np.zeros(n_topologies_cell, dtype='object_') # problems when multiple topologies
#     for i in range(a_cell_out.size):
#         a_cell_out[i] = np.array(a_cell.value).ravel()
#     return vol, a, q, U, obj_hist, a_cell_out

# def solveLP_3D_SLP_Buckling_Free_Form(myBCs: BCS.Free_form, stress_T_max, stress_C_max, s_buck, joint_cost = 0):
#     l = myBCs.ground_structure_length + joint_cost * np.max(myBCs.ground_structure_length) # indipendency from the number of cells
#     B = calcB_3D(myBCs) # equilibrium matrix
#     # VL parameter
#     r = np.array(np.where(myBCs.ground_structure[:,2]==1)).ravel()
#     repetitive = myBCs.ground_structure[r,:]
#     cell_mapping_vector = repetitive[::myBCs.N_cell,2].copy()
    
#     #Cellular        
#     a_cell = cvx.Variable((myBCs.N_cell,1), nonneg=True) # beam section cannot be negative
#     a_periodic = cvx.kron(cell_mapping_vector.reshape(-1,1), a_cell) # initialize a_periodic 
    
#     # Aperiodic
#     a_aperiodic = cvx.Variable((myBCs.ground_structure_aperiodic.shape[0],1), nonneg=True) # initialization of the design variables
    
#     a_wholestruct = cvx.vstack([a_periodic, a_aperiodic])
    
#     obj = cvx.Minimize(l.T @ a_wholestruct) # definition of the objective function
    
#     #
#     # Start of the buckling loop
#     #
#     tol, i = 1, 0
#     tol_old = 1
#     vol = 1
#     k_buck = 1
#     obj_hist = []
#     a  = np.ones(a_wholestruct.size) * np.sum(np.abs(myBCs.R))/-stress_C_max # Initialization
#     # Buckling specific parameters
    
#     buckling_length = myBCs.ground_structure_length.copy()

#     debug = False
#     reset = True
#     best_case = True
    
#     best_vol = 1e19
    
#     while ((tol>5e-03 and i<300) or reset):
#         # Buckling calculation based on the previuos iteration section
#         a_0 = a.copy()
#         a_0_buck = a.copy() # need for another variable. we cant use a_buck for the tolerance calc
        
#         if i>400: 
#             break
#         if (tol<1e-2 and tol_old<1e-4 and reset) or (i == 150):
#             reset_val = 0.8**k_buck
#             a_0_buck[(a<np.max(a_0_buck)*0.05)] = np.max(a_0_buck)*reset_val
#             k_buck *= 2
#             if best_case:
#                 candidate_best_vol = myBCs.ground_structure_length.T @ a
#                 if candidate_best_vol<=best_vol:
#                     best_vol = candidate_best_vol
#                     best_a = a.copy()
#                     best_q = q.copy()  
#             if k_buck>=8: # stopping condition
#                 reset = False
#             print('REINIT')
#         #
#         # Constraints
#         #
#         q = cvx.Variable((myBCs.ground_structure.shape[0],1))
#         cons = [B @ q == (myBCs.R * myBCs.dofs).reshape(-1,1)]
#         cons2 = q <= stress_T_max * a_wholestruct
#         cons3 = q >= stress_C_max * a_wholestruct
#         cons.extend([cons2, cons3])

        
#         try:
#             xx=(-s_buck*a_0_buck/(buckling_length)**2).reshape(-1,1)
#             b=2*a_wholestruct - a_0_buck.reshape(-1,1)
#             cons_buck = q >= cvx.atoms.affine.binary_operators.multiply(xx, b) # Linearized buckling constr
#             if i!=-1:
#                 cons.extend([cons_buck])
#         except:
#             pass
                        
#         #
#         # Problem solution
#         #
#         prob = cvx.Problem(obj, cons) # Creation of LP 
#         vol = prob.solve(verbose=False, solver='ECOS', max_iters = 2000, reltol = 5.0e-5, reltol_inacc = 1.0e-2)
#         q = np.array(q.value).ravel() # Optimized forces      
#         a_1 = np.array(a_wholestruct.value).ravel() # Optimized areas
#         a = a_1.copy()
#         tol = np.linalg.norm(a-a_0, np.inf)
#         # Eliminate the influence of the joint cost to the objective function
#         vol = myBCs.ground_structure_length.T @ a
#         obj_hist = np.append(obj_hist,vol)
#         i = i+1 
#         print('\n############### SLP ITERATION {0} ###############\n'.format(i))
#         print('vol: {0:.4f} | tol: {1:.6f}\n'.format(vol,tol))
#         if debug:
#             trussplot.plot3D.plotTruss(myBCs, a, q, max(a) * 1e-3, vol)
#             # trussplot.plot3D.plotTrussBucklingCheck(myBCs, a, q, max(a) * 1e-3, s_buck)
#             # trussplot.plot3D.plotTrussStress(myBCs, a, q, max(a) * 1e-3)
#             # input()  
            
#     if best_case:
#         candidate_best_vol = myBCs.ground_structure_length.T @ a
#         if candidate_best_vol<=best_vol:
#             best_vol = candidate_best_vol
#             best_a = a.copy()
#             best_q = q.copy()
#         try:
#             a = best_a
#             q = best_q
#             vol = best_vol
#         except:
#             pass
#     print('CONVERGED')
#     print('looped {0} times, tol = {1:.4f}'.format(i,tol))
#     print("vol SLP: {0:.5f} ".format(vol))
   
#     U = np.zeros(myBCs.dofs.size)
#     n_topologies_cell = np.max(myBCs.ground_structure[:,2])
    
#     a_cell_out = np.zeros(n_topologies_cell, dtype='object_') # problems when multiple topologies
#     for i in range(a_cell_out.size):
#         a_cell_out[i] = np.array(a_cell.value).ravel()
#     return vol, a, q, U, obj_hist, a_cell_out

# def solveNLP_3D_IPOPT_VL_Buckling_Free_Form(myBCs: BCS.Free_form, stress_T_max, stress_C_max, E, s_buck, folder, joint_cost = 0, reduceProblem = True, a_init=False, a_cell=False, tol = 1e-3, autoScale = True):
#     if reduceProblem:
#         #Problem reduction
#         import copy
#         myBCs_old = copy.deepcopy(myBCs)
#         myBCs, a_reduced = reduce_BCs(myBCs, tol, a_init, a_cell)
    
#     # SAND IPOPT cellular layout optimization 
#     B = calcB_3D(myBCs) # equilibrium matrix
#     N = myBCs.N # Number of member of the Ground Structure
#     N_cell = myBCs.N_cell # Number of member per different cell 
#     M = myBCs.M # Number of DOFs of the Groud Structure
    
#     NN = myBCs.N_cell+myBCs.N_aperiodic # nunmber of members of a single periodic unit plus aperiodic members
    
#     N_design_var = NN+N+M # area design variables + total element + DOFs
    
#     # Create the indexing variables used for splitting the design variables vector x
#     init = np.zeros(N_design_var, dtype='bool')
#     area_id = init.copy()
#     area_id[0:NN] = True
#     force_id = init.copy()
#     force_id[NN:NN+N] = True
#     U_id = init.copy()
#     U_id[NN+N:] = True
    
#     ## Design variables bounds
#     lb = np.zeros(N_design_var)
#     ub = np.zeros(N_design_var)
#     # Areas
#     lb[area_id] = 0
#     ub[area_id] = 2.0e2
#     # Forces (box constraints)
#     lb[force_id] = -1.0e4
#     ub[force_id] = 1.0e4
#     # Displacements (box constraints)
#     lb[U_id] = -5.0e2
#     ub[U_id] = 1.0e2
    
#     ## Starting point
#     if reduceProblem:
#         x0 = calculate_starting_point_free_form_on_unreduced_BCs(myBCs_old, myBCs, N_design_var, area_id, force_id, U_id, E, a_init, a_reduced)
#     else:
#         x0 = calculate_starting_point_free_form(myBCs, N_design_var, area_id, force_id, U_id, N, M, B, E, a_reduced)
        
#     ## Constraints bounds
#     cl = np.zeros(4*N+M)
#     cu = np.zeros(4*N+M)
#     # Equilibrium (M eq)
#     cl[0:M] = -1.0e-06 # Equality constr
#     cu[0:M] = 1.0e-06
#     # Stress (2*N eq)
#     # Compression
#     cl[M:N+M] = 0
#     cu[M:N+M] = 2.0e19
#     # Tension
#     cl[N+M:2*N+M] = -2.0e19
#     cu[N+M:2*N+M] = 0
#     # Compatibility (N eq)
#     cl[2*N+M:3*N+M] = -1.0e-02 # Equality constr
#     cu[2*N+M:3*N+M] = 1.0e-02
#     # Buckling (N eq)
#     cl[3*N+M:] = 0
#     cu[3*N+M:] = 2.0e19
    
    
#     if autoScale:
#         ### Objective = 10
#         obj_scale = 1/(np.sum(x0[area_id])) * 1000
        
#         ### Design Variables
#         var_scale = np.zeros(len(x0))
#         # Area
#         x_scale_area = 1/(ub[0]) * 1000
#         var_scale[area_id] = x_scale_area
#         # Force
#         x_scale_force = 1/(ub[NN]-lb[NN]) * 1000
#         var_scale[force_id] = x_scale_force
#         # Displacements
#         x_scale_U = 1/(ub[NN+N]-lb[NN+N]) * 1000
#         var_scale[U_id] = x_scale_U
        
#         ### Constraints
#         constr_scale = np.zeros(4*N+M)
#         # Equilibrium (M eq)
#         eq_scale = x_scale_force # Equality constr
#         constr_scale[0:M] = eq_scale
#         # Stress (2*N eq)
#         # Compression
#         s_c_scale = x_scale_force
#         constr_scale[M:N+M] = s_c_scale
#         # Tension
#         s_t_scale = x_scale_force
#         constr_scale[N+M:2*N+M] = s_t_scale
#         # Compatibility (N eq)
#         comp_scale = x_scale_area * x_scale_force * x_scale_U
#         constr_scale[2*N+M:3*N+M] = comp_scale
#         # Buckling (N eq)
#         buck_scale = x_scale_force * x_scale_area**2
#         constr_scale[3*N+M:] = buck_scale
        
#         x0_scaled = x0*var_scale
        
#         NLP_prob = ipopt_routines.Layopt_IPOPT_VL_Buck_Free_Form(NN, N_cell, N, M, myBCs, joint_cost, B, myBCs.R * myBCs.dofs, stress_C_max, stress_T_max, E, s_buck)
        
#         nlp = cyipopt.Problem(
#             n=len(x0),
#             m=len(cl),
#             problem_obj=NLP_prob,
#             lb=lb,
#             ub=ub,
#             cl=cl,
#             cu=cu,
#             )
        
#         nlp.set_problem_scaling(obj_scaling = obj_scale,
#                                 x_scaling = var_scale,
#                                 g_scaling = constr_scale)
#         nlp.add_option('nlp_scaling_method', 'user-scaling')
        
#     else:       
#         NLP_prob = ipopt_routines.Layopt_IPOPT_VL_Buck_Free_Form(NN, N_cell, N, M, myBCs, joint_cost, B, myBCs.R * myBCs.dofs, stress_C_max, stress_T_max, E, s_buck)
        
#         nlp = cyipopt.Problem(
#             n=len(x0),
#             m=len(cl),
#             problem_obj=NLP_prob,
#             lb=lb,
#             ub=ub,
#             cl=cl,
#             cu=cu,
#             )
#         nlp.add_option('nlp_scaling_constr_target_gradient', 100.0)
        
        
     
#     nlp.add_option('max_iter', 2000)
#     nlp.add_option('tol', 5e-4)
#     nlp.add_option('acceptable_tol', 1e-2)
#     nlp.add_option('mu_strategy', 'adaptive')
#     nlp.add_option('alpha_for_y', 'min-dual-infeas')
#     nlp.add_option('linear_solver', 'pardiso')
#     nlp.add_option('pardisolib', 'D:\estragio\PhD\98_Portable-Software\PardisoSolver\libpardiso600-WIN-X86-64.dll')
#     nlp.add_option('expect_infeasible_problem', 'yes')
#     nlp.add_option('recalc_y', 'yes')
#     nlp.add_option('bound_push', 0.0001)
#     #nlp.add_option('nlp_scaling_method', 'none')
#     #nlp.add_option('start_with_resto', 'yes')
#     #nlp.add_option('bound_relax_factor', 1e-5)
#     nlp.add_option('grad_f_constant', 'yes')
#     nlp.add_option('derivative_test', 'none')
#     nlp.add_option('print_timing_statistics', 'yes')
#     nlp.add_option('hessian_approximation', 'limited-memory')
#     nlp.add_option('limited_memory_max_history', 25)
#     #nlp.add_option('hessian_constant', 'yes')
#     nlp.add_option('print_info_string', 'yes')
#     nlp.add_option('output_file', folder+'/'+'IPOPT_out.log')
#     nlp.add_option('print_level', 5)
#     nlp.add_option('print_user_options', 'yes')   
        
#     x, info = nlp.solve(x0)
    
#     # Calculate physical areas 
#     a = NLP_prob.calc_area_phys(x[:NN])
#     a_cell = np.empty(1, dtype='object_')
#     a_cell[0] = x[:myBCs.N_cell]
    
#     q = x[force_id] # Optimized forces
#     U = x[U_id]  
    
#     obj_hist = nlp.__objective.__self__.obj_hist
#     eq_hist = nlp.__objective.__self__.constr_equilib_hist 
#     s_c_hist = nlp.__objective.__self__.constr_s_c_hist
#     s_t_hist = nlp.__objective.__self__.constr_s_t_hist
#     comp_hist = nlp.__objective.__self__.constr_comp_hist
#     buck_hist = nlp.__objective.__self__.constr_buck_hist
    
    
#     vol = myBCs.ground_structure_length.T @ a
#     return vol, a, q, U, a_cell, obj_hist, eq_hist, s_c_hist, s_t_hist, comp_hist, buck_hist    

# ############################ SECTIONS

# def solveLP_3D_SLP_Buckling_Free_Form_sections(myBCs: BCS.Free_form, stress_T_max, stress_C_max, s_buck, joint_cost = 0):
#     l = myBCs.ground_structure_length + joint_cost * np.max(myBCs.ground_structure_length) # indipendency from the number of cells
#     B = calcB_3D(myBCs) # equilibrium matrix
    
#     # VL parameter
#     cell_mapping_vector = np.ones(myBCs.number_cell, dtype='int')
    
#     divisions = np.max(myBCs.ground_structure_periodic_sections[:,3]) + 1
#     section_mapping_vector = np.ones(divisions, dtype='int')
    
#     # Cellular        
#     a_cell = cvx.Variable((myBCs.N_cell,1), nonneg=True) # beam section cannot be negative
#     a_cell_periodic = cvx.kron(cell_mapping_vector.reshape(-1,1), a_cell) # initialize a_periodic 
    
#     # Sections
#     a_section = cvx.Variable((myBCs.N_section,1), nonneg=True) # beam section cannot be negative
#     a_section_periodic = cvx.kron(section_mapping_vector.reshape(-1,1), a_section) # initialize a_periodic
    
#     a_wholestruct = cvx.vstack([a_cell_periodic, a_section_periodic])
    
#     obj = cvx.Minimize(l.T @ a_wholestruct) # definition of the objective function
    
#     #
#     # Start of the buckling loop
#     #
#     tol, i = 1, 0
#     tol_old = 1
#     vol = 1
#     k_buck = 1
#     obj_hist = []
#     best_vol = 1e19
#     a  = np.ones(a_wholestruct.size) * np.sum(np.abs(myBCs.R))/-stress_C_max # Initialization
#     # Buckling specific parameters
    
#     buckling_length = myBCs.ground_structure_length.copy()

#     debug = False
#     reset = True
#     best_case = True
    
#     while ((tol>5e-03 and i<250) or reset):
#     # while (i<2):
#         # Buckling calculation based on the previuos iteration section
#         a_0 = a.copy()
#         a_0_buck = a.copy() # need for another variable. we cant use a_buck for the tolerance calc
        
#         if i>400: 
#             break
#         if (tol<1e-2 and tol_old<1e-2 and reset) or (i == 150):
#             reset_val = 0.8**k_buck
#             a_0_buck[(a<np.max(a_0_buck)*0.05)] = np.max(a_0_buck)*reset_val
#             k_buck *= 2
#             if best_case:
#                 candidate_best_vol = myBCs.ground_structure_length.T @ a
#                 if candidate_best_vol<=best_vol:
#                     best_vol = candidate_best_vol
#                     best_a = np.array(a_wholestruct.value).ravel()
#                     best_a_cell = np.array(a_cell.value).ravel()
#                     best_a_section = np.array(a_section.value).ravel()
#                     best_q = np.array(q).ravel()
#             if k_buck>=4: # stopping condition
#                 reset = False
#             print('REINIT')
#         #
#         # Constraints
#         #
#         q = cvx.Variable((myBCs.ground_structure.shape[0],1))
#         cons = [B @ q == (myBCs.R * myBCs.dofs).reshape(-1,1)]
#         cons2 = q <= stress_T_max * a_wholestruct
#         cons3 = q >= stress_C_max * a_wholestruct
#         cons.extend([cons2, cons3])

        
#         try:
#             xx=(-s_buck*a_0_buck/(buckling_length)**2).reshape(-1,1)
#             b=2*a_wholestruct - a_0_buck.reshape(-1,1)
#             cons_buck = q >= cvx.atoms.affine.binary_operators.multiply(xx, b) # Linearized buckling constr
#             if i!=-1:
#                 cons.extend([cons_buck])
#         except:
#             pass
                        
#         #
#         # Problem solution
#         #
#         prob = cvx.Problem(obj, cons) # Creation of LP 
#         vol = prob.solve(verbose=False, solver='ECOS', max_iters = 2000, reltol = 5.0e-5, reltol_inacc = 1.0e-2)
#         q = np.array(q.value).ravel() # Optimized forces      
#         a = np.array(a_wholestruct.value).ravel() # Optimized areas
#         tol_old = tol
#         tol = np.linalg.norm(a-a_0, np.inf)
#         # Eliminate the influence of the joint cost to the objective function
#         vol = myBCs.ground_structure_length.T @ a
#         obj_hist = np.append(obj_hist,vol)
#         i = i+1
#         print('\n############### SLP ITERATION {0} ###############\n'.format(i))
#         print('vol: {0:.4f} | tol: {1:.6f}\n'.format(vol,tol))
#         if debug:
#             trussplot.plot3D.plotTruss(myBCs, a, q, max(a) * 1e-3, vol)
#             # trussplot.plot3D.plotTrussBucklingCheck(myBCs, a, q, max(a) * 1e-3, s_buck)
#             # trussplot.plot3D.plotTrussStress(myBCs, a, q, max(a) * 1e-3)
#             # input()  
            
#     if best_case:
#         candidate_best_vol = myBCs.ground_structure_length.T @ a
#         if candidate_best_vol<=best_vol:
#             best_vol = candidate_best_vol
#             best_a = np.array(a).ravel()
#             best_a_cell = np.array(a_cell.value).ravel()
#             best_a_section = np.array(a_section.value).ravel()
#             best_q = np.array(q).ravel()
#         try:
#             a = best_a
#             a_cell = best_a_cell
#             a_section = best_a_section
#             q = best_q
#             vol = best_vol
#         except:
#             pass
    
#     print('CONVERGED')
#     print('looped {0} times, tol = {1:.4f}'.format(i,tol))
#     print("vol SLP: {0:.5f} ".format(vol))
   
#     U = np.zeros(myBCs.dofs.size)
#     n_topologies_cell = 2
    
#     p_cell = np.array(np.where((myBCs.ground_structure[:,2]==1) & (myBCs.ground_structure[:,3]==0))).ravel()
#     p_sec = np.array(np.where((myBCs.ground_structure[:,2]==2) & (myBCs.ground_structure[:,3]==0))).ravel() 
    
#     a_cell_out = np.zeros(n_topologies_cell, dtype='object_') # problems when multiple topologies
#     a_cell_out[0] = a[p_cell]
#     a_cell_out[1] = a[p_sec]
#     return vol, a, q, U, obj_hist, a_cell_out

# def solveNLP_3D_IPOPT_VL_Buckling_Free_Form_sections(myBCs: BCS.Free_form, stress_T_max, stress_C_max, E, s_buck, folder, joint_cost = 0, reduceProblem = True, a_init=False, a_cell=False, tol = 1e-3, autoScale = True):
#     if reduceProblem:
#         #Problem reduction
#         import copy
#         myBCs_old = copy.deepcopy(myBCs)
#         myBCs, a_reduced = reduce_BCs_section(myBCs, tol, a_init, a_cell)
    
#     # SAND IPOPT cellular layout optimization 
#     B = calcB_3D(myBCs) # equilibrium matrix
#     N = myBCs.N # Number of member of the Ground Structure
#     N_cell = myBCs.N_cell # Number of member per different cell 
#     N_section = myBCs.N_periodic_section # Number of member per different section 
#     M = myBCs.M # Number of DOFs of the Groud Structure
    
#     NN = N_cell+N_section # nunmber of members of a single periodic unit plus aperiodic members
    
#     N_design_var = NN+N+M # area design variables + total element + DOFs
    
#     # Create the indexing variables used for splitting the design variables vector x
#     init = np.zeros(N_design_var, dtype='bool')
#     area_id = init.copy()
#     area_id[0:NN] = True
#     force_id = init.copy()
#     force_id[NN:NN+N] = True
#     U_id = init.copy()
#     U_id[NN+N:] = True
    
#     ## Design variables bounds
#     lb = np.zeros(N_design_var)
#     ub = np.zeros(N_design_var)
#     # Areas
#     lb[area_id] = 0
#     ub[area_id] = 2.0e2
#     # Forces (box constraints)
#     lb[force_id] = -1.0e4
#     ub[force_id] = 1.0e4
#     # Displacements (box constraints)
#     lb[U_id] = -5.0e2
#     ub[U_id] = 1.0e2
    
#     ## Starting point
#     if reduceProblem:
#         x0 = calculate_starting_point_free_form_on_unreduced_BCs_section(myBCs_old, myBCs, N_design_var, area_id, force_id, U_id, E, a_init, a_reduced)
#     else:
#         x0 = calculate_starting_point_free_form(myBCs, N_design_var, area_id, force_id, U_id, N, M, B, E, a_reduced)
        
#     ## Constraints bounds
#     cl = np.zeros(4*N+M)
#     cu = np.zeros(4*N+M)
#     # Equilibrium (M eq)
#     cl[0:M] = -1.0e-06 # Equality constr
#     cu[0:M] = 1.0e-06
#     # Stress (2*N eq)
#     # Compression
#     cl[M:N+M] = 0
#     cu[M:N+M] = 2.0e19
#     # Tension
#     cl[N+M:2*N+M] = -2.0e19
#     cu[N+M:2*N+M] = 0
#     # Compatibility (N eq)
#     cl[2*N+M:3*N+M] = -1.0e-02 # Equality constr
#     cu[2*N+M:3*N+M] = 1.0e-02
#     # Buckling (N eq)
#     cl[3*N+M:] = 0
#     cu[3*N+M:] = 2.0e19
    
    
#     if autoScale:
#         ### Objective = 10
#         obj_scale = 1/(np.sum(x0[area_id])) * 1000
        
#         ### Design Variables
#         var_scale = np.zeros(len(x0))
#         # Area
#         x_scale_area = 1/(ub[0]) * 1000
#         var_scale[area_id] = x_scale_area
#         # Force
#         x_scale_force = 1/(ub[NN]-lb[NN]) * 1000
#         var_scale[force_id] = x_scale_force
#         # Displacements
#         x_scale_U = 1/(ub[NN+N]-lb[NN+N]) * 1000
#         var_scale[U_id] = x_scale_U
        
#         ### Constraints
#         constr_scale = np.zeros(4*N+M)
#         # Equilibrium (M eq)
#         eq_scale = x_scale_force # Equality constr
#         constr_scale[0:M] = eq_scale
#         # Stress (2*N eq)
#         # Compression
#         s_c_scale = x_scale_force
#         constr_scale[M:N+M] = s_c_scale
#         # Tension
#         s_t_scale = x_scale_force
#         constr_scale[N+M:2*N+M] = s_t_scale
#         # Compatibility (N eq)
#         comp_scale = x_scale_area * x_scale_force * x_scale_U
#         constr_scale[2*N+M:3*N+M] = comp_scale
#         # Buckling (N eq)
#         buck_scale = x_scale_force * x_scale_area**2
#         constr_scale[3*N+M:] = buck_scale
        
#         x0_scaled = x0*var_scale
        
#         NLP_prob = ipopt_routines.Layopt_IPOPT_VL_Buck_Free_Form_Section(NN, N_cell, N_section, N, M, myBCs, joint_cost, B, myBCs.R * myBCs.dofs, stress_C_max, stress_T_max, E, s_buck)
        
#         nlp = cyipopt.Problem(
#             n=len(x0),
#             m=len(cl),
#             problem_obj=NLP_prob,
#             lb=lb,
#             ub=ub,
#             cl=cl,
#             cu=cu,
#             )
        
#         nlp.set_problem_scaling(obj_scaling = obj_scale,
#                                 x_scaling = var_scale,
#                                 g_scaling = constr_scale)
#         nlp.add_option('nlp_scaling_method', 'user-scaling')
        
#     else:       
#         NLP_prob = ipopt_routines.Layopt_IPOPT_VL_Buck_Free_Form_Section(NN, N_cell, N_section, N, M, myBCs, joint_cost, B, myBCs.R * myBCs.dofs, stress_C_max, stress_T_max, E)
        
#         nlp = cyipopt.Problem(
#             n=len(x0),
#             m=len(cl),
#             problem_obj=NLP_prob,
#             lb=lb,
#             ub=ub,
#             cl=cl,
#             cu=cu,
#             )
#         nlp.add_option('nlp_scaling_constr_target_gradient', 100.0)
        
#     nlp.add_option('max_iter', 2000)
#     nlp.add_option('tol', 5e-4)
#     nlp.add_option('acceptable_tol', 1e-2)
#     nlp.add_option('mu_strategy', 'adaptive')
#     nlp.add_option('alpha_for_y', 'min-dual-infeas')
#     nlp.add_option('linear_solver', 'pardiso')
#     nlp.add_option('pardisolib', 'D:\estragio\PhD\98_Portable-Software\PardisoSolver\libpardiso600-WIN-X86-64.dll')
#     nlp.add_option('expect_infeasible_problem', 'yes')
#     nlp.add_option('recalc_y', 'yes')
#     nlp.add_option('bound_push', 0.0001)
#     #nlp.add_option('nlp_scaling_method', 'none')
#     #nlp.add_option('start_with_resto', 'yes')
#     #nlp.add_option('bound_relax_factor', 1e-5)
#     nlp.add_option('grad_f_constant', 'yes')
#     nlp.add_option('derivative_test', 'none')
#     nlp.add_option('print_timing_statistics', 'yes')
#     nlp.add_option('hessian_approximation', 'limited-memory')
#     nlp.add_option('limited_memory_max_history', 25)
#     #nlp.add_option('hessian_constant', 'yes')
#     nlp.add_option('print_info_string', 'yes')
#     nlp.add_option('output_file', folder+'/'+'IPOPT_out.log')
#     nlp.add_option('print_level', 5)
#     nlp.add_option('print_user_options', 'yes')   
        
#     x, info = nlp.solve(x0)
    
#     # Calculate physical areas 
#     a = NLP_prob.calc_area_phys_sect(x[:NN])
#     a_cell = np.empty(1, dtype='object_')
#     a_cell[0] = x[:myBCs.N_cell]
    
#     q = x[force_id] # Optimized forces
#     U = x[U_id]  
    
#     obj_hist = nlp.__objective.__self__.obj_hist
#     eq_hist = nlp.__objective.__self__.constr_equilib_hist 
#     s_c_hist = nlp.__objective.__self__.constr_s_c_hist
#     s_t_hist = nlp.__objective.__self__.constr_s_t_hist
#     comp_hist = nlp.__objective.__self__.constr_comp_hist
#     buck_hist = nlp.__objective.__self__.constr_buck_hist
    
    
#     vol = myBCs.ground_structure_length.T @ a
#     if reduceProblem:
#         reduction_pattern = a_init>np.max(a_init)*tol
#         a_unreduced = np.ones(myBCs_old.N) * np.max(a_init)*1e-8 # to avoid K singular matrix
#         a_unreduced[reduction_pattern] = a
        
#         return vol, a, q, U, a_cell, obj_hist, eq_hist, s_c_hist, s_t_hist, comp_hist, buck_hist, a_unreduced
#     else:
#         return vol, a, q, U, a_cell, obj_hist, eq_hist, s_c_hist, s_t_hist, comp_hist, buck_hist

# ############################ ALL EQUAL

# def solveLP_3D_SLP_Buckling_Free_Form_allequal(myBCs: BCS.Free_form, stress_T_max, stress_C_max, s_buck, joint_cost = 0):
#     l = myBCs.ground_structure_length + joint_cost * np.max(myBCs.ground_structure_length) # indipendency from the number of cells
#     B = calcB_3D(myBCs) # equilibrium matrix
    
#     # VL parameter
#     cell_mapping_vector = myBCs.ground_structure_cellular[::myBCs.N_cell,2].copy()
    
#     all_equal_mapping_vector = np.ones(myBCs.N_all_equal)
    
#     # Cellular        
#     a_cell = cvx.Variable((myBCs.N_cell,1), nonneg=True) # beam section cannot be negative
#     a_cell_periodic = cvx.kron(cell_mapping_vector.reshape(-1,1), a_cell) # initialize a_periodic 
    
#     # Sections
#     a_all_equal= cvx.Variable((1,1), nonneg=True) # beam section cannot be negative
#     a_all_equal_periodic = cvx.kron(all_equal_mapping_vector.reshape(-1,1), a_all_equal) # initialize a_periodic
    
#     a_wholestruct = cvx.vstack([a_cell_periodic, a_all_equal_periodic])
    
#     obj = cvx.Minimize(l.T @ a_wholestruct) # definition of the objective function
    
#     #
#     # Start of the buckling loop
#     #
#     tol, i = 1, 0
#     tol_old = 1
#     vol = 1
#     k_buck = 1
#     obj_hist = []
#     best_vol = 1e19
#     a  = np.ones(a_wholestruct.size) * np.sum(np.abs(myBCs.R))/-stress_C_max # Initialization
#     # Buckling specific parameters
    
#     buckling_length = myBCs.ground_structure_length.copy()

#     debug = False
#     reset = True
#     best_case = True
    
#     while ((tol>1e-04 and i<250) or reset):
#         # Buckling calculation based on the previuos iteration section
#         a_0 = a.copy()
#         a_0_buck = a.copy() # need for another variable. we cant use a_buck for the tolerance calc
        
#         if i>400: 
#             break
#         if (tol<1e-2 and tol_old<1e-2 and reset) or (i == 150):
#             reset_val = 0.8**k_buck
#             a_0_buck[(a<np.max(a_0_buck)*0.05)] = np.max(a_0_buck)*reset_val
#             k_buck *= 2
#             if best_case:
#                 candidate_best_vol = myBCs.ground_structure_length.T @ a
#                 if candidate_best_vol<=best_vol:
#                     best_vol = candidate_best_vol
#                     best_a = a.copy()
#                     best_q = q.copy()  
#             if k_buck>=8: # stopping condition
#                 reset = False
#             print('REINIT')
#         #
#         # Constraints
#         #
#         q = cvx.Variable((myBCs.ground_structure.shape[0],1))
#         cons = [B @ q == (myBCs.R * myBCs.dofs).reshape(-1,1)]
#         cons2 = q <= stress_T_max * a_wholestruct
#         cons3 = q >= stress_C_max * a_wholestruct
#         cons.extend([cons2, cons3])

        
#         try:
#             xx=(-s_buck*a_0_buck/(buckling_length)**2).reshape(-1,1)
#             b=2*a_wholestruct - a_0_buck.reshape(-1,1)
#             cons_buck = q >= cvx.atoms.affine.binary_operators.multiply(xx, b) # Linearized buckling constr
#             if i!=-1:
#                 cons.extend([cons_buck])
#         except:
#             pass
                        
#         #
#         # Problem solution
#         #
#         prob = cvx.Problem(obj, cons) # Creation of LP 
#         vol = prob.solve(verbose=False, solver='ECOS', max_iters = 2000, reltol = 5.0e-5, reltol_inacc = 1.0e-2)
#         q = np.array(q.value).ravel() # Optimized forces      
#         a_1 = np.array(a_wholestruct.value).ravel() # Optimized areas
#         a = a_1.copy()
#         tol_old = tol
#         tol = np.linalg.norm(a-a_0, np.inf)
#         # Eliminate the influence of the joint cost to the objective function
#         vol = myBCs.ground_structure_length.T @ a
#         obj_hist = np.append(obj_hist,vol)
#         i = i+1
#         print('\n############### SLP ITERATION {0} ###############\n'.format(i))
#         print('vol: {0:.4f} | tol: {1:.6f}\n'.format(vol,tol))
#         if debug:
#             trussplot.plot3D.plotTruss(myBCs, a, q, max(a) * 1e-3, vol)
#             # trussplot.plot3D.plotTrussBucklingCheck(myBCs, a, q, max(a) * 1e-3, s_buck)
#             # trussplot.plot3D.plotTrussStress(myBCs, a, q, max(a) * 1e-3)
#             # input()  
            
#     if best_case:
#         candidate_best_vol = myBCs.ground_structure_length.T @ a
#         if candidate_best_vol<=best_vol:
#             best_vol = candidate_best_vol
#             best_a = a.copy()
#             best_q = q.copy()
#         try:
#             a = best_a
#             q = best_q
#             vol = best_vol
#         except:
#             pass
#     print('CONVERGED')
#     print('looped {0} times, tol = {1:.4f}'.format(i,tol))
#     print("vol SLP: {0:.5f} ".format(vol))
   
#     U = np.zeros(myBCs.dofs.size)
#     n_topologies_cell = np.max(myBCs.ground_structure[:,2])
    
#     a_cell_out = np.zeros(n_topologies_cell, dtype='object_') # problems when multiple topologies
#     for i in range(a_cell_out.size):
#         a_cell_out[i] = np.array(a_cell.value).ravel()
#     return vol, a, q, U, obj_hist, a_cell_out

# ############################ MULTI LOAD CASE

# def solveLP_2D_multi_load(myBCs: BCS.MBB2D_Symm, stress_T_max, stress_C_max, joint_cost = 0, cellsEquals: bool = False, cell_mapping_vector = False):
#     l = myBCs.ground_structure_length + joint_cost * np.max(myBCs.ground_structure_length) # indipendency from the number of cells
#     B = calcB(myBCs) # equilibrium matrix
#     # VL parameter
#     NN, n_topologies_cell, cell_mapping_matrix  = mapping_matrix_init(cell_mapping_vector, myBCs)
    
#     if cellsEquals:        
#         a_wholestruct = np.zeros(myBCs.ncel_str)
#         a_cell = np.zeros(n_topologies_cell, dtype='object_')
#         for i in range(n_topologies_cell): # starting from the lower left corner
#             a_cell[i] = cvx.Variable((int(myBCs.ground_structure.shape[0]/myBCs.ncel_str),1))
#             a_wholestruct += cvx.kron(cell_mapping_matrix[:,i].reshape(-1,1), a_cell[i]) # initialize a_phys 
#     else:
#         a_cell = cvx.Variable((myBCs.ground_structure.shape[0],1)) # initialization of the design variables
#         a_wholestruct = a_cell
#     obj = cvx.Minimize(l.T @ a_wholestruct) # definition of the objective function
#     q = []
#     # constraints
#     n_load_case = myBCs.R.shape[-1]
#     cons = [a_wholestruct>=0] # beam section cannot be negative
#     for i in range(n_load_case):
#         q.append(cvx.Variable((myBCs.ground_structure.shape[0],1)))
#         cons2 = (B @ q[i] == (myBCs.R[:,i] * myBCs.dofs).reshape(-1,1))
#         cons3 = q[i] <= stress_T_max * a_wholestruct
#         cons4 = q[i] >= stress_C_max * a_wholestruct
#         cons.extend([cons2, cons3, cons4])

#     prob = cvx.Problem(obj, cons) # Creation of LP
#     vol = prob.solve(verbose=True, solver='ECOS') # Problem solving
#     q = np.array([np.array(qi.value).ravel() for qi in q]).T
#     a = np.array(a_wholestruct.value).ravel() # Optimized areas

#     # Eliminate the influence of the joint cost to the objective function
#     vol = myBCs.ground_structure_length.T @ a
#     U = np.zeros(myBCs.dofs.size)
#     obj_hist = False
#     if cellsEquals:
#         a_cell_out = np.zeros(n_topologies_cell, dtype='object_')
#         for i in range(a_cell_out.size):
#             a_cell_out[i] = np.array(a_cell[i].value).ravel()
#         return vol, a, q, U, obj_hist, a_cell_out
#     else:
#         return vol, a, q, U, obj_hist

# def solveLP_2D_SLP_Buckling_multi_load(myBCs: BCS.MBB2D_Symm, stress_T_max, stress_C_max, s_buck, joint_cost = 0, cellsEquals: bool = False, cell_mapping_vector = False, a_init = False, foldername = False):
#     l = myBCs.ground_structure_length + joint_cost * np.max(myBCs.ground_structure_length) # indipendency from the number of cells
#     B = calcB(myBCs) # equilibrium matrix
#     #
#     # Design variable initialization
#     #
#     # VL parameter
#     NN, n_topologies_cell, cell_mapping_matrix  = mapping_matrix_init(cell_mapping_vector, myBCs)
    
#     if cellsEquals:        
#         a_wholestruct = np.zeros(myBCs.ncel_str, dtype='object_')
#         a_cell = np.zeros(n_topologies_cell, dtype='object_')
#         for i in range(n_topologies_cell): # starting from the lower left corner
#             a_cell[i] = cvx.Variable((int(myBCs.ground_structure.shape[0]/myBCs.ncel_str),1), nonneg=True)
#             a_wholestruct = cvx.kron(cell_mapping_matrix[:,i].reshape(-1,1), a_cell[i]) # initialize a_phys 
#     else:
#         a_cell = cvx.Variable((myBCs.ground_structure.shape[0],1), nonneg=True) # initialization of the design variables
#         a_wholestruct = a_cell
#     obj = cvx.Minimize(l.T @ a_wholestruct) # definition of the objective function
#     #
#     # Start of the buckling loop
#     #
#     tol, i = 1, 0
#     tol_old = 1
#     vol = 1
#     k_buck = 1
#     obj_hist = []
#     #a  = np.ones(a_wholestruct.size) * 50 # Initialization
#     if np.any(a_init!=False):
#         a = a_init # Initialization
#     else:
#         a  = np.ones(a_wholestruct.size) * 10 # Initialization
        
#     # Buckling specific parameters
    
#     buckling_length = myBCs.ground_structure_length.copy()
    
#     debug = False
#     reset = False
#     chain_control = False
#     only_first = True
#     best_case = True
#     plot_intermediate_reset = False
    
#     best_vol = 1e19
#     reset_val_old = 0
    
#     while ((tol>1e-06 and i<300) or reset):
#         # Buckling calculation based on the previuos iteration section
#         only_first_activate_constraint = False # buckling constr only on the first bar
#         a_0 = a.copy()
#         a_0_buck = a.copy() # need for another variable. we cant use a_buc for the tolerance calc
#         buckling_length = myBCs.ground_structure_length.copy()

#         if i>400: 
#             break
#         if (tol<1e-4 and tol_old<1e-4 and reset) or (i == 150):
#             reset_val = 0.8**k_buck
#             if plot_intermediate_reset:
#                 # save image before perturbation
#                 folder_case = foldername+'/LP/reset/'+'It_{0:0>4}-K_{2:0>3}-val_{1:.3f}/'.format(i,reset_val_old,k_buck)
#                 if not os.path.isdir(folder_case):
#                     os.makedirs(folder_case)
#                 trussplot.plot2D.plotTruss(myBCs, a, q, max(a) * 1e-3, myBCs.ground_structure_length.T @ a, folder_case)
#             reset_val_old = reset_val
#             a_0_buck[(a<np.max(a_0_buck)*0.05)] = np.max(a_0_buck)*reset_val
#             k_buck *= 2
#             if best_case:
#                 candidate_best_vol = myBCs.ground_structure_length.T @ a
#                 if candidate_best_vol<=best_vol:
#                     best_vol = candidate_best_vol
#                     best_a = a.copy()
#                     best_q = q.copy()
                    
                    
#             if k_buck>=2: # stopping condition
#                 reset = False
#             print('REINIT')
        
#         are_bars_in_compression = False    
#         if chain_control and i!=0:
#             # Identify if there is some chain buckling
#             reduction_pattern = a_0>(1e-3*max(a_0))
#             reduced_ground_structure = myBCs.ground_structure[reduction_pattern].copy()
#             reduced_q = q[reduction_pattern].copy()
#             unique, counts = np.unique(reduced_ground_structure, return_counts=True)
#             possible_chain_nodes = np.array(unique[np.where(counts==2)]).ravel()
#             chain_nodes = []
#             for k in range(possible_chain_nodes.shape[0]): # needed to avoid adding BC nodes
#                 candidates_id = (reduced_ground_structure[:,0]==possible_chain_nodes[k]) | (reduced_ground_structure[:,1]==possible_chain_nodes[k])
#                 candidates = reduced_ground_structure[candidates_id,:]
#                 diff = np.setxor1d(candidates[0,:],candidates[1,:])
#                 start, end = np.min(diff), np.max(diff)
#                 if np.all(reduced_q[candidates_id]<0):
#                     are_bars_in_compression = True
#                 x0,y0 = myBCs.nodes[candidates[0,0],:]
#                 x1,y1 = myBCs.nodes[candidates[0,1],:]
#                 x2,y2 = myBCs.nodes[candidates[1,0],:]
#                 x3,y3 = myBCs.nodes[candidates[1,1],:]
#                 angle1 = np.arctan2(y1-y0, x1-x0)
#                 angle2 = np.arctan2(y3-y2, x3-x2)
#                 if np.allclose(np.abs(angle1),np.abs(angle2),rtol=1e-2):
#                     if (possible_chain_nodes[k]>start) and (possible_chain_nodes[k]<end):#If not some BCs can become candidates
#                         if are_bars_in_compression: # We charge only nodes in a compression chain
#                             chain_nodes.append(possible_chain_nodes[k])
                
#             if len(chain_nodes)!=0: # augmented length approach
#                 buckling_length = myBCs.ground_structure_length.copy()
#                 total_leng = 0
#                 treated_nodes = []
#                 first_bar_chain = []
#                 member_to_add_id = []
#                 buckling_length_reduced = buckling_length[reduction_pattern]
#                 for k, node in enumerate(chain_nodes): # need to use loops and not a vectorized routine
#                     candidates_bars_id = (reduced_ground_structure[:,0]==node) | (reduced_ground_structure[:,1]==node)
#                     candidates_bars = reduced_ground_structure[candidates_bars_id,:] 
#                     # check if the 2 members have a nodes in common with the last treated chain_node, if not leng = 0
#                     if k!=0:
#                         if node in treated_nodes[-1]:
#                             treated_nodes[-1].update(candidates_bars.ravel().tolist())
#                             last = candidates_bars_id & ~member_to_add # get only the new bar and update the length based on it
#                             member_to_add += candidates_bars_id
#                             total_leng += buckling_length_reduced[last]
#                             member_to_add_id[-1].update(set(np.where(member_to_add)[0]))
#                         else:
#                             first_bar_chain.append(np.where(candidates_bars_id)[0][0])
#                             treated_nodes.append(set(candidates_bars.ravel().tolist()))
#                             total_leng = np.sum(buckling_length_reduced[candidates_bars_id])
#                             member_to_add = candidates_bars_id 
#                             member_to_add_id.append(set(np.where(member_to_add)[0]))
#                     else:
#                         first_bar_chain.append(np.where(candidates_bars_id)[0][0])
#                         treated_nodes.append(set(candidates_bars.ravel().tolist()))
#                         total_leng = np.sum(buckling_length_reduced[candidates_bars_id])
#                         member_to_add = candidates_bars_id
#                         member_to_add_id.append(set(np.where(member_to_add)[0])) 
                       
#                     buckling_length_reduced[member_to_add] = total_leng
            
#                 if only_first:
#                     only_first_activate_constraint = True
#                     first_bar_chain_pattern = np.zeros(buckling_length_reduced.size, dtype='bool')
#                     first_bar_chain_pattern[first_bar_chain] = True
#                     buckling_length_reduced[~first_bar_chain_pattern] = myBCs.ground_structure_length[reduction_pattern][~first_bar_chain_pattern]
                
#                 buckling_length[reduction_pattern] = buckling_length_reduced
                    
#                 print('Chain buckling nodes elim.')
                  
#         #
#         # Constraints
#         #
#         q = []
#         n_load_case = myBCs.R.shape[-1]
#         cons = []
#         for p in range(n_load_case):
#             q.append(cvx.Variable((myBCs.ground_structure.shape[0],1)))
#             cons2 = (B @ q[p] == (myBCs.R[:,p] * myBCs.dofs).reshape(-1,1))
#             cons3 = q[p] <= stress_T_max * a_wholestruct
#             cons4 = q[p] >= stress_C_max * a_wholestruct
#             cons.extend([cons2, cons3, cons4])
        
#         if only_first_activate_constraint:
#             for ii, leading in enumerate(first_bar_chain):
#                 for kk, other in enumerate(tuple(member_to_add_id[ii])):
#                     if leading != other:
#                         side_constr = a_wholestruct[reduction_pattern][leading] <= a_wholestruct[reduction_pattern][other]               
#                         cons.extend([side_constr])
        
#         try:
#             xx=(-s_buck*a_0_buck/(buckling_length)**2).reshape(-1,1)
#             b=2*a_wholestruct - a_0_buck.reshape(-1,1)
#             for p in range(n_load_case):
#                 cons_buck = q[p] >= cvx.atoms.affine.binary_operators.multiply(xx, b) # Linearized buckling constr
#                 cons.extend([cons_buck])
#         except:
#             pass
                 
#         #
#         # Problem solution
#         #
#         prob = cvx.Problem(obj, cons) # Creation of LP
#         vol = prob.solve(verbose=False, solver='ECOS', max_iters = 5000) # Problem solving bfs=True for mosek if you want the basic solution (simplex)
#         q = np.array([np.array(qi.value).ravel() for qi in q]).T  
#         a_1 = np.array(a_wholestruct.value).ravel() # Optimized areas
#         a = a_1.copy()
#         tol_old = tol
#         tol = np.linalg.norm(a-a_0, np.inf)
#         # Eliminate the influence of the joint cost to the objective function
#         vol = myBCs.ground_structure_length.T @ a
#         obj_hist = np.append(obj_hist,vol)
#         i = i+1 
#         print('\n############### SLP ITERATION {0} ###############\n'.format(i))
#         print('vol: {0:.4f} | tol: {1:.6f}\n'.format(vol,tol))
#         if debug:
#             trussplot.plot2D.plotTruss_multiload(myBCs, a, q, max(a) * 1e-3, vol)
#             #trussplot.plot2D.plotTrussBucklingCheck(myBCs, a, q, max(a) * 1e-3, s_buck)
#             #trussplot.plot2D.plotTrussStress(myBCs, a, q, max(a) * 1e-3)  
#             #input()
            
#     if best_case:
#         candidate_best_vol = myBCs.ground_structure_length.T @ a
#         if candidate_best_vol<=best_vol:
#             best_vol = candidate_best_vol
#             best_a = a.copy()
#             best_q = q.copy()
#         try:
#             a = best_a
#             q = best_q
#             vol = best_vol
#         except:
#             pass
#     print('CONVERGED')
#     print('looped {0} times, tol = {1:.8f}'.format(i,tol))
#     print("vol SLP: {0:.5f} ".format(vol))
    
#     U = np.zeros(myBCs.dofs.size)
#     if cellsEquals:
#         a_cell_out = np.zeros(n_topologies_cell, dtype='object_')
#         for i in range(a_cell_out.size):
#             a_cell_out[i] = np.array(a_cell[i].value).ravel()
#         return vol, a, q, U, obj_hist, a_cell_out
#     else:
#         return vol, a, q, U, obj_hist
 
# def solveLP_3D_SLP_Buckling_free_form_multi_load(myBCs: BCS.MBB2D_Symm, stress_T_max, stress_C_max, s_buck, joint_cost = 0, cellsEquals: bool = False, cell_mapping_vector = False, a_init = False, foldername = False):
#     l = myBCs.ground_structure_length + joint_cost * np.max(myBCs.ground_structure_length) # indipendency from the number of cells
#     B = calcB_3D(myBCs) # equilibrium matrix
    
#     # Aperiodic
#     a_aperiodic = cvx.Variable((myBCs.ground_structure_aperiodic.shape[0],1), nonneg=True) # initialization of the design variables
    
#     # VL parameter
#     r = np.array(np.where(myBCs.ground_structure[:,2]==1)).ravel()
#     if r.size != 0:
#         repetitive = myBCs.ground_structure[r,:]
#         cell_mapping_vector = repetitive[::myBCs.N_cell,2].copy()
        
#         #Cellular        
#         a_cell = cvx.Variable((myBCs.N_cell,1), nonneg=True) # beam section cannot be negative
#         a_periodic = cvx.kron(cell_mapping_vector.reshape(-1,1), a_cell) # initialize a_periodic 
#         a_wholestruct = cvx.vstack([a_periodic, a_aperiodic])
#     else:
#         a_periodic = np.array([])
#         a_wholestruct = a_aperiodic
          
#     obj = cvx.Minimize(l.T @ a_wholestruct) # definition of the objective function
#     #
#     # Start of the buckling loop
#     #
#     tol, i = 1, 0
#     tol_old = 1
#     vol = 1
#     k_buck = 1
#     obj_hist = []

#     if np.any(a_init!=False):
#         a = a_init # Initialization
#     else:
#         a  = np.ones(a_wholestruct.size) * 0.0001 # Initialization
        
#     # Buckling specific parameters
    
#     buckling_length = myBCs.ground_structure_length.copy()
    
#     debug = False
#     reset = True
#     chain_control = False # still not working
#     only_first = True
#     best_case = True
#     plot_intermediate_reset = True
    
#     best_vol = 1e19
#     reset_val_old = 0
    
#     while ((tol>1e-04 and i<400) or reset):
#         # Buckling calculation based on the previuos iteration section
#         only_first_activate_constraint = False # buckling constr only on the first bar
#         a_0 = a.copy()
#         a_0_buck = a.copy() # need for another variable. we cant use a_buc for the tolerance calc
#         buckling_length = myBCs.ground_structure_length.copy()

#         if i>400: 
#             break
#         if (tol<1e-4 and tol_old<1e-4 and reset) or (i == 150) or (i == 300):
#             reset_val = 0.8**k_buck
#             if plot_intermediate_reset:
#                 # save image before perturbation
#                 folder_case = foldername+'/LP/reset/'+'It_{0:0>4}-K_{2:0>3}-val_{1:.3f}/'.format(i,reset_val_old,k_buck)
#                 if not os.path.isdir(folder_case):
#                     os.makedirs(folder_case)
#                 trussplot.plot3D.plotTruss_ML(myBCs, a, q, a>1e-4*max(a), myBCs.ground_structure_length.T @ a, 2/max(a), foldername=folder_case)
#             reset_val_old = reset_val
#             a_0_buck[(a<np.max(a_0_buck)*0.05)] = np.max(a_0_buck)*reset_val
#             k_buck *= 2
#             if best_case:
#                 candidate_best_vol = myBCs.ground_structure_length.T @ a
#                 if candidate_best_vol<=best_vol:
#                     best_vol = candidate_best_vol
#                     best_a = a.copy()
#                     best_q = q.copy()
                    
                    
#             if k_buck>=32: # stopping condition
#                 reset = False
#             print('REINIT')
        
#         are_bars_in_compression = False    
#         if chain_control and i!=0:
#             # Identify if there is some chain buckling
#             reduction_pattern = a_0>(1e-3*max(a_0))
#             reduced_ground_structure = myBCs.ground_structure[reduction_pattern].copy()
#             reduced_q = q[reduction_pattern].copy()
#             unique, counts = np.unique(reduced_ground_structure, return_counts=True)
#             possible_chain_nodes = np.array(unique[np.where(counts==2)]).ravel()
#             chain_nodes = []
#             for k in range(possible_chain_nodes.shape[0]): # needed to avoid adding BC nodes
#                 candidates_id = (reduced_ground_structure[:,0]==possible_chain_nodes[k]) | (reduced_ground_structure[:,1]==possible_chain_nodes[k])
#                 candidates = reduced_ground_structure[candidates_id,:]
#                 diff = np.setxor1d(candidates[0,:],candidates[1,:])
#                 start, end = np.min(diff), np.max(diff)
#                 if np.all(reduced_q[candidates_id]<0):
#                     are_bars_in_compression = True
#                 x0,y0 = myBCs.nodes[candidates[0,0],:]
#                 x1,y1 = myBCs.nodes[candidates[0,1],:]
#                 x2,y2 = myBCs.nodes[candidates[1,0],:]
#                 x3,y3 = myBCs.nodes[candidates[1,1],:]
#                 angle1 = np.arctan2(y1-y0, x1-x0)
#                 angle2 = np.arctan2(y3-y2, x3-x2)
#                 if np.allclose(np.abs(angle1),np.abs(angle2),rtol=1e-2):
#                     if (possible_chain_nodes[k]>start) and (possible_chain_nodes[k]<end):#If not some BCs can become candidates
#                         if are_bars_in_compression: # We charge only nodes in a compression chain
#                             chain_nodes.append(possible_chain_nodes[k])
                
#             if len(chain_nodes)!=0: # augmented length approach
#                 buckling_length = myBCs.ground_structure_length.copy()
#                 total_leng = 0
#                 treated_nodes = []
#                 first_bar_chain = []
#                 member_to_add_id = []
#                 buckling_length_reduced = buckling_length[reduction_pattern]
#                 for k, node in enumerate(chain_nodes): # need to use loops and not a vectorized routine
#                     candidates_bars_id = (reduced_ground_structure[:,0]==node) | (reduced_ground_structure[:,1]==node)
#                     candidates_bars = reduced_ground_structure[candidates_bars_id,:] 
#                     # check if the 2 members have a nodes in common with the last treated chain_node, if not leng = 0
#                     if k!=0:
#                         if node in treated_nodes[-1]:
#                             treated_nodes[-1].update(candidates_bars.ravel().tolist())
#                             last = candidates_bars_id & ~member_to_add # get only the new bar and update the length based on it
#                             member_to_add += candidates_bars_id
#                             total_leng += buckling_length_reduced[last]
#                             member_to_add_id[-1].update(set(np.where(member_to_add)[0]))
#                         else:
#                             first_bar_chain.append(np.where(candidates_bars_id)[0][0])
#                             treated_nodes.append(set(candidates_bars.ravel().tolist()))
#                             total_leng = np.sum(buckling_length_reduced[candidates_bars_id])
#                             member_to_add = candidates_bars_id 
#                             member_to_add_id.append(set(np.where(member_to_add)[0]))
#                     else:
#                         first_bar_chain.append(np.where(candidates_bars_id)[0][0])
#                         treated_nodes.append(set(candidates_bars.ravel().tolist()))
#                         total_leng = np.sum(buckling_length_reduced[candidates_bars_id])
#                         member_to_add = candidates_bars_id
#                         member_to_add_id.append(set(np.where(member_to_add)[0])) 
                       
#                     buckling_length_reduced[member_to_add] = total_leng
            
#                 if only_first:
#                     only_first_activate_constraint = True
#                     first_bar_chain_pattern = np.zeros(buckling_length_reduced.size, dtype='bool')
#                     first_bar_chain_pattern[first_bar_chain] = True
#                     buckling_length_reduced[~first_bar_chain_pattern] = myBCs.ground_structure_length[reduction_pattern][~first_bar_chain_pattern]
                
#                 buckling_length[reduction_pattern] = buckling_length_reduced
                    
#                 print('Chain buckling nodes elim.')
                  
#         #
#         # Constraints
#         #
#         q = []
#         n_load_case = myBCs.R.shape[-1]
#         cons = []
#         for p in range(n_load_case):
#             q.append(cvx.Variable((myBCs.ground_structure.shape[0],1)))
#             cons2 = (B @ q[p] == (myBCs.R[:,p] * myBCs.dofs).reshape(-1,1))
#             cons3 = q[p] <= stress_T_max * a_wholestruct / myBCs.sf[p]
#             cons4 = q[p] >= stress_C_max * a_wholestruct / myBCs.sf[p]
#             cons.extend([cons2, cons3, cons4])
        
#         if only_first_activate_constraint:
#             for ii, leading in enumerate(first_bar_chain):
#                 for kk, other in enumerate(tuple(member_to_add_id[ii])):
#                     if leading != other:
#                         side_constr = a_wholestruct[reduction_pattern][leading] <= a_wholestruct[reduction_pattern][other]               
#                         cons.extend([side_constr])
        
#         try:
#             xx=(-s_buck*a_0_buck/((buckling_length)**2)).reshape(-1,1)
#             b=2*a_wholestruct - a_0_buck.reshape(-1,1)
#             for p in range(n_load_case):
#                 cons_buck = q[p] >= cvx.atoms.affine.binary_operators.multiply(xx, b/myBCs.sf[p]) # Linearized buckling constr
#                 cons.extend([cons_buck])
#         except:
#             pass
                    
#         #
#         # Problem solution
#         #
#         prob = cvx.Problem(obj, cons) # Creation of LP
#         vol = prob.solve(verbose=False, solver='ECOS', max_iters = 5000)#, reltol = 5.0e-3, feastol = 1.0e-4, reltol_inacc = 1.0e-2, feastol_inacc = 1.0e-2) 
#         q = np.array([np.array(qi.value).ravel() for qi in q]).T  
#         a_1 = np.array(a_wholestruct.value).ravel() # Optimized areas
#         a = a_1.copy()
#         tol_old = tol
#         tol = np.linalg.norm(a-a_0, np.inf)
#         # Eliminate the influence of the joint cost to the objective function
#         vol = myBCs.ground_structure_length.T @ a
#         obj_hist = np.append(obj_hist,vol)
#         i = i+1 
#         print('\n############### SLP ITERATION {0} ###############\n'.format(i))
#         print('vol: {0:.4f} | tol: {1:.6f}\n'.format(vol,tol))
#         if debug:
#             trussplot.plot3D.plotTruss_ML(myBCs, a, q, max(a) * 1e-6,vol , 3/max(a))
#             #trussplot.plot2D.plotTrussBucklingCheck(myBCs, a, q, max(a) * 1e-4, s_buck)
#             #trussplot.plot2D.plotTrussStress(myBCs, a, q, max(a) * 1e-4)  
#             #input()
            
#     if best_case:
#         candidate_best_vol = myBCs.ground_structure_length.T @ a
#         if candidate_best_vol<=best_vol:
#             best_vol = candidate_best_vol
#             best_a = a.copy()
#             best_q = q.copy()
#         try:
#             a = best_a
#             q = best_q
#             vol = best_vol
#         except:
#             pass
#     print('CONVERGED')
#     print('looped {0} times, tol = {1:.8f}'.format(i,tol))
#     print("vol SLP: {0:.5f} ".format(vol))
    
#     U = np.zeros(myBCs.dofs.size)
#     if cellsEquals:
#         a_cell_out = np.zeros(n_topologies_cell, dtype='object_')
#         for i in range(a_cell_out.size):
#             a_cell_out[i] = np.array(a_cell[i].value).ravel()
#         return vol, a, q, U, obj_hist, a_cell_out
#     else:
#         return vol, a, q, U, obj_hist

# def solveNLP_2D_IPOPT_Buckling_multi_load(myBCs: BCS.MBB2D_Symm, stress_T_max, stress_C_max, E, s_buck, folder, joint_cost = 0, a_init = False):
#     #Problem reduction
#     import copy
#     myBCs_old = copy.deepcopy(myBCs)
#     myBCs, a_reduced = reduce_BCs(myBCs, 0, a_init)
    
#     # SAND IPOPT optimization
#     B = calcB(myBCs) # equilibrium matrix
    
#     N = len(myBCs.ground_structure) # Number of member of the Ground Structure
#     M = len(myBCs.dofs) # Number of DOFs of the Groud Structure
    
#     n_load_cases = myBCs.R.shape[-1]
#     N_design_var = N+n_load_cases*N+n_load_cases*M
    
#     # Create the indexing variables used for splitting the design variables vector x
#     init = np.zeros(N_design_var,dtype='bool')
#     area_id = init.copy()
#     area_id[0:N] = True
#     force_id = init.copy()
#     force_id[N:N+n_load_cases*N] = True
#     U_id = init.copy()
#     U_id[N+n_load_cases*N:] = True
    
#     ## Design variables bounds
#     lb = np.zeros(N_design_var)
#     ub = np.zeros(N_design_var)
#     # Areas
#     lb[area_id] = 0
#     ub[area_id] = 2.0e19 # inf
#     # Forces (no box constraints)
#     lb[force_id] = -2.0e19
#     ub[force_id] = 2.0e19
#     # Displacements (no box constraints)
#     lb[U_id] = -2.0e19
#     ub[U_id] = 2.0e19
    
#     ## Starting point
#     x0 = calculate_starting_point_free_form_on_unreduced_BCs_multiload(myBCs_old, myBCs, N_design_var, area_id, force_id, U_id, E, a_init, a_reduced)
    
#     ## Constraints bounds
#     cl = np.zeros(M*n_load_cases+4*N*n_load_cases)
#     cu = np.zeros(M*n_load_cases+4*N*n_load_cases)
#     # Equilibrium (M*p eq)
#     cl[0:M*n_load_cases] = 0 # Equality constr
#     cu[0:M*n_load_cases] = 0
#     # Stress (2*N*p eq)
#     # Compression
#     cl[M*n_load_cases:M*n_load_cases+N*n_load_cases] = 0
#     cu[M*n_load_cases:M*n_load_cases+N*n_load_cases] = 2.0e19
#     # Tension
#     cl[M*n_load_cases+N*n_load_cases:M*n_load_cases+2*N*n_load_cases] = -2.0e19
#     cu[M*n_load_cases+N*n_load_cases:M*n_load_cases+2*N*n_load_cases] = 0
#     # Compatibility (N*p eq)
#     cl[M*n_load_cases+2*N*n_load_cases:M*n_load_cases+3*N*n_load_cases] = -1.0e-04 # Equality constr
#     cu[M*n_load_cases+2*N*n_load_cases:M*n_load_cases+3*N*n_load_cases] = 1.0e-04
#     # Buckling (N*p eq)
#     cl[M*n_load_cases+3*N*n_load_cases:] = 0
#     cu[M*n_load_cases+3*N*n_load_cases:] = 2.0e19
    
#     nlp = cyipopt.Problem(
#         n=len(x0),
#         m=len(cl),
#         problem_obj=ipopt_routines.Layopt_IPOPT_Buck_multiload(N, M, myBCs.ground_structure_length, joint_cost, B, myBCs.R, myBCs.dofs, stress_C_max, stress_T_max, E, s_buck),
#         lb=lb,
#         ub=ub,
#         cl=cl,
#         cu=cu,
#         )
    
#     nlp.add_option('max_iter', 3000)
#     nlp.add_option('tol', 1e-4)
#     nlp.add_option('derivative_test', 'none')
#     nlp.add_option('mu_strategy', 'adaptive')
#     nlp.add_option('alpha_for_y', 'min-dual-infeas')
#     # nlp.add_option('linear_solver', 'pardiso')
#     # nlp.add_option('pardisolib', 'D:\estragio\PhD\98_Portable-Software\PardisoSolver\libpardiso600-WIN-X86-64.dll')
#     nlp.add_option('expect_infeasible_problem', 'yes')
#     # nlp.add_option('recalc_y', 'yes')
#     # nlp.add_option('bound_push', 0.0001)
#     #nlp.add_option('nlp_scaling_method', 'none')
#     #nlp.add_option('start_with_resto', 'yes')
#     #nlp.add_option('bound_relax_factor', 1e-5)
#     nlp.add_option('grad_f_constant', 'yes')
#     nlp.add_option('print_timing_statistics', 'yes')
#     nlp.add_option('hessian_approximation', 'limited-memory')
#     # nlp.add_option('limited_memory_max_history', 25)
#     #nlp.add_option('hessian_constant', 'yes')
#     nlp.add_option('print_info_string', 'yes')

#     x, info = nlp.solve(x0)
    
#     obj_hist = nlp.__objective.__self__.obj_hist
    
#     a = x[area_id] # Optimized areas
#     q, U = [], []
#     for p in range(n_load_cases):
#         q.append(x[N+(p)*N:N+(p+1)*N]) # Optimized forces
#         U.append(x[N+n_load_cases*N+(p)*M:N+n_load_cases*N+(p+1)*M]) # Optimized disploacements

#     vol = myBCs.ground_structure_length.T @ a
#     return vol, a, np.array(q).T, np.array(U).T, obj_hist, myBCs    

# def solveNLP_3D_IPOPT_Buckling_free_form_multi_load(myBCs: BCS.MBB2D_Symm, stress_T_max, stress_C_max, E, s_buck, folder, candidates, tol, joint_cost = 0, a_init = False, q = False):
    
#     #Problem reduction
#     import copy
#     myBCs_old = copy.deepcopy(myBCs)
#     myBCs, a_reduced = reduce_BCs_candidates(myBCs, candidates, a_init, q=q)
#     candidates_dofs = np.unique(myBCs.ground_structure[:,:2].ravel())
#     candidates_dofs = np.sort(np.vstack([candidates_dofs*3,candidates_dofs*3+1,candidates_dofs*3+2]).ravel())
    
#     # SAND IPOPT optimization
#     B = calcB_3D(myBCs) # equilibrium matrix
#     N = myBCs.N # Number of member of the Ground Structure
#     M = myBCs.M # Number of DOFs of the Groud Structure
    
#     n_load_cases = myBCs.R.shape[-1]
#     N_design_var = N+n_load_cases*N+n_load_cases*M
    
#     # Create the indexing variables used for splitting the design variables vector x
#     init = np.zeros(N_design_var,dtype='bool')
#     area_id = init.copy()
#     area_id[0:N] = True
#     force_id = init.copy()
#     force_id[N:N+n_load_cases*N] = True
#     U_id = init.copy()
#     U_id[N+n_load_cases*N:] = True
   
#     ## Starting point
#     #x0 = calculate_starting_point_free_form_on_unreduced_BCs_multiload_correct_areas(myBCs_old, myBCs, N_design_var, area_id, force_id, U_id, stress_T_max, stress_C_max, E, s_buck, myBCs.sf, candidates, a_init, a_reduced, folder)
#     x0 = calculate_starting_point_free_form_on_unreduced_BCs_multiload(myBCs_old, myBCs, N_design_var, area_id, force_id, U_id, E, a_init, a_reduced)
    
#     ## Design variables bounds
#     lb = np.zeros(N_design_var)
#     ub = np.zeros(N_design_var)
#     # Areas
#     lb[area_id] = 0
#     # ub[area_id] = x0[area_id]*1.5
#     ub[area_id] = 5.0e1 
#     # Forces (no box constraints)
#     lb[force_id] = -1.0e2
#     ub[force_id] = 1.0e2
#     # Displacements (no box constraints)
#     lb[U_id] = -1.0e2
#     ub[U_id] = 2.0e2
#     # lb[np.logical_and(U_id, x0>0)] = x0[np.logical_and(U_id, x0>0)]*0.5
#     # lb[np.logical_and(U_id, x0<=0)] = x0[np.logical_and(U_id, x0<=0)]*1.5
#     # ub[np.logical_and(U_id, x0>0)] = x0[np.logical_and(U_id, x0>0)]*1.5
#     # ub[np.logical_and(U_id, x0<=0)] = x0[np.logical_and(U_id, x0<=0)]*0.5
    
    
#     ## Constraints bounds
#     cl = np.zeros(M*n_load_cases+4*N*n_load_cases)
#     cu = np.zeros(M*n_load_cases+4*N*n_load_cases)
#     # Equilibrium (M*p eq)
#     cl[0:M*n_load_cases] = 0 # Equality constr
#     cu[0:M*n_load_cases] = 0
#     # Stress (2*N*p eq)
#     # Compression
#     cl[M*n_load_cases:M*n_load_cases+N*n_load_cases] = 0
#     cu[M*n_load_cases:M*n_load_cases+N*n_load_cases] = 2.0e19
#     # Tension
#     cl[M*n_load_cases+N*n_load_cases:M*n_load_cases+2*N*n_load_cases] = -2.0e19
#     cu[M*n_load_cases+N*n_load_cases:M*n_load_cases+2*N*n_load_cases] = 0
#     # Compatibility (N*p eq)
#     cl[M*n_load_cases+2*N*n_load_cases:M*n_load_cases+3*N*n_load_cases] = -1e-6 # Equality constr
#     cu[M*n_load_cases+2*N*n_load_cases:M*n_load_cases+3*N*n_load_cases] = 1e-6
#     # cl[M*n_load_cases+2*N*n_load_cases:M*n_load_cases+3*N*n_load_cases] = -1e-8 # Equality constr
#     # cu[M*n_load_cases+2*N*n_load_cases:M*n_load_cases+3*N*n_load_cases] = 1e-8
#     # Buckling (N*p eq)
#     cl[M*n_load_cases+3*N*n_load_cases:] = 0
#     cu[M*n_load_cases+3*N*n_load_cases:] = 2.0e19
    
#     ### Objective = 1000
#     obj_scale = 1/(np.sum(x0[area_id])) * 1000
    
#     ### Design Variables
#     var_scale = np.zeros(len(x0))
#     # Area
#     x_scale_area = 1/(ub[0]) * 1000
#     # x_scale_area = 1/(5.0e1) * 1000
#     var_scale[area_id] = x_scale_area
#     # Force
#     x_scale_force = 1/(ub[N]-lb[N]) * 1000
#     var_scale[force_id] = x_scale_force
#     # Displacements
#     x_scale_U = 1/(ub[N*n_load_cases+N]-lb[N*n_load_cases+N]) * 1000
#     # x_scale_U = 1/(3.0e2) * 1000
#     var_scale[U_id] = x_scale_U
    
#     ### Constraints
#     constr_scale = np.zeros(M*n_load_cases+4*N*n_load_cases)
#     # Equilibrium (M eq)
#     eq_scale = x_scale_force # Equality constr
#     constr_scale[0:M*n_load_cases] = eq_scale
#     # Stress (2*N eq)
#     # Compression
#     s_c_scale = x_scale_force
#     constr_scale[M*n_load_cases:M*n_load_cases+N*n_load_cases] = s_c_scale
#     # Tension
#     s_t_scale = x_scale_force
#     constr_scale[M*n_load_cases+N*n_load_cases:M*n_load_cases+2*N*n_load_cases] = s_t_scale
#     # Compatibility (N eq)
#     comp_scale = x_scale_area * x_scale_force * x_scale_U
#     constr_scale[M*n_load_cases+2*N*n_load_cases:M*n_load_cases+3*N*n_load_cases] = comp_scale
#     # Buckling (N eq)
#     buck_scale = x_scale_force * x_scale_area**2
#     constr_scale[M*n_load_cases+3*N*n_load_cases:] = buck_scale
    
#     x0_scaled = x0*var_scale
    
#     NLP_prob = ipopt_routines.Layopt_IPOPT_Buck_Free_Form_multiload(N, M, myBCs.ground_structure_length, joint_cost, B, myBCs.R, myBCs.dofs, stress_C_max, stress_T_max, E, s_buck, myBCs.sf)
    
#     nlp = cyipopt.Problem(
#         n=len(x0),
#         m=len(cl),
#         problem_obj=NLP_prob,
#         lb=lb,
#         ub=ub,
#         cl=cl,
#         cu=cu,
#         )
    
#     nlp.set_problem_scaling(obj_scaling = obj_scale,
#                             x_scaling = var_scale,
#                             g_scaling = constr_scale)
#     nlp.add_option('nlp_scaling_method', 'user-scaling')
    
#     nlp.add_option('max_iter', 6000)
#     nlp.add_option('tol', 1e-4)
#     nlp.add_option('acceptable_tol', 0.1)
#     nlp.add_option('acceptable_iter', 5)
#     nlp.add_option('derivative_test', 'none')
#     nlp.add_option('mu_strategy', 'adaptive')
#     nlp.add_option('alpha_for_y', 'min-dual-infeas')
#     # nlp.add_option('required_infeasibility_reduction', 0.999999)
#     nlp.add_option('linear_solver', 'pardiso')
#     nlp.add_option('pardisolib', 'D:\estragio\PhD\98_Portable-Software\PardisoSolver\libpardiso600-WIN-X86-64.dll')
#     nlp.add_option('expect_infeasible_problem', 'yes')
#     # nlp.add_option('recalc_y', 'yes')
#     nlp.add_option('bound_push', 1e-12)
#     nlp.add_option('bound_frac', 1e-8)
#     # nlp.add_option('slack_bound_push', 1e-12)
#     # nlp.add_option('slack_bound_frac', 1e-12)
#     # nlp.add_option('nlp_scaling_method', 'none')
#     # nlp.add_option('start_with_resto', 'yes')
#     #nlp.add_option('bound_relax_factor', 1e-5)
#     nlp.add_option('grad_f_constant', 'yes')
#     nlp.add_option('print_timing_statistics', 'yes')
#     nlp.add_option('hessian_approximation', 'limited-memory')
#     # nlp.add_option('limited_memory_max_history', 25)
#     #nlp.add_option('hessian_constant', 'yes')
#     nlp.add_option('print_info_string', 'yes')
#     nlp.add_option('output_file', folder+'/'+'IPOPT_out.log')
#     nlp.add_option('print_level', 5)
#     nlp.add_option('print_user_options', 'yes') 

#     x, info = nlp.solve(x0)
    
#     obj_hist = nlp.__objective.__self__.obj_hist
    
#     a = x[area_id] # Optimized areas
#     q, U = [], []
#     for p in range(n_load_cases):
#         q.append(x[N+(p)*N:N+(p+1)*N]) # Optimized forces
#         temp = x[N+n_load_cases*N+(p)*M:N+n_load_cases*N+(p+1)*M]
#         temp[myBCs.dofs==0]=0
#         U.append(temp) # Optimized disploacements
   
#     obj_hist = nlp.__objective.__self__.obj_hist
#     eq_hist = nlp.__objective.__self__.constr_equilib_hist 
#     s_c_hist = nlp.__objective.__self__.constr_s_c_hist
#     s_t_hist = nlp.__objective.__self__.constr_s_t_hist
#     comp_hist = nlp.__objective.__self__.constr_comp_hist
#     buck_hist = nlp.__objective.__self__.constr_buck_hist
    
    
#     vol = myBCs.ground_structure_length.T @ a
#     a_unreduced = np.ones(myBCs_old.N) * 1e-16 # to avoid K singular matrix
#     a_unreduced[candidates] = a
#     q_unreduced = np.zeros((myBCs_old.N, n_load_cases))
#     q_unreduced[candidates] = np.array(q).T
#     U_unreduced = np.zeros((myBCs_old.M, n_load_cases))
#     U_unreduced[candidates_dofs] = np.array(U).T

#     return vol, a, np.array(q).T, np.array(U).T, obj_hist, eq_hist, s_c_hist, s_t_hist, comp_hist, buck_hist, myBCs, a_unreduced, q_unreduced, U_unreduced
# #####################################################################################################

if __name__ == "__main__":
    ## Parameters definition
    # Cell variables (odd if cells)
    nnodx_cel = 2
    nnody_cel = 2
    nnodz_cel = 3
    #Structure varaibles
    nelx_str = 32
    nely_str = 16
    nelz_str = 1
    # Full Struct
    nelx = nnodx_cel * nelx_str - (nelx_str-1)
    nely = nnody_cel * nely_str - (nely_str-1)
    nelz = nnodz_cel * nelz_str - (nelz_str-1)

    # Lenght of the truss
    L = [200, 100] # X, Y dimensions [mm] 
    # L = [10, 2] # X, Y dimensions [mm] # Achzingher cantilever beam
    # L = [8000, 4000] # X, Y dimensions [mm] # Shahabsafa beam
    # L = [720, 360] # X, Y dimensions [mm] 10 bar bench
    # L = [300, 300, 100] # X, Y and Z dimensions [mm]
    # L = [5500, 500, 125] # X, Y and Z dimensions [mm]
    # Force magnitude (not for Wing)
    # f = 8e3 # [N] # Shahabsafa beam
    f = 200 # [N]
    # Stiffness definition
    # E = 210000 # [N/mm2] MPa - steel
    E = 69000 # [N/mm2] MPa - alu
    # E = 150000 # [N/mm2] MPa - cf
    # E = 10000 # 10 bar bench
    # E = (12*2**0.5)/(np.pi**2) # Achzingher bar bench
    # E = 69000 # Shahabsafa beam
    nu = 0.3
    s_buck = np.pi * E / 4 # Circular sections
    # s_buck = np.pi**2 * E / 12 # Square sections
    # Stress vaulues definition
    #stress_tension_max, stress_compression_max = 355, -355 # [N/mm2] MPa - steel
    stress_tension_max, stress_compression_max = 270, -270 # [N/mm2] MPa - alu
    #stress_tension_max, stress_compression_max = 540, -540 # [N/mm2] MPa - alu2
    #stress_tension_max, stress_compression_max = 1200, -750 # [N/mm2] MPa - cf
    #stress_tension_max, stress_compression_max = 1, -1
    # stress_tension_max, stress_compression_max = 1, -1 # Achzingher cantilever beam
    #stress_tension_max, stress_compression_max = 172.36, -172.36 # Shahabsafa beam
    # stress_tension_max, stress_compression_max = 20, -20 # 10 bar bench
    # Density
    #rho = 7.85 * 1e-09 # [ton/mm3] (ton/m3 * 1e-09 = ton/mm3) - steel
    # rho = 2.7 * 1e-09 # [ton/mm3] (ton/m3 * 1e-09 = ton/mm3) - alu
    #rho = 1.6 * 1e-09 # [ton/mm3] (ton/m3 * 1e-09 = ton/mm3) - cf
    #rho = 2.768 * 1e-09 # [ton/mm3] - Shahabsafa beam
    rho = 0.1 # 10 bar bench
    # Joint cost
    joint_cost = 0 # joint cost must not change the obj function
    # Are all cells equal?
    cellsEquals = False
    # L-Shape domain parameters
    n_nod_empty_x = int(((nnodx_cel-1) * nelx_str + 1) / 2 )
    n_nod_empty_y = int(((nnody_cel-1) * nely_str + 1) / 2 )
    
    tol = 1e-05

    
    ## Main program launcher 2D
    #myBCs = BCS.L_Shape(nnodx_cel,nnody_cel,nelx_str,nely_str,f,n_nod_empty_x,n_nod_empty_y, L)
    #myBCs = BCS.T_Shape(nnodx_cel,nnody_cel,nelx_str,nely_str,f,n_nod_empty_x,n_nod_empty_y, L)
    #myBCs = BCS.MichCantilever(nnodx_cel,nnody_cel,nelx_str,nely_str,f, L)
    myBCs = BCS.MichCantilever_Tugilimana(nnodx_cel,nnody_cel,nelx_str,nely_str,f, L)
    #myBCs = BCS.MBB2D_Symm(nnodx_cel,nnody_cel,nelx_str,nely_str,f, L)
    #myBCs = BCS.Cantilever(nnodx_cel,nnody_cel,nelx_str,nely_str,f, L) 
    # myBCs = BCS.Ten_bar_benchmark()
    #myBCs = BCS.Cantilever_Low(nnodx_cel,nnody_cel,nelx_str,nely_str,f, L) # Achzingher cantilever beam
    # myBCs = BCS.Cantilever_Low_Achzingher() # Achzingher cantilever beam
    #myBCs = BCS.Multiscale_test_bridge(nnodx_cel,nnody_cel,nelx_str,nely_str,f, L)
    #myBCs = BCS.Column(nnodx_cel,nnody_cel,nelx_str,nely_str,f, L)
    #myBCs = BCS.Column_dist(nnodx_cel,nnody_cel,nelx_str,nely_str,f, L)
    # Save results
    if not cellsEquals:
        foldername = myBCs.name + '-dim{0}x{1}-{2}x{3}-{4}x{5}-JC={6}-nocell'.format(L[0], L[1], nelx_str, nely_str,nnodx_cel,nnody_cel, joint_cost)
    else:
        foldername = myBCs.name + '-dim{0}x{1}-{2}x{3}-{4}x{5}-JC={6}'.format(L[0], L[1], nelx_str, nely_str,nnodx_cel,nnody_cel, joint_cost)
    if not os.path.isdir(foldername):
        os.makedirs(foldername)
    if not os.path.isdir(foldername+'/LP'):
        os.makedirs(foldername+'/LP')
        
    
    #Launcher for monolitic optimization
    
    t = time.time()
    vol_LP, a, q, U, obj_hist_LP = solveLP_2D(myBCs, stress_tension_max, stress_compression_max, joint_cost)
    elapsed_LP = time.time() - t
    trussplot.plot2D.plotRoutine(myBCs, a, q, U, vol_LP, stress_tension_max, stress_compression_max, tol, 0, obj_hist_LP, False, cellsEquals, foldername, LP=True, GIF=False)
    save_files(a,q,U,myBCs,vol_LP,stress_tension_max, stress_compression_max, joint_cost, obj_hist_LP, 0, E, s_buck , cellsEquals, foldername, True, a>tol*np.max(a), L, rho, f, elapsed_LP, is3D = False, isL = False)
    # #     

    #trussplot.plot2D.plotGroundStructure(myBCs, foldername)

    
    # Cell mapping vector (starting from the left low corner and going horizontally)
    #cell_mapping_vector = np.ones((myBCs.ncely_str,myBCs.ncelx_str), dtype='int').ravel()
    #cell_mapping_vector[0:int(myBCs.ncel_str/2)] += 1
    #cell_mapping_vector = np.arange(8)+1
    #cell_mapping_vector = np.array([1,2,2,2,2,3])
    
    # #trussplot.plotCluster(cell_mapping_vector.reshape(nely_str,nelx_str))
    
    # # Generate random x0
    # plot_different = False   
    # n_random = 100 -1
    # x0 = 50
    # max_variation = x0
    # rng = np.random.default_rng(seed=12)
    # optimized_vol = np.zeros(n_random+1)
    # optimized_n_bars = np.zeros(n_random+1, dtype=int)
    
    # x0_area = rng.random((90,n_random)) * max_variation*2 - max_variation # Generate random areas
    # x0_area = np.hstack([np.zeros((90,1)),x0_area]) + x0
    
    # # x0_area = rng.random((90,n_random)) * max_variation*2 # Generate random areas
    # # x0_Stolpe = np.array([50,0,50,50,50,50,50,0,50,0]).reshape((10,1))
    # # x0_area = np.hstack([x0_Stolpe, x0_area]) # 10 bar
    
    # x0_vol = np.sum(x0_area.T * myBCs.ground_structure_length, axis=1)
    # vol_set = set() 
    
    # min_vol = 1e19
    # for i in range(n_random+1):
    #     a_init = x0_area[:,i] 

    #     # With problem reduction
    #     t = time.time()
    #     vol_LP, a, q, U, obj_hist_LP = solveLP_2D_SLP_Buckling(myBCs, stress_tension_max, stress_compression_max, s_buck, joint_cost, cellsEquals, a_init=a_init)
    #     # myBCs = BCS.Cantilever_Low_Achzingher() # Achzingher cantilever beam
    #     # vol_LP, a, q, U, obj_hist_LP = solveNLP_2D_IPOPT_Buckling(myBCs,stress_tension_max,stress_compression_max,E,s_buck,joint_cost,a=a_init)
    #     elapsed_LP = time.time() - t  
    #     BCs_reduced, a_reduced = reduce_BCs(myBCs, tol, a, q=q, delete_chain=True) # We count only the simplified structures
    #     # vol_LP, a = buckling_stress_pp(BCs_reduced,q_reduced,stress_tension_max,stress_compression_max,s_buck)
        
    #     optimized_vol[i] = vol_LP 
    #     optimized_n_bars[i] = a_reduced.size
    #     if vol_LP < min_vol:
    #         min_a = a.copy()
    #         # min_a = a_reduced.copy()
    #         # min_q = q_reduced.copy()
    #         min_q = q.copy()
    #         min_U = U.copy()
    #         min_obj = obj_hist_LP.copy()
    #         min_vol = vol_LP.copy()
    #         # min_BCs = copy.deepcopy(BCs_reduced)
    #     if (vol_LP.round(decimals=2) not in vol_set) and plot_different:
    #         vol_set.add(vol_LP.round(decimals=2))
    #         if not os.path.isdir(foldername+'/unoptimal_layouts/{0:0>4}'.format(i)):
    #             os.makedirs(foldername+'/unoptimal_layouts/{0:0>4}'.format(i))
    #         trussplot.plot2D.plotRoutineBuckling(myBCs, a, q, U, vol_LP, stress_tension_max, stress_compression_max, tol, s_buck, min_obj, False, cellsEquals, foldername+'/unoptimal_layouts/{0:0>4}'.format(i), LP=True, GIF=False)

    # trussplot.plotNBarsRandom(optimized_n_bars, tol, foldername)
    # trussplot.plotStartingPointRandom(x0_vol, optimized_vol, foldername)
    # save_files(min_a,min_q,min_U,myBCs,min_vol,stress_tension_max, stress_compression_max, joint_cost, min_obj, 0, E, s_buck, cellsEquals, foldername, True, min_a>tol*np.max(min_a), L, rho, f, elapsed_LP, is3D = False, isL = False, a_in=x0_area)
    # trussplot.plot2D.plotRoutineBuckling(myBCs, min_a, min_q, min_U, min_vol, stress_tension_max, stress_compression_max, tol, s_buck, min_obj, False, cellsEquals, foldername, LP=True, GIF=False)
    
    # myBCs, a_reduced = reduce_BCs(myBCs, tol, min_a, min_q, delete_chain=True) 
    # t = time.time()
    # vol, a, q, U, obj_hist = solveNLP_2D_IPOPT_Buckling(myBCs,stress_tension_max,stress_compression_max,E,s_buck,joint_cost,a=a_reduced)
    # elapsed = time.time() - t
    # trussplot.plot2D.plotRoutineBuckling(myBCs, a, q, U, vol, stress_tension_max, stress_compression_max, tol, s_buck, obj_hist, obj_hist_LP, cellsEquals, foldername, LP=False, GIF=False)
    # save_files(a,q,U,myBCs,vol,stress_tension_max, stress_compression_max, joint_cost, obj_hist, 0, E, s_buck, cellsEquals, foldername, False, a>tol*np.max(a), L, rho, f, elapsed, is3D = False, isL = False, vol_LP=min_vol, a_in=min_a)

    """ x0 = np.ones(90)*0.1
    x0_vol = np.sum(x0.T * myBCs.ground_structure_length)
    t = time.time()     
    vol_LP, a, q, U, obj_hist_LP = solveLP_2D_SLP_Buckling(myBCs, stress_tension_max, stress_compression_max, s_buck, joint_cost, cellsEquals, a_init=x0.ravel())
    elapsed_LP = time.time() - t
    save_files(a,q,U,myBCs,vol_LP,stress_tension_max, stress_compression_max, joint_cost, obj_hist_LP, 0, E, s_buck, cellsEquals, foldername, True, tol, L, rho, f, elapsed_LP, is3D = False, isL = False)
    trussplot.plot2D.plotRoutineBuckling(myBCs, a, q, U, vol_LP, stress_tension_max, stress_compression_max, 1e-3, s_buck, obj_hist_LP, False, cellsEquals, foldername, LP=True, GIF=False) """
    
    """ myBCs, a_reduced = reduce_BCs(myBCs, 1e-3, a, delete_chain=True)
    t = time.time()
    vol, a, q, U, obj_hist = solveNLP_2D_IPOPT_Buckling(myBCs,stress_tension_max,stress_compression_max,E,s_buck,joint_cost,a=a_reduced)
    elapsed = time.time() - t
    
    save_files(a,q,U,myBCs,vol,stress_tension_max, stress_compression_max, joint_cost, obj_hist, 0, E, s_buck, cellsEquals, foldername, False, tol, L, rho, f, elapsed, is3D = False, isL = False, vol_LP=vol_LP)
    trussplot.plot2D.plotRoutineBuckling(myBCs, a, q, U, vol, stress_tension_max, stress_compression_max, 1e-3, s_buck, obj_hist, obj_hist_LP, cellsEquals, foldername, LP=False, GIF=False)
    trussplot.plotStartingPointRandom(x0_vol, vol, foldername) """
    
    #trussfem.exportINP(myBCs, a, q, E, nu, stress_tension_max, stress_compression_max, rho)
    
    #cell_mapping_vector = np.array([1,1,2,2,3])
    #trussplot.plotCluster(cell_mapping_vector.reshape(nely_str,nelx_str))

    """ ## Main program launcher 3D
    #myBCs = BCS.Wing_3D_conc_center(nnodx_cel,nnody_cel,nnodz_cel,nelx_str,nely_str,nelz_str,f , L)
    #myBCs = BCS.Wing_3D_center(nnodx_cel,nnody_cel,nnodz_cel,nelx_str,nely_str,nelz_str,f,L)
    #myBCs = BCS.Wing_3D(nnodx_cel,nnody_cel,nnodz_cel,nelx_str,nely_str,nelz_str,f , L)
    #myBCs = BCS.CantileverBeam_3D(nnodx_cel,nnody_cel,nnodz_cel,nelx_str,nely_str,nelz_str,f, L)
    myBCs = BCS.CantileverBeam_3D_smear(nnodx_cel,nnody_cel,nnodz_cel,nelx_str,nely_str,nelz_str,f, L)
    #myBCs = BCS.Column_3D(nnodx_cel,nnody_cel,nnodz_cel,nelx_str,nely_str,nelz_str,f , L)
    #myBCs = BCS.TEST_sym(nnodx_cel,nnody_cel,nnodz_cel,nelx_str,nely_str,nelz_str,f , L)
    # Save results
    if not cellsEquals:
        foldername = myBCs.name + '-dim{0}x{1}x{2}-{3}x{4}x{5}-JC={6}-nocell'.format(L[0], L[1], L[2], nelx_str, nely_str, nelz_str, joint_cost)
    else:
        foldername = myBCs.name + '-dim{0}x{1}x{2}-{3}x{4}x{5}-JC={6}'.format(L[0], L[1], L[2], nelx_str, nely_str, nelz_str, joint_cost)
    if not os.path.isdir(foldername):
        os.makedirs(foldername)

    #trussplot.plot3D.plotGroundStructure(myBCs)
    t = time.time()
    #vol, a, q, U, obj_hist, a_cell = solveLP_3D(myBCs,stress_tension_max, stress_compression_max, joint_cost, cellsEquals)
    vol, a, q, U, obj_hist, a_cell = solveLP_3D_SLP_Buckling(myBCs, stress_tension_max, stress_compression_max, s_buck, joint_cost, cellsEquals)
    elapsed_LP = time.time() - t
    trussplot.plot3D.plotRoutineBuckling(myBCs, a, q, U, vol, stress_tension_max, stress_compression_max, 1e-3, s_buck, obj_hist, cellsEquals, foldername, LP=True, a_cell=a_cell, GIF=True)
    
    save_files(a,q,U,myBCs,vol,stress_tension_max, stress_compression_max, joint_cost, obj_hist, a_cell, E, s_buck, cellsEquals, foldername, True, tol, L, rho, f, elapsed_LP)
    myBCs_reduced, a_cell_reduced = reduce_BCs(myBCs, 1e-3, a, a_cell)
    
    t = time.time()
    vol, a, q, U, obj_hist, a_cell = solveNLP_3D_IPOPT_VL_Buckling(myBCs_reduced,stress_tension_max,stress_compression_max,E,joint_cost, a_cell_reduced)
    
    elapsed = time.time() - t
    
    print("Vol: {0:.2f} mm3, Weight: {1:.5f} kg".format(vol, vol*rho*1000))
    print("Vol_star: {0:.5f} PL/sigma".format(vol/(f*max(L)/((stress_tension_max-stress_compression_max)/2))))
    print("Vol_fraction: {0:.6f}%".format(vol/(L[0]*L[1]*L[2])))
    print("Compliance: {0:.2f} mJ".format(myBCs_reduced.R.T@U))
    print("Max section: {0:.3f} mm2".format(np.max(a)))
    print("Optimization SLP time: %.2f seconds" % elapsed_LP) 
    print("Optimization time: %.2f seconds" % elapsed) 
    
    trussplot.plot3D.plotRoutineBuckling(myBCs_reduced, a, q, U, vol, stress_tension_max, stress_compression_max, 1e-3, s_buck, obj_hist, cellsEquals, foldername, LP=False, a_cell=a_cell, GIF=True)
    save_files(a,q,U,myBCs_reduced,vol,stress_tension_max, stress_compression_max, joint_cost, obj_hist, a_cell, E, s_buck, cellsEquals, foldername, False, tol, L, rho, f, elapsed)

    
    
    #### No equal cells
    ## Main program launcher 3D
    #myBCs = BCS.Wing_3D_conc_center(nnodx_cel,nnody_cel,nnodz_cel,nelx_str,nely_str,nelz_str,f , L)
    #myBCs = BCS.Wing_3D_center(nnodx_cel,nnody_cel,nnodz_cel,nelx_str,nely_str,nelz_str,f,L)
    #myBCs = BCS.Wing_3D(nnodx_cel,nnody_cel,nnodz_cel,nelx_str,nely_str,nelz_str,f , L)
    #myBCs = BCS.CantileverBeam_3D(nnodx_cel,nnody_cel,nnodz_cel,nelx_str,nely_str,nelz_str,f, L)
    myBCs = BCS.CantileverBeam_3D_smear(nnodx_cel,nnody_cel,nnodz_cel,nelx_str,nely_str,nelz_str,f, L)
    #myBCs = BCS.Column_3D(nnodx_cel,nnody_cel,nnodz_cel,nelx_str,nely_str,nelz_str,f , L)
    #myBCs = BCS.TEST_sym(nnodx_cel,nnody_cel,nnodz_cel,nelx_str,nely_str,nelz_str,f , L)
    # Save results
    if not cellsEquals:
        foldername = myBCs.name + 'dim{0}x{1}x{2}-{3}x{4}x{5}-JC={6}-nocell'.format(L[0], L[1], L[2], nelx_str, nely_str, nelz_str, joint_cost)
    else:
        foldername = myBCs.name + 'dim{0}x{1}x{2}-{3}x{4}x{5}-JC={6}'.format(L[0], L[1], L[2], nelx_str, nely_str, nelz_str, joint_cost)
        
    if not os.path.isdir(foldername):
        os.makedirs(foldername)

    #trussplot.plot3D.plotGroundStructure(myBCs)
    t = time.time()
    #vol, a, q, U, obj_hist, a_cell = solveLP_3D(myBCs,stress_tension_max, stress_compression_max, joint_cost, cellsEquals)
    vol, a, q, U, obj_hist = solveLP_3D_SLP_Buckling(myBCs, stress_tension_max, stress_compression_max, s_buck, joint_cost, cellsEquals)
    elapsed_LP = time.time() - t
    trussplot.plot3D.plotRoutineBuckling(myBCs, a, q, U, vol, stress_tension_max, stress_compression_max, 1e-3, s_buck, obj_hist, cellsEquals, foldername, LP=True, GIF=False)
    
    #save_files(a,q,U,myBCs,vol,stress_tension_max, stress_compression_max, joint_cost, obj_hist, 0, foldername, True, tol, L, rho, f, elapsed_LP, is3D = True, isL = False)
    
    t = time.time()
    vol, a, q, U, obj_hist = solveNLP_3D_IPOPT_Buckling(myBCs,stress_tension_max,stress_compression_max,E,s_buck,joint_cost, a = a)
    
    elapsed = time.time() - t
    
    print("Vol: {0:.2f} mm3, Weight: {1:.5f} kg".format(vol, vol*rho*1000))
    print("Vol_star: {0:.5f} PL/sigma".format(vol/(f*max(L)/((stress_tension_max-stress_compression_max)/2))))
    print("Vol_fraction: {0:.6f}%".format(vol/(L[0]*L[1]*L[2])))
    print("Compliance: {0:.2f} mJ".format(myBCs.R.T@U))
    print("Max section: {0:.3f} mm2".format(np.max(a)))
    print("Optimization SLP time: %.2f seconds" % elapsed_LP) 
    print("Optimization time: %.2f seconds" % elapsed) 
    
    trussplot.plot3D.plotRoutineBuckling(myBCs, a, q, U, vol, stress_tension_max, stress_compression_max, 1e-3, s_buck, obj_hist, cellsEquals, foldername, LP=False, GIF=False)
    compute_displacement_and_force_error(myBCs, a, q, U)
    #save_files(a,q,U,myBCs,vol,stress_tension_max, stress_compression_max, joint_cost, obj_hist, 0, foldername, False, tol, L, rho, f, elapsed, is3D = True, isL = False)
    #trussfem.exportINP(myBCs, a, q, E, nu, stress_tension_max, stress_compression_max, rho)
     """