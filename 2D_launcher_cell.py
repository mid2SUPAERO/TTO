# LayoutOptimization - Enrico Stragiotti - Jun 2023
# mm, N, MPa

import bcs as BCS
import layopt as LAYOPT
import trussplot as PLOT
import numpy as np
import os, time

## Structure parameters definition
# Cell nodes (odd if cells)
nnodx_cel = 3
nnody_cel = 3

#Structure varaibles (number of cells)
nelx_str = 6
nely_str = 3

# Dimensions of the truss
L = [3, 1] # X, Y dimensions [mm]
f = 1 # [N]
# Material definition
E = 1 # Young's modulus
nu = 0.3 # Poisson coefficient
# Stress vaulues definition
stress_tension_max, stress_compression_max = 10, -10 # [MPa]
# Density
rho = 1
# Buckling section shape parameter: Circular sections
s_buck = np.pi * E / 4 
# Joint cost
joint_cost = 0 # Joint cost, use to penalize the number of bars. 

# Thresholding of the members
tol=1e-3

cell_mapping_vector = np.ones(int(nelx_str*nely_str))


## Main program launcher 2D
myBCs = BCS.MichCantilever(nnodx_cel,nnody_cel,nelx_str,nely_str,f, L)
# myBCs = BCS.MBB2D_Symm(nnodx_cel,nnody_cel,nelx_str,nely_str,f, L)
# myBCs = BCS.Cantilever(nnodx_cel,nnody_cel,nelx_str,nely_str,f, L) 
# myBCs = BCS.Cantilever_Low(nnodx_cel,nnody_cel,nelx_str,nely_str,f, L)

# Save results
foldername = myBCs.name + '-dim{0}x{1}-{2}x{3}-{4}x{5}-JC={6}'.format(L[0], L[1], nelx_str, nely_str,nnodx_cel,nnody_cel, joint_cost)
    
if not os.path.isdir(foldername):
    os.makedirs(foldername)
if not os.path.isdir(foldername+'/LP'):
    os.makedirs(foldername+'/LP')
    
# Problem defintion and resolution

# X0
a_init = np.ones(myBCs.ground_structure.shape[0])

t = time.time()
vol_LP, a, q, U, obj_hist_LP, a_cell = LAYOPT.solveLP_2D_SLP_Buckling(myBCs, stress_tension_max, stress_compression_max, s_buck, joint_cost, True, cell_mapping_vector=cell_mapping_vector, chain=False)
elapsed_LP = time.time() - t

LAYOPT.save_files(a,q,U,myBCs,vol_LP,stress_tension_max, stress_compression_max, joint_cost, obj_hist_LP, 0, E, s_buck, True, foldername, True, a>np.max(a)*tol, L, rho, f, elapsed_LP, is3D = False, isL = False)
PLOT.plot2D.plotRoutineBuckling(myBCs, a, q, U, vol_LP, stress_tension_max, stress_compression_max, tol, s_buck, obj_hist_LP, False, False, foldername, LP=True, a_cell=a_cell, GIF=False)
    
# Thresholding
myBCs_reduced, a_cell_reduced = LAYOPT.reduce_BCs(myBCs, tol, a, a_cell=a_cell, cell_mapping_vector=cell_mapping_vector)

t = time.time()
vol, a, q, U, obj_hist, a_cell = LAYOPT.solveNLP_2D_IPOPT_VL_Buckling(myBCs_reduced,myBCs,stress_tension_max,stress_compression_max,E,s_buck,foldername,joint_cost, a_cell=a_cell_reduced, cell_mapping_vector=cell_mapping_vector)
elapsed = time.time() - t

LAYOPT.save_files(a,q,U,myBCs_reduced,vol,stress_tension_max, stress_compression_max, joint_cost, obj_hist, 0, E, s_buck, True, foldername, False, a>np.max(a)*tol, L, rho, f, elapsed, is3D = False, isL = False, vol_LP=vol_LP)
PLOT.plot2D.plotRoutineBuckling(myBCs_reduced, a, q, U, vol, stress_tension_max, stress_compression_max, tol, s_buck, obj_hist, obj_hist_LP, False, foldername, LP=False, GIF=False)



print("Vol: {0:.2f} mm3".format(vol))
print("Compliance: {0:.2f} mJ".format(myBCs_reduced.R.T@U))
print("Max section: {0:.3f} mm2".format(np.max(a)))
print("Optimization SLP time: %.2f seconds" % elapsed_LP) 
print("Optimization time: %.2f seconds" % elapsed)