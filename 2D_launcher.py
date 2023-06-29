# LayoutOptimization - Enrico Stragiotti - Jun 2023
# mm, N, MPa

import bcs as BCS
import layopt as LAYOPT
import trussplot as PLOT
import numpy as np
import os, time

## Structure parameters definition
# Structure variables
nnodx_cel = 7
nnody_cel = 5

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


## Main program launcher 2D
# myBCs = BCS.MichCantilever(nnodx_cel,nnody_cel,1,1,f, L)
myBCs = BCS.MBB2D_Symm(nnodx_cel,nnody_cel,1,1,f, L)
# myBCs = BCS.Cantilever(nnodx_cel,nnody_cel,1,1,f, L) 
# myBCs = BCS.Cantilever_Low(nnodx_cel,nnody_cel,1,1,f, L)

# Create save folder
foldername = myBCs.name + '-dim{0}x{1}-{2}x{3}-{4}x{5}-JC={6}-nocell'.format(L[0], L[1], 1, 1,nnodx_cel,nnody_cel, joint_cost)
if not os.path.isdir(foldername):
    os.makedirs(foldername)
if not os.path.isdir(foldername+'/LP'):
    os.makedirs(foldername+'/LP')
    
# Problem defintion and resolution

# X0
a_init = np.ones(myBCs.ground_structure.shape[0])

t = time.time()
vol_LP, a, q, U, obj_hist_LP = LAYOPT.solveLP_2D_SLP_Buckling(myBCs, stress_tension_max, stress_compression_max, s_buck, joint_cost, False, a_init=a_init)
elapsed_LP = time.time() - t

LAYOPT.save_files(a,q,U,myBCs,vol_LP,stress_tension_max, stress_compression_max, joint_cost, obj_hist_LP, 0, E, s_buck, False, foldername, True, a>np.max(a)*tol, L, rho, f, elapsed_LP, is3D = False, isL = False)
PLOT.plot2D.plotRoutineBuckling(myBCs, a, q, U, vol_LP, stress_tension_max, stress_compression_max, tol, s_buck, obj_hist_LP, False, False, foldername, LP=True, GIF=False)
    

# Thresholding and compressive chain cleaning
BCs_reduced, a_reduced = LAYOPT.reduce_BCs(myBCs, tol, a, q=q, delete_chain=True)

t = time.time()
vol, a, q, U, obj_hist = LAYOPT.solveNLP_2D_IPOPT_Buckling(BCs_reduced,myBCs,stress_tension_max,stress_compression_max,E,s_buck,foldername,joint_cost,a_init=a_reduced, a_fem=a)
elapsed = time.time() - t

LAYOPT.save_files(a,q,U,BCs_reduced,vol,stress_tension_max, stress_compression_max, joint_cost, obj_hist, 0, E, s_buck, False, foldername, False, a>np.max(a)*tol, L, rho, f, elapsed, is3D = False, isL = False, vol_LP=vol_LP)
PLOT.plot2D.plotRoutineBuckling(BCs_reduced, a, q, U, vol, stress_tension_max, stress_compression_max, tol, s_buck, obj_hist, obj_hist_LP, False, foldername, LP=False, GIF=False)

print("Vol: {0:.2f} mm3".format(vol))
print("Compliance: {0:.2f} mJ".format(BCs_reduced.R.T@U))
print("Max section: {0:.3f} mm2".format(np.max(a)))
print("Optimization SLP time: %.2f seconds" % elapsed_LP) 
print("Optimization time: %.2f seconds" % elapsed)