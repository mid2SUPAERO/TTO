# Enrico Stragiotti 10/2021
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import collections as mc
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import plotly.graph_objects as go
from itertools import product, combinations, compress
import glob
import imageio
import os
import tikzplotlib
import scipy.sparse as sp
# import bpy

#matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

sample = 8
norm = matplotlib.colors.Normalize(vmin=0, vmax=sample-1) #normalize item number values to colormap
cmap = matplotlib.cm.get_cmap('coolwarm')

accent_b_1 = colors.rgb2hex((cmap(norm(0))), keep_alpha=True)
accent_b_2 = colors.rgb2hex((cmap(norm(1))), keep_alpha=True)
accent_b_3 = colors.rgb2hex((cmap(norm(2))), keep_alpha=True)
accent_b_4 = colors.rgb2hex((cmap(norm(3))), keep_alpha=True)

accent_r_1 = colors.rgb2hex((cmap(norm(7))), keep_alpha=True)
accent_r_2 = colors.rgb2hex((cmap(norm(6))), keep_alpha=True)
accent_r_3 = colors.rgb2hex((cmap(norm(5))), keep_alpha=True)
accent_r_4 = colors.rgb2hex((cmap(norm(4))), keep_alpha=True)

# Create custom colormaps

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

# Blue
cmap_original = plt.get_cmap('coolwarm')
cmap_original_r = plt.get_cmap('coolwarm_r')
Blue_mono = truncate_colormap(cmap_original_r, 0.5, 1)
# Red
Red_mono = truncate_colormap(cmap, 0.5, 1)



############################
# 2D plotting
############################

class plot2D():
    #Visualize truss
    @staticmethod
    def plotTruss(myBCs, a, forces, threshold, vol, tk, foldername=False):
        """ Plot the ground stucture optimization solution """
        nodes = myBCs.nodes
        candidates = myBCs.ground_structure

        fig = plt.figure()
        plt.clf()
        plt.axis('off')
        plt.axis('equal')
        plt.title("Final Structure - Vol = %.3f" % (vol))
        
        candidates = myBCs.ground_structure[a >= threshold,:]
        q = forces[a >= threshold]
        a = a[a >= threshold]
        
        c = np.empty(a.size, dtype = object)
        tol = np.max(q) * 1e-04
        c[:] = 'grey' # no force
        c[q>tol] = accent_r_1 # tension
        c[q<-tol] = accent_b_1 # compression
        linewidth = a * tk
        # boundaries
        if not myBCs.isLshape:
            plt.plot([0,0], [0, myBCs.L[1]], 'lightgray', linestyle="dotted", linewidth=0.2)
            plt.plot([0, myBCs.L[0]], [myBCs.L[1], myBCs.L[1]], 'lightgray', linestyle="dotted", linewidth=0.2)
            plt.plot([myBCs.L[0], myBCs.L[0]], [myBCs.L[1], 0], 'lightgray', linestyle="dotted", linewidth=0.2)
            plt.plot([myBCs.L[0], 0], [0,0], 'lightgray', linestyle="dotted", linewidth=0.2)
        # optimized beams
        for i in range(len(a)):
            pos = nodes[candidates[i, [0, 1]].astype(int), :]
            plt.plot(pos[:, 0], pos[:, 1], c[i], linewidth = linewidth[i])
            # plt.scatter(pos[:, 0], pos[:, 1], c = 'black', marker = 'o', edgecolors = 'none', zorder=2, s=5) # enforce the scatter plot after the plot
        fig.tight_layout()
        if foldername == False:
            plt.show()
        else:
            # tikzplotlib.save(foldername+'fig1-Topology.tex')
            # plt.savefig(foldername+'fig1-Topology.pgf')
            plt.savefig(foldername+'fig1-Topology.pdf')
        
    #Visualize deformed truss
    @staticmethod
    def plotTrussDeformation(myBCs, a, q, u, threshold, magnitude, tk, foldername, axis):
        """ Plot the ground stucture optimization solution """
        nodes = myBCs.nodes
        u = u.reshape(-1,2)
        candidates = myBCs.ground_structure[a >= threshold,:]
        
        q = q[a >= threshold]
        a = a[a >= threshold]

        fig = plt.figure()
        plt.clf()
        plt.axis('off')
        plt.axis('equal')
        plt.title("Final Structure deformation - Magnitude %i" % (magnitude))
        if not myBCs.isLshape:
            plt.plot([0,0], [0, myBCs.L[1]], 'lightgray')
            plt.plot([0, myBCs.L[0]], [myBCs.L[1], myBCs.L[1]], 'lightgray')
            plt.plot([myBCs.L[0], myBCs.L[0]], [myBCs.L[1], 0], 'lightgray')
            plt.plot([myBCs.L[0], 0], [0,0], 'lightgray')
        # optimized beams
        for i in [i for i in range(len(a)) if a[i] >= threshold]:
            c = 'black' 
            pos = nodes[candidates[i, [0, 1]].astype(int), :] + magnitude * u[candidates[i, [0, 1]].astype(int), :]
            plt.plot(pos[:, 0], pos[:, 1], c, linewidth = a[i] * tk)
            plt.scatter(pos[:, 0], pos[:, 1], c = 'black', marker = 'o', edgecolors = 'none', zorder=2) # enforce the scatter plot after the plot
        fig.tight_layout()
        if not axis:
            plt.title(' ')
            if foldername == False:
                plt.show()
            else:
                # tikzplotlib.save(foldername+'fig6-U_NoAxis-Mag={0}.tex'.format(magnitude))
                plt.savefig(foldername+'fig5-U_NoAxis-Mag={0}.pdf'.format(magnitude))
                # plt.savefig(foldername+'fig6-U_NoAxis-Mag={0}.pgf'.format(magnitude))
        else:
            plt.title(' ')
            if foldername == False:
                plt.show()
            else:
                # tikzplotlib.save(foldername+'fig6-U_Axis-Mag={0}.tex'.format(magnitude))
                plt.savefig(foldername+'fig6-U_Axis-Mag={0}.pdf'.format(magnitude))
                # plt.savefig(foldername+'fig6-U_Axis-Mag={0}.pgf'.format(magnitude))
     
    @staticmethod    
    def plotTrussRF(myBCs, a, q, threshold, stress_t, stress_c, tk, foldername=False):
        """ Plot the ground stucture optimization solution efficency, defined as the percentage of the max charge """
        nodes = myBCs.nodes
        candidates = myBCs.ground_structure[a >= threshold]
        fig, ax = plt.subplots()
        plt.axis('off')
        plt.axis('equal')
        plt.title("Reserve Factor")

        q[abs(q) < max(abs(q)) * 10e-3] = 0 # eliminates 0/0 cases
        efficency = q[a >= threshold]/a[a >= threshold]
        efficency[efficency > 0] /= stress_t
        efficency[efficency < 0] /= stress_c
        
        points = nodes[candidates[:, [0, 1]].astype(int), :].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1:2], points[1::2]], axis=1)
        
        ax.set_xlim(np.min(nodes[:,0]), np.max(nodes[:,0]+0.05*np.max(nodes[:,0])))
        ax.set_ylim(np.min(nodes[:,1]), np.max(nodes[:,1]+0.05*np.max(nodes[:,1])))
        
        norm = plt.Normalize(efficency.min(), efficency.max())
        lc = mc.LineCollection(segments, cmap=Blue_mono, norm=norm)
        lc.set_array(efficency)
        lc.set_linewidth(a[a >= threshold] * tk)
        line = ax.add_collection(lc)
        fig.colorbar(line, ax=[ax], location='bottom', orientation="horizontal")
        fig.tight_layout()
        # optimized beams
        if foldername == False:
                plt.show()
        else:
            # tikzplotlib.save(foldername+'fig7-safety.tex')
            plt.savefig(foldername+'fig7-safety.pdf')
            # plt.savefig(foldername+'fig7-safety.pgf')
    
    @staticmethod    
    def plotTrussStress(myBCs, a, q, threshold, s_t, s_c, tk, foldername=False):
        """ Plot the ground stucture optimization solution stress """
        nodes = myBCs.nodes
        candidates = myBCs.ground_structure[a >= threshold]
        fig, ax = plt.subplots()
        plt.axis('off')
        plt.axis('equal')
        plt.title("ABS stress")

        q = q[a >= threshold]
        a = a[a >= threshold]
        stress = q/a
        
        # Round to a specified number of significant digits
        significant_digits = 4
        decimals = significant_digits - int(math.floor(math.log10(np.abs(np.max(stress))))) - 1
        stress =  np.around(stress, decimals=decimals)
        
        points = nodes[candidates[:, [0, 1]].astype(int), :].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1:2], points[1::2]], axis=1)
        
        ax.set_xlim(np.min(nodes[:,0]), np.max(nodes[:,0]+0.05*np.max(nodes[:,0])))
        ax.set_ylim(np.min(nodes[:,1]), np.max(nodes[:,1]+0.05*np.max(nodes[:,1])))
        
        norm = plt.Normalize(s_c, s_t)
        lc = mc.LineCollection(segments, cmap='coolwarm', norm=norm)
        lc.set_array(stress[a >= threshold])
        lc.set_linewidth(a[a >= threshold] * tk)
        line = ax.add_collection(lc)
        fig.colorbar(line, ax=[ax], location='bottom', orientation="horizontal")
        
        if foldername == False:
            plt.show()
        else:
            # tikzplotlib.save(foldername+'fig2-stress.tex')
            plt.savefig(foldername+'fig2-stress.pdf')
            # plt.savefig(foldername+'fig2-stress.pgf')  
                
    #Visualize initial truss
    @staticmethod
    def plotGroundStructure(myBCs, plot_nodes=False, foldername=False, load_case = -1, mag=0.1):
        nodes = myBCs.nodes
        candidates = myBCs.ground_structure
        fig, ax = plt.subplots()
        plt.axis('off')
        plt.axis('equal')
        plt.title("Starting Ground Structure - N. Elements = " + str(len(candidates))) 
        points = nodes[candidates[:, [0, 1]].astype(int), :].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1:2], points[1::2]], axis=1)
        
        lc = mc.LineCollection(segments, colors=accent_b_2)
        ax.add_collection(lc)
        
        if plot_nodes:
            for i in range(len(segments)):
                pos = nodes[candidates[i, [0, 1]].astype(int), :]
                plt.scatter(pos[:, 0], pos[:, 1], c = accent_r_1, marker = 'o', edgecolors = 'none', zorder=100000, s=5) # enforce the scatter plot after the plot
        
        ax.set_xlim(np.min(nodes[:,0]), np.max(nodes[:,0]+0.05*np.max(nodes[:,0])))
        ax.set_ylim(np.min(nodes[:,1]), np.max(nodes[:,1]+0.05*np.max(nodes[:,1])))
        
        # BCs
        fixed_nodes = np.unique(myBCs.fixed_dofs // 2)
        x,y = myBCs.nodes[fixed_nodes,:][:,[0,1]].T
        ax.scatter(x,y, s=15, c=accent_b_1, alpha=1, zorder=2)
        # Loads
        if load_case != -1:
            load_nodes = np.nonzero(myBCs.R[:,load_case])[0] // 2
            x,y = myBCs.nodes[load_nodes,:][:,[0,1]].T
            R = myBCs.R[:,load_case].copy()
        else:
            load_nodes = np.nonzero(myBCs.R)[0] // 2
            x,y = myBCs.nodes[load_nodes,:][:,[0,1]].T
            R = myBCs.R.copy()
            
        magnit = R[R!=0]
        direction = myBCs.force_dofs % 2
        v = np.zeros((load_nodes.shape[0],2))
        v[range(load_nodes.shape[0]),direction] = magnit
        # normalization of v
        v_norm = np.max(np.abs(myBCs.R))
        v = v/v_norm * mag * np.max(myBCs.L)
        ax.quiver(x, y, v[:,0], v[:,1], color='red', zorder=30, linewidth = 0.7)

    
        fig.tight_layout()
        if foldername == False:
            plt.show()
        else:
            if load_case != -1:
                plt.savefig(foldername+'fig0-GS-LC{:03d}.pdf'.format(load_case+1))
            else:
                plt.savefig(foldername+'fig0-GS.pdf')

    @staticmethod
    def plotTrussBucklingCheck(myBCs, a, forces, threshold, s_buck, tk, foldername=False):
        """ Plot the ground stucture optimization solution """
        nodes = myBCs.nodes
        candidates = myBCs.ground_structure[a >= threshold,:]
        q = forces[a >= threshold]
        # Critical force
        q_crit = -s_buck*a[a >= threshold]**2/myBCs.ground_structure_length[a >= threshold]**2
        a = a[a >= threshold]
        color = np.empty(a.size, dtype = object)
        
        color[q<q_crit*1.01] = accent_r_1 # Buckled
        color[q>q_crit*1.01] = 'grey' # Ok
        
        fig, ax = plt.subplots()
        plt.axis('off')
        plt.axis('equal')
        plt.title("Buckling check")
        points = nodes[candidates[:, [0, 1]].astype(int), :].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1:2], points[1::2]], axis=1)
        
        # boundaries
        if not myBCs.isLshape:
            ax.set_xlim(np.min(nodes[:,0]), np.max(nodes[:,0]+0.05*np.max(nodes[:,0])))
            ax.set_ylim(np.min(nodes[:,1]), np.max(nodes[:,1]+0.05*np.max(nodes[:,1])))
        lc = mc.LineCollection(segments, colors = color)
        lc.set_linewidth(a[a >= threshold] * tk)
        ax.add_collection(lc)
        fig.tight_layout()

        if foldername == False:
            plt.show()
        else:
            plt.savefig(foldername+'fig3-B_check.pdf')
 
    @staticmethod       
    def plotTrussBuckling(myBCs, a, forces, threshold, s_buck, tk, foldername=False):
        """ Plot the ground stucture optimization solution """
        nodes = myBCs.nodes
        candidates = myBCs.ground_structure[a >= threshold,:]
        q = forces[a >= threshold]
        # Critical force
        q_crit = -s_buck*a[a >= threshold]**2/myBCs.ground_structure_length[a >= threshold]**2
        
        buck = q/q_crit
        buck[buck<0] = 0 # Member in tension cannot buckle
        
        fig, ax = plt.subplots()
        plt.axis('off')
        plt.axis('equal')
        plt.title("Buckling SF")

        # boundaries
        if not myBCs.isLshape:
            ax.set_xlim(np.min(nodes[:,0]), np.max(nodes[:,0]+0.05*np.max(nodes[:,0])))
            ax.set_ylim(np.min(nodes[:,1]), np.max(nodes[:,1]+0.05*np.max(nodes[:,1])))
        
        points = nodes[candidates[:, [0, 1]].astype(int), :].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1:2], points[1::2]], axis=1)
        
        ax.set_xlim(np.min(nodes[:,0]), np.max(nodes[:,0]*1.05))
        ax.set_ylim(np.min(nodes[:,1]), np.max(nodes[:,1]*1.05))
        
        norm = plt.Normalize(0,1)
        lc = mc.LineCollection(segments, cmap=Blue_mono, norm=norm)
        lc.set_array(buck)
        lc.set_linewidth(a[a >= threshold] * tk)
        line = ax.add_collection(lc)
        fig.colorbar(line, ax=[ax], location='bottom', orientation="horizontal")
        fig.tight_layout()

        if foldername == False:
            plt.show()
        else:
            plt.savefig(foldername+'fig4-B.pdf')

    ########### Multiload
    #Visualize truss
    @staticmethod
    def plotTruss_ML(myBCs, a, forces, threshold, vol, tk, foldername=False):
        """ Plot the ground stucture optimization solution """
        nodes = myBCs.nodes
        candidates = myBCs.ground_structure
    
        fig = plt.figure()
        plt.clf()
        plt.axis('off')
        plt.axis('equal')
        plt.title("Final Structure - Vol = %.3f" % (vol))
        
        candidates = myBCs.ground_structure[a >= threshold,:]
        q = forces[a >= threshold,:]
        grey = np.prod(forces, axis=1)[a >= threshold]
        a = a[a >= threshold]
        
        c = np.empty(a.size, dtype = object)
        tol = np.max(q) * 1e-04
        
        
        c[:] = 'grey' # no force
        c[(q[:,0]>tol) & (grey>tol)] = accent_r_1 # tension
        c[(q[:,0]<-tol)& (grey>tol)] = accent_b_1 # compression
        linewidth = a * tk
        # boundaries
        if not myBCs.isLshape:
            plt.plot([0,0], [0, myBCs.L[1]], 'lightgray', linestyle="dotted", linewidth=0.2)
            plt.plot([0, myBCs.L[0]], [myBCs.L[1], myBCs.L[1]], 'lightgray', linestyle="dotted", linewidth=0.2)
            plt.plot([myBCs.L[0], myBCs.L[0]], [myBCs.L[1], 0], 'lightgray', linestyle="dotted", linewidth=0.2)
            plt.plot([myBCs.L[0], 0], [0,0], 'lightgray', linestyle="dotted", linewidth=0.2)
        # optimized beams
        for i in range(len(a)):
            pos = nodes[candidates[i, [0, 1]].astype(int), :]
            plt.plot(pos[:, 0], pos[:, 1], c[i], linewidth = linewidth[i])
            plt.scatter(pos[:, 0], pos[:, 1], c = 'black', marker = 'o', edgecolors = 'none', zorder=2) # enforce the scatter plot after the plot
            
        fig.tight_layout()
        if foldername == False:
            plt.show()
        else:
            # tikzplotlib.save(foldername+'fig1-Topology.tex')
            plt.savefig(foldername+'fig1-Topology.pgf')
            plt.savefig(foldername+'fig1-Topology.pdf')
    
    @staticmethod    
    def plotTrussMulti_ML(myBCs, a_unred, forces, threshold, tk, foldername=False):
        """ Plot the ground stucture optimization solution stress """
        nodes = myBCs.nodes
        candidates = myBCs.ground_structure[a_unred >= threshold]
        
        n_load_cases = forces.shape[-1]
        
        for i in range(n_load_cases):
            fig, ax = plt.subplots()
            plt.axis('off')
            plt.axis('equal')
            q = forces[:,i]

            q = q[a_unred >= threshold]
            a = a_unred[a_unred >= threshold]
            
            c = np.empty(a.size, dtype = object)
        
        
            c[:] = 'darkgrey' # no force
            c[(q>1e-5*max(q))] = accent_r_1 # tension
            c[(q<1e-5*min(q))] = accent_b_1 # compression
            linewidth = a * tk
            
            points = nodes[candidates[:, [0, 1]].astype(int), :].reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1:2], points[1::2]], axis=1)
            
            ax.set_xlim(np.min(nodes[:,0]), np.max(nodes[:,0]+0.05*np.max(nodes[:,0])))
            ax.set_ylim(np.min(nodes[:,1]), np.max(nodes[:,1]+0.05*np.max(nodes[:,1])))
            
            lc = mc.LineCollection(segments, colors=c)
            lc.set_linewidth(linewidth)
            line = ax.add_collection(lc)
            
            if foldername == False:
                plt.show()
            else:
                plt.savefig(foldername+'fig1-Topology-LC{:03d}.pdf'.format(i+1))
    
    @staticmethod    
    def plotTrussStress_ML(myBCs, a_unred, forces, threshold, s_t, s_c, tk, foldername=False):
        """ Plot the ground stucture optimization solution stress """
        nodes = myBCs.nodes
        candidates = myBCs.ground_structure[a_unred >= threshold]
        
        n_load_cases = forces.shape[-1]
        
        for i in range(n_load_cases):
            fig, ax = plt.subplots()
            plt.axis('off')
            plt.axis('equal')
            plt.title("ABS stress")
            q = forces[:,i]

            q = q[a_unred >= threshold]
            a = a_unred[a_unred >= threshold]
            stress = q/a
            
            # Round to a specified number of significant digits
            significant_digits = 4
            decimals = significant_digits - int(math.floor(math.log10(np.abs(np.max(stress))))) - 1
            stress =  np.around(stress, decimals=decimals)
            
            points = nodes[candidates[:, [0, 1]].astype(int), :].reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1:2], points[1::2]], axis=1)
            
            ax.set_xlim(np.min(nodes[:,0]), np.max(nodes[:,0]+0.05*np.max(nodes[:,0])))
            ax.set_ylim(np.min(nodes[:,1]), np.max(nodes[:,1]+0.05*np.max(nodes[:,1])))
            
            norm = plt.Normalize(s_c, s_t)
            lc = mc.LineCollection(segments, cmap='coolwarm', norm=norm)
            lc.set_array(stress[a >= threshold])
            lc.set_linewidth(a[a >= threshold] * tk)
            line = ax.add_collection(lc)
            fig.colorbar(line, ax=[ax], location='bottom', orientation="horizontal")
            
            if foldername == False:
                plt.show()
            else:
                plt.savefig(foldername+'fig2-stress-LC{:03d}.pdf'.format(i+1))
    
    @staticmethod    
    def plotTrussBuckling_ML(myBCs, a_unred, forces, threshold, s_buck, tk, foldername=False):
        """ Plot the ground stucture optimization solution stress """
        nodes = myBCs.nodes
        candidates = myBCs.ground_structure[a_unred >= threshold]
        
        n_load_cases = forces.shape[-1]
        
        for i in range(n_load_cases):
            fig, ax = plt.subplots()
            plt.axis('off')
            plt.axis('equal')
            plt.title("Buckling SF")
            # Critical force
            q = forces[:,i]

            q = q[a_unred >= threshold]
            a = a_unred[a_unred >= threshold]
            q_crit = -s_buck*a**2/myBCs.ground_structure_length[a_unred >= threshold]**2
            
            buck = q/q_crit
            buck[buck<0] = 0 # Member in tension cannot buckle
            
            points = nodes[candidates[:, [0, 1]].astype(int), :].reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1:2], points[1::2]], axis=1)
            
            ax.set_xlim(np.min(nodes[:,0]), np.max(nodes[:,0]+0.05*np.max(nodes[:,0])))
            ax.set_ylim(np.min(nodes[:,1]), np.max(nodes[:,1]+0.05*np.max(nodes[:,1])))
            
            norm = plt.Normalize(0,1)
            lc = mc.LineCollection(segments, cmap=Blue_mono, norm=norm)
            lc.set_array(buck)
            lc.set_linewidth(a * tk)
            line = ax.add_collection(lc)
            fig.colorbar(line, ax=[ax], location='bottom', orientation="horizontal")
            
            if foldername == False:
                plt.show()
            else:
                plt.savefig(foldername+'fig4-B-LC{:03d}.pdf'.format(i+1))
                   
    @classmethod 
    def plotRoutine(cls, myBCs, a, q, U, vol, stress_tension_max, stress_compression_max, tol, s_buck, obj_hist, obj_hist_LP, cellsEquals, foldername, LP, GIF, a_cell=False):
        thick = 5 / max(a)
        tol *= max(a)
        if LP:
            folder = foldername + '/LP/'
            if not os.path.isdir(folder):
                    os.makedirs(folder)
        else:
            folder = foldername + '/'
                    
        cls.plotGroundStructure(myBCs, False, folder)
        cls.plotTruss(myBCs, a, q, max(a) * tol, vol, thick, folder)
        cls.plotTrussStress(myBCs, a, q, max(a) * tol, stress_tension_max, stress_compression_max, thick, folder)
        if sum(U) != 0:
            cls.plotTrussDeformation(myBCs, a, q, U, max(a) * tol, 1, folder, axis=False)
        cls.plotTrussRF(myBCs, a, q, max(a) * tol, stress_tension_max, stress_compression_max, thick, folder)
        if np.any(obj_hist) != False: 
            plotObjHistory(obj_hist, folder)
            if np.any(obj_hist_LP) != False: 
                plotObjHistory_combined(obj_hist, obj_hist_LP, folder)
        if GIF:
            generateGIF(folder,'animation')
    
    @classmethod    
    def plotRoutineBuckling(cls, myBCs, a, q, U, vol, stress_tension_max, stress_compression_max, tol, s_buck, obj_hist, obj_hist_LP, cellsEquals, foldername, LP, GIF, a_cell=False):
        thick = 5 / max(a)
        if LP:
            folder = foldername + '/LP/'
            if not os.path.isdir(folder):
                    os.makedirs(folder)
        else:
            folder = foldername + '/'
        cls.plotGroundStructure(myBCs, True, folder)
        cls.plotTruss(myBCs, a, q, max(a) * tol, vol, thick, folder)
        cls.plotTrussStress(myBCs, a, q, max(a) * tol, stress_tension_max, stress_compression_max, thick, folder)
        cls.plotTrussBucklingCheck(myBCs, a, q, max(a) * tol, s_buck, thick, folder)
        cls.plotTrussBuckling(myBCs, a, q, max(a) * tol, s_buck, thick, folder)
        if sum(U) != 0:
            cls.plotTrussDeformation(myBCs, a, q, U, max(a) * tol, 1, thick, folder, axis=False)
        cls.plotTrussRF(myBCs, a, q, max(a) * tol, stress_tension_max, stress_compression_max, thick, folder)
        if np.any(obj_hist) != False: 
            plotObjHistory(obj_hist, folder)
            if np.any(obj_hist_LP) != False: 
                plotObjHistory_combined(obj_hist, obj_hist_LP, folder)  
        if GIF:
            generateGIF(folder,'animation')
     
    @classmethod    
    def plotRoutineBuckling_ML(cls, myBCs, a, q, U, vol, stress_tension_max, stress_compression_max, tol, 
                            s_buck, obj_hist, foldername, LP, GIF, obj_hist_LP=False,
                            eq_hist=False, s_c_hist=False, s_t_hist=False, comp_hist=False, buck_hist=False):
        thick = 5 / max(a)
        tol *= max(a)
        if LP:
            folder = foldername + '/LP/'
            if not os.path.isdir(folder):
                    os.makedirs(folder)
        else:
            folder = foldername + '/'
            
        n_load_cases = myBCs.R.shape[1]
        
        for i in range(n_load_cases):
            cls.plotGroundStructure(myBCs, foldername=folder, load_case=i, mag=0.1)
        cls.plotTruss_ML(myBCs, a, q, tol, vol, thick, foldername=folder)
        cls.plotTrussMulti_ML(myBCs, a, q, tol, thick, foldername=folder)
        cls.plotTrussStress_ML(myBCs, a, q, tol, stress_tension_max, stress_compression_max, thick, foldername=folder)
        cls.plotTrussBuckling_ML(myBCs, a, q, tol, s_buck, thick, foldername=folder)
            
        if np.sum(U[:]) != 0:
            cls.plotTrussDeformation(myBCs, a, q, U, tol, 1, thick, foldername=folder, axis=False)
        
        if np.any(obj_hist) != False: 
            plotObjHistory(obj_hist, foldername=folder)
            if np.any(eq_hist) != False:
                plotConstHistory(eq_hist, s_c_hist, s_t_hist, comp_hist, buck_hist, folder)
            if np.any(obj_hist_LP) != False: 
                plotObjHistory_combined(obj_hist, obj_hist_LP, folder)
        if GIF:
            generateGIF(folder+'gif_tmp_top','topology')   
      
############################
# 3D plotting
############################
class plot3D():
    # Visualize truss
    @staticmethod
    def plotTruss(myBCs, a, forces, threshold, vol, tk, foldername = False, gif=False):
        """ Plot the ground stucture optimization solution """
        nodes = myBCs.nodes
        candidates = myBCs.ground_structure[a >= threshold,:]
        q = forces[a >= threshold]
        a = a[a >= threshold]
        color = np.empty(a.size, dtype = object)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d', proj_type = 'ortho')
        plt.axis('off')
        plt.title("Final Structure - Vol = %.3f" % (vol))
        # optimized beams
        tol = np.max(q) * 1e-04
        color[:] = 'grey' # no force
        color[q>tol] = accent_r_1 # tension
        color[q<-tol] = accent_b_1 # compression
        linewidth = a * tk
        points = nodes[candidates[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3)
        # Normalization to [0-Dim]
        points[:,:,0] = (points[:,:,0] - np.min(points[:,:,0]))
        points[:,:,1] = (points[:,:,1] - np.min(points[:,:,1]))
        points[:,:,2] = (points[:,:,2] - np.min(points[:,:,2]))
        segments = np.concatenate([points[:-1:2], points[1::2]], axis=1)

        lc = Line3DCollection(segments, colors = color)
        lc.set_linewidth(linewidth)
        ax.add_collection3d(lc, zs=nodes[:,2], zdir='z')
        
        xx = myBCs.L[0]
        yy = myBCs.L[1]
        zz = myBCs.L[2]
        
        #draw cube
        r1 = [0, xx]
        r2 = [0, yy]
        r3 = [0, zz]

        for s, e in combinations(np.array(list(product(r1,r2,r3))), 2):
            s=np.array(s)
            e=np.array(e)
            if np.linalg.norm(s-e) == r1[1] or np.linalg.norm(s-e) == r2[1] or np.linalg.norm(s-e) == r3[1]:
                ax.plot3D(*zip(s,e), color="lightgrey", linestyle="dotted", linewidth=0.2) 
        
        ax.set_xlim([0, myBCs.L[0]*1.05])
        ax.set_ylim([0, myBCs.L[1]*1.05])
        ax.set_zlim([0, myBCs.L[2]*1.05])
        ax.set_box_aspect((myBCs.L[0]*1.05,myBCs.L[1]*1.05,myBCs.L[2]*1.05))
        fig.tight_layout()

        if foldername!=False:
            plt.savefig(foldername+'fig1-Topology.pdf')
            # Orthogonal views
            plt.title(' ')
            ax.view_init(azim=0, elev=0)    # x-z plane
            plt.savefig(foldername+'fig1-Topology_xz.pdf')
            ax.view_init(azim=-90, elev=90)   # x-y plane
            plt.savefig(foldername+'fig1-Topology_xy.pdf')
            ax.view_init(azim=-90, elev=0)      # y-z plane   
            plt.savefig(foldername+'fig1-Topology_yz.pdf')
            ax.view_init(azim=-70, elev=30)      # view 1   
            plt.savefig(foldername+'fig1-Topology_ortho1.pdf')
            ax.view_init(azim=70, elev=30)      # view 1   
            plt.savefig(foldername+'fig1-Topology_ortho2.pdf')
        else:
            plt.show()
            
        if gif:
            fig.tight_layout()
            plt.title(' ')
            if not os.path.isdir(foldername+'/'+'gif_tmp_top'):
                os.makedirs(foldername+'/'+'gif_tmp_top')
            for ii in range(0,360,2):
                ax.view_init(elev=25., azim=ii)
                plt.savefig(foldername+'/'+'gif_tmp_top'+'/'+'gif{0:03d}.png'.format(ii), dpi=300) 

    # Visualize the reserve factor 
    @staticmethod   
    def plotTrussRF(myBCs, a, q, threshold, stress_t, stress_c, tk, foldername):
        """ Plot the ground stucture optimization solution reserve factor, defined as the percentage of the max charge """
        nodes = myBCs.nodes
        candidates = myBCs.ground_structure[a >= threshold]
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        plt.axis('off')
        plt.title("Reserve factor")

        q = q[a >= threshold]
        a = a[a >= threshold]
        q[abs(q) < max(abs(q)) * 10e-3] = 0 # eliminates 0/0 cases
        efficency = q/a
        efficency[efficency > 0] /= stress_t
        efficency[efficency < 0] /= stress_c
        
        points = nodes[candidates[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1:2], points[1::2]], axis=1)
        
        norm = plt.Normalize(efficency.min(), efficency.max())
        lc = Line3DCollection(segments, cmap='RdYlGn', norm=norm)
        lc.set_array(efficency[a >= threshold])
        lc.set_linewidth(a[a >= threshold] * tk)
        line = ax.add_collection(lc)
        fig.colorbar(line, ax=ax)
        ax.set_xlim([0, myBCs.L[0]*1.05])
        ax.set_ylim([0, myBCs.L[1]*1.05])
        ax.set_zlim([0, myBCs.L[2]*1.05])
        ax.set_box_aspect((myBCs.L[0],myBCs.L[1],myBCs.L[2]))
        fig.tight_layout()
        # optimized beams
        plt.savefig(foldername+'fig7-safety.pdf')
        
    # Visualize the stress  
    @staticmethod 
    def plotTrussStress(myBCs, a, q, threshold, tk, foldername = False):
        """ Plot the ground stucture optimization solution stress """
        nodes = myBCs.nodes
        candidates = myBCs.ground_structure[a >= threshold]
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d', proj_type = 'ortho')
        plt.axis('off')
        plt.title("Stress plot [MPa]")
        

        q = q[a >= threshold]
        a = a[a >= threshold]
        stress = q/a
        
        # Round to a specified number of significant digits
        significant_digits = 4
        decimals = significant_digits - int(math.floor(math.log10(np.abs(np.max(stress))))) - 1
        stress =  np.around(stress, decimals=decimals)
        
        points = nodes[candidates[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1:2], points[1::2]], axis=1)
        
        norm = plt.Normalize(stress.min(), stress.max())
        lc = Line3DCollection(segments, cmap='coolwarm', norm=norm)
        lc.set_array(stress)
        lc.set_linewidth(a * tk)
        line = ax.add_collection(lc)
    
        fig.colorbar(line, ax=[ax], location='bottom', orientation="horizontal", fraction=0.046)
        
        ax.set_xlim([0, myBCs.L[0]*1.05])
        ax.set_ylim([0, myBCs.L[1]*1.05])
        ax.set_zlim([0, myBCs.L[2]*1.05])
        ax.set_box_aspect((myBCs.L[0],myBCs.L[1],myBCs.L[2]))
        ax.view_init(azim=-70, elev=30)      # view 1
        # optimized beams
        if foldername!=False:
            plt.savefig(foldername+'fig2-stress.pdf')
        else:
            plt.show()
            
    # Visualize only the different cell topologies
    @staticmethod    
    def PlotTrussCellule(myBCs, a_cell, threshold, tk, foldername, axis, gif):
        """ Plot the ground stucture optimization cell solution """
        nodes = myBCs.nodes
        topol = a_cell.shape[0]
        xx = myBCs.cell_size[0]
        yy = myBCs.cell_size[1]
        zz = myBCs.cell_size[2]
        
        for i in np.arange(topol):
            if not np.any(myBCs.ground_stucture_list_cell[i]):
                continue
            if myBCs.isReduced:
                cellule = []
                for l in myBCs.ground_stucture_list_cell[i]:
                    a = np.array(myBCs.ground_stucture_list_not_reduced).tolist() # neded because compatibility problem between iterators and lists
                    cellule.append(a.index(l))
                    mask = a_cell[i][:len(cellule)] >= threshold
                candidates = myBCs.ground_structure_not_reduced[cellule][mask]  
            else:
                cellule = [myBCs.ground_stucture_list.index(l) for l in myBCs.ground_stucture_list_cell]
                candidates = myBCs.ground_structure[cellule][a_cell[i] >= threshold]
            a_plot = a_cell[i][a_cell[i] >= threshold]
            # checking whether Numpy array is empty
            if not np.any(a_plot):
                continue
            color = np.empty(a_plot.size, dtype = object)
            color[:] = 'black'
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d', proj_type = 'ortho')
            if not axis:
                plt.axis('off')
            else:
                ax.set_xlabel('X axis [mm]',)
                ax.set_ylabel('Y axis [mm]')
                ax.set_zlabel('Z axis [mm]')
                
            plt.title("Final cell topology {0}".format(i+1))
            linewidth = a_plot * tk
            if myBCs.isReduced:
                points = myBCs.nodes_not_reduced[candidates[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3) 
                # Normalization to [0-cellDim]
                points[:,:,0] = (points[:,:,0] - np.min(points[:,:,0])) * xx/np.max(points[:,:,0] - np.min(points[:,:,0]))
                points[:,:,1] = (points[:,:,1] - np.min(points[:,:,1])) * yy/np.max(points[:,:,1] - np.min(points[:,:,1]))
                points[:,:,2] = (points[:,:,2] - np.min(points[:,:,2])) * zz/np.max(points[:,:,2] - np.min(points[:,:,2]))
            else:
                points = nodes[candidates[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3)
                # Normalization to [0-cellDim]
                points[:,:,0] = (points[:,:,0] - np.min(points[:,:,0])) * xx/np.max(points[:,:,0] - np.min(points[:,:,0]))
                points[:,:,1] = (points[:,:,1] - np.min(points[:,:,1])) * yy/np.max(points[:,:,1] - np.min(points[:,:,1]))
                points[:,:,2] = (points[:,:,2] - np.min(points[:,:,2])) * zz/np.max(points[:,:,2] - np.min(points[:,:,2]))
            segments = np.concatenate([points[:-1:2], points[1::2]], axis=1)

            lc = Line3DCollection(segments, colors = color)
            lc.set_linewidth(linewidth)
            ax.add_collection3d(lc)
            ax.set_frame_on(True)
            ax.set_xlim([min(points[:,:,0]), max(points[:,:,0]*1.05)])
            ax.set_ylim([min(points[:,:,1]), max(points[:,:,1]*1.05)])
            ax.set_zlim([min(points[:,:,2]), max(points[:,:,2]*1.05)])
            ax.set_box_aspect((xx,yy,zz))
            
            #draw cube
            r1 = [0, xx]
            r2 = [0, yy]
            r3 = [0, zz]

            for s, e in combinations(np.array(list(product(r1,r2,r3))), 2):
                s=np.array(s)
                e=np.array(e)
                if np.linalg.norm(s-e) == r1[1] or np.linalg.norm(s-e) == r2[1] or np.linalg.norm(s-e) == r3[1]:
                    ax.plot3D(*zip(s,e), color="lightblue", linestyle="dotted")  
                        
            
            if not axis:
                fig.tight_layout()
                plt.savefig(foldername+'fig8-{:03d}_Cell-noAx.pdf'.format(i))
                # Orthogonal views
                plt.title(' ')
                ax.view_init(azim=0, elev=0)    # x-z plane
                plt.savefig(foldername+'fig8-{:03d}_Cell_xz.pdf'.format(i))
                ax.view_init(azim=-90, elev=90)   # x-y plane
                plt.savefig(foldername+'fig8-{:03d}_Cell_xy.pdf'.format(i))
                ax.view_init(azim=-90, elev=0)      # y-z plane   
                plt.savefig(foldername+'fig8-{:03d}_Cell_yz.pdf'.format(i))
                ax.view_init(azim=-70, elev=30)      # view 1   
                plt.savefig(foldername+'fig8-{:03d}_Cell_ortho1.pdf'.format(i))
                ax.view_init(azim=70, elev=30)      # view 2   
                plt.savefig(foldername+'fig8-{:03d}_Cell_ortho2.pdf'.format(i))
            else:
                plt.savefig(foldername+'fig9-{:03d}_Cell-Ax.pdf'.format(i))
                  
            if gif and not axis:
                fig.tight_layout()
                plt.title(' ')
                if not os.path.isdir(foldername+'/'+'gif_tmp_cell_{:03d}'.format(i)):
                    os.makedirs(foldername+'/'+'gif_tmp_cell_{:03d}'.format(i))
                for ii in range(0,360,2):
                    ax.view_init(elev=10., azim=ii)
                    plt.savefig(foldername+'/'+'gif_tmp_cell_{:03d}'.format(i)+'/'+'gif{0:03d}.png'.format(ii), dpi=300)  
     
    # Visualize only the different cell topologies
    @staticmethod    
    def PlotTrussCellule_Free_form(myBCs, a, threshold, tk, foldername, axis, gif):
        """ Plot the ground stucture optimization cell solution """
        topologies = np.unique(myBCs.ground_structure[:,2]) # List the different types of topologies of the structure
        for ID, top_ID in np.ndenumerate(topologies): # Evaluate the hull dimensions
            ID = ID[0]
            if top_ID == 1: 
                xx,yy,zz = myBCs.cell_size
            elif top_ID == 2:
                xx,yy,zz = myBCs.L
                xx /= myBCs.number_section 
            elif top_ID == 3:
                xx,yy,zz = myBCs.L
            else:
                break
            
            # Evaluate the setions to plot
            a_plot = a[(myBCs.ground_structure[:,2]==top_ID) & (myBCs.ground_structure[:,3]==0)]
            a_plot = a_plot[a_plot>threshold] 
            candidates = myBCs.ground_structure[(myBCs.ground_structure[:,2]==top_ID) & (myBCs.ground_structure[:,3]==0),:2] # Start and end nodes of the candidates of the top_ID topology to plot
            nodes = myBCs.nodes
            points = nodes[candidates[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3)
            # Normalization to [0-cellDim]
            points[:,:,0] = (points[:,:,0] - np.min(points[:,:,0])) * xx/np.max(points[:,:,0] - np.min(points[:,:,0]))
            points[:,:,1] = (points[:,:,1] - np.min(points[:,:,1])) * yy/np.max(points[:,:,1] - np.min(points[:,:,1]))
            points[:,:,2] = (points[:,:,2] - np.min(points[:,:,2])) * zz/np.max(points[:,:,2] - np.min(points[:,:,2]))
            segments = np.concatenate([points[:-1:2], points[1::2]], axis=1)
            
            color = np.empty(a_plot.size, dtype = object)
            color[:] = 'black'
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d', proj_type = 'ortho')
            if not axis:
                plt.axis('off')
            else:
                ax.set_xlabel('X axis [mm]',)
                ax.set_ylabel('Y axis [mm]')
                ax.set_zlabel('Z axis [mm]')
                
            plt.title("Final cell topology {0}".format(ID+1))
            linewidth = a_plot * tk
            
            lc = Line3DCollection(segments, colors = color)
            lc.set_linewidth(linewidth)
            ax.add_collection3d(lc)
            ax.set_frame_on(True)
            ax.set_xlim([min(points[:,:,0]), max(points[:,:,0]*1.05)])
            ax.set_ylim([min(points[:,:,1]), max(points[:,:,1]*1.05)])
            ax.set_zlim([min(points[:,:,2]), max(points[:,:,2]*1.05)])
            ax.set_box_aspect((xx,yy,zz))
            
            # Draw hull
            r1 = [0, xx]
            r2 = [0, yy]
            r3 = [0, zz]

            for s, e in combinations(np.array(list(product(r1,r2,r3))), 2):
                s=np.array(s)
                e=np.array(e)
                if np.linalg.norm(s-e) == r1[1] or np.linalg.norm(s-e) == r2[1] or np.linalg.norm(s-e) == r3[1]:
                    ax.plot3D(*zip(s,e), color="lightblue", linestyle="dotted")  
                        
            
            if not axis:
                fig.tight_layout()
                plt.savefig(foldername+'fig8-{:03d}_Cell-noAx.pdf'.format(ID))
                # Orthogonal views
                plt.title(' ')
                ax.view_init(azim=0, elev=0)    # x-z plane
                plt.savefig(foldername+'fig8-{:03d}_Cell_xz.pdf'.format(ID))
                ax.view_init(azim=-90, elev=90)   # x-y plane
                plt.savefig(foldername+'fig8-{:03d}_Cell_xy.pdf'.format(ID))
                ax.view_init(azim=-90, elev=0)      # y-z plane   
                plt.savefig(foldername+'fig8-{:03d}_Cell_yz.pdf'.format(ID))
                ax.view_init(azim=-70, elev=30)      # view 1   
                plt.savefig(foldername+'fig8-{:03d}_Cell_ortho1.pdf'.format(ID))
                ax.view_init(azim=70, elev=30)      # view 2   
                plt.savefig(foldername+'fig8-{:03d}_Cell_ortho2.pdf'.format(ID))
            else:
                plt.savefig(foldername+'fig9-{:03d}_Cell-Ax.pdf'.format(ID))
                  
            if gif and not axis:
                fig.tight_layout()
                plt.title(' ')
                if not os.path.isdir(foldername+'/'+'gif_tmp_cell_{:03d}'.format(ID)):
                    os.makedirs(foldername+'/'+'gif_tmp_cell_{:03d}'.format(ID))
                for ii in range(0,360,2):
                    ax.view_init(elev=10., azim=ii)
                    plt.savefig(foldername+'/'+'gif_tmp_cell_{:03d}'.format(ID)+'/'+'gif{0:03d}.png'.format(ii), dpi=300)

    #Visualize initial truss
    @staticmethod
    def plotGroundStructure(myBCs, foldername, load_case = -1, mag=0.1):
        nodes = myBCs.nodes
        candidates = myBCs.ground_structure
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d', computed_zorder=False)
        plt.axis('off')
        plt.title("Starting Ground Structure - N. Elements = " + str(len(candidates))) 
        points = nodes[candidates[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1:2], points[1::2]], axis=1)

        lc = Line3DCollection(segments, linewidths = 0.3)
        ax.add_collection3d(lc, zs=nodes[:,2], zdir='z')
        
        # BCs
        fixed_nodes = np.unique(myBCs.fixed_dofs // 3)
        x,y,z = myBCs.nodes[fixed_nodes,:][:,[0,1,2]].T
        ax.scatter(x,y,z, s=10, c='black', alpha=1, zorder=0.5)
        # Loads
        load_nodes = myBCs.force_dofs // 3
        x,y,z = myBCs.nodes[load_nodes,:][:,[0,1,2]].T
        if load_case != -1:
            R = myBCs.R[:,load_case].copy()
        else:
            R = myBCs.R.copy()
            
        magnit = R[R!=0]
        direction = myBCs.force_dofs % 3
        v = np.zeros((load_nodes.shape[0],3))
        v[range(load_nodes.shape[0]),direction] = magnit
        # normalization of v
        v_norm = np.max(np.abs(myBCs.R))
        v = v/v_norm * mag * np.max(myBCs.L)
        ax.quiver(x, y, z, v[:,0], v[:,1], v[:,2], color='red', zorder=30, linewidth = 0.7)
        #ax.scatter(x, y, z, s=200, c="red", zorder=20)
        
        ax.set_xlim([0, myBCs.L[0]*1.05])
        ax.set_ylim([0, myBCs.L[1]*1.05])
        ax.set_zlim([0, myBCs.L[2]*1.05])
        ax.set_box_aspect((myBCs.L[0],myBCs.L[1],myBCs.L[2]))
        
        fig.tight_layout()
        
        if foldername!=False:
            if load_case != -1:
                plt.savefig(foldername+'fig0-GS-LC{:03d}.pdf'.format(load_case+1))
            else:
                plt.savefig(foldername+'fig0-GS.pdf')
        else:
            plt.show()

    # Visualize ground structure deformation 
    @staticmethod 
    def plotTrussDeformation(myBCs, a, Q, U, candidates_bars, magnitude, foldername, axis):
        """ Plot the ground stucture optimized deformations """
        nodes = myBCs.nodes

        if Q.ndim == 1:
            n_load_case = 1
        else:
            n_load_case = Q.shape[-1]
        candidates = myBCs.ground_structure[candidates_bars,:]
        
        a = a[candidates_bars]
        for i in range(n_load_case):
            if Q.ndim == 1:
                u = U.reshape(-1,3)
            else:
                u = U[:,i].copy().reshape(-1,3)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d', proj_type = 'ortho')
            if not axis:
                plt.axis('off')
            else:
                ax.set_xlabel('X axis')
                ax.set_ylabel('Y axis')
                ax.set_zlabel('Z axis')

            color = np.empty(a.size, dtype = object)
            color_undef = np.empty(a.size, dtype = object)
            
            plt.title("Final Structure deformation - Magnitude %i" % (magnitude))
            # optimized beams
            color[:] = 'black'
            color_undef = 'lightgrey' # Color of undeformed
            linewidth = 0.3

            # Undeformed
            points_undef = nodes[candidates[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3)
            segments_undef = np.concatenate([points_undef[:-1:2], points_undef[1::2]], axis=1)
            # Deformed
            points = nodes[candidates[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3) + magnitude*u[candidates[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1:2], points[1::2]], axis=1)

            # Undeformed
            lc_undef = Line3DCollection(segments_undef, colors = color_undef,alpha=0.5, zorder=0)
            lc_undef.set_linewidth(linewidth)
            ax.add_collection3d(lc_undef, zs=nodes[:,2], zdir='z')
            # Deformed
            lc = Line3DCollection(segments, colors = color)
            lc.set_linewidth(linewidth)
            ax.add_collection3d(lc, zs=nodes[:,2], zdir='z')
            ax.set_xlim([min(points[:,:,0]), max(points[:,:,0]*1.05)])
            ax.set_ylim([min(points[:,:,1]), max(points[:,:,1]*1.05)])
            ax.set_zlim([min(points[:,:,2]), max(points[:,:,2]*1.05)])
            ax.set_box_aspect((float(max(points[:,:,0])-min(points[:,:,0])),float(max(points[:,:,1])-min(points[:,:,1])),float(max(points[:,:,2])-min(points[:,:,2]))))
            fig.tight_layout()
            if U.ndim == 1:
                if not axis:
                    plt.title(' ')
                    plt.savefig(foldername+'fig5-U_NoAxis-Mag={0}.pdf'.format(magnitude))
                    ax.view_init(azim=0, elev=0)    # x-z plane
                    plt.savefig(foldername+'fig5-U_NoAxis-Mag={0}_xz.pdf'.format(magnitude))
                else:
                    plt.title(' ')
                    plt.savefig(foldername+'fig6-U_Axis-Mag={0}.pdf'.format(magnitude))
            else:
                if not axis:
                    plt.title(' ')
                    plt.savefig(foldername+'fig5-U_NoAxis-LC{0:03d}-Mag={1}.pdf'.format(i+1,magnitude))
                    ax.view_init(azim=0, elev=0)    # x-z plane
                    plt.savefig(foldername+'fig5-U_NoAxis-LC{0:03d}-Mag={1}_xz.pdf'.format(i+1,magnitude))
                    ax.view_init(azim=0, elev=90)    # x-y plane
                    plt.savefig(foldername+'fig5-U_NoAxis-LC{0:03d}-Mag={1}_xy.pdf'.format(i+1,magnitude))
                else:
                    plt.title(' ')
                    plt.savefig(foldername+'fig6-U_Axis-LC{0:03d}-Mag={1}.pdf'.format(i+1,magnitude))
    
    @staticmethod 
    def plotTrussDeformation_FEM(myBCs, a, Q, U, candidates_bars, magnitude, foldername, axis=False, title=False):
        """ Plot the ground stucture optimized deformations """
        nodes = myBCs.nodes

        if Q.ndim == 1:
            n_load_case = 1
        else:
            n_load_case = Q.shape[-1]
        candidates = myBCs.ground_structure[candidates_bars,:]
        
        a = a[candidates_bars]
        for i in range(n_load_case):
            if Q.ndim == 1:
                u = U.reshape(-1,3)
            else:
                u = U[:,i].copy().reshape(-1,3)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d', proj_type = 'ortho')
            if not axis:
                plt.axis('off')
            else:
                ax.set_xlabel('X axis')
                ax.set_ylabel('Y axis')
                ax.set_zlabel('Z axis')

            color = np.empty(a.size, dtype = object)
            color_undef = np.empty(a.size, dtype = object)
            
            plt.title("Final Structure deformation - Magnitude %i" % (magnitude))
            # optimized beams
            color[:] = 'black'
            color_undef = 'lightgrey' # Color of undeformed
            linewidth = 0.3

            # Undeformed
            points_undef = nodes[candidates[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3)
            segments_undef = np.concatenate([points_undef[:-1:2], points_undef[1::2]], axis=1)
            # Deformed
            points = nodes[candidates[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3) + magnitude*u[candidates[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1:2], points[1::2]], axis=1)

            # Undeformed
            lc_undef = Line3DCollection(segments_undef, colors = color_undef,alpha=0.5, zorder=0)
            lc_undef.set_linewidth(linewidth)
            ax.add_collection3d(lc_undef, zs=nodes[:,2], zdir='z')
            # Deformed
            lc = Line3DCollection(segments, colors = color)
            lc.set_linewidth(linewidth)
            ax.add_collection3d(lc, zs=nodes[:,2], zdir='z')
            ax.set_xlim([min(points[:,:,0]), max(points[:,:,0]*1.05)])
            ax.set_ylim([min(points[:,:,1]), max(points[:,:,1]*1.05)])
            ax.set_zlim([min(points[:,:,2]), max(points[:,:,2]*1.05)])
            ax.set_box_aspect((float(max(points[:,:,0])-min(points[:,:,0])),float(max(points[:,:,1])-min(points[:,:,1])),float(max(points[:,:,2])-min(points[:,:,2]))))
            fig.tight_layout()
            if U.ndim == 1:
                if not axis:
                    plt.title(' ')
                    plt.savefig(foldername+'fig5_FEM-U_NoAxis-Mag={0}.pdf'.format(magnitude))
                    ax.view_init(azim=0, elev=0)    # x-z plane
                    plt.savefig(foldername+'fig5_FEM-U_NoAxis-Mag={0}_xz.pdf'.format(magnitude))
                else:
                    plt.title(' ')
                    plt.savefig(foldername+'fig6_FEM-U_Axis-Mag={0}.pdf'.format(magnitude))
            else:
                if not axis:
                    if title==False:
                        plt.title(' ')
                        plt.savefig(foldername+'fig5_FEM-U_NoAxis-LC{0:03d}-Mag={1}.pdf'.format(i+1,magnitude))
                        ax.view_init(azim=0, elev=0)    # x-z plane
                        plt.savefig(foldername+'fig5_FEM-U_NoAxis-LC{0:03d}-Mag={1}_xz.pdf'.format(i+1,magnitude))
                        ax.view_init(azim=0, elev=90)    # x-y plane
                        plt.savefig(foldername+'fig5_FEM-U_NoAxis-LC{0:03d}-Mag={1}_xy.pdf'.format(i+1,magnitude))
                    else:
                        plt.savefig(foldername+title+'{0:03d}.pdf'.format(i+1))
                else:
                    plt.title(' ')
                    plt.savefig(foldername+'fig6_FEM-U_Axis-LC{0:03d}-Mag={1}.pdf'.format(i+1,magnitude))
                  
    # Visualize truss buckling
    @staticmethod
    def plotTrussBucklingCheck(myBCs, a, forces, threshold, s_buck, tk, foldername = False):
        """ Plot the ground stucture optimization solution """
        nodes = myBCs.nodes
        candidates = myBCs.ground_structure[a >= threshold,:]
        q = forces[a >= threshold]
        # Critical force
        q_crit = -s_buck*a[a >= threshold]**2/myBCs.ground_structure_length[a >= threshold]**2
        a = a[a >= threshold]
        color = np.empty(a.size, dtype = object)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        plt.axis('off')
        plt.title("Buckling check")
        
        # optimized beams
        color[q<q_crit*1.01] = 'r' # Buckled
        color[q>q_crit*1.01] = 'black' # Ok
        linewidth = a * tk
        points = nodes[candidates[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1:2], points[1::2]], axis=1)

        lc = Line3DCollection(segments, colors = color)
        lc.set_linewidth(linewidth)
        ax.add_collection3d(lc, zs=nodes[:,2], zdir='z')
        ax.set_xlim([0, myBCs.L[0]*1.05])
        ax.set_ylim([0, myBCs.L[1]*1.05])
        ax.set_zlim([0, myBCs.L[2]*1.05])
        ax.set_box_aspect((myBCs.L[0],myBCs.L[1],myBCs.L[2]))
        fig.tight_layout()
        
        if foldername!=False:
            plt.savefig(foldername+'fig3-B_check.pdf')
        else:
            plt.show()
        
    # Visualize the buckling factor 
    @staticmethod   
    def plotTrussBuckling(myBCs, a, forces, threshold, s_buck, tk, foldername = False):
        """ Plot the ground stucture optimization solution efficency, defined as the percentage of the max charge """
        nodes = myBCs.nodes
        candidates = myBCs.ground_structure[a >= threshold,:]
        q = forces[a >= threshold]
        # Critical force
        q_crit = -s_buck*a[a >= threshold]**2/myBCs.ground_structure_length[a >= threshold]**2
        a = a[a >= threshold]
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d', proj_type = 'ortho')
        plt.axis('off')
        plt.title("Buckling SF")
        
        # optimized beams
        buck = q/q_crit
        buck[buck<0] = 0 # Member in tension cannot buckle
        
        points = nodes[candidates[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1:2], points[1::2]], axis=1)
        
        norm = plt.Normalize(buck.min(), buck.max())
        lc = Line3DCollection(segments, cmap='copper', norm=norm)
        lc.set_array(buck[a >= threshold])
        lc.set_linewidth(a[a >= threshold] * tk)
        line = ax.add_collection(lc)
        fig.colorbar(line, ax=[ax], location='bottom', orientation="horizontal", fraction=0.046)
        ax.set_xlim([0, myBCs.L[0]*1.05])
        ax.set_ylim([0, myBCs.L[1]*1.05])
        ax.set_zlim([0, myBCs.L[2]*1.05])
        ax.set_box_aspect((myBCs.L[0],myBCs.L[1],myBCs.L[2]))
        # optimized beams
        ax.view_init(azim=-70, elev=30)      # view 1
        plt.savefig(foldername+'fig4-B.pdf')
    
    ########### Multiload
    #Visualize truss
    @staticmethod
    def plotTruss_ML(myBCs, a_unred, forces, candidates_bars, vol, tk, gif=False, foldername=False, title=False):
        """ Plot the ground stucture optimization solution """
        nodes = myBCs.nodes
        n_load_cases = myBCs.R.shape[-1]
    
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d', proj_type = 'ortho')
        plt.axis('off')
        plt.title("Final Structure - Vol = %.3f" % (vol))
        
        candidates = myBCs.ground_structure[candidates_bars,:]
        q = forces[candidates_bars]
        sum_grey = np.sum(np.sign(q), axis=1).astype(int)
        grey = np.ones(sum_grey.size,dtype=bool)
        grey[(sum_grey==n_load_cases) | (sum_grey==-n_load_cases)] = False
        
        a = a_unred[candidates_bars]
        
        c = np.empty(a.size, dtype = object)
        
        
        c[:] = 'darkgrey' # no force
        c[(q[:,0]>0) & (~grey)] = accent_r_1 # tension
        c[(q[:,0]<0) & (~grey)] = accent_b_1 # compression
        linewidth = a * tk
        
        points = nodes[candidates[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3)


        # Normalization to [0-Dim]
        points[:,:,0] = (points[:,:,0] - np.min(points[:,:,0]))
        points[:,:,1] = (points[:,:,1] - np.min(points[:,:,1]))
        points[:,:,2] = (points[:,:,2] - np.min(points[:,:,2]))
        segments = np.concatenate([points[:-1:2], points[1::2]], axis=1)
        # optimized beams
        lc = Line3DCollection(segments, colors = c)
        lc.set_linewidth(linewidth)
        ax.add_collection3d(lc, zs=nodes[:,2], zdir='z')
         
        ax.set_xlim([0, myBCs.L[0]*1.05])
        ax.set_ylim([0, myBCs.L[1]*1.05])
        ax.set_zlim([0, myBCs.L[2]*1.05])
        ax.set_box_aspect((myBCs.L[0]*1.05,myBCs.L[1]*1.05,myBCs.L[2]*1.05))
        fig.tight_layout()
        
        if foldername!=False:
            if title==False:
                plt.savefig(foldername+'fig1-Topology.pdf')
                # Orthogonal views
                plt.title(' ')
                ax.view_init(azim=0, elev=0)    # x-z plane
                plt.savefig(foldername+'fig1-Topology_XZ.pdf')
                ax.view_init(azim=0, elev=90)   # x-y plane
                plt.savefig(foldername+'fig1-Topology_XY.pdf')
                ax.view_init(azim=-90, elev=0)      # y-z plane   
                plt.savefig(foldername+'fig1-Topology_YZ.pdf')
                ax.view_init(azim=0, elev=30)      # view 1   
                plt.savefig(foldername+'fig1-Topology_V1.pdf')
                ax.view_init(azim=70, elev=30)      # view 1   
                plt.savefig(foldername+'fig1-Topology_V2.pdf')
            else:
                plt.savefig(foldername+title)
        else:
            plt.show()
            
        if gif:
            fig.tight_layout()
            plt.title(' ')
            if not os.path.isdir(foldername+'/'+'gif_tmp_top'):
                os.makedirs(foldername+'/'+'gif_tmp_top')
            for ii in range(0,360,2):
                ax.view_init(elev=25., azim=ii)
                plt.savefig(foldername+'/'+'gif_tmp_top'+'/'+'gif{0:03d}.png'.format(ii), dpi=300)
                
    @staticmethod    
    def plotTrussMulti_ML(myBCs, a_unred, forces, candidates_bars, tk, foldername=False):
        """ Plot the ground stucture optimization solution stress """
        nodes = myBCs.nodes
        candidates = myBCs.ground_structure[candidates_bars]
        
        n_load_cases = forces.shape[-1]
        
        for i in range(n_load_cases):
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d', proj_type = 'ortho')
            plt.axis('off')
            plt.title("Topology")
            q = forces[:,i]

            q = q[candidates_bars]         
            a = a_unred[candidates_bars]
            
            c = np.empty(a.size, dtype = object)
        
        
            c[:] = 'darkgrey' # no force
            c[(q>0)] = accent_r_1 # tension
            c[(q<0)] = accent_b_1 # compression
            linewidth = a * tk
            
            
            points = nodes[candidates[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3)
            # Normalization to [0-Dim]
            points[:,:,0] = (points[:,:,0] - np.min(points[:,:,0]))
            points[:,:,1] = (points[:,:,1] - np.min(points[:,:,1]))
            points[:,:,2] = (points[:,:,2] - np.min(points[:,:,2]))
            segments = np.concatenate([points[:-1:2], points[1::2]], axis=1)

            # optimized beams
            lc = Line3DCollection(segments, colors = c)
            lc.set_linewidth(linewidth)
            ax.add_collection3d(lc, zs=nodes[:,2], zdir='z')
            
            ax.set_xlim([min(points[:,:,0]), max(points[:,:,0]*1.05)])
            ax.set_ylim([min(points[:,:,1]), max(points[:,:,1]*1.05)])
            ax.set_zlim([min(points[:,:,2]), max(points[:,:,2]*1.05)])
            ax.set_box_aspect((myBCs.L[0]*1.05,myBCs.L[1]*1.05,myBCs.L[2]*1.05))
            fig.tight_layout()
            
            if foldername == False:
                plt.show()
            else:
                # tikzplotlib.save(foldername+'fig2-stress-LC{:03d}.tex'.format(i+1))
                plt.savefig(foldername+'fig1-topology-LC{:03d}.pdf'.format(i+1))
                # plt.savefig(foldername+'fig2-stress-LC{:03d}.pgf'.format(i+1))
                # Orthogonal views
                plt.title(' ')
                ax.view_init(azim=0, elev=0)    # x-z plane
                plt.savefig(foldername+'fig1-topology-XZ-LC{:03d}.pdf'.format(i+1))
                ax.view_init(azim=0, elev=90)   # x-y plane
                plt.savefig(foldername+'fig1-topology-XY-LC{:03d}.pdf'.format(i+1))
                ax.view_init(azim=-90, elev=0)      # y-z plane   
                plt.savefig(foldername+'fig1-topology-YZ-LC{:03d}.pdf'.format(i+1))
                ax.view_init(azim=0, elev=30)      # view 1   
                plt.savefig(foldername+'fig1-topology-V1-LC{:03d}.pdf'.format(i+1))
                ax.view_init(azim=70, elev=30)      # view 1   
                plt.savefig(foldername+'fig1-topology-V2-LC{:03d}.pdf'.format(i+1))
    
    @staticmethod    
    def plotTrussStress_ML(myBCs, a_unred, forces, candidates_bars, s_t, s_c, tk, foldername=False):
        """ Plot the ground stucture optimization solution stress """
        nodes = myBCs.nodes
        candidates = myBCs.ground_structure[candidates_bars]
        
        n_load_cases = forces.shape[-1]
        
        for i in range(n_load_cases):
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d', proj_type = 'ortho')
            plt.axis('off')
            plt.title("ABS stress")
            q = forces[:,i]

            q = q[candidates_bars]
            a = a_unred[candidates_bars]
            stress = q/a
            
            # Round to a specified number of significant digits
            significant_digits = 4
            decimals = significant_digits - int(math.floor(math.log10(np.abs(np.max(stress))))) - 1
            stress =  np.around(stress, decimals=decimals)
            
            points = nodes[candidates[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3)
            # Normalization to [0-Dim]
            points[:,:,0] = (points[:,:,0] - np.min(points[:,:,0]))
            points[:,:,1] = (points[:,:,1] - np.min(points[:,:,1]))
            points[:,:,2] = (points[:,:,2] - np.min(points[:,:,2]))
            segments = np.concatenate([points[:-1:2], points[1::2]], axis=1)
            # optimized beams                      
            norm = plt.Normalize(s_c/ myBCs.sf[i], s_t/ myBCs.sf[i])
            lc = Line3DCollection(segments, cmap='coolwarm', norm=norm)
            lc.set_array(stress)
            lc.set_linewidth(a * tk)
            line = ax.add_collection(lc)
            fig.colorbar(line, ax=[ax], location='bottom', orientation="horizontal")
            
            ax.set_xlim([min(points[:,:,0]), max(points[:,:,0]*1.05)])
            ax.set_ylim([min(points[:,:,1]), max(points[:,:,1]*1.05)])
            ax.set_zlim([min(points[:,:,2]), max(points[:,:,2]*1.05)])
            ax.set_box_aspect((myBCs.L[0]*1.05,myBCs.L[1]*1.05,myBCs.L[2]*1.05))
            fig.tight_layout()
            
            if foldername == False:
                plt.show()
            else:
                # tikzplotlib.save(foldername+'fig2-stress-LC{:03d}.tex'.format(i+1))
                plt.savefig(foldername+'fig2-stress-LC{:03d}.pdf'.format(i+1))
                # Orthogonal views
                plt.title(' ')
                ax.view_init(azim=0, elev=0)    # x-z plane
                plt.savefig(foldername+'fig2-stress-XZ-LC{:03d}.pdf'.format(i+1))
                ax.view_init(azim=0, elev=90)   # x-y plane
                plt.savefig(foldername+'fig2-stress-XY-LC{:03d}.pdf'.format(i+1))
                ax.view_init(azim=-90, elev=0)      # y-z plane   
                plt.savefig(foldername+'fig2-stress-YZ-LC{:03d}.pdf'.format(i+1))
                ax.view_init(azim=0, elev=30)      # view 1   
                plt.savefig(foldername+'fig2-stress-V1-LC{:03d}.pdf'.format(i+1))
                ax.view_init(azim=70, elev=30)      # view 1   
                plt.savefig(foldername+'fig2-stress-V2-LC{:03d}.pdf'.format(i+1))
                # plt.savefig(foldername+'fig2-stress-LC{:03d}.pgf'.format(i+1))
    
    @staticmethod    
    def plotTrussBuckling_ML(myBCs, a_unred, forces, candidates_bars, s_buck, tk, foldername=False):
        """ Plot the ground stucture optimization solution stress """
        nodes = myBCs.nodes
        candidates = myBCs.ground_structure[candidates_bars]
        
        n_load_cases = forces.shape[-1]
        
        for i in range(n_load_cases):
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d', proj_type = 'ortho')
            plt.axis('off')
            plt.title("Buckling SF")
            q = forces[:,i]

            q = q[candidates_bars]
            # Critical force
            q_crit = -s_buck*a_unred[candidates_bars]**2/myBCs.ground_structure_length[candidates_bars]**2 / myBCs.sf[i]
            a = a_unred[candidates_bars]
            
            # optimized beams
            buck = q/q_crit
            buck[buck<0] = 0 # Member in tension cannot buckle
            
            points = nodes[candidates[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3)
            # Normalization to [0-Dim]
            points[:,:,0] = (points[:,:,0] - np.min(points[:,:,0]))
            points[:,:,1] = (points[:,:,1] - np.min(points[:,:,1]))
            points[:,:,2] = (points[:,:,2] - np.min(points[:,:,2]))
            segments = np.concatenate([points[:-1:2], points[1::2]], axis=1)
            # optimized beams                      
            norm = plt.Normalize(0, 1)
            lc = Line3DCollection(segments, cmap=Blue_mono, norm=norm)
            lc.set_array(buck)
            lc.set_linewidth(a * tk)
            line = ax.add_collection(lc)
            fig.colorbar(line, ax=[ax], location='bottom', orientation="horizontal")
            
            ax.set_xlim([min(points[:,:,0]), max(points[:,:,0]*1.05)])
            ax.set_ylim([min(points[:,:,1]), max(points[:,:,1]*1.05)])
            ax.set_zlim([min(points[:,:,2]), max(points[:,:,2]*1.05)])
            ax.set_box_aspect((myBCs.L[0]*1.05,myBCs.L[1]*1.05,myBCs.L[2]*1.05))
            fig.tight_layout()
            
            if foldername == False:
                plt.show()
            else:
                # tikzplotlib.save(foldername+'fig4-B-LC{:03d}.tex'.format(i+1))
                plt.savefig(foldername+'fig4-B-LC{:03d}.pdf'.format(i+1))
                # Orthogonal views
                plt.title(' ')
                ax.view_init(azim=0, elev=0)    # x-z plane
                plt.savefig(foldername+'fig4-B-XZ-LC{:03d}.pdf'.format(i+1))
                ax.view_init(azim=0, elev=90)   # x-y plane
                plt.savefig(foldername+'fig4-B-XY-LC{:03d}.pdf'.format(i+1))
                ax.view_init(azim=-90, elev=0)      # y-z plane   
                plt.savefig(foldername+'fig4-B-YZ-LC{:03d}.pdf'.format(i+1))
                ax.view_init(azim=0, elev=30)      # view 1   
                plt.savefig(foldername+'fig4-B-V1-LC{:03d}.pdf'.format(i+1))
                ax.view_init(azim=70, elev=30)      # view 1   
                plt.savefig(foldername+'fig4-B-V2-LC{:03d}.pdf'.format(i+1))
                # plt.savefig(foldername+'fig4-B-LC{:03d}.pgf'.format(i+1))
     
    @staticmethod 
    def plotTrussMulti_deformed_ML(myBCs, myBCs_unred, a_unred, forces, U, candidates_bars, tk, foldername):
        """ Plot the ground stucture optimization solution stress """
        nodes = myBCs_unred.nodes
        candidates = myBCs_unred.ground_structure[candidates_bars]
        candidates_unred = myBCs_unred.ground_structure
        
        n_load_cases = forces.shape[-1]
        
        for i in range(n_load_cases):
            if forces.ndim == 1:
                u = U.reshape(-1,3)
            else:
                u = U[:,i].copy().reshape(-1,3)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d', proj_type = 'ortho')
            plt.axis('off')
            q = forces[:,i]

            q = q[candidates_bars]         
            a = a_unred[candidates_bars]
            
            c = np.empty(a.size, dtype = object)
        

            color_undef = 'lightgrey'
            c[:] = 'darkgrey' # no force
            c[(q>0)] = accent_r_1 # tension
            c[(q<0)] = accent_b_1 # compression
            linewidth = a * tk
            linewidth_undef = 0.2

            # Undeformed
            points_undef = nodes[candidates_unred[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3)
            segments_undef = np.concatenate([points_undef[:-1:2], points_undef[1::2]], axis=1)
            # Deformed
            points = nodes[candidates[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3) + u[candidates[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1:2], points[1::2]], axis=1)

            # Undeformed
            lc_undef = Line3DCollection(segments_undef, colors = color_undef,alpha=0.5, zorder=0)
            lc_undef.set_linewidth(linewidth_undef)
            ax.add_collection3d(lc_undef, zs=nodes[:,2], zdir='z')
            # Deformed
            lc = Line3DCollection(segments, colors = c)
            lc.set_linewidth(linewidth)
            ax.add_collection3d(lc, zs=nodes[:,2], zdir='z')
            ax.set_xlim([min(points[:,:,0]), max(points[:,:,0]*1.05)])
            ax.set_ylim([min(points[:,:,1]), max(points[:,:,1]*1.05)])
            ax.set_zlim([min(points[:,:,2]), max(points[:,:,2]*1.05)])
            ax.set_box_aspect((float(max(points[:,:,0])-min(points[:,:,0])),float(max(points[:,:,1])-min(points[:,:,1])),float(max(points[:,:,2])-min(points[:,:,2]))))
            fig.tight_layout()
            plt.title(' ')
            ax.view_init(azim=0, elev=30)
            plt.savefig(foldername+'fig1-topology-def-LC{0:03d}-V1.pdf'.format(i+1))
            ax.view_init(azim=0, elev=0)    # x-z plane
            plt.savefig(foldername+'fig1-topology-def-LC{0:03d}-XZ.pdf'.format(i+1))
            ax.view_init(azim=0, elev=90)    # x-y plane
            plt.savefig(foldername+'fig1-topology-def-LC{0:03d}-XY.pdf'.format(i+1))
    
    @staticmethod 
    def plotTrussStress_deformed_ML(myBCs, myBCs_unred, a_unred, forces, U, candidates_bars, s_t, s_c, tk, foldername):
        """ Plot the ground stucture optimization solution stress """
        nodes = myBCs_unred.nodes
        candidates = myBCs_unred.ground_structure[candidates_bars]
        candidates_unred = myBCs_unred.ground_structure
        
        
        n_load_cases = forces.shape[-1]
        
        for i in range(n_load_cases):
            if forces.ndim == 1:
                u = U.reshape(-1,3)
            else:
                u = U[:,i].copy().reshape(-1,3)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d', proj_type = 'ortho')
            plt.axis('off')
            q = forces[:,i]
            
            q = q[candidates_bars]
            a = a_unred[candidates_bars]
            stress = q/a
            
            # Round to a specified number of significant digits
            significant_digits = 4
            decimals = significant_digits - int(math.floor(math.log10(np.abs(np.max(stress))))) - 1
            stress =  np.around(stress, decimals=decimals)
            
            c = np.empty(a.size, dtype = object)
        

            color_undef = 'lightgrey'
            linewidth = a * tk
            linewidth_undef = 0.2

            # Undeformed
            points_undef = nodes[candidates_unred[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3)
            segments_undef = np.concatenate([points_undef[:-1:2], points_undef[1::2]], axis=1)
            # Deformed
            points = nodes[candidates[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3) + u[candidates[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1:2], points[1::2]], axis=1)

            # Undeformed
            lc_undef = Line3DCollection(segments_undef, colors = color_undef,alpha=0.5, zorder=0)
            lc_undef.set_linewidth(linewidth_undef)
            ax.add_collection3d(lc_undef, zs=nodes[:,2], zdir='z')
            # Deformed                     
            norm = plt.Normalize(s_c / myBCs.sf[i], s_t / myBCs.sf[i])
            lc = Line3DCollection(segments, cmap='coolwarm', norm=norm)
            lc.set_array(stress)
            lc.set_linewidth(linewidth)
            line = ax.add_collection(lc)
            fig.colorbar(line, ax=[ax], location='bottom', orientation="horizontal")
            ax.set_xlim([min(points[:,:,0]), max(points[:,:,0]*1.05)])
            ax.set_ylim([min(points[:,:,1]), max(points[:,:,1]*1.05)])
            ax.set_zlim([min(points[:,:,2]), max(points[:,:,2]*1.05)])
            ax.set_box_aspect((float(max(points[:,:,0])-min(points[:,:,0])),float(max(points[:,:,1])-min(points[:,:,1])),float(max(points[:,:,2])-min(points[:,:,2]))))
            fig.tight_layout()
            plt.title(' ')
            ax.view_init(azim=0, elev=30)
            plt.savefig(foldername+'fig2-stress-def-LC{0:03d}-V1.pdf'.format(i+1))
            ax.view_init(azim=0, elev=0)    # x-z plane
            plt.savefig(foldername+'fig2-stress-def-LC{0:03d}-XZ.pdf'.format(i+1))
            ax.view_init(azim=0, elev=90)    # x-y plane
            plt.savefig(foldername+'fig2-stress-def-LC{0:03d}-XY.pdf'.format(i+1))
   
    @staticmethod 
    def plotTrussBuckling_deformed_ML(myBCs, myBCs_unred, a_unred, forces, U, candidates_bars, s_buck, tk, foldername):
        """ Plot the ground stucture optimization solution buckling """
        nodes = myBCs_unred.nodes
        candidates = myBCs_unred.ground_structure[candidates_bars]
        candidates_unred = myBCs_unred.ground_structure

        
        n_load_cases = forces.shape[-1]
        
        for i in range(n_load_cases):
            if forces.ndim == 1:
                u = U.reshape(-1,3)
            else:
                u = U[:,i].copy().reshape(-1,3)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d', proj_type = 'ortho')
            plt.axis('off')
            q = forces[:,i]
            
            q = q[candidates_bars]
            # Critical force
            q_crit = -s_buck*a_unred[candidates_bars]**2/myBCs_unred.ground_structure_length[candidates_bars]**2 / myBCs.sf[i]
            a = a_unred[candidates_bars]
            
            # optimized beams
            buck = q/q_crit
            buck[buck<0] = 0 # Member in tension cannot buckle      

            color_undef = 'lightgrey'
            linewidth = a * tk
            linewidth_undef = 0.2

            # Undeformed
            points_undef = nodes[candidates_unred[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3)
            segments_undef = np.concatenate([points_undef[:-1:2], points_undef[1::2]], axis=1)
            # Deformed
            points = nodes[candidates[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3) + u[candidates[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1:2], points[1::2]], axis=1)

            # Undeformed
            lc_undef = Line3DCollection(segments_undef, colors = color_undef,alpha=0.5, zorder=0)
            lc_undef.set_linewidth(linewidth_undef)
            ax.add_collection3d(lc_undef, zs=nodes[:,2], zdir='z')
            # Deformed                                       
            norm = plt.Normalize(0, 1)
            lc = Line3DCollection(segments, cmap=Blue_mono, norm=norm)
            lc.set_array(buck)
            lc.set_linewidth(linewidth)
            line = ax.add_collection(lc)
            fig.colorbar(line, ax=[ax], location='bottom', orientation="horizontal")
            ax.set_xlim([min(points[:,:,0]), max(points[:,:,0]*1.05)])
            ax.set_ylim([min(points[:,:,1]), max(points[:,:,1]*1.05)])
            ax.set_zlim([min(points[:,:,2]), max(points[:,:,2]*1.05)])
            ax.set_box_aspect((float(max(points[:,:,0])-min(points[:,:,0])),float(max(points[:,:,1])-min(points[:,:,1])),float(max(points[:,:,2])-min(points[:,:,2]))))
            fig.tight_layout()
            plt.title(' ')
            ax.view_init(azim=0, elev=30)
            plt.savefig(foldername+'fig4-B-def-LC{0:03d}-V1.pdf'.format(i+1))
            ax.view_init(azim=0, elev=0)    # x-z plane
            plt.savefig(foldername+'fig4-B-def-LC{0:03d}-XZ.pdf'.format(i+1))
            ax.view_init(azim=0, elev=90)    # x-y plane
            plt.savefig(foldername+'fig4-B-def-LC{0:03d}-XY.pdf'.format(i+1))         
    
    @staticmethod 
    def plotTopologyForceThreshold(myBCs, a, K, B, E, tol, foldername):
        """ Plot the ground stucture for different force threshold """
        tk = 3 / max(a)
        M = myBCs.M
        N = myBCs.N
        n_load_cases = myBCs.R.shape[-1]
        nodes = myBCs.nodes
        # FEM
        q = np.zeros((N,1)) 
        U = np.zeros((M,1)) 
        for p in range(n_load_cases):    
        # Initial forces and displacements are calculated using FEM
            U_lc = np.zeros(M)
            keep = myBCs.free_dofs
            K_free = K[keep, :][:, keep]
            U_lc[keep] = sp.linalg.spsolve(K_free, myBCs.R_free[:,p]) # FEM analysis linear sistem
            U = np.hstack([U,U_lc.reshape((-1,1))])
            # Forces   
            q_lc = a*E/myBCs.ground_structure_length * (B.T @ U_lc) 
            q = np.hstack([q,q_lc.reshape((-1,1))])
            
        q_max = np.max(np.abs(q), axis = 1)
        threshold = tol*np.max(q_max)
      
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d', proj_type = 'ortho')
        plt.axis('off')
                
        candidates = myBCs.ground_structure[q_max >= threshold,:]
        q = q[q_max >= threshold]
        sum_grey = np.sum(np.sign(q), axis=1).astype(int)
        grey = np.ones(sum_grey.size,dtype=bool)
        grey[(sum_grey==n_load_cases) | (sum_grey==-n_load_cases)] = False
        
        a = a[q_max >= threshold]
        
        c = np.empty(a.size, dtype = object)
        
        
        c[:] = 'darkgrey' # no force
        c[(q[:,0]>0) & (~grey)] = accent_r_1 # tension
        c[(q[:,0]<0) & (~grey)] = accent_b_1 # compression
        linewidth = a * tk
        
        points = nodes[candidates[:, [0, 1]].astype(int), :][:,:,[0,1,2]].reshape(-1, 1, 3)


        # Normalization to [0-Dim]
        points[:,:,0] = (points[:,:,0] - np.min(points[:,:,0]))
        points[:,:,1] = (points[:,:,1] - np.min(points[:,:,1]))
        points[:,:,2] = (points[:,:,2] - np.min(points[:,:,2]))
        segments = np.concatenate([points[:-1:2], points[1::2]], axis=1)
        # optimized beams
        lc = Line3DCollection(segments, colors = c)
        lc.set_linewidth(linewidth)
        ax.add_collection3d(lc, zs=nodes[:,2], zdir='z')
         
        ax.set_xlim([0, myBCs.L[0]*1.05])
        ax.set_ylim([0, myBCs.L[1]*1.05])
        ax.set_zlim([0, myBCs.L[2]*1.05])
        ax.set_box_aspect((myBCs.L[0]*1.05,myBCs.L[1]*1.05,myBCs.L[2]*1.05))
        plt.title('Active bars: {0}'.format(candidates.shape[0]))
        fig.tight_layout()
        
        if foldername!=False:
            plt.savefig(foldername+'/fig1-Topology-{0}.pdf'.format(tol))     
   
    @classmethod
    def plotRoutine(cls, myBCs, a, q, U, vol, stress_tension_max, stress_compression_max, tol, s_buck, obj_hist, cellsEquals, foldername, LP, GIF, a_cell=False):
        thick = 3 / max(a)
        if LP:
            folder = foldername + '/LP/'
            if not os.path.isdir(folder):
                    os.makedirs(folder)
        else:
            folder = foldername + '/'
        cls.plotGroundStructure(myBCs, folder)   
        cls.plotTruss(myBCs, a, q, max(a) * tol, vol, thick, folder)
        cls.plotTrussStress(myBCs, a, q, max(a) * tol, thick, folder)
        if sum(U) != 0:
            for mag in (0.1,1,5,10):
                cls.plotTrussDeformation(myBCs, a, q, U, max(a) * tol, mag, folder, axis=True)
                cls.plotTrussDeformation(myBCs, a, q, U, max(a) * tol, mag, folder, axis=False)
        cls.plotTrussRF(myBCs, a, q, max(a) * tol, stress_tension_max, stress_compression_max, thick, folder)
        if cellsEquals:
            try:
                if 1 in myBCs.ground_structure[:,2]:
                    cls.PlotTrussCellule_Free_form(myBCs, a_cell, max(a) * tol, thick, folder, axis=True, gif=GIF) 
                    cls.PlotTrussCellule_Free_form(myBCs, a_cell, max(a) * tol,  thick, folder, axis=False, gif=GIF)
            except:
                cls.PlotTrussCellule(myBCs, a_cell, max(a) * tol, thick, folder, axis=True, gif=GIF)
                cls.PlotTrussCellule(myBCs, a_cell, max(a) * tol,  thick, folder, axis=False, gif=GIF)
        if np.any(obj_hist) != False: 
            plotObjHistory(obj_hist, folder)
        if GIF:
            generateGIF(folder,'animation')
     
    @classmethod    
    def plotRoutineBuckling(cls, myBCs, a, q, U, vol, stress_tension_max, stress_compression_max, tol, 
                            s_buck, obj_hist, cellsEquals, foldername, LP, GIF, a_cell=False, obj_hist_LP=False,
                            eq_hist=False, s_c_hist=False, s_t_hist=False, comp_hist=False, buck_hist=False):
        thick = 3 / max(a)
        if LP:
            folder = foldername + '/LP/'
            if not os.path.isdir(folder):
                    os.makedirs(folder)
        else:
            folder = foldername + '/'
        cls.plotGroundStructure(myBCs, folder)
        cls.plotTruss(myBCs, a, q, max(a) * tol, vol, thick, folder, gif=GIF)
        cls.plotTrussStress(myBCs, a, q, max(a) * tol, thick, folder)
        cls.plotTrussBucklingCheck(myBCs, a, q, max(a) * tol, s_buck, thick, folder)
        cls.plotTrussBuckling(myBCs, a, q, max(a) * tol, s_buck, thick, folder)
        if sum(U) != 0:
            for mag in (0.1,1,5,10):
                cls.plotTrussDeformation(myBCs, a, q, U, a>max(a) * tol, mag, folder, axis=True)
                cls.plotTrussDeformation(myBCs, a, q, U, a>max(a) * tol, mag, folder, axis=False)
        cls.plotTrussRF(myBCs, a, q, max(a) * tol, stress_tension_max, stress_compression_max, thick, folder)
        if cellsEquals:
            try:
                if 1 in myBCs.ground_structure[:,2]:
                    cls.PlotTrussCellule_Free_form(myBCs, a, max(a) * tol, thick, folder, axis=True, gif=GIF) 
                    cls.PlotTrussCellule_Free_form(myBCs, a, max(a) * tol, thick, folder, axis=False, gif=GIF)
            except:
                cls.PlotTrussCellule(myBCs, a_cell, max(a) * tol, thick, folder, axis=True, gif=GIF)
                cls.PlotTrussCellule(myBCs, a_cell, max(a) * tol, thick, folder, axis=False, gif=GIF)
        if np.any(obj_hist) != False: 
            plotObjHistory(obj_hist, folder)
            if np.any(eq_hist) != False:
                plotConstHistory(eq_hist, s_c_hist, s_t_hist, comp_hist, buck_hist, folder)
            if np.any(obj_hist_LP) != False: 
                plotObjHistory_combined(obj_hist, obj_hist_LP, folder)
        if GIF:
            generateGIF(folder+'gif_tmp_top','topology')
        if GIF and cellsEquals:
            root = os.listdir(folder) 
            ind = [x.startswith('gif_tmp_cell') for x in os.listdir(folder)]
            folders = list(compress(root, ind))
            for fold in folders:
                generateGIF(folder+fold,'cell')
            
    @classmethod    
    def plotRoutineBuckling_ML(cls, myBCs, a, q, U, vol, stress_tension_max, stress_compression_max, candidates, K, B, E,
                            s_buck, obj_hist, foldername, LP, GIF, obj_hist_LP=False,
                            eq_hist=False, s_c_hist=False, s_t_hist=False, comp_hist=False, buck_hist=False, myBCs_unred=False):
        thick = 3 / max(a)
        if LP:
            folder = foldername + '/LP/'
            if not os.path.isdir(folder):
                    os.makedirs(folder)
        else:
            folder = foldername + '/'
            
        M = myBCs_unred.M
        N = myBCs_unred.N
        n_load_cases = myBCs_unred.R.shape[-1]
        # FEM
        q_fem = np.zeros((N,1)) 
        U_fem = np.zeros((M,1)) 
        for p in range(n_load_cases):    
        # Initial forces and displacements are calculated using FEM
            U_lc = np.zeros(M)
            keep = myBCs_unred.free_dofs
            K_free = K[keep, :][:, keep]
            U_lc[keep] = sp.linalg.spsolve(K_free, myBCs_unred.R_free[:,p]) # FEM analysis linear sistem
            U_fem = np.hstack([U_fem,U_lc.reshape((-1,1))])
            # Forces   
            q_fem_lc = a*E/myBCs_unred.ground_structure_length * (B.T @ U_lc) 
            q_fem = np.hstack([q_fem,q_fem_lc.reshape((-1,1))])
        
        
        cls.plotTruss_ML(myBCs_unred, a, q, candidates, vol, thick, GIF, folder)
        cls.plotTrussDeformation_FEM(myBCs_unred, a, q_fem[:,1:], U_fem[:,1:], candidates, 1, folder, axis=False)
        for p in range(n_load_cases):
            cls.plotGroundStructure(myBCs_unred, folder, load_case=p, mag=0.1)
        
        cls.plotTrussMulti_ML(myBCs_unred, a, q, candidates, thick, folder)
        cls.plotTrussStress_ML(myBCs_unred, a, q, candidates, stress_tension_max, stress_compression_max, thick, folder)
        cls.plotTrussBuckling_ML(myBCs_unred, a, q, candidates, s_buck, thick, folder)
            
        if np.sum(U[:]) != 0:
            cls.plotTrussMulti_deformed_ML(myBCs, myBCs_unred, a, q, U, candidates, thick, folder)
            cls.plotTrussStress_deformed_ML(myBCs, myBCs_unred, a, q, U, candidates, stress_tension_max, stress_compression_max, thick, folder)
            cls.plotTrussBuckling_deformed_ML(myBCs, myBCs_unred, a, q, U, candidates, s_buck, thick, folder)
            for mag in (1,5):
                cls.plotTrussDeformation(myBCs_unred, a, q, U, candidates, mag, folder, axis=True)
                cls.plotTrussDeformation(myBCs_unred, a, q, U, candidates, mag, folder, axis=False)
        
        if np.any(obj_hist) != False: 
            plotObjHistory(obj_hist, folder)
            if np.any(eq_hist) != False:
                plotConstHistory(eq_hist, s_c_hist, s_t_hist, comp_hist, buck_hist, folder)
            if np.any(obj_hist_LP) != False: 
                plotObjHistory_combined(obj_hist, obj_hist_LP, folder)
        if GIF:
            generateGIF(folder+'gif_tmp_top','topology')
            
        return candidates
            
############################
# Various
############################

def plotNBarsRandom(n_bars, tol, foldername):
    """ Plot the objective function history trought iterations """
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    # Calculate statistical measure of interest
    mean = np.mean(n_bars)
    std_opt = np.std(n_bars)
    
    # Plot mean
    #ax1.axhline(y = mean, color = 'red', linestyle = 'dashed', label = "Mean: {0:.2f}".format(mean))
    
    ax1.scatter(np.arange(n_bars.size, dtype='int')+1, n_bars, marker='.', color=accent_b_1, label='n_bars')
    plt.title('Convergence plot\nMean: {0:.2f}, Std: {1:.2f}'.format(mean, std_opt))
    ax1.set_xlabel('Random starting point N')
    ax1.set_ylabel('Number of bars of the solution')
    ax1.set_axisbelow(True)
    
    ax1.legend()
        
    np.save(foldername+'/fig99_n_bars', n_bars)
    # tikzplotlib.save(foldername+'/fig99_n_bars.tex')
    plt.savefig(foldername+'/fig99_n_bars.pdf')
    plt.savefig(foldername+'/fig99_n_bars.pgf')

def plotStartingPointRandom(vol_x0, vol_opt, foldername):
    """ Plot the objective function history trought iterations """
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    # Calculate statistical measure of interest
    mean = np.mean(vol_opt)
    std_opt = np.std(vol_opt)
    
    # Plot mean
    #ax1.axhline(y = mean, color = 'red', linestyle = 'dashed', label = "Mean: {0:.2f}".format(mean))
    
    ax1.scatter(np.arange(vol_x0.size, dtype='int')+1, vol_x0, marker='.', color=accent_b_1, label='x0')
    ax1.scatter(np.arange(vol_opt.size, dtype='int')+1, vol_opt, marker='.', color=accent_r_1, label='Optim.')
    plt.title('Convergence plot\nMean: {0:.2f}, Std: {1:.2f}'.format(mean, std_opt))
    ax1.set_xlabel('Random starting point N')
    ax1.set_ylabel('Volume')
    ax1.set_axisbelow(True)
    
    ax1.legend()
        
    np.save(foldername+'/fig99_Convergence_plot_data_x0', vol_x0)
    np.save(foldername+'/fig99_Convergence_plot_data', vol_opt)
    # tikzplotlib.save(foldername+'/fig99_Convergence_plot.tex')
    plt.savefig(foldername+'/fig99_Convergence_plot.pdf')
    plt.savefig(foldername+'/fig99_Convergence_plot.pgf')

def plotObjHistory(obj_hist, foldername):
    """ Plot the objective function history trought iterations """
    fig,ax = plt.subplots()
    plt.plot(np.arange(obj_hist.size), obj_hist, color=accent_r_1)
    plt.title('Convergence history')
    plt.xlabel('Iteration N')
    plt.ylabel('Objective function')
    plt.grid(True)
    plt.savefig(foldername+'fig10_obj_hist.pdf')
    
def plotObjHistory_combined(obj_hist, obj_hist_LP,foldername):
    """ Plot the objective function history trought iterations """
    fig,ax = plt.subplots()
    #trans = obj_hist[0] - obj_hist_LP[-1]
    trans = 0
    plt.plot(np.arange(obj_hist_LP.size), obj_hist_LP, color=accent_r_1, label = 'SLP')
    plt.plot((np.arange(obj_hist.size)+obj_hist_LP.size), (obj_hist-trans), color=accent_b_1, label = 'NLP')
    ax.legend()
    plt.title('Convergence history')
    plt.xlabel('Iteration N')
    plt.ylabel('Volume')
    plt.grid(True)
    plt.savefig(foldername+'fig11_obj_hist_comb.pdf')
    
def plotConstHistory(eq, s_c, s_t, comp, buck, foldername):
    """ Plot the constraints function history trought iterations """
    fig,ax = plt.subplots()
    plt.plot(np.arange(eq.size), eq, label = 'Equilibrium')
    plt.plot(np.arange(s_c.size), s_c, label = 'Stress Compression')
    plt.plot(np.arange(s_t.size), s_t, label = 'Stress Tension')
    plt.plot(np.arange(comp.size), comp, label = 'Compatibility')
    plt.plot(np.arange(buck.size), buck, label = 'Euler Buckling')
    ax.legend()
    plt.title('Constraints history')
    plt.xlabel('Iteration N')
    plt.ylabel('Constraint violation')
    plt.ylim([-0.1, 1.05])
    plt.grid(True)
    plt.savefig(foldername+'fig12_constr_hist.pdf')
    
def plotCluster(matrix):
    fig,ax = plt.subplots()
    ax.imshow(matrix, cmap='Set1',\
        interpolation='none')
    plt.title('Optimization clusters')
    plt.axis('off')
    plt.axis('equal')
    plt.show(block=False)
    
def generateGIF(folder, name):
    # GIF generation
    filenames = glob.glob(folder+'/*.png')
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(folder+'/'+name+'.gif', images)