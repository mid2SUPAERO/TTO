import numpy as np
import scipy.sparse as sp

#################################
## IPOPT classes definition
#################################
#https://stackoverflow.com/questions/68266571/how-to-speed-up-the-ipopt-solver
class IPOPT_Problem():
    """ Parent class used to store common methods for the different IPOPT classes  """
    
    def calc_reduction_pattern_mapped(self):
        """ This method calculates the reduced mapping pattern of the distribution of the area variables """
        if self.myBCs.isReduced:
            reduction_pattern_map = np.zeros(self.myBCs.ncel_str*self.myBCs.N_cell_max)
            for i in range(self.n_topologies_cell): # starting from the lower left corner
               reduction_pattern_map += np.kron(self.cell_mapping_matrix[:,i], self.myBCs.reduction_pattern[i,:]) 
            return reduction_pattern_map==1
        else:
            raise ValueError("This method can't be used on unreduced problems")
        
    def calc_area_phys(self, area):
        """ This function is used to calculate the true physical distribution of the sections of the truss starting from the mapping matrix and the areas of the cells """
        if self.myBCs.isReduced:
            reduction_pattern_map = self.calc_reduction_pattern_mapped()
            a_cell = np.zeros(self.myBCs.reduction_pattern.size).ravel()
            tmp = np.zeros(self.myBCs.ncel_str*self.myBCs.N_cell_max)
            a_cell[self.myBCs.reduction_pattern.ravel()] = area
            a_cell = a_cell.reshape((-1,self.myBCs.N_cell_max))
            a_cell_out = a_cell.copy()
            for i in range(self.n_topologies_cell): # starting from the lower left corner
                tmp += np.kron(self.cell_mapping_matrix[:,i], a_cell[i,:])
            area_phys = tmp[reduction_pattern_map] # initialize a_phys
        else:
            area_phys = np.zeros(self.N)
            a_cell_out = np.zeros(self.n_topologies_cell, dtype='object_')
            for i in range(self.n_topologies_cell): # starting from the lower left corner
                area_phys += np.kron(self.cell_mapping_matrix[:,i], area[i*self.N_cell[0]:(i+1)*self.N_cell[0]]) # initialize a_phys
                a_cell_out[i] = np.array(area[i*self.N_cell[0]:(i+1)*self.N_cell[0]])
            
        return area_phys, a_cell_out
    
    def calc_area_phys_id(self):
        """ This function is used to calculate the true physical distribution of the sections IDs of the truss starting from the mapping matrix """
        if self.myBCs.isReduced:
            reduction_pattern_map = self.calc_reduction_pattern_mapped()
            IDs = np.ones(self.myBCs.reduction_pattern.size, dtype='int').ravel() * -1
            tmp = np.zeros(self.myBCs.ncel_str*self.myBCs.N_cell_max, dtype='int')
            IDs[self.myBCs.reduction_pattern.ravel()] = np.arange(self.NN, dtype='int')
            IDs = IDs.reshape((-1,self.myBCs.N_cell_max))
            for i in range(self.n_topologies_cell): # starting from the lower left corner
                tmp += np.kron(self.cell_mapping_matrix[:,i], IDs[i,:]) # initialize a_phys
            area_phys_id = tmp[reduction_pattern_map]
        else:
            area_phys_id = np.zeros(self.N, dtype='int')
            for i in range(self.n_topologies_cell): # starting from the lower left corner
                area_phys_id += np.kron(self.cell_mapping_matrix[:,i], np.arange(self.NN, dtype='int')[i*self.N_cell[0]:(i+1)*self.N_cell[0]])
        return area_phys_id
    
    def calc_len_phys(self):
        """ This function is used to calculate the sum of the true physical distribution of the length of the truss starting from the mapping matrix.
        Used for gradient calculation """
        if self.myBCs.isReduced:            
            g = []
            for i in range(self.n_topologies_cell):
                su = np.sum(np.kron(self.cell_mapping_matrix[:,i].T, (self.l_cell[i][self.myBCs.reduction_pattern_len[i,:]]).reshape((-1, 1))), axis=1)
                g = np.append(g,su[su>0])
        else:
            g = []
            for i in range(self.n_topologies_cell):
                su = np.sum(np.kron(self.cell_mapping_matrix[:,i].T, self.l_cell.reshape((-1, 1))), axis=1)
                g = np.append(g,su[su>0])
        return g
          
class Layopt_IPOPT(IPOPT_Problem):
    def __init__(self, N, M, l_true, joint_cost, B, f, s_c, s_t, E):
        # Use the class constructor to declare the variable used by the optimizer during the optimization
        self.N = N # Number of member of the Ground Structure
        self.M = M # Number of DOFs of the Groud Structure
        self.l_true = l_true # Physical member lenghts [mm]
        self.l = l_true + joint_cost * np.max(l_true) # indipendency from the number of cells [mm]
        self.B = B # Equilibrium forces matrix 
        self.f = f # External forces [N]
        self.s_c = s_c # Max compression allowable stress [MPa]
        self.s_t = s_t # Max tension allowable stress [MPa]
        self.E = E # Young's modulus [MPa]
        
        # Sparsity pattern of B matrix
        self.B_coo = self.B.tocoo()
        self.B_row, self.B_column = self.B_coo.row,self.B_coo.col 
        self.B_T_coo = self.B.T.tocoo() # Horrible but necessary to conserve the very same order of dComp_dU
        self.B_T_row, self.B_T_column = self.B_T_coo.row,self.B_T_coo.col       
        
        # Create the indexing variables used for splitting the design variables vector x
        self.area = slice(0,N)
        self.force = slice(N,2*N)
        self.U = slice(2*N,2*N+M)
        
        # History values
        self.it = 0
        self.obj_hist = []

    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        self.volume_phys = self.l_true.T @ x[self.area]
        return self.l.T @ x[self.area]

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        grad = np.zeros(x.size)
        grad[self.area] = self.l
        return grad

    def constraints(self, x):
        """Returns the constraints."""
        ### Equilibrium (M eq)
        equilibrium = self.B @ x[self.force] - self.f
        ### Stress (2*N eq)
        stress_c = x[self.force] - self.s_c * x[self.area]
        stress_t = x[self.force] - self.s_t * x[self.area]
        stress = np.append(stress_c, stress_t)
        ### Compatibility (N eq)
        compatibility = x[self.area] * self.E/self.l * (self.B.T @ x[self.U]) - x[self.force]
        cons = np.concatenate((equilibrium, stress, compatibility))
        return cons
    
    def jacobianstructure(self):
        """ Returns the row and column indices for non-zero vales of the
        Jacobian. """
        # DIM = Number of constraint functions X number of design variables
        ### Equilibrium (M eq)
        row, column = self.B_row, self.B_column+self.N # the indexes are translated to the corresponding design variables (force)
        ### Stress (2*N eq)
        row = np.append(row,range(self.M,self.M+self.N*2))
        column = np.append(column,np.r_[self.area,self.area])
        row = np.append(row,range(self.M,self.M+self.N*2))
        column = np.append(column,np.r_[self.force,self.force])
        ### Compatibility (N eq)
        row = np.append(row,range(self.M+self.N*2,self.M+self.N*3))
        column = np.append(column,np.r_[self.area])
        row = np.append(row,range(self.M+self.N*2,self.M+self.N*3))
        column = np.append(column,np.r_[self.force])
        row = np.append(row,self.B_T_row + self.M+self.N*2) # B is transposed
        column = np.append(column,self.B_T_column+self.N*2)
        return row,column

    def jacobian(self, x):
        """ Returns the Jacobian of the constraints with respect to x. """
        # DIM = Number of constraint functions X number of design variables
        ### Equilibrium (M eq)
        Jacobian = self.B_coo.data
        ### Stress (2*N eq)
        Jacobian = np.append(Jacobian,np.ones(self.N)*(- self.s_c))
        Jacobian = np.append(Jacobian,np.ones(self.N)*(- self.s_t))
        Jacobian = np.append(Jacobian,np.ones(self.N))
        Jacobian = np.append(Jacobian,np.ones(self.N))
        ### Compatibility (N eq)
        dComp_dA = self.B.T @ x[self.U] * self.E / self.l # Diagonal term, i==j
        dComp_dU = (sp.diags(x[self.area] * self.E / self.l) @ self.B.T) # Multiply the column of B.T and the main diagnonal of diag
        dComp_dU.sort_indices() # Needed to match the order of B.T given in jacobianstructure
        dComp_dU = dComp_dU.tocoo()
        # Area
        Jacobian = np.append(Jacobian,dComp_dA)
        # Force
        Jacobian = np.append(Jacobian,np.ones(self.N)*-1)
        # Displacements
        Jacobian = np.append(Jacobian,dComp_dU.data)
        return Jacobian

    """ def hessianstructure(self):
        Returns the row and column indices for non-zero vales of the
        Hessian.

        # NOTE: The default hessian structure is of a lower triangular matrix,
        # therefore this function is redundant. It is included as an example
        # for structure callback.

        return np.nonzero(np.tril(np.ones((4, 4))))

    def hessian(self, x, lagrange, obj_factor):
        Returns the non-zero values of the Hessian.

        H = obj_factor*np.array((
            (2*x[3], 0, 0, 0),
            (x[3],   0, 0, 0),
            (x[3],   0, 0, 0),
            (2*x[0]+x[1]+x[2], x[0], x[0], 0)))

        H += lagrange[0]*np.array((
            (0, 0, 0, 0),
            (x[2]*x[3], 0, 0, 0),
            (x[1]*x[3], x[0]*x[3], 0, 0),
            (x[1]*x[2], x[0]*x[2], x[0]*x[1], 0)))

        H += lagrange[1]*2*np.eye(4)

        row, col = self.hessianstructure()

        return H[row, col] """

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        """Prints information at every Ipopt iteration."""

        self.it = iter_count
        self.obj_hist = np.append(self.obj_hist, self.volume_phys)

class Layopt_IPOPT_Buck(IPOPT_Problem):
    def __init__(self, N, M, l_true, joint_cost, B, f, s_c, s_t, E, s_buck):
        # Use the class constructor to declare the variable used by the optimizer during the optimization
        self.N = N # Number of member of the Ground Structure
        self.M = M # Number of DOFs of the Groud Structure
        self.l_true = l_true # Physical member lenghts [mm]
        self.l = l_true + joint_cost * np.max(l_true) # indipendency from the number of cells [mm]
        self.B = B # Equilibrium forces matrix 
        self.f = f # External forces [N]
        self.s_c = s_c # Max compression allowable stress [MPa]
        self.s_t = s_t # Max tension allowable stress [MPa]
        self.E = E # Young's modulus [MPa]
        
        # Buckling specific parameters
        self.s_buck = s_buck
        self.a_cr = -s_c * l_true**2 / self.s_buck # Evaluation of the critical section for the members
        
        # Sparsity pattern of B matrix
        self.B_coo = self.B.tocoo()
        self.B_row, self.B_column = self.B_coo.row,self.B_coo.col 
        self.B_T_coo = self.B.T.tocoo() # Horrible but necessary to conserve the very same order of dComp_dU
        self.B_T_row, self.B_T_column = self.B_T_coo.row,self.B_T_coo.col       
        
        # Create the indexing variables used for splitting the design variables vector x
        self.area = slice(0,N)
        self.force = slice(N,2*N)
        self.U = slice(2*N,2*N+M)
        
        # History values
        self.it = 0
        self.obj_hist = []

    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        self.volume_phys = self.l_true.T @ x[self.area]
        return self.l.T @ x[self.area]

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        grad = np.zeros(x.size)
        grad[self.area] = self.l
        return grad

    def constraints(self, x):
        """Returns the constraints."""
        ### Equilibrium (M eq)
        equilibrium = self.B @ x[self.force] - self.f
        ### Stress (2*N eq)
        stress_c = x[self.force] - self.s_c * x[self.area]
        stress_t = x[self.force] - self.s_t * x[self.area]
        stress = np.append(stress_c, stress_t)
        ### Compatibility (N eq)
        compatibility = x[self.area] * self.E/self.l_true * (self.B.T @ x[self.U]) - x[self.force]
        ### Buckling (N eq)
        buckling = x[self.force] + (self.s_buck/self.l_true**2) * x[self.area]**2
        cons = np.concatenate((equilibrium, stress, compatibility, buckling))
        return cons
    
    def jacobianstructure(self):
        """ Returns the row and column indices for non-zero vales of the
        Jacobian. """
        # DIM = Number of constraint functions X number of design variables
        ### Equilibrium (M eq)
        row, column = self.B_row, self.B_column+self.N # the indexes are translated to the corresponding design variables (force)
        ### Stress (2*N eq)
        row = np.append(row,range(self.M,self.M+self.N*2))
        column = np.append(column,np.r_[self.area,self.area])
        row = np.append(row,range(self.M,self.M+self.N*2))
        column = np.append(column,np.r_[self.force,self.force])
        ### Compatibility (N eq)
        row = np.append(row,range(self.M+self.N*2,self.M+self.N*3))
        column = np.append(column,np.r_[self.area])
        row = np.append(row,range(self.M+self.N*2,self.M+self.N*3))
        column = np.append(column,np.r_[self.force])
        row = np.append(row,self.B_T_row + self.M+self.N*2) # B is transposed
        column = np.append(column,self.B_T_column+self.N*2)
        ### Buckling (N eq)
        row = np.append(row,range(self.M+self.N*3,self.M+self.N*4))
        column = np.append(column,np.r_[self.area])
        row = np.append(row,range(self.M+self.N*3,self.M+self.N*4))
        column = np.append(column,np.r_[self.force])
        return row,column

    def jacobian(self, x):
        """ Returns the Jacobian of the constraints with respect to x. """
        # DIM = Number of constraint functions X number of design variables
        ### Equilibrium (M eq)
        Jacobian = self.B_coo.data
        ### Stress (2*N eq)
        Jacobian = np.append(Jacobian,np.ones(self.N)*(- self.s_c))
        Jacobian = np.append(Jacobian,np.ones(self.N)*(- self.s_t))
        Jacobian = np.append(Jacobian,np.ones(self.N))
        Jacobian = np.append(Jacobian,np.ones(self.N))
        ### Compatibility (N eq) x[self.area] * self.E/self.l_true * (self.B.T @ x[self.U]) - x[self.force]
        dComp_dA = self.B.T @ x[self.U] * self.E / self.l_true # Diagonal term, i==j
        dComp_dU = (sp.diags(x[self.area] * self.E / self.l_true) @ self.B.T) # Multiply the column of B.T and the main diagnonal of diag
        dComp_dU.sort_indices() # Needed to match the order of B.T given in jacobianstructure
        dComp_dU = dComp_dU.tocoo()
        # Area
        Jacobian = np.append(Jacobian,dComp_dA)
        # Force
        Jacobian = np.append(Jacobian,np.ones(self.N)*-1)
        # Displacements
        Jacobian = np.append(Jacobian,dComp_dU.data)
        ### Buckling (N eq)
        # Area
        Jacobian = np.append(Jacobian,2*x[self.area]*self.s_buck/self.l_true**2)
        # Force
        Jacobian = np.append(Jacobian,np.ones(self.N))
        return Jacobian
    
    def hessianstructure(self):
        """ Returns the row and column indices for non-zero vales of the
        Hessian. """
        # DIM = Number of design variables X number of design variables
        M = self.M
        N = self.N
        
        ### Compatibility (N eq) x[self.area] * self.E/self.l_true * (self.B.T @ x[self.U]) - x[self.force]
        ## Area self.B.T @ x[self.U] * self.E / self.l_true
        # Area -
        
        # Force -

        # Displacements 
        i, j = np.meshgrid(np.arange(2*N,2*N+M, dtype='int'), np.arange(N, dtype='int'))
        
        ### Buckling (N eq) buckling = x[self.force] + (self.s_buck/self.l_true**2) * x[self.area]**2
        ## Area 2*x[self.area]*self.s_buck/self.l_true**2
        # Area
        i = np.append(i.ravel(),np.arange(N, dtype='int'))
        j = np.append(j.ravel(),np.arange(N, dtype='int'))

        return i, j

    def hessian(self, x, l, obj_factor):
        """ Returns the non-zero values of the Hessian. """
        # DIM = Number of design variables X number of design variables
        M = self.M
        N = self.N
        
        # ID of the constraints
        comp_id = slice(M+N*2, M+N*3)
        buck_id = slice(M+N*3, M+N*4)
        
        ### Objective function (N eq) the hessian of the obj function is empty
        # Hessian = obj_factor * NULL
        
        ### Equilibrium (M eq) self.B @ x[self.force] - self.f
        ## Area -

        ## Force self.B_coo.data

        ## Displacements -
        
        ### Stress (2*N eq)
        ## Area -
        
        ## Force -

        ## Displacements -

        ### Compatibility (N eq) x[self.area] * self.E/self.l_true * (self.B.T @ x[self.U]) - x[self.force]
        ## Area self.B.T @ x[self.U] * self.E / self.l_true
        # Area -
        
        # Force -

        # Displacements 
        temp = (sp.diags(l[comp_id] * self.E / self.l_true) @ self.B.T)
        temp.sort_indices() # Needed to match the order of B.T given in jacobianstructure
        temp = temp.tocoo()
        j, i = temp.row, temp.col + (2*N)
        data = temp.data

        ## Force np.ones(self.N)*-1

        ## Displacements (sp.diags(x[self.area] * self.E / self.l_true) @ self.B.T)
        # Area
        # Symmetric, already considered
        
        # Force -

        # Displacements -
        
        ### Buckling (N eq) buckling = x[self.force] + (self.s_buck/self.l_true**2) * x[self.area]**2
        ## Area 2*x[self.area]*self.s_buck/self.l_true**2
        # Area
        i = np.append(i.ravel(),np.arange(N, dtype='int'))
        j = np.append(j.ravel(),np.arange(N, dtype='int'))
        data = np.append(data,2*self.s_buck/self.l_true**2 * l[buck_id]) 
        # Force -

        # Displacements -
        
        ## Force np.ones(self.N)

        ## Displacements -
        
        Hessian = sp.coo_matrix((data, (i, j)), shape=(x.size, x.size)).tocsc() # check which one is better 
        row, col = self.hessianstructure()
        out = np.array(Hessian[row, col]).ravel()
        
        return out
    
    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        """Prints information at every Ipopt iteration."""

        self.it = iter_count
        self.obj_hist = np.append(self.obj_hist, self.volume_phys)
    
class Layopt_IPOPT_VL(IPOPT_Problem):
    def __init__(self, NN, N_cell, N, M, myBCs, joint_cost, B, f, s_c, s_t, E, cell_mapping_matrix):
        # Use the class constructor to declare the variable used by the optimizer during the optimization
        self.NN = NN # Number of members of the cells
        self.N_cell = N_cell # Number of members of a single cell
        self.N = N # Number of members of the Ground Structure
        self.M = M # Number of DOFs of the Groud Structure
        self.myBCs = myBCs # BC class
        self.l_true = myBCs.ground_structure_length # Physical member lenghts [mm]
        self.l = self.l_true + joint_cost * np.max(self.l_true) # indipendency from the number of cells [mm]
        self.B = B # Equilibrium forces matrix 
        self.f = f # External forces [N]
        self.s_c = s_c # Max compression allowable stress [MPa]
        self.s_t = s_t # Max tension allowable stress [MPa]
        self.E = E # Young's modulus [MPa]
        self.cell_mapping_matrix = cell_mapping_matrix # Mapping function used to distribute the cells over the structure domain
        self.n_topologies_cell = np.size(cell_mapping_matrix, 1) # Number of different cell topologies 

        # VL parameters
        if self.myBCs.isReduced:
            self.l_cell = self.myBCs.l_cell_full.copy() + joint_cost * np.max(self.l_true) # change, different cell size per different reduced cell
        else:
            self.l_cell = self.l[:N_cell[0]].copy() # Length of every member of one cell 
        
        # Sparsity pattern of B matrix
        self.B_coo = self.B.tocoo()
        self.B_row, self.B_column = self.B_coo.row,self.B_coo.col 
        self.B_T_coo = self.B.T.tocoo() # Horrible but necessary to conserve the very same order of dComp_dU
        self.B_T_row, self.B_T_column = self.B_T_coo.row,self.B_T_coo.col      
                
        # Create the indexing variables used for splitting the design variables vector x
        self.area = slice(0,NN)
        self.force = slice(NN,NN+N)
        self.U = slice(NN+N,NN+N+M)
        
        # History values
        self.it = 0
        self.obj_hist = []

    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        area_phys,_ = self.calc_area_phys(x[self.area])
        self.volume_phys = self.l_true.T @ area_phys
        return self.l.T @ area_phys

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        g = self.calc_len_phys()
        grad = np.zeros(x.size)
        grad[self.area] = g
        return grad

    def constraints(self, x):
        """Returns the constraints."""
        area_phys, = self.calc_area_phys(x[self.area]) # initialize a_phys
        ### Equilibrium (M eq)
        equilibrium = self.B @ x[self.force] - self.f
        ### Stress (2*N eq)
        stress_c = x[self.force] - self.s_c * area_phys
        stress_t = x[self.force] - self.s_t * area_phys
        stress = np.append(stress_c, stress_t)
        ### Compatibility (N eq)
        compatibility = area_phys * self.E/self.l_true * (self.B.T @ x[self.U]) - x[self.force] 
        cons = np.concatenate((equilibrium, stress, compatibility))
        return cons
    
    def jacobianstructure(self):
        """ Returns the row and column indices for non-zero vales of the
        Jacobian. """
        # DIM = Number of constraint functions X number of design variables
        area_phys_id = self.calc_area_phys_id()
        ### Equilibrium (M eq)
        row, column = self.B_row, self.B_column+self.NN # the indexes are translated to the corresponding design variables (force)
        ### Stress (2*N eq)
        row = np.append(row,range(self.M,self.M+self.N*2))
        column = np.append(column,np.tile(area_phys_id,2))
        row = np.append(row,range(self.M,self.M+self.N*2))
        column = np.append(column,np.r_[self.force,self.force])
        ### Compatibility (N eq)
        row = np.append(row,range(self.M+self.N*2,self.M+self.N*3))
        column = np.append(column,area_phys_id)
        row = np.append(row,range(self.M+self.N*2,self.M+self.N*3))
        column = np.append(column,np.r_[self.force])
        row = np.append(row,self.B_T_row + self.M+self.N*2) # B is transposed
        column = np.append(column,self.B_T_column+self.N+self.NN)
        return row,column

    def jacobian(self, x):
        """ Returns the Jacobian of the constraints with respect to x. """
        # DIM = Number of constraint functions X number of design variables
        # Physical areas
        area_phys,_ = self.calc_area_phys(x[self.area]) # initialize a_phys
        ### Equilibrium (M eq)
        Jacobian = self.B_coo.data
        ### Stress (2*N eq)
        Jacobian = np.append(Jacobian,np.ones(self.N)*(- self.s_c))
        Jacobian = np.append(Jacobian,np.ones(self.N)*(- self.s_t))
        Jacobian = np.append(Jacobian,np.ones(self.N))
        Jacobian = np.append(Jacobian,np.ones(self.N))
        ### Compatibility (N eq)
        dComp_dA = []
        dComp_dA = self.B.T @ x[self.U] * self.E / self.l_true # Diagonal term, i==j
        dComp_dU = (sp.diags(area_phys * self.E / self.l_true) @ self.B.T) # Multiply the column of B.T and the main diagnonal of diag
        dComp_dU.sort_indices() # Needed to match the order of B.T given in jacobianstructure
        dComp_dU = dComp_dU.tocoo()
        # Area
        Jacobian = np.append(Jacobian,dComp_dA)
        # Force
        Jacobian = np.append(Jacobian,np.ones(self.N)*-1)
        # Displacements
        Jacobian = np.append(Jacobian,dComp_dU.data)
        return Jacobian
    
    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        """Prints information at every Ipopt iteration."""

        self.it = iter_count
        self.obj_hist = np.append(self.obj_hist, self.volume_phys)
    
class Layopt_IPOPT_VL_Buck(IPOPT_Problem):
    def __init__(self, NN, N_cell, N, M, myBCs, joint_cost, B, f, s_c, s_t, E, s_buck, cell_mapping_matrix):
        # Use the class constructor to declare the variable used by the optimizer during the optimization
        self.NN = NN # Number of members of the cells
        self.N_cell = N_cell # Number of members of a single cell
        self.N = N # Number of members of the Ground Structure
        self.M = M # Number of DOFs of the Groud Structure
        self.myBCs = myBCs # BC class
        self.l_true = myBCs.ground_structure_length # Physical member lengths [mm]
        self.l = self.l_true + joint_cost * np.max(self.l_true) # indipendency from the number of cells [mm]
        self.B = B # Equilibrium forces matrix 
        self.f = f # External forces [N]
        self.s_c = s_c # Max compression allowable stress [MPa]
        self.s_t = s_t # Max tension allowable stress [MPa]
        self.E = E # Young's modulus [MPa]
        self.cell_mapping_matrix = cell_mapping_matrix # Mapping function used to distribute the cells over the structure domain
        self.n_topologies_cell = np.size(cell_mapping_matrix, 1) # Number of different cell topologies
        
        # VL parameters
        if self.myBCs.isReduced:
            self.l_cell = self.myBCs.l_cell_full.copy() + joint_cost * np.max(self.l_true) # change, different cell size per different reduced cell
        else:
            self.l_cell = self.l[:N_cell[0]].copy() # Length of every member of one cell
        
        # Buckling specific parameters
        self.s_buck = s_buck
        self.a_cr = -s_c * self.l_true**2 / self.s_buck # Evaluation of the critical section for the members
        
        # Sparsity pattern of B matrix
        self.B_coo = self.B.tocoo()
        self.B_row, self.B_column = self.B_coo.row,self.B_coo.col 
        self.B_T_coo = self.B.T.tocoo() # Horrible but necessary to keep the very same order of dComp_dU
        self.B_T_row, self.B_T_column = self.B_T_coo.row,self.B_T_coo.col       
        
        # Create the indexing variables used for splitting the design variables vector x
        self.area = slice(0,NN)
        self.force = slice(NN,NN+N)
        self.U = slice(NN+N,NN+N+M)
        
        # History values
        self.it = 0
        self.obj_hist = []

    def objective(self, x):
        """Returns the scalar value of the objective given x."""        
        area_phys,_ = self.calc_area_phys(x[self.area])
        self.volume_phys = self.l_true.T @ area_phys
        return self.l.T @ area_phys

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        g = self.calc_len_phys()
        grad = np.zeros(x.size)
        grad[self.area] = g
        return grad

    def constraints(self, x):
        """Returns the constraints."""
        area_phys,_ = self.calc_area_phys(x[self.area]) # initialize a_phys
        ### Equilibrium (M eq)
        equilibrium = self.B @ x[self.force] - self.f
        ### Stress (2*N eq)
        stress_c = x[self.force] - self.s_c * area_phys
        stress_t = x[self.force] - self.s_t * area_phys
        stress = np.append(stress_c, stress_t)
        ### Compatibility (N eq)
        compatibility = area_phys * self.E/self.l_true * (self.B.T @ x[self.U]) - x[self.force] 
        ### Buckling (N eq)
        buckling = x[self.force] + (self.s_buck/self.l_true**2) * area_phys**2
        cons = np.concatenate((equilibrium, stress, compatibility, buckling))

        return cons
    
    def jacobianstructure(self):
        """ Returns the row and column indices for non-zero vales of the
        Jacobian. """
        # DIM = Number of constraint functions X number of design variables
        area_phys_id = self.calc_area_phys_id()
        ### Equilibrium (M eq)
        row, column = self.B_row, self.B_column+self.NN # the indexes are translated to the corresponding design variables (force)
        ### Stress (2*N eq)
        row = np.append(row,range(self.M,self.M+self.N*2))
        column = np.append(column,np.tile(area_phys_id,2))
        row = np.append(row,range(self.M,self.M+self.N*2))
        column = np.append(column,np.r_[self.force,self.force])
        ### Compatibility (N eq)
        row = np.append(row,range(self.M+self.N*2,self.M+self.N*3))
        column = np.append(column,area_phys_id)
        row = np.append(row,range(self.M+self.N*2,self.M+self.N*3))
        column = np.append(column,np.r_[self.force])
        row = np.append(row,self.B_T_row + self.M+self.N*2) # B is transposed
        column = np.append(column,self.B_T_column+self.N+self.NN)
        ### Buckling (N eq)
        row = np.append(row,range(self.M+self.N*3,self.M+self.N*4))
        column = np.append(column,area_phys_id)
        row = np.append(row,range(self.M+self.N*3,self.M+self.N*4))
        column = np.append(column,np.r_[self.force])
        return row,column

    def jacobian(self, x):
        """ Returns the Jacobian of the constraints with respect to x. """
        # DIM = Number of constraint functions X number of design variables
        # Physical areas
        area_phys,_ = self.calc_area_phys(x[self.area]) # initialize a_phys
        ### Equilibrium (M eq)
        Jacobian = self.B_coo.data
        ### Stress (2*N eq)
        Jacobian = np.append(Jacobian,np.ones(self.N)*(- self.s_c))
        Jacobian = np.append(Jacobian,np.ones(self.N)*(- self.s_t))
        Jacobian = np.append(Jacobian,np.ones(self.N))
        Jacobian = np.append(Jacobian,np.ones(self.N))
        ### Compatibility (N eq)
        dComp_dA = []
        dComp_dA = self.B.T @ x[self.U] * self.E / self.l_true # Diagonal term, i==j
        dComp_dU = (sp.diags(area_phys * self.E / self.l_true) @ self.B.T) # Multiply the column of B.T and the main diagnonal of diag
        dComp_dU.sort_indices() # Needed to match the order of B.T given in jacobianstructure
        dComp_dU = dComp_dU.tocoo()
        # Area
        Jacobian = np.append(Jacobian,dComp_dA)
        # Force
        Jacobian = np.append(Jacobian,np.ones(self.N)*-1)
        # Displacements
        Jacobian = np.append(Jacobian,dComp_dU.data)
        ### Buckling (N eq)
        # Area
        Jacobian = np.append(Jacobian,2*area_phys*self.s_buck/self.l_true**2)
        # Force
        Jacobian = np.append(Jacobian,np.ones(self.N))
        return Jacobian
    
    def hessianstructure(self):
        """ Returns the row and column indices for non-zero vales of the
        Hessian. """
        # DIM = Number of design variables X number of design variables
        M = self.M
        N = self.N
        NN = self.NN
        
        ### Compatibility (N eq) x[self.area] * self.E/self.l_true * (self.B.T @ x[self.U]) - x[self.force]
        ## Area self.B.T @ x[self.U] * self.E / self.l_true
        # Area -
        
        # Force -

        # Displacements 
        temp_i, temp_j = np.meshgrid(np.arange(NN+N,NN+N+M, dtype='int'), np.arange(NN, dtype='int'))
        i = temp_i.ravel()
        j = temp_j.ravel()
        
        ### Buckling (N eq) buckling = x[self.force] + (self.s_buck/self.l_true**2) * x[self.area]**2
        ## Area 2*x[self.area]*self.s_buck/self.l_true**2
        # Area
        i = np.append(i,np.arange(NN, dtype='int'))
        j = np.append(j,np.arange(NN, dtype='int'))

        # here you have to sort and to eliminate redundant variables
        return i, j

    def hessian(self, x, l, obj_factor):
        """  Returns the non-zero values of the Hessian. """ 
        # DIM = Number of design variables X number of design variables
        M = self.M
        N = self.N
        NN = self.NN
        
        # ID of the constraints
        comp_id = slice(M+N*2, M+N*3)
        buck_id = slice(M+N*3, M+N*4)
        
        ### Objective function (N eq) the hessian of the obj function is empty
        # Hessian = obj_factor * NULL
        
        ### Equilibrium (M eq) self.B @ x[self.force] - self.f
        ## Area -

        ## Force self.B_coo.data

        ## Displacements -
        
        ### Stress (2*N eq)
        ## Area -
        
        ## Force -

        ## Displacements -

        ### Compatibility (N eq) x[self.area] * self.E/self.l_true * (self.B.T @ x[self.U]) - x[self.force]
        ## Area self.B.T @ x[self.U] * self.E / self.l_true
        # Area -
        
        # Force -

        # Displacements 
        temp = (sp.diags(l[comp_id] * self.E / self.l_true) @ self.B.T).tocoo()
        j, i = temp.row, temp.col + (NN+N)
        try:
            j[j>=NN] = j % NN # Only the first cellule is important for sensitivity
        except:
            pass
        data = temp.data 

        ## Force np.ones(self.N)*-1

        ## Displacements (sp.diags(x[self.area] * self.E / self.l_true) @ self.B.T)
        # Area
        # Symmetric, already considered
        
        # Force -

        # Displacements -
        
        ### Buckling (N eq) buckling = x[self.force] + (self.s_buck/self.l_true**2) * x[self.area]**2
        ## Area 2*x[self.area]*self.s_buck/self.l_true**2
        # Area
        i = np.append(i.ravel(),np.tile(np.arange(NN, dtype='int'),self.myBCs.ncel_str))
        j = np.append(j.ravel(),np.tile(np.arange(NN, dtype='int'),self.myBCs.ncel_str))
        data = np.append(data, 2*self.s_buck/self.l_true**2 * l[buck_id]) 
        # Force -

        # Displacements -
        
        ## Force np.ones(self.N)

        ## Displacements -
        
        
        Hessian = sp.coo_matrix((data, (i, j)), shape=(x.size, x.size)).tocsc() # check which one is better 
        row, col = self.hessianstructure()
        out = np.array(Hessian[row, col]).ravel()
        
        return out
    
    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        """Prints information at every Ipopt iteration."""

        self.it = iter_count
        self.obj_hist = np.append(self.obj_hist, self.volume_phys)
        
class Layopt_IPOPT_Buck_multiload(IPOPT_Problem):
    def __init__(self, N, M, l_true, joint_cost, B, f, dofs, s_c, s_t, E, s_buck):
        # Use the class constructor to declare the variable used by the optimizer during the optimization
        self.N = N # Number of member of the Ground Structure
        self.M = M # Number of DOFs of the Groud Structure
        self.l_true = l_true # Physical member lenghts [mm]
        self.l = l_true + joint_cost * np.max(l_true) # indipendency from the number of cells [mm]
        self.B = B # Equilibrium forces matrix 
        self.f = f # External forces [N]
        self.dofs = dofs
        self.s_c = s_c # Max compression allowable stress [MPa]
        self.s_t = s_t # Max tension allowable stress [MPa]
        self.E = E # Young's modulus [MPa]
        self.n_load_cases = f.shape[-1]
        
        # Buckling specific parameters
        self.s_buck = s_buck
        self.a_cr = -s_c * l_true**2 / self.s_buck # Evaluation of the critical section for the members
        
        # Sparsity pattern of B matrix
        self.B_coo = self.B.tocoo()
        self.B_row, self.B_column = self.B_coo.row,self.B_coo.col 
        self.B_T_coo = self.B.T.tocoo() # Horrible but necessary to conserve the very same order of dComp_dU
        self.B_T_row, self.B_T_column = self.B_T_coo.row,self.B_T_coo.col       
        
        # Create the indexing variables used for splitting the design variables vector x
        self.area = slice(0,N)
        self.force = []
        self.U = []
        for p in range(self.n_load_cases):
            self.force.append(slice(N+(p)*N,N+(p+1)*N))
            self.U.append(slice(N+self.n_load_cases*N+(p)*M,N+self.n_load_cases*N+(p+1)*M))
        
        # History values
        self.it = 0
        self.obj_hist = []

    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        self.volume_phys = self.l_true.T @ x[self.area]
        return self.l.T @ x[self.area]

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        grad = np.zeros(x.size)
        grad[self.area] = self.l
        return grad

    def constraints(self, x):
        """Returns the constraints."""
        ### Equilibrium (M*p eq)
        equilibrium = np.array([])
        for p in range(self.n_load_cases):
            eq = self.B @ x[self.force[p]] - self.f[:,p]*self.dofs
            equilibrium = np.append(equilibrium, eq)
        ### Stress (2*N*p eq)
        stress = np.array([])
        for p in range(self.n_load_cases):
            stress_c = x[self.force[p]] - self.s_c * x[self.area]
            stress = np.append(stress, stress_c)
        for p in range(self.n_load_cases):
            stress_t = x[self.force[p]] - self.s_t * x[self.area]
            stress = np.append(stress, stress_t)
        ### Compatibility (N*p eq)
        compatibility = np.array([])
        for p in range(self.n_load_cases):
            comp = x[self.area] * self.E/self.l_true * (self.B.T @ x[self.U[p]]) - x[self.force[p]]
            compatibility = np.append(compatibility, comp)
        ### Buckling (N eq)
        buckling = np.array([])
        for p in range(self.n_load_cases):
            buck = x[self.force[p]] + (self.s_buck/self.l_true**2) * x[self.area]**2
            buckling = np.append(buckling, buck)
        cons = np.concatenate((equilibrium, stress, compatibility, buckling))
        return cons
    
    def jacobianstructure(self):
        """ Returns the row and column indices for non-zero vales of the
        Jacobian. """
        # DIM = Number of constraint functions X number of design variables
        ### Equilibrium (M*p eq)
        row, column = self.B_row, self.B_column+self.N # the indexes are translated to the corresponding design variables (force)
        for p in range(self.n_load_cases-1):
            row = np.append(row,self.B_row+self.M+p*self.M)
            column = np.append(column,self.B_column+self.N+(p+1)*self.N)
        ### Stress (2*N eq)
        for p in range(self.n_load_cases): 
            row = np.append(row,range(self.M*self.n_load_cases+p*self.N, self.M*self.n_load_cases+(p+1)*self.N))
            column = np.append(column,np.r_[self.area])
            row = np.append(row,range(self.M*self.n_load_cases+p*self.N, self.M*self.n_load_cases+(p+1)*self.N))
            column = np.append(column,np.r_[self.force[p]])
        for p in range(self.n_load_cases):
            row = np.append(row,range(self.M*self.n_load_cases+self.n_load_cases*self.N+p*self.N, self.M*self.n_load_cases+self.n_load_cases*self.N+(p+1)*self.N))
            column = np.append(column,np.r_[self.area])
            row = np.append(row,range(self.M*self.n_load_cases+self.n_load_cases*self.N+p*self.N, self.M*self.n_load_cases+self.n_load_cases*self.N+(p+1)*self.N))
            column = np.append(column,np.r_[self.force[p]])
        ### Compatibility (N eq)
        for p in range(self.n_load_cases):
            row = np.append(row,range(self.M*self.n_load_cases+self.N*2*self.n_load_cases+p*self.N, self.M*self.n_load_cases+self.N*2*self.n_load_cases+(p+1)*self.N))
            column = np.append(column,np.r_[self.area])
            row = np.append(row,range(self.M*self.n_load_cases+self.N*2*self.n_load_cases+p*self.N, self.M*self.n_load_cases+self.N*2*self.n_load_cases+(p+1)*self.N))
            column = np.append(column,np.r_[self.force[p]])
            row = np.append(row,self.B_T_row + self.M*self.n_load_cases+self.N*2*self.n_load_cases+p*self.N) # B is transposed
            column = np.append(column,self.B_T_column+self.N+self.N*self.n_load_cases+p*self.M) # U
        ### Buckling (N eq)
        for p in range(self.n_load_cases):
            row = np.append(row,range(self.M*self.n_load_cases+self.N*3*self.n_load_cases+p*self.N, self.M*self.n_load_cases+self.N*3*self.n_load_cases+(p+1)*self.N))
            column = np.append(column,np.r_[self.area])
            row = np.append(row,range(self.M*self.n_load_cases+self.N*3*self.n_load_cases+p*self.N, self.M*self.n_load_cases+self.N*3*self.n_load_cases+(p+1)*self.N))
            column = np.append(column,np.r_[self.force[p]])
        return row,column

    def jacobian(self, x):
        """ Returns the Jacobian of the constraints with respect to x. """
        # DIM = Number of constraint functions X number of design variables
        ### Equilibrium (M eq)
        Jacobian = self.B_coo.data
        for p in range(self.n_load_cases-1): 
            Jacobian = np.append(Jacobian,self.B_coo.data)   
        ### Stress (2*N eq)
        for p in range(self.n_load_cases):
            Jacobian = np.append(Jacobian,np.ones(self.N)*(- self.s_c))
            Jacobian = np.append(Jacobian,np.ones(self.N))
        for p in range(self.n_load_cases):
            Jacobian = np.append(Jacobian,np.ones(self.N)*(- self.s_t))
            Jacobian = np.append(Jacobian,np.ones(self.N))
        ### Compatibility (N eq) x[self.area] * self.E/self.l_true * (self.B.T @ x[self.U]) - x[self.force]
        for p in range(self.n_load_cases):
            dComp_dA = self.B.T @ x[self.U[p]] * self.E / self.l_true # Diagonal term, i==j
            dComp_dU = (sp.diags(x[self.area] * self.E / self.l_true) @ self.B.T) # Multiply the column of B.T and the main diagnonal of diag
            dComp_dU.sort_indices() # Needed to match the order of B.T given in jacobianstructure
            dComp_dU = dComp_dU.tocoo()
            # Area
            Jacobian = np.append(Jacobian,dComp_dA)
            # Force
            Jacobian = np.append(Jacobian,np.ones(self.N)*-1)
            # Displacements
            Jacobian = np.append(Jacobian,dComp_dU.data)
        ### Buckling (N eq)
        for p in range(self.n_load_cases):
            # Area
            Jacobian = np.append(Jacobian,2*x[self.area]*self.s_buck/self.l_true**2)
            # Force
            Jacobian = np.append(Jacobian,np.ones(self.N))
        return Jacobian
    
    def hessianstructure(self):
        """ Returns the row and column indices for non-zero vales of the
        Hessian. NO HESSIAN, NOT WORKING"""
        # DIM = Number of design variables X number of design variables
        M = self.M
        N = self.N
        
        ### Compatibility (N eq) x[self.area] * self.E/self.l_true * (self.B.T @ x[self.U]) - x[self.force]
        ## Area self.B.T @ x[self.U] * self.E / self.l_true
        # Area -
        
        # Force -

        # Displacements 
        i, j = np.meshgrid(np.arange(2*N,2*N+M, dtype='int'), np.arange(N, dtype='int'))
        
        ### Buckling (N eq) buckling = x[self.force] + (self.s_buck/self.l_true**2) * x[self.area]**2
        ## Area 2*x[self.area]*self.s_buck/self.l_true**2
        # Area
        i = np.append(i.ravel(),np.arange(N, dtype='int'))
        j = np.append(j.ravel(),np.arange(N, dtype='int'))

        return i, j

    def hessian(self, x, l, obj_factor):
        """ Returns the non-zero values of the Hessian. """
        # DIM = Number of design variables X number of design variables
        M = self.M
        N = self.N
        
        # ID of the constraints
        comp_id = slice(M+N*2, M+N*3)
        buck_id = slice(M+N*3, M+N*4)
        
        ### Objective function (N eq) the hessian of the obj function is empty
        # Hessian = obj_factor * NULL
        
        ### Equilibrium (M eq) self.B @ x[self.force] - self.f
        ## Area -

        ## Force self.B_coo.data

        ## Displacements -
        
        ### Stress (2*N eq)
        ## Area -
        
        ## Force -

        ## Displacements -

        ### Compatibility (N eq) x[self.area] * self.E/self.l_true * (self.B.T @ x[self.U]) - x[self.force]
        ## Area self.B.T @ x[self.U] * self.E / self.l_true
        # Area -
        
        # Force -

        # Displacements 
        temp = (sp.diags(l[comp_id] * self.E / self.l_true) @ self.B.T)
        temp.sort_indices() # Needed to match the order of B.T given in jacobianstructure
        temp = temp.tocoo()
        j, i = temp.row, temp.col + (2*N)
        data = temp.data

        ## Force np.ones(self.N)*-1

        ## Displacements (sp.diags(x[self.area] * self.E / self.l_true) @ self.B.T)
        # Area
        # Symmetric, already considered
        
        # Force -

        # Displacements -
        
        ### Buckling (N eq) buckling = x[self.force] + (self.s_buck/self.l_true**2) * x[self.area]**2
        ## Area 2*x[self.area]*self.s_buck/self.l_true**2
        # Area
        i = np.append(i.ravel(),np.arange(N, dtype='int'))
        j = np.append(j.ravel(),np.arange(N, dtype='int'))
        data = np.append(data,2*self.s_buck/self.l_true**2 * l[buck_id]) 
        # Force -

        # Displacements -
        
        ## Force np.ones(self.N)

        ## Displacements -
        
        Hessian = sp.coo_matrix((data, (i, j)), shape=(x.size, x.size)).tocsc() # check which one is better 
        row, col = self.hessianstructure()
        out = np.array(Hessian[row, col]).ravel()
        
        return out
    
    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        """Prints information at every Ipopt iteration."""

        self.it = iter_count
        self.obj_hist = np.append(self.obj_hist, self.volume_phys)

######################################################################## 
 
class IPOPT_Problem_Free_Form():
    """ Parent class used to store common methods for the different IPOPT classes  """
        
    def calc_area_phys(self, area_optim):
        """ This function is used to calculate the true physical distribution of the sections of the truss starting from the areas of the cells """
        area_cell = area_optim[:self.myBCs.N_cell]
        area_aperiodic = area_optim[self.myBCs.N_cell:]
        temp = np.concatenate([np.tile(area_cell,self.myBCs.ncel_str),area_aperiodic])
        return temp
    
    def calc_area_phys_sect(self, area_optim):
        """ This function is used to calculate the true physical distribution of the sections of the truss starting from the areas of the cells and repetitive section """
        area_cell = area_optim[:self.myBCs.N_cell]
        area_sect = area_optim[self.myBCs.N_cell:]
        temp = np.concatenate([np.tile(area_cell,self.myBCs.ncel_str),np.tile(area_sect,self.myBCs.number_section)])
        return temp
    
    def calc_area_optim(self, area_phys):
        """ This function is used to calculate the distribution of the sections of the truss for the optimization starting from the true members area """
        p_cell = np.array(np.where((self.myBCs.ground_structure[:,2]!=-1) & (self.myBCs.ground_structure[:,3]==0))).ravel()
        a = np.array(np.where(self.myBCs.ground_structure[:,2]==-1)).ravel()  
        temp = np.concatenate([area_phys[p_cell], area_phys[a]])    
        return temp
    
    def calc_area_phys_id(self):
        """ This function is used to calculate the distribution of the sections IDs of the truss """
        id_area_cell = np.arange(self.NN)[:self.myBCs.N_cell]
        id_area_aperiodic = np.arange(self.NN)[self.myBCs.N_cell:]
        temp = np.concatenate([np.tile(id_area_cell,self.myBCs.ncel_str),id_area_aperiodic])
        return temp
    
    def calc_area_phys_id_sect(self):
        """ This function is used to calculate the distribution of the sections IDs of the truss """
        id_area_cell = np.arange(self.NN)[:self.myBCs.N_cell]
        id_area_sect = np.arange(self.NN)[self.myBCs.N_cell:]
        temp = np.concatenate([np.tile(id_area_cell,self.myBCs.ncel_str),np.tile(id_area_sect,self.myBCs.number_section)])
        return temp
    
    def calc_len_optim(self):
        """ This function is used to calculate the sum of the distribution of the length of the truss
        Used for gradient calculation """
        a = np.array(np.where(self.myBCs.ground_structure[:,2]==-1)).ravel()        
        temp = np.concatenate([self.l_cell * self.myBCs.ncel_str, self.l[a]])     
        return temp
    
    def calc_len_optim_sect(self):
        """ This function is used to calculate the sum of the distribution of the length of the truss for the periodic section case
        Used for gradient calculation """
        a = np.array(np.where(self.myBCs.ground_structure[:,2]==-1)).ravel()        
        temp = np.concatenate([self.l_cell * self.myBCs.ncel_str, self.l_sec * self.myBCs.number_section])     
        return temp
        
class Layopt_IPOPT_VL_Buck_Free_Form(IPOPT_Problem_Free_Form):
    def __init__(self, NN, N_cell, N, M, myBCs, joint_cost, B, f, s_c, s_t, E, s_buck):
        # Use the class constructor to declare the variable used by the optimizer during the optimization
        self.NN = NN # Number of members of the area design variables
        self.N_cell = N_cell # Number of members of a single cell
        self.N = N # Number of members of the Ground Structure
        self.M = M # Number of DOFs of the Groud Structure
        self.myBCs = myBCs # BC class
        self.l_true = myBCs.ground_structure_length # Physical member lengths [mm]
        self.l = self.l_true + joint_cost * np.max(self.l_true) # indipendency from the number of cells [mm]
        self.B = B # Equilibrium forces matrix 
        self.f = f # External forces [N]
        self.s_c = s_c # Max compression allowable stress [MPa]
        self.s_t = s_t # Max tension allowable stress [MPa]
        self.E = E # Young's modulus [MPa]
        
        # VL parameters 
        self.l_cell = self.myBCs.ground_structure_length_cell.copy() + joint_cost * np.max(self.l_true)
        
        # Buckling specific parameters
        self.s_buck = s_buck
        self.a_cr = -s_c * self.l_true**2 / self.s_buck # Evaluation of the critical section for the members
        
        # Sparsity pattern of B matrix
        self.B_coo = self.B.tocoo()
        self.B_row, self.B_column = self.B_coo.row,self.B_coo.col 
        self.B_T_coo = self.B.T.tocoo() # Horrible but necessary to keep the very same order of dComp_dU
        self.B_T_row, self.B_T_column = self.B_T_coo.row,self.B_T_coo.col       
        
        # Create the indexing variables used for splitting the design variables vector x
        self.area = slice(0,NN)
        self.force = slice(NN,NN+N)
        self.U = slice(NN+N,NN+N+M)
        
        # History values
        self.it = 0
        self.obj_hist = []
        self.constr_equilib_hist = []
        self.constr_s_c_hist = []
        self.constr_s_t_hist = []
        self.constr_comp_hist = []
        self.constr_buck_hist = []

    def objective(self, x):
        """Returns the scalar value of the objective given x."""        
        area_phys = self.calc_area_phys(x[self.area])
        self.volume_phys = self.l_true.T @ area_phys
        return self.l.T @ area_phys

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        g = self.calc_len_optim()
        grad = np.zeros(x.size)
        grad[self.area] = g
        return grad

    def constraints(self, x):
        """Returns the constraints."""
        forceee = x[self.force]
        displ = x[self.U]
        area_phys = self.calc_area_phys(x[self.area]) # initialize a_phys
        ### Equilibrium (M eq)
        equilibrium = self.B @ x[self.force] - self.f
        self.viol_equilib = np.max(np.abs(equilibrium))
        ### Stress (2*N eq)
        stress_c = x[self.force] - self.s_c * area_phys
        stress_t = x[self.force] - self.s_t * area_phys
        self.viol_s_c = np.min(np.abs(np.append(stress_c,0)))
        self.viol_s_t = np.max(np.append(stress_t,0))
        stress = np.append(stress_c, stress_t)
        ### Compatibility (N eq)
        compatibility = (area_phys * self.E/self.l_true * (self.B.T @ x[self.U]) - x[self.force])# / 1e4 # scaled because very different with respect to other entries
        self.viol_comp = np.max(np.abs(compatibility))
        ### Buckling (N eq)
        buckling = x[self.force] + (self.s_buck/self.l_true**2) * area_phys**2
        self.viol_buck = np.min(np.abs(np.append(buckling,0)))
        cons = np.concatenate((equilibrium, stress, compatibility, buckling))

        return cons
    
    def jacobianstructure(self):
        """ Returns the row and column indices for non-zero vales of the
        Jacobian. """
        # DIM = Number of constraint functions X number of design variables
        area_phys_id = self.calc_area_phys_id()
        ### Equilibrium (M eq)
        row, column = self.B_row, self.B_column+self.NN # the indexes are translated to the corresponding design variables (force)
        ### Stress (2*N eq)
        row = np.append(row,range(self.M,self.M+self.N*2))
        column = np.append(column,np.tile(area_phys_id,2))
        row = np.append(row,range(self.M,self.M+self.N*2))
        column = np.append(column,np.r_[self.force,self.force])
        ### Compatibility (N eq)
        row = np.append(row,range(self.M+self.N*2,self.M+self.N*3))
        column = np.append(column,area_phys_id)
        row = np.append(row,range(self.M+self.N*2,self.M+self.N*3))
        column = np.append(column,np.r_[self.force])
        row = np.append(row,self.B_T_row + self.M+self.N*2) # B is transposed
        column = np.append(column,self.B_T_column+self.N+self.NN)
        ### Buckling (N eq)
        row = np.append(row,range(self.M+self.N*3,self.M+self.N*4))
        column = np.append(column,area_phys_id)
        row = np.append(row,range(self.M+self.N*3,self.M+self.N*4))
        column = np.append(column,np.r_[self.force])
        return row,column

    def jacobian(self, x):
        """ Returns the Jacobian of the constraints with respect to x. """
        # DIM = Number of constraint functions X number of design variables
        # Physical areas
        area_phys = self.calc_area_phys(x[self.area]) # initialize a_phys
        ### Equilibrium (M eq)
        Jacobian = self.B_coo.data
        ### Stress (2*N eq)
        Jacobian = np.append(Jacobian,np.ones(self.N)*(- self.s_c))
        Jacobian = np.append(Jacobian,np.ones(self.N)*(- self.s_t))
        Jacobian = np.append(Jacobian,np.ones(self.N))
        Jacobian = np.append(Jacobian,np.ones(self.N))
        ### Compatibility (N eq)
        dComp_dA = []
        dComp_dA = self.B.T @ x[self.U] * self.E / self.l_true # Diagonal term, i==j
        dComp_dU = (sp.diags(area_phys * self.E / self.l_true) @ self.B.T) # Multiply the column of B.T and the main diagnonal of diag
        dComp_dU.sort_indices() # Needed to match the order of B.T given in jacobianstructure
        dComp_dU = dComp_dU.tocoo()
        # Area
        Jacobian = np.append(Jacobian,dComp_dA)
        # Force
        Jacobian = np.append(Jacobian,np.ones(self.N)*-1)
        # Displacements
        Jacobian = np.append(Jacobian,dComp_dU.data)
        ### Buckling (N eq)
        # Area
        Jacobian = np.append(Jacobian,2*area_phys*self.s_buck/self.l_true**2)
        # Force
        Jacobian = np.append(Jacobian,np.ones(self.N))
        return Jacobian
    
    def hessianstructure(self):
        """ Returns the row and column indices for non-zero vales of the
        Hessian. """
        # DIM = Number of design variables X number of design variables
        M = self.M
        N = self.N
        NN = self.NN
        
        ### Compatibility (N eq) x[self.area] * self.E/self.l_true * (self.B.T @ x[self.U]) - x[self.force]
        ## Area self.B.T @ x[self.U] * self.E / self.l_true
        # Area -
        
        # Force -

        # Displacements 
        temp_i, temp_j = np.meshgrid(np.arange(NN+N,NN+N+M, dtype='int'), np.arange(NN, dtype='int'))
        i = temp_i.ravel()
        j = temp_j.ravel()
        
        ### Buckling (N eq) buckling = x[self.force] + (self.s_buck/self.l_true**2) * x[self.area]**2
        ## Area 2*x[self.area]*self.s_buck/self.l_true**2
        # Area
        i = np.append(i,np.arange(NN, dtype='int'))
        j = np.append(j,np.arange(NN, dtype='int'))

        # here you have to sort and to eliminate redundant variables
        return i, j

    def hessian(self, x, l, obj_factor):
        """  Returns the non-zero values of the Hessian. """ 
        # DIM = Number of design variables X number of design variables
        M = self.M
        N = self.N
        NN = self.NN
        N_cell = self.N_cell
        NN_cell = N_cell * self.myBCs.ncel_str
        
        # ID of the constraints
        comp_id = slice(M+N*2, M+N*3)
        buck_id = slice(M+N*3, M+N*4)
        
        ### Objective function (N eq) the hessian of the obj function is empty
        # Hessian = obj_factor * NULL
        
        ### Equilibrium (M eq) self.B @ x[self.force] - self.f
        ## Area -

        ## Force self.B_coo.data

        ## Displacements -
        
        ### Stress (2*N eq)
        ## Area -
        
        ## Force -

        ## Displacements -

        ### Compatibility (N eq) x[self.area] * self.E/self.l_true * (self.B.T @ x[self.U]) - x[self.force]
        ## Area self.B.T @ x[self.U] * self.E / self.l_true
        # Area -
        
        # Force -

        # Displacements 
        temp = (sp.diags(l[comp_id] * self.E / self.l_true) @ self.B.T).tocoo()
        j, i = temp.row, temp.col + (NN+N)
        try:
            j[(j<NN_cell) & (j>=N_cell)] = j % N_cell # Only the first cellule is important for sensitivity
        except:
            pass
        try:
            j[j>=NN_cell] = j - NN_cell + N_cell # Only the first cellule is important for sensitivity
        except:
            pass
        data = temp.data 

        ## Force np.ones(self.N)*-1

        ## Displacements (sp.diags(x[self.area] * self.E / self.l_true) @ self.B.T)
        # Area
        # Symmetric, already considered
        
        # Force -

        # Displacements -
        
        ### Buckling (N eq) buckling = x[self.force] + (self.s_buck/self.l_true**2) * x[self.area]**2
        ## Area 2*x[self.area]*self.s_buck/self.l_true**2
        # Area
        i = np.append(i.ravel(),np.concatenate([np.tile(np.arange(N_cell, dtype='int'),self.myBCs.ncel_str),np.arange(N_cell, NN, dtype='int')]))
        j = np.append(j.ravel(),np.concatenate([np.tile(np.arange(N_cell, dtype='int'),self.myBCs.ncel_str),np.arange(N_cell, NN, dtype='int')]))
        data = np.append(data, 2*self.s_buck/self.l_true**2 * l[buck_id]) 
        # Force -

        # Displacements -
        
        ## Force np.ones(self.N)

        ## Displacements -
        
        
        Hessian = sp.coo_matrix((data, (i, j)), shape=(x.size, x.size)).tocsc() # check which one is better 
        row, col = self.hessianstructure()
        out = np.array(Hessian[row, col]).ravel()
        
        return out
    
    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        """Prints information at every Ipopt iteration."""

        self.it = iter_count
        self.obj_hist = np.append(self.obj_hist, self.volume_phys)
        self.constr_equilib_hist = np.append(self.constr_equilib_hist, self.viol_equilib)
        self.constr_s_c_hist = np.append(self.constr_s_c_hist, self.viol_s_c)
        self.constr_s_t_hist = np.append(self.constr_s_t_hist, self.viol_s_t)
        self.constr_comp_hist = np.append(self.constr_comp_hist, self.viol_comp)
        self.constr_buck_hist = np.append(self.constr_buck_hist, self.viol_buck)
        
        print("Phys. Vol: {0:.5f},  Eq. Cons: {1:.2e},  SC. Cons: {2:.2e},  ST. Cons: {3:.2e},  Comp. Cons: {4:.2e},  Buck. Cons: {5:.2e}\n".format(\
					self.obj_hist[-1], self.constr_equilib_hist[-1], self.constr_s_c_hist[-1], \
                    self.constr_s_t_hist[-1], self.constr_comp_hist[-1], self.constr_buck_hist[-1]))
        
class Layopt_IPOPT_VL_Buck_Free_Form_Section(IPOPT_Problem_Free_Form):
    def __init__(self, NN, N_cell, N_sec, N, M, myBCs, joint_cost, B, f, s_c, s_t, E, s_buck):
        # Use the class constructor to declare the variable used by the optimizer during the optimization
        self.NN = NN # Number of members of the area design variables
        self.N_cell = N_cell # Number of members of a single cell
        self.N_sec = N_sec # Number of members of a single section
        self.N = N # Number of members of the Ground Structure
        self.M = M # Number of DOFs of the Groud Structure
        self.myBCs = myBCs # BC class
        self.l_true = myBCs.ground_structure_length # Physical member lengths [mm]
        self.l = self.l_true + joint_cost * np.max(self.l_true) # indipendency from the number of cells [mm]
        self.B = B # Equilibrium forces matrix 
        self.f = f # External forces [N]
        self.s_c = s_c # Max compression allowable stress [MPa]
        self.s_t = s_t # Max tension allowable stress [MPa]
        self.E = E # Young's modulus [MPa]
        
        # VL parameters 
        self.l_cell = self.myBCs.ground_structure_length_cell.copy() + joint_cost * np.max(self.l_true)
        self.l_sec = self.myBCs.ground_structure_length_section.copy() + joint_cost * np.max(self.l_true)
        
        # Buckling specific parameters
        self.s_buck = s_buck
        self.a_cr = -s_c * self.l_true**2 / self.s_buck # Evaluation of the critical section for the members
        
        # Sparsity pattern of B matrix
        self.B_coo = self.B.tocoo()
        self.B_row, self.B_column = self.B_coo.row,self.B_coo.col 
        self.B_T_coo = self.B.T.tocoo() # Horrible but necessary to keep the very same order of dComp_dU
        self.B_T_row, self.B_T_column = self.B_T_coo.row,self.B_T_coo.col       
        
        # Create the indexing variables used for splitting the design variables vector x
        self.area = slice(0,NN)
        self.force = slice(NN,NN+N)
        self.U = slice(NN+N,NN+N+M)
        
        # History values
        self.it = 0
        self.obj_hist = []
        self.constr_equilib_hist = []
        self.constr_s_c_hist = []
        self.constr_s_t_hist = []
        self.constr_comp_hist = []
        self.constr_buck_hist = []

    def objective(self, x):
        """Returns the scalar value of the objective given x."""        
        area_phys = self.calc_area_phys_sect(x[self.area])
        self.volume_phys = self.l_true.T @ area_phys
        return self.l.T @ area_phys

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        g = self.calc_len_optim_sect()
        grad = np.zeros(x.size)
        grad[self.area] = g
        return grad

    def constraints(self, x):
        """Returns the constraints."""
        forceee = x[self.force]
        displ = x[self.U]
        area_phys = self.calc_area_phys_sect(x[self.area]) # initialize a_phys
        ### Equilibrium (M eq)
        equilibrium = self.B @ x[self.force] - self.f
        self.viol_equilib = np.max(np.abs(equilibrium))
        ### Stress (2*N eq)
        stress_c = x[self.force] - self.s_c * area_phys
        stress_t = x[self.force] - self.s_t * area_phys
        self.viol_s_c = np.min(np.abs(np.append(stress_c,0)))
        self.viol_s_t = np.max(np.append(stress_t,0))
        stress = np.append(stress_c, stress_t)
        ### Compatibility (N eq)
        compatibility = (area_phys * self.E/self.l_true * (self.B.T @ x[self.U]) - x[self.force])# / 1e4 # scaled because very different with respect to other entries
        self.viol_comp = np.max(np.abs(compatibility))
        ### Buckling (N eq)
        buckling = x[self.force] + (self.s_buck/self.l_true**2) * area_phys**2
        self.viol_buck = np.min(np.abs(np.append(buckling,0)))
        cons = np.concatenate((equilibrium, stress, compatibility, buckling))

        return cons
    
    def jacobianstructure(self):
        """ Returns the row and column indices for non-zero vales of the
        Jacobian. """
        # DIM = Number of constraint functions X number of design variables
        area_phys_id = self.calc_area_phys_id_sect()
        ### Equilibrium (M eq)
        row, column = self.B_row, self.B_column+self.NN # the indexes are translated to the corresponding design variables (force)
        ### Stress (2*N eq)
        row = np.append(row,range(self.M,self.M+self.N*2))
        column = np.append(column,np.tile(area_phys_id,2))
        row = np.append(row,range(self.M,self.M+self.N*2))
        column = np.append(column,np.r_[self.force,self.force])
        ### Compatibility (N eq)
        row = np.append(row,range(self.M+self.N*2,self.M+self.N*3))
        column = np.append(column,area_phys_id)
        row = np.append(row,range(self.M+self.N*2,self.M+self.N*3))
        column = np.append(column,np.r_[self.force])
        row = np.append(row,self.B_T_row + self.M+self.N*2) # B is transposed
        column = np.append(column,self.B_T_column+self.N+self.NN)
        ### Buckling (N eq)
        row = np.append(row,range(self.M+self.N*3,self.M+self.N*4))
        column = np.append(column,area_phys_id)
        row = np.append(row,range(self.M+self.N*3,self.M+self.N*4))
        column = np.append(column,np.r_[self.force])
        return row,column

    def jacobian(self, x):
        """ Returns the Jacobian of the constraints with respect to x. """
        # DIM = Number of constraint functions X number of design variables
        # Physical areas
        area_phys = self.calc_area_phys_sect(x[self.area]) # initialize a_phys
        ### Equilibrium (M eq)
        Jacobian = self.B_coo.data
        ### Stress (2*N eq)
        Jacobian = np.append(Jacobian,np.ones(self.N)*(- self.s_c))
        Jacobian = np.append(Jacobian,np.ones(self.N)*(- self.s_t))
        Jacobian = np.append(Jacobian,np.ones(self.N))
        Jacobian = np.append(Jacobian,np.ones(self.N))
        ### Compatibility (N eq)
        dComp_dA = []
        dComp_dA = self.B.T @ x[self.U] * self.E / self.l_true # Diagonal term, i==j
        dComp_dU = (sp.diags(area_phys * self.E / self.l_true) @ self.B.T) # Multiply the column of B.T and the main diagnonal of diag
        dComp_dU.sort_indices() # Needed to match the order of B.T given in jacobianstructure
        dComp_dU = dComp_dU.tocoo()
        # Area
        Jacobian = np.append(Jacobian,dComp_dA)
        # Force
        Jacobian = np.append(Jacobian,np.ones(self.N)*-1)
        # Displacements
        Jacobian = np.append(Jacobian,dComp_dU.data)
        ### Buckling (N eq)
        # Area
        Jacobian = np.append(Jacobian,2*area_phys*self.s_buck/self.l_true**2)
        # Force
        Jacobian = np.append(Jacobian,np.ones(self.N))
        return Jacobian
    
    def hessianstructure(self):
        """ Returns the row and column indices for non-zero vales of the
        Hessian. """
        # DIM = Number of design variables X number of design variables
        M = self.M
        N = self.N
        NN = self.NN
        
        ### Compatibility (N eq) x[self.area] * self.E/self.l_true * (self.B.T @ x[self.U]) - x[self.force]
        ## Area self.B.T @ x[self.U] * self.E / self.l_true
        # Area -
        
        # Force -

        # Displacements 
        temp_i, temp_j = np.meshgrid(np.arange(NN+N,NN+N+M, dtype='int'), np.arange(NN, dtype='int'))
        i = temp_i.ravel()
        j = temp_j.ravel()
        
        ### Buckling (N eq) buckling = x[self.force] + (self.s_buck/self.l_true**2) * x[self.area]**2
        ## Area 2*x[self.area]*self.s_buck/self.l_true**2
        # Area
        i = np.append(i,np.arange(NN, dtype='int'))
        j = np.append(j,np.arange(NN, dtype='int'))

        # here you have to sort and to eliminate redundant variables
        return i, j

    def hessian(self, x, l, obj_factor):
        """  Returns the non-zero values of the Hessian. """ 
        # DIM = Number of design variables X number of design variables
        M = self.M
        N = self.N
        NN = self.NN
        N_cell = self.N_cell
        NN_cell = N_cell * self.myBCs.ncel_str
        
        # ID of the constraints
        comp_id = slice(M+N*2, M+N*3)
        buck_id = slice(M+N*3, M+N*4)
        
        ### Objective function (N eq) the hessian of the obj function is empty
        # Hessian = obj_factor * NULL
        
        ### Equilibrium (M eq) self.B @ x[self.force] - self.f
        ## Area -

        ## Force self.B_coo.data

        ## Displacements -
        
        ### Stress (2*N eq)
        ## Area -
        
        ## Force -

        ## Displacements -

        ### Compatibility (N eq) x[self.area] * self.E/self.l_true * (self.B.T @ x[self.U]) - x[self.force]
        ## Area self.B.T @ x[self.U] * self.E / self.l_true
        # Area -
        
        # Force -

        # Displacements 
        temp = (sp.diags(l[comp_id] * self.E / self.l_true) @ self.B.T).tocoo()
        j, i = temp.row, temp.col + (NN+N)
        try:
            j[(j<NN_cell) & (j>=N_cell)] = j % N_cell # Only the first cellule is important for sensitivity
        except:
            pass
        try:
            j[j>=NN_cell] = j - NN_cell + N_cell # Only the first cellule is important for sensitivity
        except:
            pass
        data = temp.data 

        ## Force np.ones(self.N)*-1

        ## Displacements (sp.diags(x[self.area] * self.E / self.l_true) @ self.B.T)
        # Area
        # Symmetric, already considered
        
        # Force -

        # Displacements -
        
        ### Buckling (N eq) buckling = x[self.force] + (self.s_buck/self.l_true**2) * x[self.area]**2
        ## Area 2*x[self.area]*self.s_buck/self.l_true**2
        # Area
        i = np.append(i.ravel(),np.concatenate([np.tile(np.arange(N_cell, dtype='int'),self.myBCs.ncel_str),np.arange(N_cell, NN, dtype='int')]))
        j = np.append(j.ravel(),np.concatenate([np.tile(np.arange(N_cell, dtype='int'),self.myBCs.ncel_str),np.arange(N_cell, NN, dtype='int')]))
        data = np.append(data, 2*self.s_buck/self.l_true**2 * l[buck_id]) 
        # Force -

        # Displacements -
        
        ## Force np.ones(self.N)

        ## Displacements -
        
        
        Hessian = sp.coo_matrix((data, (i, j)), shape=(x.size, x.size)).tocsc() # check which one is better 
        row, col = self.hessianstructure()
        out = np.array(Hessian[row, col]).ravel()
        
        return out
    
    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        """Prints information at every Ipopt iteration."""

        self.it = iter_count
        self.obj_hist = np.append(self.obj_hist, self.volume_phys)
        self.constr_equilib_hist = np.append(self.constr_equilib_hist, self.viol_equilib)
        self.constr_s_c_hist = np.append(self.constr_s_c_hist, self.viol_s_c)
        self.constr_s_t_hist = np.append(self.constr_s_t_hist, self.viol_s_t)
        self.constr_comp_hist = np.append(self.constr_comp_hist, self.viol_comp)
        self.constr_buck_hist = np.append(self.constr_buck_hist, self.viol_buck)
        
        print("Phys. Vol: {0:.5f},  Eq. Cons: {1:.2e},  SC. Cons: {2:.2e},  ST. Cons: {3:.2e},  Comp. Cons: {4:.2e},  Buck. Cons: {5:.2e}\n".format(\
					self.obj_hist[-1], self.constr_equilib_hist[-1], self.constr_s_c_hist[-1], \
                    self.constr_s_t_hist[-1], self.constr_comp_hist[-1], self.constr_buck_hist[-1]))
               
class Layopt_IPOPT_Buck_Free_Form_multiload(IPOPT_Problem_Free_Form):
    def __init__(self, N, M, l_true, joint_cost, B, f, dofs, s_c, s_t, E, s_buck, sf):
        # Use the class constructor to declare the variable used by the optimizer during the optimization
        self.N = N # Number of member of the Ground Structure
        self.M = M # Number of DOFs of the Groud Structure
        self.l_true = l_true # Physical member lenghts [mm]
        self.l = l_true + joint_cost * np.max(l_true) # indipendency from the number of cells [mm]
        self.B = B # Equilibrium forces matrix 
        self.f = f # External forces [N]
        self.dofs = dofs
        self.s_c = s_c # Max compression allowable stress [MPa]
        self.s_t = s_t # Max tension allowable stress [MPa]
        self.E = E # Young's modulus [MPa]
        self.sf = sf # Safety factor
        self.n_load_cases = f.shape[-1]
        
        # Buckling specific parameters
        self.s_buck = s_buck
        self.a_cr = -s_c * l_true**2 / self.s_buck # Evaluation of the critical section for the members
        
        # Sparsity pattern of B matrix
        self.B_coo = self.B.tocoo()
        self.B_row, self.B_column = self.B_coo.row,self.B_coo.col 
        self.B_T_coo = self.B.T.tocoo() # Horrible but necessary to conserve the very same order of dComp_dU
        self.B_T_row, self.B_T_column = self.B_T_coo.row,self.B_T_coo.col       
        
        # Create the indexing variables used for splitting the design variables vector x
        self.area = slice(0,N)
        self.force = []
        self.U = []
        for p in range(self.n_load_cases):
            self.force.append(slice(N+(p)*N,N+(p+1)*N))
            self.U.append(slice(N+self.n_load_cases*N+(p)*M,N+self.n_load_cases*N+(p+1)*M))
        
        # History values
        self.it = 0
        self.obj_hist = []
        self.constr_equilib_hist = []
        self.constr_s_c_hist = []
        self.constr_s_t_hist = []
        self.constr_comp_hist = []
        self.constr_buck_hist = []

    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        aaa = x[self.area]
        bbb = np.abs(x[self.force[0]])
        ccc = np.abs(x[self.U[0]])
        self.volume_phys = self.l_true.T @ x[self.area]
        return self.l.T @ x[self.area]

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        grad = np.zeros(x.size)
        grad[self.area] = self.l
        return grad

    def constraints(self, x):
        """Returns the constraints."""
        # for p in range(self.n_load_cases):
        #     x[self.U[p]][self.dofs==0] = 0
        ### Equilibrium (M*p eq)
        equilibrium = np.array([])
        for p in range(self.n_load_cases):
            eq = self.B @ x[self.force[p]] - self.f[:,p]*self.dofs
            equilibrium = np.append(equilibrium, eq)
        indexes = np.argsort(equilibrium)[::-1][:20]
        val = equilibrium[indexes]
        self.viol_equilib = np.max(np.abs(equilibrium))
        ### Stress (2*N*p eq)
        stress = np.array([])
        for p in range(self.n_load_cases):
            stress_c = x[self.force[p]] - self.s_c * x[self.area] / self.sf[p]
            stress = np.append(stress, stress_c)
        self.viol_s_c = np.abs(np.min(np.append(stress_c,0)))
        for p in range(self.n_load_cases):
            stress_t = x[self.force[p]] - self.s_t * x[self.area] / self.sf[p]
            stress = np.append(stress, stress_t)
        self.viol_s_t = np.max(np.append(stress_t,0))
        ### Compatibility (N*p eq)
        compatibility = np.array([])
        for p in range(self.n_load_cases):
            comp = x[self.area] * self.E/self.l_true * (self.B.T @ x[self.U[p]]) - x[self.force[p]]
            compatibility = np.append(compatibility, comp)
        self.viol_comp = np.max(np.abs(compatibility))
        ### Buckling (N eq)
        buckling = np.array([])
        for p in range(self.n_load_cases):
            buck = x[self.force[p]] + (self.s_buck/self.l_true**2) * x[self.area]**2 / self.sf[p]
            buckling = np.append(buckling, buck)
        self.viol_buck = np.abs(np.min(np.append(buckling,0)))
        cons = np.concatenate((equilibrium, stress, compatibility, buckling))
        return cons
    
    def jacobianstructure(self):
        """ Returns the row and column indices for non-zero vales of the
        Jacobian. """
        # DIM = Number of constraint functions X number of design variables
        ### Equilibrium (M*p eq)
        row, column = self.B_row, self.B_column+self.N # the indexes are translated to the corresponding design variables (force)
        for p in range(self.n_load_cases-1):
            row = np.append(row,self.B_row+self.M+p*self.M)
            column = np.append(column,self.B_column+self.N+(p+1)*self.N)
        ### Stress (2*N eq)
        for p in range(self.n_load_cases): 
            row = np.append(row,range(self.M*self.n_load_cases+p*self.N, self.M*self.n_load_cases+(p+1)*self.N))
            column = np.append(column,np.r_[self.area])
            row = np.append(row,range(self.M*self.n_load_cases+p*self.N, self.M*self.n_load_cases+(p+1)*self.N))
            column = np.append(column,np.r_[self.force[p]])
        for p in range(self.n_load_cases):
            row = np.append(row,range(self.M*self.n_load_cases+self.n_load_cases*self.N+p*self.N, self.M*self.n_load_cases+self.n_load_cases*self.N+(p+1)*self.N))
            column = np.append(column,np.r_[self.area])
            row = np.append(row,range(self.M*self.n_load_cases+self.n_load_cases*self.N+p*self.N, self.M*self.n_load_cases+self.n_load_cases*self.N+(p+1)*self.N))
            column = np.append(column,np.r_[self.force[p]])
        ### Compatibility (N eq)
        for p in range(self.n_load_cases):
            row = np.append(row,range(self.M*self.n_load_cases+self.N*2*self.n_load_cases+p*self.N, self.M*self.n_load_cases+self.N*2*self.n_load_cases+(p+1)*self.N))
            column = np.append(column,np.r_[self.area])
            row = np.append(row,range(self.M*self.n_load_cases+self.N*2*self.n_load_cases+p*self.N, self.M*self.n_load_cases+self.N*2*self.n_load_cases+(p+1)*self.N))
            column = np.append(column,np.r_[self.force[p]])
            row = np.append(row,self.B_T_row + self.M*self.n_load_cases+self.N*2*self.n_load_cases+p*self.N) # B is transposed
            column = np.append(column,self.B_T_column+self.N+self.N*self.n_load_cases+p*self.M) # U
        ### Buckling (N eq)
        for p in range(self.n_load_cases):
            row = np.append(row,range(self.M*self.n_load_cases+self.N*3*self.n_load_cases+p*self.N, self.M*self.n_load_cases+self.N*3*self.n_load_cases+(p+1)*self.N))
            column = np.append(column,np.r_[self.area])
            row = np.append(row,range(self.M*self.n_load_cases+self.N*3*self.n_load_cases+p*self.N, self.M*self.n_load_cases+self.N*3*self.n_load_cases+(p+1)*self.N))
            column = np.append(column,np.r_[self.force[p]])
        return row,column

    def jacobian(self, x):
        """ Returns the Jacobian of the constraints with respect to x. """
        # DIM = Number of constraint functions X number of design variables
        ### Equilibrium (M eq)
        Jacobian = self.B_coo.data
        for p in range(self.n_load_cases-1): 
            Jacobian = np.append(Jacobian,self.B_coo.data)   
        ### Stress (2*N eq)
        for p in range(self.n_load_cases):
            Jacobian = np.append(Jacobian,np.ones(self.N)*(- self.s_c / self.sf[p]))
            Jacobian = np.append(Jacobian,np.ones(self.N))
        for p in range(self.n_load_cases):
            Jacobian = np.append(Jacobian,np.ones(self.N)*(- self.s_t / self.sf[p]))
            Jacobian = np.append(Jacobian,np.ones(self.N))
        ### Compatibility (N eq) x[self.area] * self.E/self.l_true * (self.B.T @ x[self.U]) - x[self.force]
        for p in range(self.n_load_cases):
            dComp_dA = self.B.T @ x[self.U[p]] * self.E / self.l_true # Diagonal term, i==j
            dComp_dU = (sp.diags(x[self.area] * self.E / self.l_true) @ self.B.T) # Multiply the column of B.T and the main diagnonal of diag
            dComp_dU.sort_indices() # Needed to match the order of B.T given in jacobianstructure
            dComp_dU = dComp_dU.tocoo()
            # Area
            Jacobian = np.append(Jacobian,dComp_dA)
            # Force
            Jacobian = np.append(Jacobian,np.ones(self.N)*-1)
            # Displacements
            Jacobian = np.append(Jacobian,dComp_dU.data)
        ### Buckling (N eq)
        for p in range(self.n_load_cases):
            # Area
            Jacobian = np.append(Jacobian,2*x[self.area]*self.s_buck/self.l_true**2/self.sf[p])
            # Force
            Jacobian = np.append(Jacobian,np.ones(self.N))
            magnitude_test = np.abs(Jacobian)
        return Jacobian
    
    def hessianstructure(self):
        """ Returns the row and column indices for non-zero vales of the
        Hessian. NO HESSIAN, NOT WORKING"""
        # DIM = Number of design variables X number of design variables
        M = self.M
        N = self.N
        
        ### Compatibility (N eq) x[self.area] * self.E/self.l_true * (self.B.T @ x[self.U]) - x[self.force]
        ## Area self.B.T @ x[self.U] * self.E / self.l_true
        # Area -
        
        # Force -

        # Displacements 
        i, j = np.meshgrid(np.arange(2*N,2*N+M, dtype='int'), np.arange(N, dtype='int'))
        
        ### Buckling (N eq) buckling = x[self.force] + (self.s_buck/self.l_true**2) * x[self.area]**2
        ## Area 2*x[self.area]*self.s_buck/self.l_true**2
        # Area
        i = np.append(i.ravel(),np.arange(N, dtype='int'))
        j = np.append(j.ravel(),np.arange(N, dtype='int'))

        return i, j

    def hessian(self, x, l, obj_factor):
        """ Returns the non-zero values of the Hessian. """
        # DIM = Number of design variables X number of design variables
        M = self.M
        N = self.N
        
        # ID of the constraints
        comp_id = slice(M+N*2, M+N*3)
        buck_id = slice(M+N*3, M+N*4)
        
        ### Objective function (N eq) the hessian of the obj function is empty
        # Hessian = obj_factor * NULL
        
        ### Equilibrium (M eq) self.B @ x[self.force] - self.f
        ## Area -

        ## Force self.B_coo.data

        ## Displacements -
        
        ### Stress (2*N eq)
        ## Area -
        
        ## Force -

        ## Displacements -

        ### Compatibility (N eq) x[self.area] * self.E/self.l_true * (self.B.T @ x[self.U]) - x[self.force]
        ## Area self.B.T @ x[self.U] * self.E / self.l_true
        # Area -
        
        # Force -

        # Displacements 
        temp = (sp.diags(l[comp_id] * self.E / self.l_true) @ self.B.T)
        temp.sort_indices() # Needed to match the order of B.T given in jacobianstructure
        temp = temp.tocoo()
        j, i = temp.row, temp.col + (2*N)
        data = temp.data

        ## Force np.ones(self.N)*-1

        ## Displacements (sp.diags(x[self.area] * self.E / self.l_true) @ self.B.T)
        # Area
        # Symmetric, already considered
        
        # Force -

        # Displacements -
        
        ### Buckling (N eq) buckling = x[self.force] + (self.s_buck/self.l_true**2) * x[self.area]**2
        ## Area 2*x[self.area]*self.s_buck/self.l_true**2
        # Area
        i = np.append(i.ravel(),np.arange(N, dtype='int'))
        j = np.append(j.ravel(),np.arange(N, dtype='int'))
        data = np.append(data,2*self.s_buck/self.l_true**2 * l[buck_id]) 
        # Force -

        # Displacements -
        
        ## Force np.ones(self.N)

        ## Displacements -
        
        Hessian = sp.coo_matrix((data, (i, j)), shape=(x.size, x.size)).tocsc() # check which one is better 
        row, col = self.hessianstructure()
        out = np.array(Hessian[row, col]).ravel()
        
        return out
    
    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        """Prints information at every Ipopt iteration."""

        self.it = iter_count
        self.obj_hist = np.append(self.obj_hist, self.volume_phys)
        self.constr_equilib_hist = np.append(self.constr_equilib_hist, self.viol_equilib)
        self.constr_s_c_hist = np.append(self.constr_s_c_hist, self.viol_s_c)
        self.constr_s_t_hist = np.append(self.constr_s_t_hist, self.viol_s_t)
        self.constr_comp_hist = np.append(self.constr_comp_hist, self.viol_comp)
        self.constr_buck_hist = np.append(self.constr_buck_hist, self.viol_buck)
        
        print("Phys. Vol: {0:.5f},  Eq. Cons: {1:.2e},  SC. Cons: {2:.2e},  ST. Cons: {3:.2e},  Comp. Cons: {4:.2e},  Buck. Cons: {5:.2e}".format(\
					self.obj_hist[-1], self.constr_equilib_hist[-1], self.constr_s_c_hist[-1], \
                    self.constr_s_t_hist[-1], self.constr_comp_hist[-1], self.constr_buck_hist[-1]))
