#################################
## Boundary Conditions
#################################
import itertools

import numpy as np

############################
# Utilities
############################


def view_as_windows_2D(arr_in, cell_size_x, cell_size_y, cell_number_x, cell_number_y):
    """Crop a matrix with overlap"""
    s0, s1 = arr_in.strides
    arr_out = np.lib.stride_tricks.as_strided(
        arr_in,
        shape=(cell_number_y, cell_number_x, cell_size_y, cell_size_x),
        strides=(s0 * (cell_size_y - 1), s1 * (cell_size_x - 1), s0, s1),
    )
    return arr_out


def view_as_windows_3D(
    arr_in,
    cell_size_x,
    cell_size_y,
    cell_size_z,
    cell_number_x,
    cell_number_y,
    cell_number_z,
):
    """Crop a matrix with overlap"""
    s0, s1, s2 = arr_in.strides
    arr_out = np.lib.stride_tricks.as_strided(
        arr_in,
        shape=(
            cell_number_x,
            cell_number_y,
            cell_number_z,
            cell_size_x,
            cell_size_y,
            cell_size_z,
        ),
        strides=(
            s0 * (cell_size_x - 1),
            s1 * (cell_size_y - 1),
            s2 * (cell_size_z - 1),
            s0,
            s1,
            s2,
        ),
    )
    return arr_out


### 2D
class BoundaryConditions2D:
    """
    Class used to define the boundary conditions of a 2D problem.
    Functionalty for setting fixed dof and forces.
    """

    def __init__(
        self,
        nnodx_cel: int,
        nnody_cel: int,
        ncelx_str: int = 1,
        ncely_str: int = 1,
        f: float = 1,
        L: int = 1000,
    ):
        """
        Create the mesh for the problem.

        Parameters
        ----------
        nnodx:
            The number of elements in the x direction.
        nnody:
            The number of elements in the y direction.
        F:
            Magnitude of the total force applied to the structure.

        """
        self.L = L
        self.is3D = False
        self.isReduced = False
        self.isFreeForm = False
        # Elements

        self.nnodx_cel = nnodx_cel
        self.nnody_cel = nnody_cel
        self.nnod_cel = self.nnodx_cel * self.nnody_cel
        self.ncelx_str = ncelx_str
        self.ncely_str = ncely_str
        self.ncel_str = self.ncelx_str * self.ncely_str

        # Multiscale parameters
        self.isMultiscale = False
        if self.ncelx_str != 1 or self.ncely_str != 1:
            self.isMultiscale = True

        self.nnodx = self.nnodx_cel * self.ncelx_str - (self.ncelx_str - 1)
        self.nnody = self.nnody_cel * self.ncely_str - (self.ncely_str - 1)
        self.nnod = self.nnodx * self.nnody

        # Nodes matrix
        self.nodes_index = np.zeros((self.nnod, 2), dtype=int)
        self.nodes = np.zeros((self.nnod, 2), dtype=float)
        x, y = np.meshgrid(np.arange(self.nnodx), np.arange(self.nnody), indexing="xy")
        self.nodes_index[:, 0] = x.ravel()
        self.nodes_index[:, 1] = y.ravel()

        # Scaling to the desired dimensions
        self.nodes[:, 0] = (
            self.nodes_index[:, 0] / np.max(self.nodes_index[:, 0]) * L[0]
        )  # X
        self.nodes[:, 1] = (
            self.nodes_index[:, 1] / np.max(self.nodes_index[:, 1]) * L[1]
        )  # Y

        self.dofs_list = np.arange(2 * self.nnod)  # Dofs initialization
        # Ground structure assembly
        if self.isMultiscale:
            self.nodes_list = np.arange(self.nnod).reshape((self.nnody, self.nnodx))
            self.nodes_cell = view_as_windows_2D(
                self.nodes_list,
                self.nnodx_cel,
                self.nnody_cel,
                self.ncelx_str,
                self.ncely_str,
            ).reshape(
                self.ncely_str * self.ncelx_str, self.nnodx_cel * self.nnody_cel
            )  # one row = one cellule
            # creation of the ground structure
            self.ground_stucture_list = []  # initialization of the list containing all the beam
            for i in range(self.nodes_cell.shape[0]):  # cycling trough all the cells
                self.ground_stucture_list.append(
                    (list(itertools.combinations(self.nodes_cell[i, :], 2)))
                )
            self.ground_stucture_list = list(
                itertools.chain.from_iterable(self.ground_stucture_list)
            )
            self.ground_stucture_list_cell = list(
                itertools.combinations(self.nodes_cell[0, :], 2)
            )  # ground structure of the first cellule
        else:
            self.ground_stucture_list = list(
                itertools.combinations(range(len(self.nodes_index)), 2)
            )
        self.ground_structure = np.array(self.ground_stucture_list)
        self.N_cell = [
            int(self.ground_structure.shape[0] / self.ncel_str)
        ]  # Number of elements per cell
        self.N_cell_max = self.N_cell[0]  # Used for reduced problems
        # using 2-norm to calculate the beams length [mm]
        self.ground_structure_length = np.linalg.norm(
            (
                self.nodes[self.ground_structure[:, 0], 1]
                - self.nodes[self.ground_structure[:, 1], 1],
                self.nodes[self.ground_structure[:, 0], 0]
                - self.nodes[self.ground_structure[:, 1], 0],
            ),
            axis=0,
        )
        # Force magnitude
        self.f = f

        # Nodes and members number
        self.N = int(self.ground_structure.shape[0])
        self.M = int(self.nnod) * 2

        # Is L-shape param initialization
        self.isLshape = False


class MBB2D_Symm(BoundaryConditions2D):
    """Symmetric MBB"""

    def __init__(
        self,
        nnodx_cel: int,
        nnody_cel: int,
        ncelx_str: int,
        ncely_str: int,
        f: float,
        L,
    ):
        super().__init__(nnodx_cel, nnody_cel, ncelx_str, ncely_str, f, L)
        self.name = "MBB2D_Symm_2D"
        ## DOFS
        # Odd = vertical, even = horizontal
        self.fixed_dofs = np.sort(
            np.array(
                list(range(0, 2 * (self.nnodx * (self.nnody)), 2 * self.nnodx))
                + [2 * self.nnodx - 1]
            )
        )
        self.free_dofs = np.setdiff1d(self.dofs_list, self.fixed_dofs)
        self.dofs = np.ones(self.dofs_list.shape)
        self.dofs[self.fixed_dofs] = 0
        self.dofs_nodes_connectivity = self.dofs.reshape((-1, 2))
        ## FORCES
        self.force_dofs = np.array(2 * (self.nnodx * (self.nnody - 1)) + 1)
        self.force = -self.f / self.force_dofs.size
        # fixed dofs elimination
        self.R = np.zeros(self.dofs_list.shape)
        self.R[self.force_dofs] = self.force
        self.R_free = self.R[self.free_dofs]


class Cantilever(BoundaryConditions2D):
    """Cantilever Beam"""

    def __init__(
        self,
        nnodx_cel: int,
        nnody_cel: int,
        ncelx_str: int,
        ncely_str: int,
        f: float,
        L,
    ):
        super().__init__(nnodx_cel, nnody_cel, ncelx_str, ncely_str, f, L)
        self.name = "Cantilever_2D"
        ## DOFS
        # Odd = vertical, even = horizontal
        self.fixed_dofs = np.sort(
            np.array(
                list(range(0, 2 * (self.nnodx * (self.nnody)), 2 * self.nnodx))
                + list(range(1, 2 * (self.nnodx * (self.nnody)) + 1, 2 * self.nnodx))
            )
        )
        self.free_dofs = np.setdiff1d(self.dofs_list, self.fixed_dofs)
        self.dofs = np.ones(self.dofs_list.shape)
        self.dofs[self.fixed_dofs] = 0
        self.dofs_nodes_connectivity = self.dofs.reshape((-1, 2))
        ## FORCES
        self.force_dofs = np.array(
            2 * (self.nnodx * np.ceil(self.nnody / 2) - 1) + 1, dtype=int
        ).reshape(-1)  # This should be a 1D np.array object
        self.force = self.f / self.force_dofs.size
        # fixed dofs elimination
        self.R = np.zeros(self.dofs_list.shape)
        self.R[self.force_dofs] = -self.force
        self.R_free = self.R[self.free_dofs]


class Cantilever_Low(BoundaryConditions2D):
    """Cantilever Beam"""

    def __init__(
        self,
        nnodx_cel: int,
        nnody_cel: int,
        ncelx_str: int,
        ncely_str: int,
        f: float,
        L,
    ):
        super().__init__(nnodx_cel, nnody_cel, ncelx_str, ncely_str, f, L)
        self.name = "Cantilever_Low_2D"
        ## DOFS
        # Odd = vertical, even = horizontal
        self.fixed_dofs = np.sort(
            np.array(
                list(range(0, 2 * (self.nnodx * (self.nnody)), 2 * self.nnodx))
                + list(range(1, 2 * (self.nnodx * (self.nnody)) + 1, 2 * self.nnodx))
            )
        )
        self.free_dofs = np.setdiff1d(self.dofs_list, self.fixed_dofs)
        self.dofs = np.ones(self.dofs_list.shape)
        self.dofs[self.fixed_dofs] = 0
        self.dofs_nodes_connectivity = self.dofs.reshape((-1, 2))
        ## FORCES
        self.force_dofs = np.array(2 * (self.nnodx - 1) + 1, dtype=int).reshape(
            -1
        )  # This should be a 1D np.array object
        self.force = self.f / self.force_dofs.size
        # fixed dofs elimination
        self.R = np.zeros(self.dofs_list.shape)
        self.R[self.force_dofs] = -self.force
        self.R_free = self.R[self.free_dofs]


class Multiscale_test_bridge(BoundaryConditions2D):
    """Multiscale test to assess the equivalence between multiscale and multiload"""

    def __init__(
        self,
        nnodx_cel: int,
        nnody_cel: int,
        ncelx_str: int,
        ncely_str: int,
        f: float,
        L,
    ):
        super().__init__(nnodx_cel, nnody_cel, ncelx_str, ncely_str, f, L)
        self.name = "Multiscale_test_2D_bridge_4"
        ## DOFS
        # Odd = vertical, even = horizontal
        self.fixed_dofs = list(
            range(0, 2 * (self.nnodx * (self.nnody)), 2 * self.nnodx)
        ) + [1]
        self.fixed_dofs = (
            self.fixed_dofs
            + list(
                range(
                    2 * self.nnodx - 2, 2 * (self.nnodx * (self.nnody)), 2 * self.nnodx
                )
            )
            + [2 * (self.nnodx - 1) + 1]
        )
        self.fixed_dofs = self.fixed_dofs + [self.nnodx]
        self.fixed_dofs = np.sort(np.array(self.fixed_dofs))
        self.free_dofs = np.setdiff1d(self.dofs_list, self.fixed_dofs)
        self.dofs = np.ones(self.dofs_list.shape)
        self.dofs[self.fixed_dofs] = 0
        self.dofs_nodes_connectivity = self.dofs.reshape((-1, 2))
        ## FORCES
        self.force_dofs = np.array(
            [
                2 * np.floor(self.nnodx * 1 / 5) + 1,
                2 * np.floor(self.nnodx * 4 / 5) + 1,
            ],
            dtype=int,
        ).reshape(-1)  # This should be a 1D np.array object
        self.f = self.f / self.force_dofs.size
        self.force = np.array([self.f, self.f]).reshape(-1)
        # fixed dofs elimination
        self.R = np.zeros(self.dofs_list.shape)
        self.R[self.force_dofs] = -self.force
        self.R_free = self.R[self.free_dofs]


class DistributedCantilever(BoundaryConditions2D):
    """Distributed loads cantilever Beam"""

    def __init__(
        self,
        nnodx_cel: int,
        nnody_cel: int,
        ncelx_str: int,
        ncely_str: int,
        f: float,
        L,
    ):
        super().__init__(nnodx_cel, nnody_cel, ncelx_str, ncely_str, f, L)
        self.name = "DistCantilever_2D"
        ## DOFS
        # Odd = vertical, even = horizontal
        self.fixed_dofs = np.sort(
            np.array(
                list(range(0, 2 * (self.nnodx * (self.nnody)), 2 * self.nnodx))
                + list(range(1, 2 * (self.nnodx * (self.nnody)) + 1, 2 * self.nnodx))
            )
        )
        self.free_dofs = np.setdiff1d(self.dofs_list, self.fixed_dofs)
        self.dofs = np.ones(self.dofs_list.shape)
        self.dofs[self.fixed_dofs] = 0
        self.dofs_nodes_connectivity = self.dofs.reshape((-1, 2))
        ## FORCES
        self.force_dofs = np.array(
            2 * (self.nnodx * np.ceil(self.nnody / 2) - 1) + 1, dtype=int
        ).reshape(-1)  # This should be a 1D np.array object
        self.force = self.f / self.force_dofs.size
        # fixed dofs elimination
        self.R = np.zeros(self.dofs_list.shape)
        self.R[self.force_dofs] = -self.force
        self.R_free = self.R[self.free_dofs]


class MichCantilever(BoundaryConditions2D):
    """Michell Cantilever Beam"""

    def __init__(
        self,
        nnodx_cel: int,
        nnody_cel: int,
        ncelx_str: int,
        ncely_str: int,
        f: float,
        L,
    ):
        super().__init__(nnodx_cel, nnody_cel, ncelx_str, ncely_str, f, L)
        self.name = "MichCantilever_2D"
        ## DOFS
        # Odd = vertical, even = horizontal
        self.x_mich = int(self.nnody / 3)
        self.fixed_dofs = np.sort(
            np.array(
                [
                    2 * self.x_mich * self.nnodx,
                    2 * self.x_mich * self.nnodx + 1,
                    2 * (self.nnody - self.x_mich - 1) * self.nnodx,
                    2 * (self.nnody - self.x_mich - 1) * self.nnodx + 1,
                ],
                dtype=int,
            )
        )
        self.free_dofs = np.setdiff1d(self.dofs_list, self.fixed_dofs)
        self.dofs = np.ones(self.dofs_list.shape)
        self.dofs[self.fixed_dofs] = 0
        self.dofs_nodes_connectivity = self.dofs.reshape((-1, 2))
        ## FORCES
        self.force_dofs = np.array(
            2 * (self.nnodx * np.ceil(self.nnody / 2) - 1) + 1, dtype=int
        ).reshape(-1)  # This should be a 1D np.array object
        self.force = self.f / self.force_dofs.size
        # fixed dofs elimination
        self.R = np.zeros(self.dofs_list.shape)
        self.R[self.force_dofs] = -self.force
        self.R_free = self.R[self.free_dofs]


class L_Shape:
    # make differences of the above class
    """L_Shape"""

    def __init__(
        self,
        nnodx_cel: int,
        nnody_cel: int,
        ncelx_str: int,
        ncely_str: int,
        f: float,
        n_nod_empty_x: int,
        n_nod_empty_y: int,
        L,
    ):
        self.name = "L_Shape_2D"
        self.is3D = False
        self.isReduced = False
        # Elements
        self.nnodx_cel = nnodx_cel
        self.nnody_cel = nnody_cel
        self.nnod_cel = self.nnodx_cel * self.nnody_cel
        self.ncelx_str = ncelx_str
        self.ncely_str = ncely_str
        self.ncel_str = int(np.ceil(self.ncelx_str * self.ncely_str * 0.75))
        self.nnodx_empty = n_nod_empty_x
        self.nnody_empty = n_nod_empty_y
        self.nnod_empty = self.nnodx_empty * self.nnody_empty

        # Multiscale parameters
        self.isMultiscale = False
        if self.ncelx_str != 1 or self.ncely_str != 1:
            self.isMultiscale = True

        self.nnodx = self.nnodx_cel * self.ncelx_str - (self.ncelx_str - 1)
        self.nnody = self.nnody_cel * self.ncely_str - (self.ncely_str - 1)
        self.nnod = self.nnodx * self.nnody - self.nnod_empty

        # Nodes matrix
        self.nodes_index_temp = np.zeros((self.nnodx * self.nnody, 2), dtype=int)
        self.nodes_index = np.zeros((self.nnod, 2), dtype=int)
        self.nodes = np.zeros((self.nnod, 2), dtype=float)
        self.nodes_index_temp[:, 0] = np.tile((np.arange(0, self.nnodx)), self.nnody)
        self.nodes_index_temp[:, 1] = (
            np.arange(0, self.nnody) + np.zeros((self.nnodx, 1))
        ).reshape(-1, order="F")
        # Delete empty nodes from the nodes array
        self.nodes_active = (
            self.nodes_index_temp[:, 0] < (self.nnodx - self.nnodx_empty)
        ) | (self.nodes_index_temp[:, 1] < (self.nnody - self.nnody_empty))
        self.nodes_index = self.nodes_index_temp[self.nodes_active]
        # Scaling to the desired dimensions
        self.nodes[:, 0] = (
            self.nodes_index[:, 0] / np.max(self.nodes_index[:, 0]) * L[0]
        )  # X
        self.nodes[:, 1] = (
            self.nodes_index[:, 1] / np.max(self.nodes_index[:, 1]) * L[1]
        )  # Y

        # Ground structure assembly for Lshape
        if self.isMultiscale:
            self.nodes_list = np.ones((self.nnodx, self.nnody), dtype=int)
            self.nodes_list[-self.nnodx_empty :, -self.nnody_empty :] = -1
            self.nodes_list[self.nodes_list == 1] = np.arange(self.nnod)
            self.nodes_cell_full = view_as_windows_2D(
                self.nodes_list.reshape((self.nnody, self.nnodx)),
                self.nnodx_cel,
                self.nnody_cel,
                self.ncelx_str,
                self.ncely_str,
            ).reshape(
                self.ncely_str * self.ncelx_str, self.nnodx_cel * self.nnody_cel
            )  # one row = one cellule
            self.nodes_cell = self.nodes_cell_full[
                np.all(self.nodes_cell_full != -1, axis=1), :
            ]  # Eliminate all the rows where a -1 is found
            # creation of the ground structure
            self.ground_stucture_list = []  # initialization of the list containing all the bars
            for i in range(self.nodes_cell.shape[0]):  # cycling trough all the cells
                self.ground_stucture_list.append(
                    (list(itertools.combinations(self.nodes_cell[i, :], 2)))
                )
            self.ground_stucture_list = list(
                itertools.chain.from_iterable(self.ground_stucture_list)
            )
        else:
            self.ground_stucture_list = list(
                itertools.combinations(range(len(self.nodes_index)), 2)
            )
        # Eliminate all the bars that are going outside the domain
        x0 = self.nnodx - self.nnodx_empty - 1
        y0 = self.nnody - self.nnody_empty - 1
        # Needed to cycle through the candidates
        self.ground_stucture_list_t = self.ground_stucture_list.copy()
        for l in self.ground_stucture_list_t:
            x1 = int(self.nodes_index[l[0], 0])
            x2 = int(self.nodes_index[l[1], 0])
            y1 = int(self.nodes_index[l[0], 1])
            y2 = int(self.nodes_index[l[1], 1])
            # Remove out of domain members
            k = ((y2 - y1) * (x0 - x1) - (x2 - x1) * (y0 - y1)) / (
                (y2 - y1) ** 2 + (x2 - x1) ** 2
            )
            xm = x0 - k * (y2 - y1)
            ym = (
                y0 + k * (x2 - x1)
            )  # Looking for the intersection between the line P1 P2 and the perpendicular distance between the line and the corner

            if (
                xm > x0
                and ym > y0
                and (x1 - xm) * (x2 - xm) < 0
                and (y1 - ym) * (y2 - ym) < 0
            ):
                self.ground_stucture_list.remove(l)

        self.ground_structure = np.array(self.ground_stucture_list)
        self.N_cell = [
            int(self.ground_structure.shape[0] / self.ncel_str)
        ]  # Number of elements per cell
        self.N_cell_max = self.N_cell[0]  # Used for reduced problems
        # using 2-norm to calculate the beams length [mm]
        self.ground_structure_length = np.linalg.norm(
            (
                self.nodes[self.ground_structure[:, 0], 1]
                - self.nodes[self.ground_structure[:, 1], 1],
                self.nodes[self.ground_structure[:, 0], 0]
                - self.nodes[self.ground_structure[:, 1], 0],
            ),
            axis=0,
        )
        # Force magnitude
        self.f = f

        self.isLshape = True

        ## DOFS
        self.dofs_list = np.arange(2 * self.nnod)  # Dofs initialization
        # Odd = vertical, even = horizontal
        self.fixed_dofs = self.dofs_list[-2 * (self.nnodx - self.nnodx_empty) :]
        self.free_dofs = np.setdiff1d(self.dofs_list, self.fixed_dofs)
        self.dofs = np.ones(self.dofs_list.shape)
        self.dofs[self.fixed_dofs] = 0
        self.dofs_nodes_connectivity = self.dofs.reshape((-1, 2))
        ## FORCES
        self.force_dofs = np.array(
            (((self.nnody - self.nnody_empty) * self.nnodx) - 1) * 2 + 1, dtype=int
        ).reshape(-1)  # This should be a 1D np.array object
        self.force = self.f
        # fixed dofs elimination
        self.R = np.zeros(self.dofs_list.shape)
        self.R[self.force_dofs] = -self.force
        self.R_free = self.R[self.free_dofs]


class T_Shape:
    """T_Shape"""

    def __init__(
        self,
        nnodx_cel: int,
        nnody_cel: int,
        ncelx_str: int,
        ncely_str: int,
        f: float,
        n_nod_empty_x: int,
        n_nod_empty_y: int,
        L,
    ):
        self.name = "T_Shape_2D"
        self.is3D = False
        # Elements
        self.nnodx_cel = nnodx_cel
        self.nnody_cel = nnody_cel
        self.nnod_cel = self.nnodx_cel * self.nnody_cel
        self.ncelx_str = ncelx_str
        self.ncely_str = ncely_str
        self.ncel_str = int(self.ncelx_str + self.ncely_str)
        self.nnodx_empty = n_nod_empty_x
        self.nnody_empty = n_nod_empty_y
        self.nnod_empty = self.nnodx_empty * self.nnody_empty * 2

        # Multiscale parameters
        self.isMultiscale = False
        if self.ncelx_str != 1 or self.ncely_str != 1:
            self.isMultiscale = True

        self.nnodx = self.nnodx_cel * self.ncelx_str - (self.ncelx_str - 1)
        self.nnody = self.nnody_cel * self.ncely_str - (self.ncely_str - 1)
        self.nnodx_web = self.nnodx - self.nnodx_empty * 2
        self.nnody_web = self.nnody_empty + 1
        self.nnodx_flange = self.nnodx
        self.nnody_flange = self.nnody - self.nnody_empty
        self.nnod = self.nnodx * self.nnody - self.nnod_empty

        # Nodes matrix
        self.nodes_index_temp = np.zeros((self.nnodx * self.nnody, 2), dtype=int)
        self.nodes_index = np.zeros((self.nnod, 2), dtype=int)
        self.nodes = np.zeros((self.nnod, 2), dtype=float)
        self.nodes_index_temp[:, 0] = np.tile((np.arange(0, self.nnodx)), self.nnody)
        self.nodes_index_temp[:, 1] = (
            np.arange(0, self.nnody) + np.zeros((self.nnodx, 1))
        ).reshape(-1, order="F")
        # Delete empty nodes from the nodes array
        self.nodes_active = (
            (self.nodes_index_temp[:, 0] >= (self.nnodx_empty))
            & (self.nodes_index_temp[:, 0] < (self.nnodx - self.nnodx_empty))
        ) | (self.nodes_index_temp[:, 1] >= (self.nnody_empty))
        self.nodes_index = self.nodes_index_temp[self.nodes_active]
        # Scaling to the desired dimensions
        self.nodes[:, 0] = (
            self.nodes_index[:, 0] / np.max(self.nodes_index[:, 0]) * L[0]
        )  # X
        self.nodes[:, 1] = (
            self.nodes_index[:, 1] / np.max(self.nodes_index[:, 1]) * L[1]
        )  # Y

        # Ground structure assembly for Lshape
        if self.isMultiscale:
            self.nodes_list = np.ones((self.nnodx, self.nnody), dtype=int)
            self.nodes_list[-self.nnodx_empty :, -self.nnody_empty :] = -1
            self.nodes_list[self.nodes_list == 1] = np.arange(self.nnod)
            self.nodes_cell_full = view_as_windows_2D(
                self.nodes_list.reshape((self.nnody, self.nnodx)),
                self.nnodx_cel,
                self.nnody_cel,
                self.ncelx_str,
                self.ncely_str,
            ).reshape(
                self.ncely_str * self.ncelx_str, self.nnodx_cel * self.nnody_cel
            )  # one row = one cellule
            self.nodes_cell = self.nodes_cell_full[
                np.all(self.nodes_cell_full != -1, axis=1), :
            ]  # Eliminate all the rows where a -1 is found
            # creation of the ground structure
            self.ground_stucture_list = []  # initialization of the list containing all the bars
            for i in range(self.nodes_cell.shape[0]):  # cycling trough all the cells
                self.ground_stucture_list.append(
                    (list(itertools.combinations(self.nodes_cell[i, :], 2)))
                )
            self.ground_stucture_list = list(
                itertools.chain.from_iterable(self.ground_stucture_list)
            )
        else:
            self.ground_stucture_list = list(
                itertools.combinations(range(len(self.nodes_index)), 2)
            )
        # Eliminate all the bars that are going outside the domain
        # Corners coordinates
        x0_1 = self.nnodx_empty
        y0_1 = self.nnody_empty
        x0_2 = self.nnodx - self.nnodx_empty - 1
        y0_2 = self.nnody_empty
        # Needed to cycle through the candidates
        self.ground_stucture_list_t = self.ground_stucture_list.copy()
        for l in self.ground_stucture_list_t:
            x1 = int(self.nodes_index[l[0], 0])
            x2 = int(self.nodes_index[l[1], 0])
            y1 = int(self.nodes_index[l[0], 1])
            y2 = int(self.nodes_index[l[1], 1])
            # Remove out of domain members
            # Looking for the intersection between the line P1 P2 and the perpendicular distance between the line and the corner
            k_1 = ((y2 - y1) * (x0_1 - x1) - (x2 - x1) * (y0_1 - y1)) / (
                (y2 - y1) ** 2 + (x2 - x1) ** 2
            )  #
            xm_1 = x0_1 - k_1 * (y2 - y1)
            ym_1 = y0_1 + k_1 * (x2 - x1)

            if (
                xm_1 < x0_1
                and ym_1 < y0_1
                and (x1 - xm_1) * (x2 - xm_1) < 0
                and (y1 - ym_1) * (y2 - ym_1) < 0
            ):
                self.ground_stucture_list.remove(l)

            k_2 = ((y2 - y1) * (x0_2 - x1) - (x2 - x1) * (y0_2 - y1)) / (
                (y2 - y1) ** 2 + (x2 - x1) ** 2
            )  #
            xm_2 = x0_2 - k_2 * (y2 - y1)
            ym_2 = y0_2 + k_2 * (x2 - x1)

            if (
                xm_2 > x0_2
                and ym_2 < y0_2
                and (x1 - xm_2) * (x2 - xm_2) < 0
                and (y1 - ym_2) * (y2 - ym_2) < 0
            ):
                self.ground_stucture_list.remove(l)

        self.ground_structure = np.array(self.ground_stucture_list)
        # using 2-norm to calculate the beams length [mm]
        self.ground_structure_length = np.linalg.norm(
            (
                self.nodes[self.ground_structure[:, 0], 1]
                - self.nodes[self.ground_structure[:, 1], 1],
                self.nodes[self.ground_structure[:, 0], 0]
                - self.nodes[self.ground_structure[:, 1], 0],
            ),
            axis=0,
        )
        # Force magnitude
        self.f = f

        self.isLshape = False
        self.isTshape = True

        ## DOFS
        self.dofs_list = np.arange(2 * self.nnod)  # Dofs initialization
        # Odd = vertical, even = horizontal
        self.fixed_dofs = self.dofs_list[: self.nnodx_web * 2]
        self.free_dofs = np.setdiff1d(self.dofs_list, self.fixed_dofs)
        self.dofs = np.ones(self.dofs_list.shape)
        self.dofs[self.fixed_dofs] = 0
        self.dofs_nodes_connectivity = self.dofs.reshape((-1, 2))
        ## FORCES
        self.nnod_web = self.nnodx_web * (self.nnody_web - 1)
        self.force_dofs = (
            2
            * np.array(
                (self.nnod_web, self.nnod_web + self.nnodx_flange - 1), dtype=int
            ).reshape(-1)
            + 1
        )  # This should be a 1D np.array object
        self.force = self.f / self.force_dofs.size
        # fixed dofs elimination
        self.R = np.zeros(self.dofs_list.shape)
        self.R[self.force_dofs] = -self.force
        self.R_free = self.R[self.free_dofs]


class Column(BoundaryConditions2D):
    """Column Problem"""

    def __init__(
        self,
        nnodx_cel: int,
        nnody_cel: int,
        ncelx_str: int,
        ncely_str: int,
        f: float,
        L,
    ):
        super().__init__(nnodx_cel, nnody_cel, ncelx_str, ncely_str, f, L)
        self.name = "Column_2D"
        ## DOFS
        # Odd = vertical, even = horizontal
        self.fixed_dofs = np.arange(0, 2 * self.nnodx)
        self.free_dofs = np.setdiff1d(self.dofs_list, self.fixed_dofs)
        self.dofs = np.ones(self.dofs_list.shape)
        self.dofs[self.fixed_dofs] = 0
        self.dofs_nodes_connectivity = self.dofs.reshape((-1, 2))
        ## FORCES
        self.force_dofs = np.array(
            2 * (self.nnod - np.ceil(self.nnodx / 2)) + 1, dtype=int
        ).reshape(-1)  # This should be a 1D np.array object
        self.force = self.f / self.force_dofs.size
        # fixed dofs elimination
        self.R = np.zeros(self.dofs_list.shape)
        self.R[self.force_dofs] = -self.force
        self.R_free = self.R[self.free_dofs]


class Column_dist(BoundaryConditions2D):
    """Column Problem"""

    def __init__(
        self,
        nnodx_cel: int,
        nnody_cel: int,
        ncelx_str: int,
        ncely_str: int,
        f: float,
        L,
    ):
        super().__init__(nnodx_cel, nnody_cel, ncelx_str, ncely_str, f, L)
        self.name = "Column_Dist_2D"
        ## DOFS
        # Odd = vertical, even = horizontal
        self.fixed_dofs = np.arange(0, 2 * self.nnodx)
        self.free_dofs = np.setdiff1d(self.dofs_list, self.fixed_dofs)
        self.dofs = np.ones(self.dofs_list.shape)
        self.dofs[self.fixed_dofs] = 0
        self.dofs_nodes_connectivity = self.dofs.reshape((-1, 2))
        ## FORCES
        self.force_dofs = np.arange(
            2 * (self.nnod - self.nnodx) + 1, 2 * self.nnod, 2
        )  # This should be a 1D np.array object
        self.force = self.f / self.force_dofs.size
        # fixed dofs elimination
        self.R = np.zeros(self.dofs_list.shape)
        self.R[self.force_dofs] = -self.force
        self.R_free = self.R[self.free_dofs]


class Ten_bar_benchmark:
    """Ten bar benchmarck cantilever beam from "A new approach for the solution of singular optima in truss
    topology optimization with stress and local buckling constraints" by X. Guo, G. Cheng, K. Yamazaki
    We use the very same order"""

    def __init__(self):
        self.L = [720, 360]
        self.is3D = False
        self.isReduced = False
        self.isFreeForm = False
        # Elements

        self.nnodx_cel = 2
        self.nnody_cel = 2
        self.nnod_cel = self.nnodx_cel * self.nnody_cel
        self.ncelx_str = 2
        self.ncely_str = 1
        self.ncel_str = self.ncelx_str * self.ncely_str

        # Multiscale parameters
        self.isMultiscale = False

        self.nnodx = 3
        self.nnody = 2
        self.nnod = 6

        # Nodes matrix
        self.nodes_index = np.zeros((self.nnod, 2), dtype=int)
        self.nodes = np.zeros((self.nnod, 2), dtype=float)
        x, y = np.meshgrid(np.arange(self.nnodx), np.arange(self.nnody), indexing="xy")
        self.nodes_index[:, 0] = x.ravel()
        self.nodes_index[:, 1] = y.ravel()

        # Scaling to the desired dimensions
        self.nodes[:, 0] = (
            self.nodes_index[:, 0] / np.max(self.nodes_index[:, 0]) * self.L[0]
        )  # X
        self.nodes[:, 1] = (
            self.nodes_index[:, 1] / np.max(self.nodes_index[:, 1]) * self.L[1]
        )  # Y

        self.dofs_list = np.arange(2 * self.nnod)  # Dofs initialization
        # Ground structure assembly
        self.ground_stucture_list = list(
            [
                [3, 4],
                [4, 5],
                [0, 1],
                [1, 2],
                [1, 4],
                [2, 5],
                [1, 3],
                [0, 4],
                [2, 4],
                [1, 5],
            ]
        )

        self.ground_structure = np.array(self.ground_stucture_list)
        self.N_cell = [
            int(self.ground_structure.shape[0] / self.ncel_str)
        ]  # Number of elements per cell
        self.N_cell_max = self.N_cell[0]  # Used for reduced problems
        # using 2-norm to calculate the beams length [mm]
        self.ground_structure_length = np.linalg.norm(
            (
                self.nodes[self.ground_structure[:, 0], 1]
                - self.nodes[self.ground_structure[:, 1], 1],
                self.nodes[self.ground_structure[:, 0], 0]
                - self.nodes[self.ground_structure[:, 1], 0],
            ),
            axis=0,
        )
        # Force magnitude
        self.f = 200

        # Is L-shape param initialization
        self.isLshape = False

        self.name = "Ten_Bar_bench"
        ## DOFS
        # Odd = vertical, even = horizontal
        self.fixed_dofs = np.sort(
            np.array(
                list(range(0, 2 * (self.nnodx * (self.nnody)), 2 * self.nnodx))
                + list(range(1, 2 * (self.nnodx * (self.nnody)) + 1, 2 * self.nnodx))
            )
        )
        self.free_dofs = np.setdiff1d(self.dofs_list, self.fixed_dofs)
        self.dofs = np.ones(self.dofs_list.shape)
        self.dofs[self.fixed_dofs] = 0
        self.dofs_nodes_connectivity = self.dofs.reshape((-1, 2))
        ## FORCES
        self.force_dofs = np.array([-3, -1], dtype=int).reshape(
            -1
        )  # This should be a 1D np.array object
        self.force = self.f / self.force_dofs.size
        # fixed dofs elimination
        self.R = np.zeros(self.dofs_list.shape)
        self.R[self.force_dofs] = -self.force
        self.R_free = self.R[self.free_dofs]


class Ten_bar_benchmark_multi:
    """Ten bar benchmarck cantilever beam from "A new approach for the solution of singular optima in truss
    topology optimization with stress and local buckling constraints" by X. Guo, G. Cheng, K. Yamazaki
    We use the very same order. Multi load case version"""

    def __init__(self):
        self.L = [720, 360]
        self.is3D = False
        self.isReduced = False
        self.isFreeForm = False
        # Elements

        self.nnodx_cel = 2
        self.nnody_cel = 2
        self.nnod_cel = self.nnodx_cel * self.nnody_cel
        self.ncelx_str = 2
        self.ncely_str = 1
        self.ncel_str = self.ncelx_str * self.ncely_str

        # Multiscale parameters
        self.isMultiscale = False

        self.nnodx = 3
        self.nnody = 2
        self.nnod = 6

        # Nodes matrix
        self.nodes_index = np.zeros((self.nnod, 2), dtype=int)
        self.nodes = np.zeros((self.nnod, 2), dtype=float)
        x, y = np.meshgrid(np.arange(self.nnodx), np.arange(self.nnody), indexing="xy")
        self.nodes_index[:, 0] = x.ravel()
        self.nodes_index[:, 1] = y.ravel()

        # Scaling to the desired dimensions
        self.nodes[:, 0] = (
            self.nodes_index[:, 0] / np.max(self.nodes_index[:, 0]) * self.L[0]
        )  # X
        self.nodes[:, 1] = (
            self.nodes_index[:, 1] / np.max(self.nodes_index[:, 1]) * self.L[1]
        )  # Y

        self.dofs_list = np.arange(2 * self.nnod)  # Dofs initialization
        # Ground structure assembly
        self.ground_stucture_list = list(
            [
                [3, 4],
                [4, 5],
                [0, 1],
                [1, 2],
                [1, 4],
                [2, 5],
                [1, 3],
                [0, 4],
                [2, 4],
                [1, 5],
            ]
        )

        self.ground_structure = np.array(self.ground_stucture_list)
        self.N_cell = [
            int(self.ground_structure.shape[0] / self.ncel_str)
        ]  # Number of elements per cell
        self.N_cell_max = self.N_cell[0]  # Used for reduced problems
        # using 2-norm to calculate the beams length [mm]
        self.ground_structure_length = np.linalg.norm(
            (
                self.nodes[self.ground_structure[:, 0], 1]
                - self.nodes[self.ground_structure[:, 1], 1],
                self.nodes[self.ground_structure[:, 0], 0]
                - self.nodes[self.ground_structure[:, 1], 0],
            ),
            axis=0,
        )
        # Force magnitude
        self.f = 200

        # Is L-shape param initialization
        self.isLshape = False

        self.name = "Ten_Bar_bench_multi"
        ## DOFS
        # Odd = vertical, even = horizontal
        self.fixed_dofs = np.sort(
            np.array(
                list(range(0, 2 * (self.nnodx * (self.nnody)), 2 * self.nnodx))
                + list(range(1, 2 * (self.nnodx * (self.nnody)) + 1, 2 * self.nnodx))
            )
        )
        self.free_dofs = np.setdiff1d(self.dofs_list, self.fixed_dofs)
        self.dofs = np.ones(self.dofs_list.shape)
        self.dofs[self.fixed_dofs] = 0
        self.dofs_nodes_connectivity = self.dofs.reshape((-1, 2))
        ## FORCES
        self.force_dofs = np.array([-7, -1], dtype=int).reshape(
            -1
        )  # This should be a 1D np.array object
        self.force = self.f / 2
        # fixed dofs elimination
        self.R = np.zeros((self.dofs_list.size, 2))
        self.R[-7, 0] = -self.force
        self.R[-1, 1] = self.force
        self.R_free = self.R[self.free_dofs]


class Cantilever_Low_Achzingher(BoundaryConditions2D):
    """Cantilever Beam Achzingher"""

    def __init__(self):
        self.L = [10, 2]
        self.is3D = False
        self.isReduced = False
        self.f = 1
        # Elements
        self.nnodx_cel = 2
        self.nnody_cel = 2
        self.nnod_cel = self.nnodx_cel * self.nnody_cel
        self.ncelx_str = 10
        self.ncely_str = 2
        self.ncel_str = self.ncelx_str * self.ncely_str

        # Multiscale parameters
        self.isMultiscale = False

        super().__init__(
            self.nnodx_cel,
            self.nnody_cel,
            self.ncelx_str,
            self.ncely_str,
            self.f,
            self.L,
        )

        # Nodes
        self.nnodx = 11
        self.nnody = 3
        self.nnod = 33

        self.ground_stucture_list = list(
            map(sorted, self.ground_stucture_list)
        )  # Sort the lists in the list
        self.ground_stucture_list = list(
            map(list, set(map(tuple, self.ground_stucture_list)))
        )  # Remove duplicates in a list of lists
        self.ground_stucture_list.sort(key=lambda x: x[0])
        self.ground_stucture_list.remove([0, 11])
        self.ground_stucture_list.remove([11, 22])

        self.ground_structure = np.array(self.ground_stucture_list)

        self.ground_structure_length = np.linalg.norm(
            (
                self.nodes[self.ground_structure[:, 0], 1]
                - self.nodes[self.ground_structure[:, 1], 1],
                self.nodes[self.ground_structure[:, 0], 0]
                - self.nodes[self.ground_structure[:, 1], 0],
            ),
            axis=0,
        )
        # Force magnitude
        self.name = "Cantilever_Low_2D_ACH"

        ## DOFS
        # Odd = vertical, even = horizontal
        self.fixed_dofs = np.sort(
            np.array(
                list(range(0, 2 * (self.nnodx * (self.nnody)), 2 * self.nnodx))
                + list(range(1, 2 * (self.nnodx * (self.nnody)) + 1, 2 * self.nnodx))
            )
        )
        self.free_dofs = np.setdiff1d(self.dofs_list, self.fixed_dofs)
        self.dofs = np.ones(self.dofs_list.shape)
        self.dofs[self.fixed_dofs] = 0
        self.dofs_nodes_connectivity = self.dofs.reshape((-1, 2))
        ## FORCES
        self.force_dofs = np.array(2 * (self.nnodx - 1) + 1, dtype=int).reshape(
            -1
        )  # This should be a 1D np.array object
        self.force = self.f / self.force_dofs.size
        # fixed dofs elimination
        self.R = np.zeros(self.dofs_list.shape)
        self.R[self.force_dofs] = -self.force
        self.R_free = self.R[self.free_dofs]


class Cantilever_Low_Achzingher_mod(BoundaryConditions2D):
    """Cantilever Beam Achzingher with Tyas mod"""

    def __init__(self):
        self.L = [10, 2]
        self.is3D = False
        self.isReduced = False
        self.f = 1
        # Elements
        self.nnodx_cel = 2
        self.nnody_cel = 2
        self.nnod_cel = self.nnodx_cel * self.nnody_cel
        self.ncelx_str = 10
        self.ncely_str = 2
        self.ncel_str = self.ncelx_str * self.ncely_str

        # Multiscale parameters
        self.isMultiscale = False

        super().__init__(
            self.nnodx_cel,
            self.nnody_cel,
            self.ncelx_str,
            self.ncely_str,
            self.f,
            self.L,
        )

        # Nodes
        self.nnodx = 11
        self.nnody = 3
        self.nnod = 33

        # Modify initial GS
        start = np.arange(20)
        end = np.arange(2, 22)
        start_end = np.vstack([start, end]).T.tolist()
        start_end.remove([9, 11])
        start_end.remove([10, 12])
        self.ground_stucture_list.extend(start_end)
        append = np.array(list([[0, 24], [0, 22], [2, 22]]))

        append = append.reshape((3, 2, 1)) + np.arange(9).reshape((1, 1, -1))
        append = append.transpose((2, 0, 1)).reshape(-1, 2)
        append = np.vstack([append, np.array([[9, 31], [10, 32]])])

        self.ground_stucture_list.extend(append)

        self.ground_stucture_list = list(
            map(sorted, self.ground_stucture_list)
        )  # Sort the lists in the list
        self.ground_stucture_list = list(
            map(list, set(map(tuple, self.ground_stucture_list)))
        )  # Remove duplicates in a list of lists
        self.ground_stucture_list.sort(key=lambda x: x[0])

        self.ground_structure = np.array(self.ground_stucture_list)

        self.ground_structure_length = np.linalg.norm(
            (
                self.nodes[self.ground_structure[:, 0], 1]
                - self.nodes[self.ground_structure[:, 1], 1],
                self.nodes[self.ground_structure[:, 0], 0]
                - self.nodes[self.ground_structure[:, 1], 0],
            ),
            axis=0,
        )
        # Force magnitude
        self.name = "Cantilever_Low_2D_ACH_mod"

        ## DOFS
        # Odd = vertical, even = horizontal
        self.fixed_dofs = np.sort(
            np.array(
                list(range(0, 2 * (self.nnodx * (self.nnody)), 2 * self.nnodx))
                + list(range(1, 2 * (self.nnodx * (self.nnody)) + 1, 2 * self.nnodx))
            )
        )
        self.free_dofs = np.setdiff1d(self.dofs_list, self.fixed_dofs)
        self.dofs = np.ones(self.dofs_list.shape)
        self.dofs[self.fixed_dofs] = 0
        self.dofs_nodes_connectivity = self.dofs.reshape((-1, 2))
        ## FORCES
        self.force_dofs = np.array(2 * (self.nnodx - 1) + 1, dtype=int).reshape(
            -1
        )  # This should be a 1D np.array object
        self.force = self.f / self.force_dofs.size
        # fixed dofs elimination
        self.R = np.zeros(self.dofs_list.shape)
        self.R[self.force_dofs] = -self.force
        self.R_free = self.R[self.free_dofs]


class MichCantilever_Tugilimana(BoundaryConditions2D):
    """Michell Cantilever Beam"""

    def __init__(
        self,
        nnodx_cel: int,
        nnody_cel: int,
        n_sub_x_str: int,
        n_sub_y_str: int,
        f: float,
        L,
    ):
        super().__init__(nnodx_cel, nnody_cel, n_sub_x_str, n_sub_y_str, f, L)
        self.name = "MichCantilever_Tug_2D"
        ## DOFS
        # Odd = vertical, even = horizontal
        self.x_mich = int(self.nnody / 4)
        self.fixed_dofs = np.sort(
            np.array(
                list(
                    range(
                        2 * self.x_mich * self.nnodx,
                        2 * (self.nnody - self.x_mich) * self.nnodx,
                        2 * self.nnodx,
                    )
                )
                + list(
                    range(
                        2 * self.x_mich * self.nnodx + 1,
                        2 * (self.nnody - self.x_mich) * self.nnodx + 1,
                        2 * self.nnodx,
                    )
                )
            )
        ).astype(int)
        self.free_dofs = np.setdiff1d(self.dofs_list, self.fixed_dofs)
        self.dofs = np.ones(self.dofs_list.shape)
        self.dofs[self.fixed_dofs] = 0
        self.dofs_nodes_connectivity = self.dofs.reshape((-1, 2))
        ## FORCES
        self.force_dofs = np.array(
            2 * (self.nnodx * np.ceil(self.nnody / 2) - 1) + 1, dtype=int
        ).reshape(-1)  # This should be a 1D np.array object
        self.force = f / self.force_dofs.size
        # fixed dofs elimination
        self.R = np.zeros(self.dofs_list.shape)
        self.R[self.force_dofs] = -self.force
        self.R_free = self.R[self.free_dofs]


### 3D
class BoundaryConditions3D:
    """
    Class used to define the boundary conditions of a 3D problem.
    Functionalty for setting fixed dof and forces.
    """

    def __init__(
        self,
        nnodx_cel: int,
        nnody_cel: int,
        nnodz_cel: int,
        ncelx_str: int = 1,
        ncely_str: int = 1,
        ncelz_str: int = 1,
        f: float = 1,
        L: int = 1000,
    ):
        """
        Create the mesh for the problem.

        Parameters
        ----------
        nnodx:
            The number of elements in the x direction.
        nnody:
            The number of elements in the y direction.
        F:
            Magnitude of the total force applied to the structure.

        """
        self.L = L  # Sizes of the structure
        self.is3D = True  # Used for trussfem
        self.isReduced = False
        self.isFreeForm = False

        # Elements

        self.nnodx_cel = nnodx_cel
        self.nnody_cel = nnody_cel
        self.nnodz_cel = nnodz_cel
        self.nnod_cel = self.nnodx_cel * self.nnody_cel * self.nnodz_cel
        self.ncelx_str = ncelx_str
        self.ncely_str = ncely_str
        self.ncelz_str = ncelz_str
        self.ncel_str = self.ncelx_str * self.ncely_str * self.ncelz_str

        self.cell_size = np.array(
            [
                self.L[0] / self.ncelx_str,
                self.L[1] / self.ncely_str,
                self.L[2] / self.ncelz_str,
            ]
        )

        # Multiscale parameters
        self.isMultiscale = False
        if self.ncelx_str != 1 or self.ncely_str != 1 or self.ncelz_str != 1:
            self.isMultiscale = True

        self.nnodx = self.nnodx_cel * self.ncelx_str - (self.ncelx_str - 1)
        self.nnody = self.nnody_cel * self.ncely_str - (self.ncely_str - 1)
        self.nnodz = self.nnodz_cel * self.ncelz_str - (self.ncelz_str - 1)

        self.nnod = self.nnodx * self.nnody * self.nnodz

        # Nodes matrix
        self.nodes_index = np.zeros((self.nnod, 3), dtype=int)
        self.nodes = np.zeros((self.nnod, 3), dtype=float)
        x, y, z = np.meshgrid(
            np.arange(self.nnodx),
            np.arange(self.nnody),
            np.arange(self.nnodz),
            indexing="ij",
        )
        self.nodes_index[:, 0] = x.ravel()
        self.nodes_index[:, 1] = y.ravel()
        self.nodes_index[:, 2] = z.ravel()

        # Scaling to the desired dimensions
        self.nodes[:, 0] = (
            self.nodes_index[:, 0] / np.max(self.nodes_index[:, 0]) * L[0]
        )  # X
        self.nodes[:, 1] = (
            self.nodes_index[:, 1] / np.max(self.nodes_index[:, 1]) * L[1]
        )  # Y
        self.nodes[:, 2] = (
            self.nodes_index[:, 2] / np.max(self.nodes_index[:, 2]) * L[2]
        )  # Z

        self.dofs_list = np.arange(3 * self.nnod)  # Dofs initialization
        self.M = 3 * self.nnod
        # Ground structure assembly
        if self.isMultiscale:
            self.nodes_list = np.arange(self.nnod).reshape(
                (self.nnodx, self.nnody, self.nnodz)
            )
            self.nodes_cell = view_as_windows_3D(
                self.nodes_list,
                self.nnodx_cel,
                self.nnody_cel,
                self.nnodz_cel,
                self.ncelx_str,
                self.ncely_str,
                self.ncelz_str,
            ).reshape(
                self.ncely_str * self.ncelx_str * self.ncelz_str,
                self.nnodx_cel * self.nnody_cel * self.nnodz_cel,
            )  # one row = one cellule
            self.ground_stucture_list = []  # initialization of the list containing all the beam
            for i in range(self.nodes_cell.shape[0]):  # cycling trough all the cells
                self.ground_stucture_list.append(
                    (list(itertools.combinations(self.nodes_cell[i, :], 2)))
                )
            self.ground_stucture_list = list(
                itertools.chain.from_iterable(self.ground_stucture_list)
            )
            self.ground_stucture_list_cell = list(
                itertools.combinations(self.nodes_cell[0, :], 2)
            )  # ground structure of the first cellule
        else:
            self.ground_stucture_list = list(
                itertools.combinations(range(len(self.nodes_index)), 2)
            )
        self.ground_structure = np.array(self.ground_stucture_list)
        self.N_cell = [
            int(self.ground_structure.shape[0] / self.ncel_str)
        ]  # Number of elements per cell
        self.N_cell_max = self.N_cell[0]  # Used for reduced problems
        # using 2-norm to calculate the beams length
        self.ground_structure_length = np.linalg.norm(
            (
                self.nodes[self.ground_structure[:, 0], 2]
                - self.nodes[self.ground_structure[:, 1], 2],
                self.nodes[self.ground_structure[:, 0], 1]
                - self.nodes[self.ground_structure[:, 1], 1],
                self.nodes[self.ground_structure[:, 0], 0]
                - self.nodes[self.ground_structure[:, 1], 0],
            ),
            axis=0,
        )
        # Force magnitude
        self.f = f


class CantileverBeam_3D(BoundaryConditions3D):
    """Cantilever Beam"""

    def __init__(
        self,
        nnodx_cel: int,
        nnody_cel: int,
        nnodz_cel: int,
        ncelx_str: int,
        ncely_str: int,
        ncelz_str: int,
        f: float,
        L,
    ):
        super().__init__(
            nnodx_cel, nnody_cel, nnodz_cel, ncelx_str, ncely_str, ncelz_str, f, L
        )
        self.name = "CantileverBeam_3D"
        ## DOFS
        # 0 = x, 1 = y, 2 = z
        self.fixed_dofs = np.concatenate(
            [
                np.arange(0, 3 * self.nnodz),
                np.arange(
                    3 * self.nnodz * self.nnody - 3 * self.nnodz,
                    3 * self.nnodz * self.nnody,
                ),
            ]
        )
        self.free_dofs = np.setdiff1d(self.dofs_list, self.fixed_dofs)
        self.dofs = np.ones(self.dofs_list.shape)
        self.dofs[self.fixed_dofs] = 0
        self.dofs_nodes_connectivity = self.dofs.reshape((-1, 3))
        ## FORCES
        self.force_dofs = np.array(
            -int((self.nnody - 1) / 2 * self.nnodz + (self.nnodz - 1) / 2 + 1) * 3 + 2
        ).reshape(-1)  # This should be a 1D np.array object
        self.force = -self.f
        # fixed dofs elimination
        self.R = np.zeros(self.dofs_list.shape)
        self.R[self.force_dofs] = self.force
        self.R_free = self.R[self.free_dofs]


class CantileverBeam_3D_smear(BoundaryConditions3D):
    """Cantilever Beam with smeared forces"""

    def __init__(
        self,
        nnodx_cel: int,
        nnody_cel: int,
        nnodz_cel: int,
        ncelx_str: int,
        ncely_str: int,
        ncelz_str: int,
        f: float,
        L,
    ):
        super().__init__(
            nnodx_cel, nnody_cel, nnodz_cel, ncelx_str, ncely_str, ncelz_str, f, L
        )
        self.name = "CantileverBeam_3D"
        ## DOFS
        # 0 = x, 1 = y, 2 = z
        fix_x = np.concatenate(
            [
                np.arange(
                    0, 3 * self.nnodz * self.nnody - 3 * self.nnodz + 1, 3 * self.nnodz
                ),
                np.arange(
                    3 * self.nnodz - 3,
                    3 * self.nnodz * self.nnody - 3 + 1,
                    3 * self.nnodz,
                ),
            ]
        )
        self.fixed_dofs = np.concatenate([fix_x, fix_x + 1, fix_x + 2])
        self.free_dofs = np.setdiff1d(self.dofs_list, self.fixed_dofs)
        self.dofs = np.ones(self.dofs_list.shape)
        self.dofs[self.fixed_dofs] = 0
        self.dofs_nodes_connectivity = self.dofs.reshape((-1, 3))
        ## FORCES
        first = 3 * self.nnodz * self.nnody * (self.nnodx - 1) + 2
        last = (
            3 * self.nnodz * self.nnody * (self.nnodx - 1)
            + 3 * self.nnody * self.nnodz
            + 2
        )
        self.force_dofs = np.arange(first, last, 3 * self.nnodz).reshape(
            -1
        )  # This should be a 1D np.array object
        self.force = -self.f / self.force_dofs.size
        # fixed dofs elimination
        self.R = np.zeros(self.dofs_list.shape)
        self.R[self.force_dofs] = self.force
        self.R_free = self.R[self.free_dofs]


class SimplySuppTruss_3D(BoundaryConditions3D):
    """Simply supported truss"""

    def __init__(
        self,
        nnodx_cel: int,
        nnody_cel: int,
        nnodz_cel: int,
        ncelx_str: int,
        ncely_str: int,
        ncelz_str: int,
        f: float,
        L,
    ):
        super().__init__(
            nnodx_cel, nnody_cel, nnodz_cel, ncelx_str, ncely_str, ncelz_str, f, L
        )
        self.name = "SimplySuppTruss_3D"
        ## DOFS
        # 0 = x, 1 = y, 2 = z
        sx = np.concatenate(
            [
                np.array([0, 1, 2]),
                np.array(
                    [
                        3 * self.nnodz * (self.nnody - 1),
                        3 * self.nnodz * (self.nnody - 1) + 1,
                        3 * self.nnodz * (self.nnody - 1) + 2,
                    ]
                ),
            ]
        )
        dx = np.concatenate(
            [
                np.array(
                    [
                        3 * self.nnod - 3 * self.nnodz * self.nnody,
                        3 * self.nnod - 3 * self.nnodz * self.nnody + 1,
                        3 * self.nnod - 3 * self.nnodz * self.nnody + 2,
                    ]
                ),
                np.array(
                    [
                        3 * self.nnod - 3 * self.nnodz,
                        3 * self.nnod - 3 * self.nnodz + 1,
                        3 * self.nnod - 3 * self.nnodz + 2,
                    ]
                ),
            ]
        )
        # dx = np.concatenate([np.arange(3*self.nnod-3*self.nnodz*self.nnody,3*self.nnod,3*self.nnodz),np.arange(3*self.nnod-3*self.nnodz*self.nnody,3*self.nnod,3*self.nnodz)+2])
        self.fixed_dofs = np.concatenate([sx, dx])
        self.free_dofs = np.setdiff1d(self.dofs_list, self.fixed_dofs)
        self.dofs = np.ones(self.dofs_list.shape)
        self.dofs[self.fixed_dofs] = 0
        self.dofs_nodes_connectivity = self.dofs.reshape((-1, 3))
        ## FORCES
        force_dofs_temp = np.arange(
            3 * self.nnodz * self.nnody + 2,
            3 * self.nnod - 3 * self.nnodz * self.nnody,
            3 * self.nnodz,
        )  # Z axis
        node_forces = force_dofs_temp // 3
        mean = self.nodes[node_forces, 1] == self.L[1] / 2
        xxx = (
            (self.nodes[node_forces, 0] == self.L[0] / 6)
            | (self.nodes[node_forces, 0] == 2 * self.L[0] / 6)
            | (self.nodes[node_forces, 0] == 3 * self.L[0] / 6)
            | (self.nodes[node_forces, 0] == 4 * self.L[0] / 6)
            | (self.nodes[node_forces, 0] == 5 * self.L[0] / 6)
        )
        self.force_dofs = force_dofs_temp[np.logical_and(mean, xxx)]

        self.force = -self.f / self.force_dofs.size
        # fixed dofs elimination
        self.R = np.zeros(self.dofs_list.shape)
        self.R[self.force_dofs] = self.force
        self.R_free = self.R[self.free_dofs]


class SimplySuppTruss_3D_symmetric(BoundaryConditions3D):
    """Simply supported truss x y symmetry"""

    def __init__(
        self,
        nnodx_cel: int,
        nnody_cel: int,
        nnodz_cel: int,
        ncelx_str: int,
        ncely_str: int,
        ncelz_str: int,
        f: float,
        L,
    ):
        if ncelx_str * ncely_str * ncelz_str > 1:
            ncelx_str_bc = ncelx_str / 2
            ncely_str_bc = ncely_str / 2
            nnodx_cel_bc = nnodx_cel
            nnody_cel_bc = nnody_cel
        else:
            ncelx_str_bc = ncelx_str
            ncely_str_bc = ncely_str
            nnodx_cel_bc = (nnodx_cel + 1) / 2
            nnody_cel_bc = (nnody_cel + 1) / 2
        L_bc = L.copy()
        L_bc[0] = L[0] / 2
        L_bc[1] = L[1] / 2
        super().__init__(
            int(nnodx_cel_bc),
            int(nnody_cel_bc),
            nnodz_cel,
            int(ncelx_str_bc),
            int(ncely_str_bc),
            ncelz_str,
            f,
            list(map(int, L_bc)),
        )
        self.name = "SimplySuppTruss_3D_sym_"
        ## DOFS
        # 0 = x, 1 = y, 2 = z
        sx = np.array([0, 1, 2])

        nod_symm_xz = np.where(self.nodes[:, 1] == self.L[1])[0]
        symm_xz = nod_symm_xz * 3 + 1  # DOF Y=0

        nod_symm_yz = np.where(self.nodes[:, 0] == self.L[0])[0]
        symm_yz = nod_symm_yz * 3  # DOF X=0

        self.fixed_dofs = np.concatenate([sx, symm_xz, symm_yz])
        self.free_dofs = np.setdiff1d(self.dofs_list, self.fixed_dofs)
        self.dofs = np.ones(self.dofs_list.shape)
        self.dofs[self.fixed_dofs] = 0
        self.dofs_nodes_connectivity = self.dofs.reshape((-1, 3))
        ## FORCES
        force_dofs_temp = np.arange(
            3 * self.nnodz * self.nnody + 2, 3 * self.nnod, 3 * self.nnodz
        )  # Z axis
        node_forces = force_dofs_temp // 3
        mean = self.nodes[node_forces, 1] == self.L[1]
        xxx = (
            (self.nodes[node_forces, 0] == self.L[0] / 3)
            | (self.nodes[node_forces, 0] == 2 * self.L[0] / 3)
            | (self.nodes[node_forces, 0] == 3 * self.L[0] / 3)
        )
        self.force_dofs = force_dofs_temp[np.logical_and(mean, xxx)]

        self.force = -self.f / 5
        # fixed dofs elimination
        self.R = np.zeros(self.dofs_list.shape)
        self.R[self.force_dofs] = self.force
        self.R_free = self.R[self.free_dofs]


class Column_3D(BoundaryConditions3D):
    """Compression column Beam"""

    def __init__(
        self,
        nnodx_cel: int,
        nnody_cel: int,
        nnodz_cel: int,
        ncelx_str: int,
        ncely_str: int,
        ncelz_str: int,
        f: float,
        L,
    ):
        super().__init__(
            nnodx_cel, nnody_cel, nnodz_cel, ncelx_str, ncely_str, ncelz_str, f, L
        )
        self.name = "Column_3D"
        ## DOFS
        # 0 = x, 1 = y, 2 = z
        self.fixed_dofs = np.concatenate([np.arange(0, 3 * self.nnodz * self.nnody)])
        self.free_dofs = np.setdiff1d(self.dofs_list, self.fixed_dofs)
        self.dofs = np.ones(self.dofs_list.shape)
        self.dofs[self.fixed_dofs] = 0
        self.dofs_nodes_connectivity = self.dofs.reshape((-1, 3))
        ## FORCES
        self.force_dofs = np.array(
            -int((self.nnody - 1) / 2 * self.nnodz + (self.nnodz - 1) / 2 + 1) * 3
        ).reshape(-1)  # This should be a 1D np.array object
        self.force = -self.f
        # fixed dofs elimination
        self.R = np.zeros(self.dofs_list.shape)
        self.R[self.force_dofs] = self.force
        self.R_free = self.R[self.free_dofs]


class Wing_3D(BoundaryConditions3D):
    """STARAC Beam"""

    def __init__(
        self,
        nnodx_cel: int,
        nnody_cel: int,
        nnodz_cel: int,
        ncelx_str: int,
        ncely_str: int,
        ncelz_str: int,
        f: float,
        L: int,
    ):
        super().__init__(
            nnodx_cel, nnody_cel, nnodz_cel, ncelx_str, ncely_str, ncelz_str, f, L
        )
        self.name = "Wing_3D"
        ## DOFS
        # 0 = x, 1 = y, 2 = z
        self.fixed_dofs = np.arange(0, 3 * self.nnodz * self.nnody)
        self.free_dofs = np.setdiff1d(self.dofs_list, self.fixed_dofs)
        self.dofs = np.ones(self.dofs_list.shape)
        self.dofs[self.fixed_dofs] = 0
        ## FORCES
        self.force_dofs_lift = np.arange(
            3 * self.nnodz - 1, 3 * self.nnod, 3 * self.nnodz
        )
        self.force_dofs_torque = np.array(
            [
                wing_3D.calculate_ribs_dof(self.nnody, self.nnodz, i)
                for i in range(self.nnodx)
            ]
        ).ravel()
        # fixed dofs elimination
        self.R = np.zeros(self.dofs_list.shape)
        self.lift, self.torque = wing_3D.calculateLoads_dist(
            self.nnodx, self.nnody, self.ncelx_str, self.ncely_str, L[0], L[2]
        )

        self.R[self.force_dofs_lift] = self.lift
        self.R[self.force_dofs_torque] = self.torque
        self.R_free = self.R[self.free_dofs]


class Wing_3D_center(BoundaryConditions3D):
    """STARAC Beam"""

    def __init__(
        self,
        nnodx_cel: int,
        nnody_cel: int,
        nnodz_cel: int,
        ncelx_str: int,
        ncely_str: int,
        ncelz_str: int,
        f: float,
        L: int,
    ):
        super().__init__(
            nnodx_cel, nnody_cel, nnodz_cel, ncelx_str, ncely_str, ncelz_str, f, L
        )
        self.name = "Wing_3D_center"
        ## DOFS
        # 0 = x, 1 = y, 2 = z
        self.fixed_dofs = np.arange(0, 3 * self.nnodz * self.nnody)
        self.free_dofs = np.setdiff1d(self.dofs_list, self.fixed_dofs)
        self.dofs = np.ones(self.dofs_list.shape)
        self.dofs[self.fixed_dofs] = 0
        ## FORCES
        start_center = self.nnody * int(np.floor(self.nnodx_cel / 2)) + int(
            np.floor(self.nnody_cel / 2)
        )
        first_row = start_center + np.arange(self.ncely_str) * int(self.nnody_cel - 1)
        center_nodes_mask = (
            first_row.reshape(-1, 1)
            + (
                np.arange(self.ncelx_str) * int(self.nnody * (self.nnodx_cel - 1))
            ).reshape(1, self.ncelx_str)
        ).T.ravel()  # select the center nodes on the xy plane
        upp_nodes_z = np.arange(3 * self.nnodz - 1, 3 * self.nnod, 3 * self.nnodz)
        self.force_dofs_lift = upp_nodes_z[center_nodes_mask]
        x_center = int(np.floor(self.nnodx_cel / 2)) + np.arange(
            self.ncelx_str, dtype="int"
        ) * (self.nnodx_cel - 1)
        self.force_dofs_torque = np.array(
            [wing_3D.calculate_ribs_dof(self.nnody, self.nnodz, i) for i in x_center]
        ).ravel()
        # fixed dofs elimination
        self.R = np.zeros(self.dofs_list.shape)
        self.lift, self.torque = wing_3D.calculateLoads_dist_centered(
            self.ncelx_str, self.ncely_str, L[0], L[1], L[2]
        )

        self.R[self.force_dofs_lift] = self.lift
        # self.R[self.force_dofs_torque] = self.torque
        self.R_free = self.R[self.free_dofs]


class Wing_3D_conc_center(BoundaryConditions3D):
    """STARAC Beam"""

    def __init__(
        self,
        nnodx_cel: int,
        nnody_cel: int,
        nnodz_cel: int,
        ncelx_str: int,
        ncely_str: int,
        ncelz_str: int,
        f: float,
        L: int,
    ):
        super().__init__(
            nnodx_cel, nnody_cel, nnodz_cel, ncelx_str, ncely_str, ncelz_str, f, L
        )
        self.name = "Wing_3D_conc_center"
        ## DOFS
        # 0 = x, 1 = y, 2 = z
        self.fixed_dofs = np.arange(0, 3 * self.nnodz * self.nnody)
        self.free_dofs = np.setdiff1d(self.dofs_list, self.fixed_dofs)
        self.dofs = np.ones(self.dofs_list.shape)
        self.dofs[self.fixed_dofs] = 0
        ## FORCES
        start_center = self.nnodz * int(np.floor(self.nnody_cel / 2)) + int(
            np.floor(self.nnodz_cel / 2)
        )
        first_row = start_center + np.arange(self.ncelz_str) * int(self.nnodz_cel - 1)
        center_nodes_mask = (
            first_row.reshape(-1, 1)
            + (
                np.arange(self.ncely_str) * int(self.nnodz * (self.nnody_cel - 1))
            ).reshape(1, self.ncely_str)
        ).T.ravel()  # select the center nodes on the xy plane
        right_nodes_z = np.arange(
            3 * self.nnod - 3 * self.nnodz * self.nnody + 2, 3 * self.nnod, 3
        )
        self.force_dofs_lift = right_nodes_z[center_nodes_mask]
        x_center = int(np.floor(self.nnodx_cel / 2)) + np.arange(
            self.ncelx_str, dtype="int"
        ) * (self.nnodx_cel - 1)
        self.force_dofs_torque = np.array(
            [wing_3D.calculate_ribs_dof(self.nnody, self.nnodz, i) for i in x_center]
        ).ravel()
        # fixed dofs elimination
        self.R = np.zeros(self.dofs_list.shape)
        self.lift, self.torque = wing_3D.calculateLoads_conc(
            self.ncely_str, self.ncelz_str, L[0]
        )

        self.R[self.force_dofs_lift] = self.lift
        # self.R[self.force_dofs_torque] = self.torque
        self.R_free = self.R[self.free_dofs]


# Free form
class Free_form:
    """
    Boundary conditions class

    N: Number of members of the ground Structure
    M: Number of nodes of the mesh

    Input:
    - Nodes         (M x 4)         [x,y,z,ID]                                       Matrix of nodes of the GS, ID = 1 upper skin, 0 internal and -1 lower skin node
    - Connectivity  (N x 5)         [1 node, 2 node, # topology, # cell, # bar]     -1 when aperiodic mesh, 2 when repetitive section, 3 allequal
    - Loads         (M_l x 3)       [node, direction (0 x, 1 y, 2 z), magnitude]
    - Boundary      (M_l x 2)       [node, direction (0 x, 1 y, 2 z)]
    - Name                          Name of the load case

    Output:
    - Name of the load case
    - Nodes (M x 4) Matrix of nodes of the GS
    - DOFs (ndofs)
    - groundstructure (N x 5) legacy name, same as Connectivity
    - groundstructure_lenght (N)
    - R (ndofs) right term of the equation Ku = f
    -
    -"""

    def __init__(self, nodes, connectivity, loads, boundary, name, cell_size=False):
        from scipy.spatial import ConvexHull

        self.name = name

        self.is3D = True  # Used for trussfem
        self.isReduced = False
        self.isFreeForm = True

        ## NODES
        self.nodes = nodes
        self.nnod = self.nodes.shape[0]
        # Max bounding box
        self.L = np.zeros(3)
        self.L[0] = np.abs(np.max(nodes[:, 0]) - np.min(nodes[:, 0]))
        self.L[1] = np.abs(np.max(nodes[:, 1]) - np.min(nodes[:, 1]))
        self.L[2] = np.abs(np.max(nodes[:, 2]) - np.min(nodes[:, 2]))
        hull = ConvexHull(nodes[:, :2])
        self.volume = hull.volume

        ## GROUND STRUCTURE
        self.ground_structure = connectivity

        p = np.array(np.where(self.ground_structure[:, 2] == 1)).ravel()
        p_cell = np.array(
            np.where(
                (self.ground_structure[:, 2] == 1) & (self.ground_structure[:, 3] == 0)
            )
        ).ravel()
        a = np.array(np.where(self.ground_structure[:, 2] == -1)).ravel()
        ps = np.array(np.where(self.ground_structure[:, 2] == 2)).ravel()
        ps_cell = np.array(
            np.where(
                (self.ground_structure[:, 2] == 2) & (self.ground_structure[:, 3] == 0)
            )
        ).ravel()
        ae = np.array(np.where(self.ground_structure[:, 2] == 3)).ravel()
        self.ground_structure_cellular = self.ground_structure[p, :].copy()
        self.ground_structure_aperiodic = self.ground_structure[a, :].copy()
        self.ground_structure_periodic_sections = self.ground_structure[ps, :].copy()
        self.ground_structure_all_equal = self.ground_structure[ae, :].copy()
        self.ncel_str = np.max(self.ground_structure_cellular[:, 3], initial=0) + 1

        self.cell_size = cell_size

        # Legacy variables
        self.ground_stucture_list = self.ground_structure[
            p, :2
        ].tolist()  # initialization of the list containing all the beams
        self.ground_stucture_list_cell = self.ground_structure[
            p_cell, :2
        ].tolist()  # ground structure of the first cellule

        # Using 2-norm to calculate the beams length
        self.ground_structure_length = np.linalg.norm(
            (
                self.nodes[self.ground_structure[:, 0], 2]
                - self.nodes[self.ground_structure[:, 1], 2],
                self.nodes[self.ground_structure[:, 0], 1]
                - self.nodes[self.ground_structure[:, 1], 1],
                self.nodes[self.ground_structure[:, 0], 0]
                - self.nodes[self.ground_structure[:, 1], 0],
            ),
            axis=0,
        )
        self.ground_structure_length_cell = self.ground_structure_length[p_cell]
        self.ground_structure_length_section = self.ground_structure_length[ps_cell]

        ## DOFS
        # 0 = x, 1 = y, 2 = z
        self.dofs_list = np.arange(3 * self.nnod, dtype=int)  # Dofs initialization
        self.fixed_dofs = boundary
        self.free_dofs = np.setdiff1d(self.dofs_list, self.fixed_dofs)
        self.dofs = np.ones(self.dofs_list.shape, dtype=int)
        self.dofs[self.fixed_dofs] = 0
        # self.dofs_nodes_connectivity = self.dofs.reshape((-1,3))

        ## FORCES
        self.force_dofs = (
            (loads[:, 0] * 3 + loads[:, 1]).reshape(-1).astype(int)
        )  # This should be a 1D np.array object
        # fixed dofs elimination
        self.R = np.zeros(self.dofs_list.shape)
        self.R[self.force_dofs] = loads[:, 2]
        self.R_free = self.R[self.free_dofs]

        ## OPTIMIZATION PARAMETERS
        self.N = len(self.ground_structure)  # Number of member of the Ground Structure
        self.M = len(self.dofs)  # Number of DOFs of the Groud Structure
        self.N_cellular = len(self.ground_structure_cellular)
        self.N_aperiodic = len(self.ground_structure_aperiodic)
        self.N_periodic_section = len(self.ground_structure_periodic_sections)
        self.N_cell = int(
            np.max(self.ground_structure_cellular[:, 4]) + 1
        )  # number of members per cell
        self.number_cell = int(
            np.max(self.ground_structure_cellular[:, 3]) + 1
        )  # number of members per cell
        try:
            self.N_section = int(
                np.max(self.ground_structure_periodic_sections[:, 4]) + 1
            )  # number of members per section
            self.number_section = int(
                np.max(self.ground_structure_periodic_sections[:, 3]) + 1
            )  # number of sections
        except:
            self.N_section = 0
            self.number_section = 0
        self.N_all_equal = len(
            self.ground_structure_all_equal
        )  # number of members all_equal
        self.N_cell_max = self.N_cell  # Used for reduced problems


class Free_form_multi_load:
    """
    Boundary conditions class

    N: Number of members of the ground Structure
    M: Number of nodes of the mesh

    Input:
    - Nodes         (M x 3)         [x,y,z]                                       Matrix of nodes of the GS
    - Connectivity  (N x 5)         [1 node, 2 node, # topology, # cell, # bar]     -1 when aperiodic mesh, 2 when repetitive section, 3 allequal
    - Loads         (M_l x n_loads) [dof]
    - Boundary      (M_l)           [dof]
    - Name                          Name of the load case
    - Safety factor

    Output:
    - Name of the load case
    - Nodes (M x 3) Matrix of nodes of the GS
    - DOFs (ndofs)
    - groundstructure (N x 5) legacy name, same as Connectivity
    - groundstructure_lenght (N)
    - R (ndofs) right term of the equation Ku = f
    -"""

    def __init__(
        self, nodes, connectivity, loads, boundary, name, cell_size=False, sf=1
    ):
        from scipy.spatial import ConvexHull

        self.name = name
        self.sf = sf

        self.is3D = True  # Used for trussfem
        self.isReduced = False
        self.isFreeForm = True

        ## NODES
        self.nodes = nodes
        self.nnod = self.nodes.shape[0]
        # Max bounding box
        self.L = np.zeros(3)
        self.L[0] = np.abs(np.max(nodes[:, 0]) - np.min(nodes[:, 0]))
        self.L[1] = np.abs(np.max(nodes[:, 1]) - np.min(nodes[:, 1]))
        self.L[2] = np.abs(np.max(nodes[:, 2]) - np.min(nodes[:, 2]))
        hull = ConvexHull(nodes[:, :2])
        self.volume = hull.volume

        ## GROUND STRUCTURE
        self.ground_structure = connectivity

        p = np.array(np.where(self.ground_structure[:, 2] == 1)).ravel()
        p_cell = np.array(
            np.where(
                (self.ground_structure[:, 2] == 1) & (self.ground_structure[:, 3] == 0)
            )
        ).ravel()
        a = np.array(np.where(self.ground_structure[:, 2] == -1)).ravel()
        ps = np.array(np.where(self.ground_structure[:, 2] == 2)).ravel()
        ps_cell = np.array(
            np.where(
                (self.ground_structure[:, 2] == 2) & (self.ground_structure[:, 3] == 0)
            )
        ).ravel()
        ae = np.array(np.where(self.ground_structure[:, 2] == 3)).ravel()
        self.ground_structure_cellular = self.ground_structure[p, :].copy()
        self.ground_structure_aperiodic = self.ground_structure[a, :].copy()
        self.ground_structure_periodic_sections = self.ground_structure[ps, :].copy()
        self.ground_structure_all_equal = self.ground_structure[ae, :].copy()
        self.ncel_str = np.max(self.ground_structure_cellular[:, 3], initial=0) + 1

        self.cell_size = cell_size

        # Legacy variables
        self.ground_stucture_list = self.ground_structure[
            p, :2
        ].tolist()  # initialization of the list containing all the beams
        self.ground_stucture_list_cell = self.ground_structure[
            p_cell, :2
        ].tolist()  # ground structure of the first cellule

        # Using 2-norm to calculate the beams length
        self.ground_structure_length = np.linalg.norm(
            (
                self.nodes[self.ground_structure[:, 0], 2]
                - self.nodes[self.ground_structure[:, 1], 2],
                self.nodes[self.ground_structure[:, 0], 1]
                - self.nodes[self.ground_structure[:, 1], 1],
                self.nodes[self.ground_structure[:, 0], 0]
                - self.nodes[self.ground_structure[:, 1], 0],
            ),
            axis=0,
        )
        self.ground_structure_length_cell = self.ground_structure_length[p_cell]
        self.ground_structure_length_section = self.ground_structure_length[ps_cell]

        ## DOFS
        # 0 = x, 1 = y, 2 = z
        self.dofs_list = np.arange(3 * self.nnod, dtype=int)  # Dofs initialization
        self.fixed_dofs = (
            np.array([boundary * 3, boundary * 3 + 1, boundary * 3 + 2])
            .T.reshape(-1)
            .astype(int)
        )
        self.free_dofs = np.setdiff1d(self.dofs_list, self.fixed_dofs)
        self.dofs = np.ones(self.dofs_list.shape, dtype=int)
        self.dofs[self.fixed_dofs] = 0
        # self.dofs_nodes_connectivity = self.dofs.reshape((-1,3))

        ## FORCES
        self.force_dofs = np.unique(
            np.nonzero(loads)[0]
        )  # This should be a 1D np.array object

        # fixed dofs elimination
        self.R = loads
        self.R_free = self.R[self.free_dofs]

        ## OPTIMIZATION PARAMETERS
        self.N = len(self.ground_structure)  # Number of member of the Ground Structure
        self.M = len(self.dofs)  # Number of DOFs of the Groud Structure
        self.N_cellular = len(self.ground_structure_cellular)
        self.N_aperiodic = len(self.ground_structure_aperiodic)
        self.N_periodic_section = len(self.ground_structure_periodic_sections)
        self.N_cell = int(
            np.max(self.ground_structure_cellular[:, 4], initial=-1) + 1
        )  # number of members per cell
        self.number_cell = int(
            np.max(self.ground_structure_cellular[:, 3], initial=-1) + 1
        )  # number of members per cell
        try:
            self.N_section = int(
                np.max(self.ground_structure_periodic_sections[:, 4], initial=-1) + 1
            )  # number of members per section
            self.number_section = int(
                np.max(self.ground_structure_periodic_sections[:, 3], initial=-1) + 1
            )  # number of sections
        except:
            self.N_section = 0
            self.number_section = 0
        self.N_all_equal = len(
            self.ground_structure_all_equal
        )  # number of members all_equal
        self.N_cell_max = self.N_cell  # Used for reduced problems
