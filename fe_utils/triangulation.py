import numpy as np
from scipy.spatial import ConvexHull
from basis import LagrangeBasis
from utils import create_adj_list_and_neighbor_dct, \
    polynomial_vct, polynomial_eval, min_points_for_poly


class FunctionalEstimateOnTriangulation:
    def __init__(self, points, func_vals, faces,
                 degree=1,
                 grad_est_at_nodes=None,
                 hess_est_at_nodes=None,
                 ):
        self.points = points
        self.faces = faces
        self.func_vals = func_vals
        self.dim = points.shape[1]
        self.degree = degree
        self.adj_list, self.neighbors = \
            create_adj_list_and_neighbor_dct(faces)

        self._convex_hull = ConvexHull(self.points)
        self.grad_est_at_nodes = grad_est_at_nodes
        self.hess_est_at_nodes = hess_est_at_nodes
        self.basis = LagrangeBasis(self.points, self.adj_list, degree=degree)

    def isin_hull(self, coords, tol=np.double(1e-12)):
        return np.all(self._convex_hull.
        raise NotImplementedError

    def get_neighbors_by_levels(self, idx, level=1):
        cur_level_idx, tmp={idx}, set()
        res=set()
        while level > 0:
            while cur_level_idx:
                cur_idx=cur_level_idx.pop()
                for nbr in self.neighbors[cur_idx]:
                    res.add(nbr)
                    tmp.add(nbr)
            level -= 1
            cur_level_idx=tmp.copy()
        return res

    def calc_pointwise_grad_est_from_func(self, idx, f_vals):
        min_req_pts=min_points_for_poly(self.dim, self.degree + 1)
        level=1
        nbr_indices=self.get_neighbors_by_levels(idx, level)
        while len(nbr_indices) < 16:
            level += 1
            nbr_indices=self.get_neighbors_by_levels(idx, level)
        A=np.empty((min_req_pts, len(nbr_indices))).T
        b=np.empty(len(nbr_indices))
        for i, nbr_i in enumerate(nbr_indices):
            nbr_node=self.points[nbr_i]
            A[i]=polynomial_vct(self.degree + 1, nbr_node)
            b[i]=f_vals[nbr_i]
        coefs=np.linalg.pinv(A) @ b
        return polynomial_eval(self.degree + 1, coefs, self.points[idx],
                               grad=True)

    def calculate_pointwise_grad_est(self, idx):
        return self.calc_pointwise_grad_est_from_func(idx, self.func_vals)

    def calc_entire_grad_est_from_func(self, f_vals):
        grads=np.empty((len(self.points), self.dim),
                         dtype=np.double
                         )
        for i in range(len(self.points)):
            grads[i]=self.calc_pointwise_grad_est_from_func(
                i, self.func_vals)
        return grads

    def calculate_entire_grad_est(self, return_val=False):
        self.grad_est_at_nodes=self.calc_entire_grad_est_from_func(
            self.func_vals)
        return self.grad_est_at_nodes.copy() if return_val else None

    def calc_ppr(self, grad_est, coords):
        basis_vals=self.basis.eval_at(coords)
        return basis_vals @ grad_est

    def calculate_grad_ppr(self, coords):
        if self.grad_est_at_nodes is None:
            self.calculate_entire_grad_est()
        return self.calc_ppr(self.grad_est_at_nodes, coords)

    def calculate_entire_hess_est(self, return_val=False):
        if self.grad_est_at_nodes is None:
            self.calculate_entire_grad_est()
        grad_x, grad_y=self.grad_est_at_nodes.T
        hess_x=np.empty((len(self.points), self.dim),
                          dtype=np.double
                          )
        hess_y=np.empty((len(self.points), self.dim),
                          dtype=np.double
                          )

        for i in range(len(self.points)):
            hess_x[i]=self.calc_pointwise_grad_est_from_func(i, grad_x)
            hess_y[i]=self.calc_pointwise_grad_est_from_func(i, grad_y)

        self.hess_est_at_nodes=np.stack([hess_x, hess_y], axis=1)
        return self.hess_est_at_nodes.copy() if return_val else None
