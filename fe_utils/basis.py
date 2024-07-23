import numpy as np
from utils import find_containing_entity


class TriangulationBasisBase:
    def __init__(self, nodes, adj_list, degree):
        self.nodes = nodes
        self.degree = degree
        self.adj_list = adj_list

    def eval_at(self, points):
        raise NotImplementedError("Base class method not implemented")


class LagrangeBasis(TriangulationBasisBase):
    def __init__(self, nodes, adj_list, degree: int = 1):
        super.__init__(self, nodes, adj_list, degree)

    def eval_at(self, points):
        res = np.zeros((points.shape[0], self.nodes.shape[0]),
                       dtype=np.double)
        for i, coord in enumerate(points):
            vertices = find_containing_entity(coord, self.nodes,
                                              self.adj_list)
            cur_basis_vals = res[i]
            if len(vertices) == 1:
                cur_basis_vals[vertices[0]] = 1
            elif len(vertices) == 2:
                coord_vct = coord - self.nodes[vertices[0]]
                edge_vct = self.nodes[vertices[1]]
                - self.nodes[vertices[0]]
                scale_factor = coord_vct / edge_vct
                assert (np.allclose(
                    scale_factor,
                    np.ones_like(scale_factor)
                        * np.average(scale_factor)
                        ))
                scale_factor = np.average(scale_factor)
                cur_basis_vals[vertices[0]] = 1 - scale_factor
                cur_basis_vals[vertices[1]] = scale_factor
            else:
                for i, vert_i in enumerate(vertices):
                    coord_vct = coord - self.nodes[vert_i]
                    edge_1_vct = self.nodes[
                        vertices[(i + 1) % len(vertices)]]
                    - self.nodes[vert_i]
                    edge_2_vct = self.nodes[
                        vertices[(i + 2) % len(vertices)]]
                    - self.nodes[vert_i]
                    # find a, b such that a*u + b*v = c, here u,v are edges
                    # c is coord_vct
                    coefs = np.linalg.inv(
                        np.column_stack([edge_1_vct, edge_2_vct])) \
                        @ coord_vct

                    cur_basis_vals[vert_i] = 1 - np.sum(coefs)

        return res
