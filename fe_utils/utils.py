import numpy as np
import matplotlib.pyplot as plt
from math import comb
from collections import defaultdict


def _generate_sum_exactly_k(n_vars: int, k: int):
    if n_vars < 1:
        raise Exception("cannot sum less than 0 elements")
    elif n_vars == 1:
        yield (k,)
    else:
        for i in range(k + 1):
            for alpha in _generate_sum_exactly_k(n_vars - 1, k - i):
                yield alpha + (i,)


def _generate_sum_upto_k(n_vars: int, k: int):
    if n_vars < 1:
        raise Exception("cannot sum less than 0 elements")
    for cur_sum in range(k + 1):
        for alpha in _generate_sum_exactly_k(n_vars, cur_sum):
            yield alpha


def create_adj_list_and_neighbor_dct(faces):
    adj_list = defaultdict(list)
    nbr_dct = defaultdict(set)
    for u, v, w in faces:
        if (v, w) not in adj_list[u] and (w, v) not in adj_list[u]:
            adj_list[u].append((v, w))
        nbr_dct[u].add(v)
        nbr_dct[u].add(w)
        if (w, u) not in adj_list[v] and (u, w) not in adj_list[v]:
            adj_list[v].append((w, u))
        nbr_dct[v].add(w)
        nbr_dct[v].add(u)
        if (u, v) not in adj_list[w] and (v, u) not in adj_list[w]:
            adj_list[w].append((u, v))
        nbr_dct[w].add(u)
        nbr_dct[w].add(v)
    return adj_list, nbr_dct


def min_points_for_poly(dim: int, degree: int):
    return comb(dim + degree, degree)


def polynomial_vct(degree: int, coord: np.array,
                   grad=False):
    # NOTE: Check if same as np.polynomial vandermonde func
    dim = coord.shape[0]
    if grad:
        polynomial_terms = np.ones(
            (min_points_for_poly(dim, degree), dim), dtype=np.double)
        for j, alpha in enumerate(_generate_sum_upto_k(dim, degree)):
            for i, alpha_i in enumerate(alpha):
                partial_powers = np.array(alpha)
                partial_powers[i] = max(partial_powers[i] - 1, 0)

                partial_factors = np.ones_like(partial_powers)
                partial_factors[i] = alpha_i

                polynomial_terms[j, i] *= np.prod(
                    partial_factors
                    * (coord ** partial_powers),
                )
    else:
        polynomial_terms = np.fromiter(
            (np.prod(coord ** alpha)
             for alpha in _generate_sum_upto_k(dim, degree)),
            dtype=np.double
        )

    return polynomial_terms


def polynomial_eval(degree: int, coeff: np.array, coord: np.array,
                    grad=False):
    return polynomial_vct(degree, coord, grad).T @ coeff


# NOTE: Implement this function
def __experimental_find_containing_entity_from_coords(coords, nodes,
                                                      adj_list):
    dists = np.linalg.norm(coords[:, np.newaxis] - nodes, axis=2)
    print(dists)
    closest_indices = np.argmin(dists, axis=1)
    print(closest_indices)

    closest_nodes = nodes[closest_indices]
    raise NotImplementedError
    return closest_nodes


def find_containing_entity(coord: np.array, nodes: np.array,
                           adj_list: dict):
    dist = np.linalg.norm(nodes - coord, axis=1)
    closest_idx = np.argmin(dist)
    if dist[closest_idx] == 0:
        # the entity containing coord is a vertex/node
        return (closest_idx,)

    closest_node = nodes[closest_idx]
    for i, (nbr1, nbr2) in enumerate(adj_list[closest_idx]):
        nbr1_vct = nodes[nbr1] - closest_node
        nbr2_vct = nodes[nbr2] - closest_node
        coefs = np.linalg.inv(
            np.column_stack([nbr1_vct, nbr2_vct])
        ) @ (coord - closest_node)

        if np.all(coefs >= 0):
            return closest_idx, nbr1, nbr2
    raise Exception("Containing entity not found")
