import cvxpy as cp
import numpy as np
import networkx as nx
import warnings

def graph_union(maxcut_graphs, unit_weights=True, c=None):
    """
        Returns a union graph g from `maxcut_graphs`, i.e. g contains all edges that are `maxcut_graphs`.
        If an edge occurs in multiple graphs, the individual edge weights are summed up.
        `unit_weights`: all weights in the union graph are 1
        `c`: multiply a symbolic parameter c[i] to weights of graph with index i in `maxcut_graphs`
    """
    edge_union = set().union(*[g.edges() for g in maxcut_graphs])
    node_union = list(range(len(set().union(*[g.nodes() for g in maxcut_graphs]))))
    g = nx.Graph()
    g.add_nodes_from(node_union)
    g.add_edges_from(edge_union)
    for u, v in g.edges:
        if unit_weights:
            g[u][v]['weight'] = 1.0
        else:
            if c is None:
                g[u][v]['weight'] = sum([gi[u][v]['weight'] for gi in maxcut_graphs if gi.has_edge(u, v)])
            else:
                g[u][v]['weight'] = sum([c[i]*gi[u][v]['weight'] for i, gi in enumerate(maxcut_graphs) if gi.has_edge(u, v)])
    return g


def _aggregate(alpha, measurements):
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1] but was {alpha}")

    # sort by values
    sorted_measurements = sorted(measurements, key=lambda x: x[1])
    accumulated_percent = 0.0  # once alpha is reached, stop
    cvar = 0.0
    for probability, value in sorted_measurements:
        cvar += value * min(probability, alpha - accumulated_percent)
        accumulated_percent += probability
        if accumulated_percent >= alpha:
            break

    return cvar / alpha


def relax_psd(g, objective='max'):
    """
        Determine an approximative max-cut solution for the graph `g`.
        Note that we generate and solve a positive semidefinite program here that always upper/lower
        bounds the true max-cut solution.
    """
    nnodes = len(g.nodes())
    X = cp.Variable((nnodes, nnodes), symmetric=True)
    constraints = [X >> 0]
    constraints += [X[i, i] == 1 for i in range(nnodes)]
    #print([f"{e[2]['weight']}; {e[0]}{e[1]}" for e in g.edges(data=True)])    
    fn_obj = 0.5*sum(e[2]['weight']*(1-X[e[0], e[1]]) for e in g.edges(data=True))
    if objective == 'max':
        prob = cp.Problem(cp.Maximize(fn_obj), constraints)
    elif objective == 'min':
        prob = cp.Problem(cp.Minimize(fn_obj), constraints)
    else:
        print("unknown objective function")
    ub = prob.solve()
    return ub


def get_bounds(maxcut_graphs):
    """
        Determine lower and upper bounds of the max-cut value in each graph of `maxcut_graphs`
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u_bounds = [relax_psd(g) for g in maxcut_graphs]
        u_bounds = np.ceil(u_bounds)

        l_bounds = [relax_psd(g, 'min') for g in maxcut_graphs]
        l_bounds = np.floor(l_bounds)
        return l_bounds, u_bounds



from qmoo_utils import graph_union, _aggregate
from qopt_best_practices.cost_function.cost_utils import counts_to_maxcut_cost 
from qiskit_aer.primitives import SamplerV2 as SamplerAerV2
from scipy.optimize import minimize as scipy_min

# TODO make the params function constant/static when fixing the input graphs
def train_qaoa_ansatz_params(qaoa_qc, maxcut_graphs, bond_dimension=20):
    n_obj = len(maxcut_graphs)
    # Construct the graph union g of `moo_graphs`, i.e. if an edge occurs
    # in any graph of `moo_graphs`, add the edge with unit weight to `g`` 
    g = graph_union(maxcut_graphs)
    cs = {f"c[{i}]": 1 for i in range(n_obj)}
    qc = qaoa_qc.assign_parameters(cs, inplace=False)

    # variational optimization
    def _min_func(params, ansatz, g, sampler, aggregation):
        job = sampler.run([(ansatz, params)])
        sampler_result = job.result()
        counts = sampler_result[0].data.meas.get_counts()
        shots = sum(counts.values())
        # a dictionary containing: {state: (measurement probability, value)}
        evaluated = {
            bitstr: (freq/shots, -next(iter(counts_to_maxcut_cost(g, {bitstr: 1}).keys())))
            for bitstr, freq in counts.items()
        }
        result = _aggregate(aggregation, evaluated.values())
        return result
    
    # parameter initialization according to https://arxiv.org/pdf/2101.05742
    dt = 0.75
    p = len(qc.parameters) // 2
    grid = np.arange(1, p + 1) - 0.5
    init_params = np.concatenate((1 - grid * dt / p, grid * dt / p))

    sampler = SamplerAerV2(options={'backend_options':{'method': 'matrix_product_state','matrix_product_state_max_bond_dimension':bond_dimension}})

    # use scipy's COBYLA method to find optimal parameters \beta, \gamma
    result = scipy_min(
        _min_func,
        init_params,
        args=(qc, g, sampler, 0.5),
        method="COBYLA",
    )
    return {str(p): result.x[i] for i, p in enumerate(qc.parameters)}
