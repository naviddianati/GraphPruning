"""
This module implements functions for filtering edges of a graph
based on the significance.
"""

from statsmodels.stats.proportion import binom_test
import numpy as np
from math import log
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger()

def pvalue(mode="undirected", **params):
    """
    Compute the p-value of a given edge according the MLF significance
    filter. 

    @param mode: can be C{"directed"} or C{"undirected"}.
    @kwarg w: integer weight of the edge.
    @kwarg ku: weighted degree of one end node.
    @kwarg kv: weighted degree of the other end node.
    @kwarg q: sum of all weighted degrees in graph dividd by 2.

    Other parameters are different for the B{directed} and B{undirected} cases.
    See L{__pvalue_directed} and L{__pvalue_undirected} for detailed description of parameters.
    """

    if mode == "undirected":
        return __pvalue_undirected(**params)
    elif mode == "directed":
        return __pvalue_directed(**params)
    else:
        raise ValueError("mode must be either 'directed' or 'undirected'.")


def __pvalue_undirected(**params):
    """
    Compute the pvalue for the undirected edge null model.
    Use a standard binomial test from the statsmodels package.

    @keyword w: weight of the undirected edge.
    @keyword ku: total incident weight (strength) of the first vertex.
    @keyword kv: total incident weight (strength) of the second vertex.
    @keyword q: total incident weight of all vertices divided by two. Similar to the total number of edges in the graph.
    """
    w = params.get("w")
    ku = params.get("ku")
    kv = params.get("kv")
    q = params.get("q")

    if not (w and ku and kv and q):
        raise ValueError

    p = ku * kv * 1.0 / q / q / 2.0
    return binom_test(count=w, nobs=q, prop=p, alternative="larger")


def __pvalue_directed(**params):
    """
    Compute the pvalue for the directed edge null model.
    Use a standard binomial test from the statsmodels package

    @keyword w_uv: Weight of the directe edge.
    @keyword ku_out: Total outgoing weight of the source vertex.
    @keyword kv_in: Total incoming weight of the destination vertex.
    @keyword q: Total sum of all edge weights in the graph.
    """

    w_uv = params.get("w_uv")
    ku_out = params.get("ku_out")
    kv_in = params.get("kv_in")
    q = params.get("q")

    p = 1.0 * ku_out * kv_in / q / q / 1.0
    return binom_test(count=w_uv, nobs=q, prop=p, alternative="larger")


def prune(G, field='significance', percent=None, num_remove=None):
    """
    Remove all but the top x percent of the edges of the graph
    with respect to an edge attribute.

    @param G: an C{igraph.Graph} instance.
    @param field: the edge attribute to prune with respect to.
    @param percent: percentage of the edges with the highest field value to retain.
    @param num_remove: number of edges to remove. Used only if C{percent} is C{None}.
    """
    if percent:
        deathrow = []
        n = len(G.es)
        threshold_index = n - n * percent / 100
        threshold_value = sorted(G.es[field])[threshold_index]

        for e in G.es:
            if e[field] < threshold_value:
                deathrow.append(e.index)
        G.delete_edges(deathrow)
    elif num_remove:
        sorted_indices = np.argsort(G.es[field])
        G.delete_edges(sorted_indices[:num_remove])
    return G

def compute_significance(G):
    '''
    Compute the significance for each edge of a weighted
    graph according to the U{Marginal Likelihood Filter (MLF) <http://www.naviddianati.com/research>}.
    @param G: C{igraph.Graph} instance. Edges must have the
    C{'weight'} attribute set. C{G} can be directed or undirected.
    Each case is treated separately. For each edge a new
    C{"significance"} attribute will be set.
    '''
    if G.is_directed():
        __compute_significance_directed(G)
    else:
        __compute_significance_undirected(G)

def __compute_significance_directed(G):
    """
    Compute the edge significance for the edges of the
    given graph C{G} in place. C{G.es['weight']} is expected 
    to have been set already.

    TODO: implement the directed case as well.
    @param G: C{igraph.Graph} instance. C{G} is assumed to be directed.
    """
    ks = G.strength(weights='weight')
    total_degree = sum(ks)

    for e in G.es:
        i0, i1 = e.source, e.target
        v0, v1 = G.vs[i0], G.vs[i1]
        try:
            p = pvalue(w=e['weight'], ku=ks[
                       i0], kv=ks[i1], q=total_degree / 2.0)
            e['significance'] = -log(p)
        except ValueError as error:
            logger.debug("warning: ValueError {}".format(str(error)))
            logger.debug("ValueError weight: {} ks[i0]:{} ks[i1]:{} total_degree:{} p:{}".format(e['weight'], ks[i0], ks[i1], total_degree, p))
            e['significance'] = None
        except Exception  as error:
            logger.debug("warning: Exception {}".format(str(error)))
            e['significance'] = None
            # print "error computing significance", p

    max_sig = max([s for s in G.es['significance'] if s is not None])
    for e in G.es:
        if e['significance'] is None:
            e['significance'] = max_sig




def __compute_significance_undirected(G):
    """
    Compute the edge significance for the edges of the
    given graph C{G} in place. C{G.es['weight']} is expected
    to have been set already.
    @param G: C{igraph.Graph} instance. C{G} is assumed to be undirected.
    """
    ks = G.strength(weights='weight')
    total_degree = sum(ks)

    for e in G.es:
        i0, i1 = e.source, e.target
        v0, v1 = G.vs[i0], G.vs[i1]
        try:
            p = pvalue(w=e['weight'], ku=ks[
                       i0], kv=ks[i1], q=total_degree / 2.0)
            e['significance'] = -log(p)
        except ValueError as error:
            logger.debug("warning: ValueError {}".format(str(error)))
            logger.debug("ValueError weight: {} ks[i0]:{} ks[i1]:{} total_degree:{} p:{}".format(e['weight'], ks[i0], ks[i1], total_degree, p))
            e['significance'] = None
        except Exception  as error:
            logger.debug("warning: Exception {}".format(str(error)))
            e['significance'] = None


    max_sig = max([s for s in G.es['significance'] if s is not None])
    for e in G.es:
        if e['significance'] is None:
            e['significance'] = max_sig




if __name__ == "__main__":
    pass
