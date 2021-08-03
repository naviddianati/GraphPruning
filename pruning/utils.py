import numpy as np


def prune(G, field='significance', percent=None, num_remove=None):
    """
    Remove all but the top N edges or top x percent of the edges 
    of the graph with respect to an edge attribute.

    @param G: an C{igraph.Graph} instance.
    @param field: the edge attribute to prune with respect to.
    @param percent: percentage of the edges with the highest 
    field value to retain.
    @param num_remove: number of edges to remove. Used only if 
    C{percent} is C{None}.
    """
    if percent:
        deathrow = []
        n = G.ecount()
        threshold_index = int(n - n * percent / 100.)
        threshold_value = sorted(G.es[field])[threshold_index]

        for e in G.es:
            if e[field] < threshold_value:
                deathrow.append(e.index)
        G.delete_edges(deathrow)
    elif num_remove:
        sorted_indices = np.argsort(G.es[field])
        G.delete_edges(sorted_indices[:num_remove])
