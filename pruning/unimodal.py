'''
Created on May 6, 2021

@author: Navid Dianati

Implements the Marginal Likelihood Filter which computes a significance score for each
edge of an integer-weighted graph based on a maximum likelihood null model derived
from the configuration model
'''

import igraph as ig
import pandas as pd
import numpy as np
from statsmodels.stats.proportion import binom_test

import logging

logger = logging.getLogger()
logger.setLevel('DEBUG')

# clip log of binomtest p-values to this value.
MAX_NEG_LOG = np.log(np.finfo(np.float64).max)


class MLF:
    '''
    Under the hood, if graph is not an instance of igraph.Graph, first it will
    be converted to one before the filter is applied. 
         
    '''

    def __init__(self, directed=True):
        self.directed = directed
    
    def fit_transform(self, graph):
        '''
         Receive a representation of the graph and return a similar
         representation, only with a significance score calculated
         for each edge. 
         
         Graph is one of the following:
         - A list of tuples: (node_id1, node_id_2, weight) where node_id1 and node_id2
         can be integers or strings and weight is integer. The output list will consist
         of 4-tuples and the 4th tuple is the significance score.
         - an igraph.Graph instance where each edge has an integer attribute "weight".
         The output will be a Graph instance with an additional edge attribute
         "significance"
         - a pandas.DataFrame with three columns: "source", "target", "weight" where
         "weight" values are positive integers. The output will have an additional
         column "significance".
         Graph must be simple: no loops and no multiple edges
         '''
        self._check_types(graph)
        dtype = type(graph)
        
        # Convert to igraph.Graph
        g = self._convert_to_graph(graph, directed=self.directed)
        
        if g.is_directed():
            g = self._compute_significance_directed(g)
        else:
            g = self._compute_significance_undirected(g)
        
        output = self._cast(g, dtype)
        return output
    
    def _cast(self, graph, dtype):
        '''
         Convert the igraph.Graph instance back to the 
         original format of the input. 
         '''
        if dtype == ig.Graph:
            return graph
        elif dtype == pd.DataFrame:
            # graph was constructed from a dataframe and
            # therefore each vertex has a "name" attribute 
            edgelist = [(graph.vs[e.source]['name'], graph.vs[e.target]['name'], e['weight'], e['significance']) for e in graph.es]
            df = pd.DataFrame(edgelist, columns=['source', "target", "weight", "significance"])
            return df
        elif dtype == list:
            # graph was constructed from a list of tuples
            # and therefore each vertex has a "name" attribute
            edgelist = [(graph.vs[e.source]['name'], graph.vs[e.target]['name'], e['weight'], e['significance']) for e in graph.es]
            return edgelist
        else:
            raise(TypeError('Can only recast the graph into one of: igraph.Graph, a DataFrame or a list of 3-tuples'))
    
    def _compute_significance_directed(self, G):
        """
        Compute the edge significance for the edges of the
        given graph C{G} in place. C{G.es['weight']} is expected 
        to have been set already.
    
        @param G: C{igraph.Graph} instance. C{G} is assumed to be directed.
        """
        ks = G.strength(weights='weight')
        total_degree = sum(ks)
    
        for e in G.es:
            i0, i1 = e.source, e.target
            try:
                p = _pvalue_directed(
                    w=e['weight'], ku=ks[i0],
                    kv=ks[i1], q=total_degree / 2.0,
                    )
                # Due to instabilities in binomtest at p-values
                # near "tiny", we clip all significance values
                # to MAX_NEG_LOG
                e['significance'] = min(
                    MAX_NEG_LOG, MAX_NEG_LOG if p <= 0 else -np.log(p)
                    )
            except ValueError as error:
                logger.warning("warning: ValueError {}".format(str(error)))
                logger.debug("ValueError weight: {} ks[i0]:{} ks[i1]:{} total_degree:{} p:{}".format(e['weight'], ks[i0], ks[i1], total_degree, p))
                e['significance'] = None
            except Exception  as error:
                logger.warning("warning: Exception {}".format(str(error)))
                e['significance'] = None
                # print "error computing significance", p
    
        max_sig = max([s for s in G.es['significance'] if s is not None])
        for e in G.es:
            if e['significance'] is None:
                e['significance'] = max_sig
        
        return G

    def _compute_significance_undirected(self, G):
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
            try:
                p = _pvalue_undirected(
                    w=e['weight'], ku=ks[i0],
                    kv=ks[i1], q=total_degree / 2.0,
                    )
                
                # Due to instabilities in binomtest at p-values
                # near "tiny", we clip all significance values
                # to MAX_NEG_LOG
                e['significance'] = min(
                    MAX_NEG_LOG, MAX_NEG_LOG if p <= 0 else -np.log(p)
                    )
            except ValueError as error:
                logger.warning("warning: ValueError {}".format(str(error)))
                logger.debug("ValueError weight: {} ks[i0]:{} ks[i1]:{} total_degree:{} p:{}".format(e['weight'], ks[i0], ks[i1], total_degree, p))
                e['significance'] = None
            except Exception  as error:
                logger.warning("warning: Exception {}".format(str(error)))
                e['significance'] = None
    
        max_sig = max([s for s in G.es['significance'] if s is not None])
        for e in G.es:
            if e['significance'] is None:
                e['significance'] = max_sig
        return G

    def _convert_to_graph(self, graph, directed=False):
        '''
         If graph is not an instance of igraph.Graph, then convert
         it to one.
         '''
        if isinstance(graph, ig.Graph):
            if not graph.is_simple():
                raise ValueError('Expected a simple graph: no loops and no multiple edges')
            if (graph.is_directed() and not directed):
                raise ValueError('Provided graph is directed but the "directed" parameter is False')
            if (not graph.is_directed() and directed):
                raise ValueError('Provided graph is undirected but the "directed" parameter is True')
            return graph
        
        if isinstance(graph, list):
            g = ig.Graph.TupleList(graph, directed=directed, weights=True)
            if not g.is_simple():
                raise ValueError('Expected a simple graph: no loops and no multiple edges')
            return g
        elif isinstance(graph, pd.DataFrame):
            g = ig.Graph.TupleList(
                graph[['source', 'target', 'weight']].itertuples(index=False),
                directed=directed,
                weights=True)
            if not g.is_simple():
                raise ValueError('Expected a simple graph: no loops and no multiple edges')
            return g
        else:
            raise TypeError(
                'graph must be an instance of one of the following: igraph.Graph, list, pandas.DataFrame.'
                )
    
    def _check_types(self, graph):
        if type(graph) == pd.DataFrame:
            for column in ["source", "target", "weight"]:
                if column not in graph.columns:
                    raise ValueError('{} must be in the DataFrame\'s columns'.format(column))
            if graph["weight"].min() < 0:
                raise ValueError("All weights must be non-negative")
            
        elif type(graph) == list:
            if set([type(x) for x in graph]) != {tuple}:
                raise TypeError("Each element of the edgelist must be a tuple")
        
            if set([len(x) for x in graph]) != {3}:
                raise ValueError("Each element of the edgelist must have length 3")
            
            if set([type(x[2]) for x in graph]) != {int}:
                raise TypeError("The 3rd element of each edge tuple must be an int")
            
            if min([x[2] for x in graph]) < 0:
                raise ValueError("All weights must be non-negative")
        
        elif type(graph) == ig.Graph:
            if "weight" not in graph.es.attributes():
                raise ValueError("The EdgeSequence must have a 'weight' attribute")
            types = set([type(e['weight']) for e in graph.es])
            if types not in  [{int}, {int, np.int64}, {np.int64}]:
                raise TypeError('The weight attribute of the edges must be an int')
            if min([e['weight'] for e in graph.es]) < 0:
                raise ValueError('All weights must be non-negative')
            
        else:
            raise(ValueError('The graph must be given as a, igraph.Graph, a DataFrame or a list of 3-tuples'))


def _pvalue_undirected(w, ku, kv, q):
    """
    Compute the pvalue for the undirected edge null model.
    Use a standard binomial test from the statsmodels package.

    @param w: weight of the undirected edge.
    @param ku: total incident weight (strength) of the first vertex.
    @param kv: total incident weight (strength) of the second vertex.
    @keyparamword q: total incident weight of all vertices divided by two. Similar to the total number of edges in the graph.
    """
    if not all(v is not None for v in [w, ku, kv, q]):
        raise ValueError

    p = ku * kv * 1.0 / q / q / 2.0
    return binom_test(count=w, nobs=q, prop=p, alternative="larger")


def _pvalue_directed(w_uv, ku_out, kv_in, q):
    """
    Compute the pvalue for the directed edge null model.
    Use a standard binomial test from the statsmodels package

    @param w_uv: Weight of the directe edge.
    @param ku_out: Total outgoing weight of the source vertex.
    @param kv_in: Total incoming weight of the destination vertex.
    @param q: Total sum of all edge weights in the graph.
    """
    if not all(v is not None for v in [w_uv, ku_out, kv_in, q]):
        raise ValueError
    
    p = 1.0 * ku_out * kv_in / q / q / 1.0
    return binom_test(count=w_uv, nobs=q, prop=p, alternative="larger")
