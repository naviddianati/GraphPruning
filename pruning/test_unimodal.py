'''
Created on May 6, 2021

@author: navid
'''
import unittest
import igraph as ig
import numpy as np
import pandas as pd
from . import unimodal


class Test_MLF(unittest.TestCase):
    
    def test_convert_to_graph(self):
        
        # Must raise ValueError if graph not simple
        edgelist1 = [(0, 1, 1), (0, 1, 2)]
        edgelist2 = [(0, 1, 1), (1, 1, 2)]
        edgelist3 = pd.DataFrame(edgelist1, columns=['source', 'target', 'weight'])
        g = ig.Graph.TupleList(edgelist1, directed=False, weights=True)
        mlf = unimodal.MLF()
        with self.assertRaises(ValueError):
            mlf._convert_to_graph(edgelist1)
        with self.assertRaises(ValueError):
            mlf._convert_to_graph(edgelist2)
        with self.assertRaises(ValueError):
            mlf._convert_to_graph(edgelist3)
        with self.assertRaises(ValueError):
            mlf._convert_to_graph(g)
        
        g = ig.Graph.Barabasi(100, 3)
        g.es['weight'] = np.random.randint(0, 20, g.ecount())
        
        # For Graph result should be identity 
        mlf = unimodal.MLF()
        G = mlf._convert_to_graph(g)
        self.assertEqual(g, G)

        mlf = unimodal.MLF()
        df = g.get_edge_dataframe()
        G = mlf._convert_to_graph(df)
        
        df = g.get_edge_dataframe()
        edgelist = [tuple(x) for x in df.values]
        
        G = mlf._convert_to_graph(df)
        assert_df_equal_graph(self, df, G, edge_attribute="weight")
    
        G = mlf._convert_to_graph(edgelist)
        assert_edgelist_equal_graph(self, edgelist, G, edge_attribute="weight", edge_attribute_index=2)

    def test_fit_transform(self):
        N = 1000
        g = ig.Graph.Barabasi(N, 3)
        g.es['weight'] = np.random.randint(0, 200, g.ecount())
        # Add a "name" attribute to the vs
        g.vs['name'] = [v.index for v in g.vs]
        
        df = g.get_edge_dataframe()
        edgelist = [tuple([int(x) for x in tuple(x)]) for x in df.values]
        
        mlf = unimodal.MLF(directed=False)
        g_tr = mlf.fit_transform(g)

        mlf = unimodal.MLF(directed=False)
        df_tr = mlf.fit_transform(df)
        
        mlf = unimodal.MLF(directed=False)
        edgelist_tr = mlf.fit_transform(edgelist)
        
        assert_df_equal_graph(self, df_tr, g_tr, edge_attribute="significance")
        assert_edgelist_equal_graph(self, edgelist_tr, g_tr, edge_attribute="significance", edge_attribute_index=3)
        # The 3 results must be equivalent

    def test_check_types(self):
        
        g = ig.Graph.Barabasi(100, 3)
        mlf = unimodal.MLF()
        with self.assertRaises(ValueError):
            mlf._check_types(g)
        
        g.es['weight'] = 'adsf'
        with self.assertRaises(TypeError):
            mlf._check_types(g)
            
        g.es['weight'] = range(g.ecount())
        try:
            mlf._check_types(g)
        except Exception as e:
            self.fail('Failed on valid input with message: {}'.format(str(e)))
            
        # when g is a list
        edgelist = [(edge[0], edge[1], np.random.randint(0, 20)) for edge in g.get_edgelist()]
        try:
            mlf._check_types(edgelist)
        except Exception as e:
            self.fail('Failed on valid input with message: {}'.format(str(e)))
            
        # edgelist tuples don't all have length 3
        edgelist = [(edge[0], edge[1]) for edge in g.get_edgelist()]
        with self.assertRaises(ValueError):
            mlf._check_types(edgelist)
        
        edgelist = [(edge[0], edge[1]) for edge in g.get_edgelist()]
        edgelist[-1] = (1, 2, 3)
        with self.assertRaises(ValueError):
            mlf._check_types(edgelist)
            
        # Not all edgelist elements are tuples
        edgelist = [(edge[0], edge[1]) for edge in g.get_edgelist()]
        edgelist[-1] = "text"
        with self.assertRaises(TypeError):
            mlf._check_types(edgelist)
        
        # DataFrame
        g = ig.Graph.Barabasi(100, 3)
        edgelist = g.get_edge_dataframe()
        with self.assertRaises(ValueError):
            mlf._check_types(edgelist)
        
        edgelist['weight'] = np.random.randint(0, 20, (g.ecount(),))
        try:
            mlf._check_types(edgelist)
        except Exception as e:
            self.fail('Failed on valid input with message: {}'.format(str(e)))
        
        self.assertIn
#         
#         edgelist1 = g.get_edge_dataframe()
#         edgelist1['weight'] = np.random.randint(0, 20, (g.ecount(),))
#         edgelist1

    def test_equivalent_results(self):
        
        g = ig.Graph.Barabasi(1000, 3)
        g.es['weight'] = np.random.randint(0, 200, g.ecount())
        df = g.get_edge_dataframe()
        
        mlf = unimodal.MLF(directed=False)
        G = mlf.fit_transform(g)
        df_edgelist_1 = G.get_edge_dataframe()
        
        mlf = unimodal.MLF(directed=False)
        df_edgelist_2 = mlf.fit_transform(df)
        assert_df_edgelists_equal(self, df_edgelist_1, df_edgelist_2)


def assert_df_equal_graph(testcase, df, graph, edge_attribute):
    '''
    Using a testcase instance, assert that a DataFrame and a Graph
    are equivalent representations of the same graph with respect
    to a given edge attribute 
    '''
    G = graph
    testcase.assertIn(edge_attribute, df.columns)
    testcase.assertIn(edge_attribute, G.es.attributes())
    testcase.assertEqual(df.shape[0], G.ecount())
    testcase.assertEqual(
        len(set(df['source'].to_list() + df['target'].to_list())),
        G.vcount()
        )
    for i in range(G.ecount()):
        e = G.es[i]
        
        name1, name2 = G.vs[e.source]['name'], G.vs[e.target]['name']
        res = df[(df['source'] == name1) & (df['target'] == name2)]
        testcase.assertEqual(len(res), 1)
        testcase.assertIn(edge_attribute, res.columns)
        attr = res[edge_attribute].values[0]
        testcase.assertEqual(attr, e[edge_attribute])


def assert_edgelist_equal_graph(testcase, edgelist, graph, edge_attribute, edge_attribute_index):
    '''
    Using a testcase instance, assert that an edgelist and a Graph
    are equivalent representations of the same graph with respect
    to a given edge attribute .
    '''
    G = graph
    testcase.assertIn(edge_attribute, G.es.attributes())
    testcase.assertEqual(len(set([len(x) for x in edgelist])), 1)
    for i in range(G.ecount()):
        e = G.es[i]
        name1, name2 = G.vs[e.source]['name'], G.vs[e.target]['name']
        res = [rec for rec in edgelist if (rec[0] == name1) and (rec[1] == name2)]
        res += [rec for rec in edgelist if (rec[0] == name2) and (rec[1] == name1)]
        testcase.assertEqual(len(res), 1)
        attr = res[0][edge_attribute_index]
        testcase.assertEqual(attr, e[edge_attribute])


def assert_df_edgelists_equal(testcase, df1, df2, columns=['source', 'target', 'weight', 'significance']):
    ''' Assert that two dataframes represent the same
    edgelist exactly. '''
    testcase.assertEqual(df1.shape, df2.shape)
    for c in columns:
        testcase.assertIn(c, df1.columns)
        testcase.assertIn(c, df2.columns)
    
    # build the reference weight 
    # and significance dictionaries
    w = {}
    sig = {}
    for x in df1[columns].values:
        edge = (int(x[0]), int(x[1]))
        w[edge] = x[2]
        sig[edge] = x[3]
    
    for x in df2[columns].values:
        edge = (int(x[0]), int(x[1]))
        testcase.assertEqual(w[edge], x[2])
        testcase.assertEqual(sig[edge], x[3])


if __name__ == "__main__":
    unittest.main()
