'''
This packaged contains a Python implementation of the U{Marginal Likelihood Filter<http://arxiv.org/abs/1503.04085>} for 
weighted networks (graphs). This filter works by computing a statistical significance 
measure for each weighted edge of the graph and thresholding based on this measure. 
This significance measure can be compouted for directed or undirected weighted graphs.
If you use this filter in your research, please scite the following paper:

Navid Dianati, I{Unwinding the hairball graph: pruning algorithms for weighted complex networks}, U{arXiv:1503.04085 [physics.soc-ph]<http://arxiv.org/abs/1503.04085>}

@requires: U{statsmodels <statsmodels.sourceforge.net>}
@requires: U{igraph<http://igraph.org/python/>}
@version: 1.0
@author: Navid Dianati
@contact: navid.dianati@gmail.com
@copyright: All Rights Reserved
@license:GNU General Public License, version 3 (GPL-3.0)
'''



from filters import prune,pvalue,compute_significance
