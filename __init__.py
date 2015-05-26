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

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public Licensa as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''



from filters import prune,pvalue,compute_significance
