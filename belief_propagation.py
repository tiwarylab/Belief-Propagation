#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.covariance import GraphicalLasso
import copy


def FG_from_glasso(data, alpha):
    """ Compute a factor graph using graphical lasso and clique factorization
    
    Parameters
    ----------
    data: pandas DataFrame
        Input trajectory for graphical lasso.

    alpha: float
        Penalty parameter for graphical lasso.

    Returns
    -------
    networkx Graph
        Factor graph inheriting variable names from the input DataFrame.
    
    """
    
    lasso = GraphicalLasso(alpha=alpha).fit(data)

    inv_cov = lasso.precision_
    lasso_net = np.abs(inv_cov) > 0 # entries greater than 0 
    lasso_net = lasso_net.astype('int')

    np.fill_diagonal(lasso_net,0)
    graph = nx.from_numpy_matrix(lasso_net)
    col = data.columns
    label_dict = {i:col[i] for i in range(len(col))}
    graph = nx.relabel_nodes(graph,label_dict)


    max_cliques = nx.find_cliques(graph)

    clique_list = []
    for clique in max_cliques:
        clique_list.append(clique)

    FG = nx.Graph()
    FG.add_nodes_from(list(graph.nodes), bipartite=0)

    factors = []
    edges = []
    index = 1
    for clique in clique_list:
        factor = "$f_" + str(index) +"$"
        factors.append(factor)
        for node in clique:
            edges.append((factor, node))
        index += 1

    FG.add_nodes_from(factors, bipartite=1)
    FG.add_edges_from(edges)
    
    return FG


def show_FG_components(FG, figsize=None, dpi=None, savename=None):
    """ Display a factor graph structure, showing each connected component.
    
    Parameters
    ----------
    FG: networkx Graph
        Factor graph where node type is distinguisehd with the 'bipartite' attribute.
        
    figsize: tuple
        Width and height of the output figures.
        
    dpi: int
        DPI of the output figures.
        
    savename: str
        Name to save the figure when the graph is a single component.
        
    Returns
    -------
    None 
        This function only shows matplotlib figures.
    
    
    """
    
    components = [FG.subgraph(c).copy() for c in nx.connected_components(FG)]

    for C in components:
        plt.figure(figsize=figsize,dpi=dpi)
        pos = nx.kamada_kawai_layout(C)
        var = [x for x,y in C.nodes(data=True) if y['bipartite']==0]
        func = [x for x,y in C.nodes(data=True) if y['bipartite']==1]
        nx.draw_networkx_nodes(C, nodelist = var, pos=pos, node_size=900)
        nx.draw_networkx_nodes(C, nodelist = func, pos=pos, node_shape='s', node_size=900)
        nx.draw_networkx(C,pos=pos,with_labels=True,node_size=0, font_color='w')
        
    if savename != None:
        plt.savefig(savename)
        
    plt.show()

    
def outer_multi(*arrays):
    """ Compute the outer product of a set of arrays.
    
    Parameters
    ----------
    arrays: numpy arrays
        Set of arrays to be multiplied.
        
    Returns
    -------
    numpy array 
        N-dim tensor where N is the number of input arrays.
    
    """
    
    prod = arrays[0]
    for i in range(1,len(arrays)):
        prod = np.tensordot(prod,arrays[i], axes=((),()))
    return prod


def marginalize(tensor, axis):
    """ Marginalize out all but one axes of a probability tensor.
    
    Parameters
    ----------
    tensor: numpy array
        Probability tensor to be marginalized.
        
    axis: int
        Axis that will not be marginalized.
        
    Returns
    -------
    numpy array 
        Marginalized probability array with shape (n,) where n is the length of axis.
    
    """
    
    sum_dims = [i for i in range(len(tensor.shape))]
    sum_dims.remove(axis)
    sum_dims = tuple(sum_dims)
    return np.sum(tensor, axis=sum_dims)


def standardize_df(df, return_stats=False):
    """ Standardize each column of a DataFrame.
    
    Parameters
    ----------
    df: pandas DataFrame
        DataFrame to be standardized.
        
    return_stats: bool
        Option to return the original means and standard deviations.
        
    Returns
    -------
    pandas DataFrame 
        Standardized DataFrame.
    
    pandas Series (optional)
        Means of the original DataFrame.
    
    pandas Series (optional)
        Standard deviations of the original DataFrame.
        
    """
    
    df_mean = np.mean(df)
    df_std = np.std(df)
    standardized = (df - df_mean) / df_std
    if return_stats: 
        return standardized, df_mean, df_std
    else:
        return standardized

    
def set_bandwidth(n, d, kernel, weights=None, rule='scott', print_bw=False):
    """ Calculates the bandwidth consistent with Scott or Silverman's
    rules of thumb for bandwidth selection.
    
    Parameters
    ----------
    n : int
        Number of data points.
    
    d : int
        Dimensionality of the data.
        
    kernel : str
        Kernel name for kernel density estimation.
        
    weights : list of floats or numpy array
        The weights associated with each data point after reweighting an enhanced 
        sampling trajectory.
        
    rule : str
        Rule used for bandwidth selection.
        
    Returns
    -------
    float
        Bandwidth from the rules of thumb (they're the same for 2D KDE).
        
    """
    
    if kernel == 'epanechnikov':
        bw_constant = 2.2
    else:
        bw_constant = 1

    if type(weights) != type(None):
        weights = np.array(weights)
        n = np.sum(weights)**2 / np.sum(weights**2)

    if rule == 'scott':
        bandwidth = bw_constant*n**(-1/(d+4))
    elif rule == 'silverman':
        bandwidth = bw_constant*(n*(d+2)/4)**(-1/(d+4))
    else:
        print('Please use scott or silverman as the rule for bandwidth selection.')


    if print_bw:
        print('Selected bandwidth: ' + str(bandwidth)+ '\n')

    return bandwidth


class Factor:
    """ Handles the joint probability modeling for each factor and stores a grid of their values.
    
    Attributes
    ----------
    dims: list
        The factor's neighbors in the order they appear in the probability tensors.
        
    data: pandas DataFrame
        Data used to define grids and to fit KDE.
        
    bins: int
        Number of bins for the KDE and factor grids.
        
    kernel: str
        Kernel type used in KDE.
        
    weights: numpy array
        Weights used for weighted KDE. Used for modeling biased simualtions.
        
    rule: str
        Rule for KDE bandwidth selection.
    
    """
    
    def __init__(self, dims, data, bins=20, kernel='gaussian', weights=None, rule='scott'):
        self.dims = dims
        self.bins = bins
        self.bounds = self.bin_boundaries(data)
        self.midpoints = [(b[1:]+b[:-1])/2 for b in self.bounds]
        self.KD = self.KDE_fit(data, kernel, weights, rule)
        self.joint_grid = self.KDE_grid()
        self.factor_grid = self.joint_grid
       
    
    def bin_boundaries(self,data):
        """ Defines gridpoints along each dimension of the input data.
        
        Parameters
        ----------
        data: pandas DataFrame
            Data used to calculate the gridpoints.
            
        Returns
        -------
        list of numpy arrays
            List of gridpoints for the dimensions in self.dims.
            
        """
        
        bounds = []
        for col in self.dims:
            bounds.append(np.linspace(np.min(data[col]),np.max(data[col]),self.bins+1))
        return bounds
    
    
    def KDE_fit(self, data, kernel='gaussian', weights=None, rule='scott'):
        """ Fit KDE to input data.
        
        Parameters
        ----------
        data: pandas DataFrame
            Data used for KDE fitting.
            
        kernel: str
            Kernel type used in KDE.
            
        weights: numpy array
            Weights used for weighted KDE. Used for modeling biased simualtions.
        
        rule: str
            Rule for KDE bandwidth selection.
            
        Returns
        -------
        sklearn KernelDensity
            Object containing the KDE fit.
        
        """
        
        shape = np.shape(data[self.dims])
        n = shape[0]
        if len(shape) == 1:
            d = 1
        else:
            d = shape[1]


        bw = set_bandwidth(n, d, kernel, weights, rule)

        KD = KernelDensity(bandwidth=bw,kernel=kernel)
        KD.fit(data[self.dims], sample_weight=weights)

    
        return KD

    
    def KDE_grid(self):
        """ Calculate the KDE probability at each point in the factor's grid.
        
        Returns
        -------
        numpy array
            Probability tensor calculated using the KDE values on the factor's gridpoints.
        
        """
        
        points = np.array(np.meshgrid(*self.midpoints,indexing='ij')).reshape(len(self.midpoints),-1).T
        samp = self.KD.score_samples(points)
        samp = samp.reshape([self.bins]*len(self.dims))
        p = np.exp(samp)/np.sum(np.exp(samp))

        return p


class MessagePasser:
    """Class containing the factor graph and handling message passing operations.
    
    Attributes
    ----------
    FG: bipartite networkx Graph
        Factor graph used for message passing.
    
    data: pandas DataFrame
        Data used for KDE fitting.
        
    bins: int
        Number of bins for the KDE and factor grids.
        
    kernel: str
        Kernel type used in KDE.
        
    rule: str
        Rule for KDE bandwidth selection.
    
    """
    
    def __init__(self, FG, data, bins=20, kernel='gaussian', rule='scott'):
        self.FG = FG
        self.bins = bins
        self.binned = self.digitize_df(data)
        self.data, self.orig_mean, self.orig_std = standardize_df(data,return_stats=True)
        self.kernel = kernel
        self.rule = rule
        self.funcs = self.get_func_nodes()
        self.vars = self.get_var_nodes()
        self.init_factors()
        self.init_messages()
        
        
    def get_var_nodes(self):
        """Creates a list of all variable nodes.
        
        Returns
        -------
        list
            List of variable nodes in self.FG.
            
        """
        
        return [x for x,y in self.FG.nodes(data=True) if y['bipartite']==0]
    
    
    def get_func_nodes(self):
        """Creates a list of all factor nodes.
        
        Returns
        -------
        list
            List of factor nodes in self.FG.
            
        """
        
        return [x for x,y in self.FG.nodes(data=True) if y['bipartite']==1]
    
    
    def digitize_df(self, df):
        """Digitizes a DataFrame creating a new DataFrame replacing data points with their bins.
        
        Parameters
        ----------
        df: pandas DataFrame
            DataFrame to be digitized.
            
        Returns
        -------
        pandas DataFrame
            DataFrame storing binning for each data point.
            
        """
        
        df_array = df.to_numpy()
        num_col = df_array.shape[1]

        binned_cols = []
        for i in range(num_col):
            col = df_array[:,i]
            grid = np.linspace(np.min(col), np.max(col), self.bins+1)
            binned = np.digitize(col, grid[:-1])-1
            binned_cols.append(binned)

        return pd.DataFrame(np.column_stack(binned_cols),columns=df.columns)
    
    
    def init_factors(self):
        """Initializes the Factor objects for each factor node, fitting KDE for each factor.
        
        Returns
        -------
        None
        
        """
        
        factor_properties = {}
        for node in self.funcs:
            factor_properties[node] = {}
            
            neighbors = list(self.FG[node])
            
            F = Factor(neighbors, self.data, bins=self.bins, kernel=self.kernel, rule=self.rule)
            factor_properties[node]['factor'] = F

        nx.set_node_attributes(self.FG,factor_properties)
        
        
    def init_messages(self):
        """Initializes the messages for message passing.
        
        Returns
        -------
        None
        
        """
        
        factor_properties = dict(self.FG.nodes())
        for node in self.FG.nodes:
            messages = dict(self.FG[node])
            for edge in messages:
                messages[edge] = np.ones(self.bins)
            factor_properties[node]['messages'] = messages
        nx.set_node_attributes(self.FG, factor_properties)
      
    
    def cross_entropy(self, q, dims):
        """Calculates the cross entropy between a model q and the observed data.
        
        Parameters
        ----------
        q: numpy array
            Model probability.
        
        dims: list
            Dimensions used to access columns of self.data.
            
        Returns
        -------
        float
            Value of the cross entropy.
        
        """
        
        indices = tuple(self.binned[dims].to_numpy().T)
        cross_entropy = -np.mean(np.log(q[indices]))

        return cross_entropy
            
        
    def factor_message_update(self, node):
        """Updates a single factor node and its outgoing messages.
        
        Parameters
        ----------
        node: str
            Name of the node to be updated.
        
        Retutns
        -------
        None
        
        """
        
        factor = self.FG.nodes[node]['factor']
        neighbors = factor.dims
        incoming = []
        for n in neighbors:
            incoming.append(self.FG.nodes[n]['messages'][node])
            
            
        f_grid = factor.joint_grid/outer_multi(*incoming)
        factor.factor_grid = f_grid   
        
        for i in range(len(incoming)):
            m = [x if j!=i else x/x for j,x in enumerate(incoming)] 
            self.FG.nodes[node]['messages'][neighbors[i]] = marginalize(f_grid*outer_multi(*m),i)
           
        
    def var_message_update(self, node):
        """Updates  a single variable node and its outgoing messages.
        
        Parameters
        ----------
        node: str
            Name of the node to be updated.
        
        Returns
        -------
        None
        
        """
        
        neighbors = list(self.FG[node])
        incoming = {n:self.FG.nodes[n]['messages'][node] for n in neighbors}
        product = np.product(list(incoming.values()),axis=0)
        
        product = product / np.sum(product)
        
        for n in neighbors:
            self.FG.nodes[node]['messages'][n] = product / incoming[n]

            
    def factor_update_cycle(self, node):
        """Updates a single factor node, all adjacent variable nodes, and their outgoing messages.
        
        Parameters
        ----------
        node: str
            Name of the factor node to be updated.
            
        Returns
        -------
        None
        
        """
        
        factor = self.FG.nodes[node]['factor']
        neighbors = factor.dims

        self.factor_message_update(node)
        
        for n in neighbors:
            self.var_message_update(n)
 

    def factor_CE(self, node):
        """Calculates the contribution to the model's cross entropy from one factor.
        
        Parameters
        ----------
        node: str
            Name of the factor to calculate cross entropy.
            
        Returns
        -------
        float
            The cross entropy from one factor node.
            
        """
        
        F = self.FG.nodes[node]['factor']
        vals = F.factor_grid
        return self.cross_entropy(vals, F.dims)

    
    def global_CE(self):
        """Calculates the cross entropy of the factorized joint probability model.
        
        Returns
        -------
        float
            Model cross entropy.
            
        """
        
        CE = 0
        for f in self.funcs:
            CE += self.factor_CE(f)
        return CE

    
class FactorPeriodic(Factor):
    """Factor class with manually set grid range.
    
    Attributes
    ----------
    dims: list
        The factor's neighbors in the order they appear in the probability tensors.
        
    data: pandas DataFrame
        Data used to define grids and to fit KDE.
        
    period_range: list
        List of standardized ranges (min, max) for each dimension.
            
    bins: int
        Number of bins for the KDE and factor grids.
        
    kernel: str
        Kernel type used in KDE.
        
    weights: numpy array
        Weights used for weighted KDE. Used for modeling biased simualtions.
        
    rule: str
        Rule for KDE bandwidth selection.
    
    """

    def __init__(self, dims, data, period_range, bins=21, kernel='gaussian',
                 weights=None, rule='scott'):
        self.dims = dims
        self.bins = bins
        self.gridpoints = [np.linspace(*p, bins) for p in period_range]
        self.KD = self.KDE_fit(data, kernel, weights, rule)
        self.joint_grid = self.KDE_grid()
        self.factor_grid = self.joint_grid

        
    def KDE_grid(self):
        """ Calculate the KDE probability at each point in the factor's grid.
        
        Returns
        -------
        numpy array
            Probability tensor calculated using the KDE values on the factor's gridpoints.
        
        """
        
        points = np.array(np.meshgrid(*self.gridpoints,indexing='ij')).reshape(len(self.gridpoints),-1).T
        samp = self.KD.score_samples(points)
        samp = samp.reshape([self.bins]*len(self.dims))
        p = np.exp(samp)/np.sum(np.exp(samp))

        return p
    
    
class MessagePasserPeriodic(MessagePasser):
    """Class containing the factor graph and handling message passing operations.
    
    Attributes
    ----------
    FG: bipartite networkx Graph
        Factor graph used for message passing.
    
    data: pandas DataFrame
        Data used for KDE fitting.
        
    bins: int
        Number of bins for the KDE and factor grids.
        
    period_range: tuple
        Min and max for the variables. Currently uses the same range for each variable.    
        
    kernel: str
        Kernel type used in KDE.
        
    rule: str
        Rule for KDE bandwidth selection.
    
    """
    
    def __init__(self, FG, data, bins=21, period_range=(-np.pi,np.pi), kernel='gaussian',
                 rule='scott'):
        self.FG = FG
        self.range = period_range
        self.bins = bins
        self.binned = self.digitize_df(data)
        self.data, self.orig_mean, self.orig_std = standardize_df(data,return_stats=True)
        self.kernel = kernel
        self.rule = rule
        self.funcs = self.get_func_nodes()
        self.vars = self.get_var_nodes()
        self.init_factors()
        self.init_messages()
        
        
    def init_factors(self):
        """Initializes the Factor objects for each factor node, fitting KDE for each factor.
        
        Returns
        -------
        None
        
        """
        
        factor_properties = {}
        for node in self.funcs:
            factor_properties[node] = {}
            
            neighbors = list(self.FG[node])
            
            period_range = [(self.range-self.orig_mean[n])/self.orig_std[n] for n in neighbors]
            
            F = FactorPeriodic(neighbors, self.data, period_range=period_range, bins=self.bins,
                                   kernel=self.kernel, rule=self.rule)
            factor_properties[node]['factor'] = F

        nx.set_node_attributes(self.FG,factor_properties)
        
        
    def digitize_df(self, df):
        """Digitizes a DataFrame creating a new DataFrame replacing data points with their bins.
        
        Parameters
        ----------
        df: pandas DataFrame
            DataFrame to be digitized.
            
        Returns
        -------
        pandas DataFrame
            DataFrame storing binning for each data point using the range self.range.
            
        """
        
        df_array = df.to_numpy()
        num_col = df_array.shape[1]

        binned_cols = []
        for i in range(num_col):
            col = df_array[:,i]
            grid = np.linspace(*self.range, self.bins+1)
            binned = np.digitize(col, grid[:-1])-1
            binned_cols.append(binned)

        return pd.DataFrame(np.column_stack(binned_cols),columns=df.columns)


def show_2d_update(MP,node,figsize=(25,5),levels=30,titlesize=20):
    """Update a single factor and show how it and its associated probability change.
    
    Parameters
    ----------
    MP: MessagePasser or MessagePasserPeriodic
        Object to be updated.
    
    node: str
        Name of the factor in the MessagePasser to update.
    
    figsize: tuple
        Size of the output figure.
    
    levels: int
        Number of contour levels on each subplot.
    
    titlesize: float
        Font size for each subplot's title.
        
    Returns
    -------
    None
        
    """
    
    F = MP.FG.nodes[node]['factor']
    
    neighbors = F.dims
    incoming = []
    for n in neighbors:
        incoming.append(MP.FG.nodes[n]['messages'][node])

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5,figsize=figsize)
    
    ax1.contourf((F.factor_grid*outer_multi(*incoming)).T,levels=levels,cmap='Spectral_r')
    ax1.set_title('initial model prob',fontsize=titlesize)
    
    init_factor = F.factor_grid

    ax2.contourf(F.joint_grid.T,levels=levels,cmap='Spectral_r')
    ax2.set_title('KDE',fontsize=titlesize)

    MP.factor_update_cycle(node)

    ax3.contourf((F.factor_grid*outer_multi(*incoming)).T,levels=levels,cmap='Spectral_r')
    ax3.set_title('final model prob',fontsize=titlesize)
    
    
    ax4.contourf(init_factor.T,levels=levels,cmap='Spectral_r')
    ax4.set_title('initial factor',fontsize=titlesize)
    ax5.contourf(F.factor_grid.T,levels=levels,cmap='Spectral_r')
    ax5.set_title('final factor',fontsize=titlesize)
    plt.show()

    
def update_show_if_2d(MP,node,figsize=(25,5),levels=30,titlesize=20):
    """Update a single factor and show how it and its associated probability change if it is 2D.
    
    Parameters
    ----------
    MP: MessagePasser or MessagePasserPeriodic
        Object to be updated.
    
    node: str
        Name of the factor in the MessagePasser to update.
    
    figsize: tuple
        Size of the output figure.
    
    levels: int
        Number of contour levels on each subplot.
    
    titlesize: float
        Font size for each subplot's title.
        
    Returns
    -------
    None
        
    """
    
    F = MP.FG.nodes[node]['factor']
    if len(F.dims) == 2:
        show_2d_update(MP, node, figsize, levels, titlesize)
    else:
        print('Updated factor ' + str(node) + ' with dimension ' + str(len(F.dims)) )
        MP.factor_update_cycle(node)


def periodic_padding(grid):
    """Create a periodic boundary around a tensor such that the ends are adjacent.
    For example a 1D tensor will return [grid[-1],grid[0],...,grid[-1],grid[0]]
    
    Parameters
    ----------
    grid: numpy array
        Any dimensional tensor to add a periodic boundary to.
        
    Returns
    -------
    numpy array
        Tensor with a periodic boundary added.
        
    """
    
    shape = grid.shape
    padded = np.zeros(np.array(shape)+2)
    length = len(shape)
    center_ind = tuple([slice(1,-1,1)]*length)
    full_ind = tuple([slice(None)]*length)
    padded[center_ind] = grid
    
    for i in range(length):
        pad_edge_ind = list(center_ind)
        pad_edge_ind[i] = 0
        pad_edge_ind = tuple(pad_edge_ind)

        orig_edge_ind = list(full_ind)
        orig_edge_ind[i] = -1
        orig_edge_ind = tuple(orig_edge_ind)
    
        padded[pad_edge_ind] = grid[orig_edge_ind]

        pad_edge_ind = list(center_ind)
        pad_edge_ind[i] = -1
        pad_edge_ind = tuple(pad_edge_ind)

        orig_edge_ind = list(full_ind)
        orig_edge_ind[i] = 0
        orig_edge_ind = tuple(orig_edge_ind)
        
        padded[pad_edge_ind] = grid[orig_edge_ind]

    return padded


def periodic_grad(grid, *varargs, **kwargs):
    """Calculate the gradient of a tensor representing a periodic function
        where the first and last bin are adjacent.
        
    Parameters
    ----------
    grid: numpy array
        Tensor to have its gradient calculated.
        
    varargs : list of scalar, optional
        N scalars specifying the sample distances for each dimension.
        
    kwargs
        Additional keywords passed to np.gradient. See numpy documentation for more details.
        
    Returns
    -------
    numpy array or list
        Gradient of the input tensor considering periodic boundaries. List if dimension >= 2.
    
    """
    
    periodic_grad = np.gradient(periodic_padding(grid), *varargs, **kwargs)
    center_ind = tuple([slice(1,-1,1)]*len(grid.shape))
    
    if type(periodic_grad) == list:
        for dim in range(len(periodic_grad)):
            periodic_grad[dim] = periodic_grad[dim][center_ind]
    else:
        periodic_grad = periodic_grad[center_ind]
        
    return periodic_grad


def bias_from_p(p, T, cutoff=0, gamma=None):
    """Calculate a bias potential from a probability tensor.
    
    Parameters
    ----------
    p: numpy array
        Probability tensor to convert to bias.
        
    T: float
        Temperature.
        
    cutoff: float
        Set all probability below this point to 0.
        
    gamma: float or None
        Gamma used for well-tempered bias.
        
    Returns
    -------
    numpy array
        Tensor containing the bias potential.
    
    """
    
    kb = 8.31446261815324e-3
    p[p<cutoff] = 0
    E = -kb*T*np.ma.log(p)
    V = -(E-np.max(E))
    V[V.mask] = 0
    if gamma !=None:
        V *= (1-1/gamma)
    
    return V


def clean_names(names):
    """Clean factor names to remove formatting characters.
    
    Parameters
    ----------
    names: list of str
        Names to be cleaned.
    
    Returns
    -------
    list of str
        List of strings without formatting characters.
    
    """
    
    new_names = copy.copy(names)
    for i in range(len(names)):
        name = new_names[i]
        name = name.replace('_','')
        name = name.replace('$','')
        name = name.replace('\\','')
        new_names[i] = name
    return new_names


def write_header(file, F, bias_name):
    """Write the header for a PLUMED external bias file.
    
    Parameters
    ----------
    file: TextIOWrapper
        File to write the header.
        
    F: Factor
        Factor used for this bias.
    
    bias_name: str
        Name of the bias in the header (eg. factor1.bias).
    
    Returns
    -------
    None
    
    """
    
    dims = clean_names(F.dims)
    line = '#! FIELDS {0} {1} der_{2} \n'.format(' '.join(dims), bias_name, ' der_'.join(dims))
    file.write(line)
    for i in range(len(dims)):
        dim = dims[i]
        line = '#! SET min_{0} {1:s} \n'.format(dim, '-pi')
        file.write(line)
        line = '#! SET max_{0} {1:s} \n'.format(dim, 'pi')
        file.write(line)
        line = '#! SET nbins_{0} {1} \n'.format(dim, len(F.gridpoints[i])-1)
        file.write(line)
        line = '#! SET periodic_{0} true \n'.format(dim)
        file.write(line)
        
        
def rescale_gridpoints(F, means, stds):
    """Remove standardization from gridpoints.
    
    Parameters
    ----------
    F: Factor
        Factor to get gridpoints from.
        
    means: pandas Series
        Means that were used for standardization.
        
    stds: pandas Series
        Standard deviations that were used for standardization.
        
    Returns
    -------
    list of numpy arrays
        Gridpoints with standardization removed.
        
    """
    
    rescaled = []
    for i in range(len(F.dims)):
        dim = F.dims[i]
        mean = means[dim]
        std = stds[dim]
        
        dim_grids = np.copy(F.gridpoints[i])
        dim_grids = dim_grids * std + mean
        rescaled.append(dim_grids)
        
    return rescaled


def trim_last_gridpoint(gridpoints):
    """Remove the last gridpoint for each dimension. 
        This is done so that the first and last point are one bin away for periodic grids.  
    
    Parameters
    ----------
    gridpoints: list of numpy arrays
        Gridpoints to be trimmed.
        
    Returns
    -------
    list of numpy arrays
        List of trimmed gridpoints. None the original grids are not modified.
    
    """
    
    new_gridpoints = copy.copy(gridpoints)
    for i in range(len(new_gridpoints)):
        new_gridpoints[i] = new_gridpoints[i][:-1]
        
    return new_gridpoints


def trim_factor_grid(F):
    """Trim the last gridpoint in each dimension from the factor grid.
    
    Parameters
    ----------
    F: Factor
        Factor containing the grid.
        
    Returns
    -------
    numpy array
        Grid with the last gridpoints removed. Note the original Factor is not modified.
    
    """
    
    factor_grid_new = F.factor_grid[tuple([slice(None,-1)]*len(F.dims))]
    
    return factor_grid_new


def write_grid(file, F, means, stds, T, cutoff, gamma):
    """Write a PLUMED external bias grid from a factor.
    
    Parameters
    ----------
    F: Factor
        Factor used for the bias potential.
    
    means: pandas Series
        Means that were used for standardization.
        
    stds: pandas Series
        Standard deviations that were used for standardization.
    
    T: float
        Temperature.
    
    cutoff: float
        Set probability below this to 0 for the bias calculation.
    
    gamma: float
        Gamma for well-tempered bias.
        
    Returns
    -------
    None
    
    """
    
    rescaled_points = rescale_gridpoints(F, means, stds)
    rescaled_points = trim_last_gridpoint(rescaled_points)
    mesh = np.array(np.meshgrid(*rescaled_points,indexing='ij'))
    mesh_points = mesh.reshape(len(rescaled_points),-1).T
    p = trim_factor_grid(F)
    V = bias_from_p(p, T, cutoff, gamma)
    mesh_vals = V.reshape(-1)
    
    d_grid = [grid[1]-grid[0] for grid in rescaled_points]
    grad = periodic_grad(V,*d_grid)
    mesh_grad = np.array(grad).reshape(len(rescaled_points),-1).T

    format_string = ''
    ndims = len(F.dims)
    

    col_range = 2*ndims+1

    
    for j in range(col_range):
        if j < ndims:
            format_string += '{' + str(j) + ':.6f} '
        else:
            format_string += '{' + str(j) + ':.6e} '
            
    format_string += '\n'

    for i in range(len(mesh_vals)):

        format_list = []

        for j in range(ndims):
            format_list.append(mesh_points[i,j])
        format_list.append(mesh_vals[i])
        for j in range(ndims):
            format_list.append(mesh_grad[i,j])

        file.write(format_string.format(*format_list))
        
        
def factor2bias(F, filename, means, stds, T, cutoff, gamma):
    """Converts a factor to a PLUMED external bias file for enhanced sampling.
    
    Parameters
    ----------
    F: Factor
        Factor used for the bias potential.
        
    filename: str
        File to write the external bias.
    
    means: pandas Series
        Means that were used for standardization.
        
    stds: pandas Series
        Standard deviations that were used for standardization.
    
    T: float
        Temperature.
    
    cutoff: float
        Set probability below this to 0 for the bias calculation.
    
    gamma: float
        Gamma for well-tempered bias.
        
    Returns
    -------
    None
    
    
    """
    
    with open(filename, 'w') as bias_file:
        write_header(bias_file, F, filename.replace('.grid','.bias'))
        write_grid(bias_file, F, means, stds, T, cutoff, gamma)
     
    
def MP2biases(MP, T, cutoff=0, gamma=None):
    """Convert a factor graph to a series of external bias files.
    
    Parameters
    ----------
    T: float
        Temperature.
    
    cutoff: float
        Set probability below this to 0 for the bias calculation.
    
    gamma: float
        Gamma for well-tempered bias.
        
    Returns
    -------
    None
    
    """
    
    bias_num = 1
    for f in MP.funcs:
        F = MP.FG.nodes[f]['factor']
        filename = 'factor' + str(bias_num) + '.grid'
        factor2bias(F, filename, MP.orig_mean, MP.orig_std, T, cutoff, gamma)
        bias_num += 1
