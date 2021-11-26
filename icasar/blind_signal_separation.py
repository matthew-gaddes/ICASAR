#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:59:13 2019

@author: matthew
"""

#%%
    


def fastica_MEG(X, n_comp=None,
            algorithm="parallel", whiten=True, fun="logcosh", fun_prime='', 
            fun_args={}, maxit=200, tol=1e-04, w_init=None, verbose = True):
    """Perform Fast Independent Component Analysis.
    Parameters
    ----------
    X : (p, n) array
        Array with n observations (statistical units) measured on p variables.
    n_comp : int, optional
        Number of components to extract. If None no dimension reduction
        is performed.
    algorithm : {'parallel','deflation'}
        Apply an parallel or deflational FASTICA algorithm.
    whiten: boolean, optional
        If true perform an initial whitening of the data. Do not set to 
        false unless the data is already white, as you will get incorrect 
        results.
        If whiten is true, the data is assumed to have already been
        preprocessed: it should be centered, normed and white.
    fun : String or Function
          The functional form of the G function used in the
          approximation to neg-entropy. Could be either 'logcosh', 'exp', 
          or 'cube'.
          You can also provide your own function but in this case, its 
          derivative should be provided via argument fun_prime
    fun_prime : Empty string ('') or Function
                See fun.
    fun_args : Optional dictionnary
               If empty and if fun='logcosh', fun_args will take value 
               {'alpha' : 1.0}
    maxit : int
            Maximum number of iterations to perform
    tol : float
          A positive scalar giving the tolerance at which the
          un-mixing matrix is considered to have converged
    w_init : (n_comp,n_comp) array
             Initial un-mixing array of dimension (n.comp,n.comp).
             If None (default) then an array of normal r.v.'s is used
 
    Results
    -------
    K : (n_comp, p) array
        pre-whitening matrix that projects data onto th first n.comp
        principal components. Returned only if whiten is True
    W : (n_comp, n_comp) array
        estimated un-mixing matrix
        The mixing matrix can be obtained by::
            w = np.asmatrix(W) * K.T
            A = w.T * (w * w.T).I
    S : (n_comp, n) array
        estimated source matrix
    Notes
    -----
    The data matrix X is considered to be a linear combination of
    non-Gaussian (independent) components i.e. X = SA where columns of S
    contain the independent components and A is a linear mixing
    matrix. In short ICA attempts to `un-mix' the data by estimating an
    un-mixing matrix W where S = W K X.
    Implemented using FastICA:
      A. Hyvarinen and E. Oja, Independent Component Analysis:
      Algorithms and Applications, Neural Networks, 13(4-5), 2000,
      pp. 411-430
      
      2017/07/19 | Merged into one function by MEG and included a PCA function for whitening
      2017/07/20 | fixed bug when giving the function whitened data
      2018/02/22 | Return a boolean flag describing if algorithm converged or not (only works with symetric estimation)
      
    """
    
    import numpy as np
    from scipy import linalg 
    
    
    def _ica_def(X, tol, g, gprime, fun_args, maxit, w_init):
        """Deflationary FastICA using fun approx to neg-entropy function
        Used internally by FastICA.
        """
        def _gs_decorrelation(w, W, j):
            """ Gram-Schmidt-like decorrelation. """
            t = np.zeros_like(w)
            for u in range(j):
                t = t + np.dot(w, W[u]) * W[u]
                w -= t
            return w
        
        n_comp = w_init.shape[0]
        W = np.zeros((n_comp, n_comp), dtype=float)
        # j is the index of the extracted component
        for j in range(n_comp):
            w = w_init[j, :].copy()
            w /= np.sqrt((w**2).sum())
            n_iterations = 0
            # we set lim to tol+1 to be sure to enter at least once in next while
            lim = tol + 1 
            while ((lim > tol) & (n_iterations < (maxit-1))):
                wtx = np.dot(w.T, X)
                gwtx = g(wtx, fun_args)
                g_wtx = gprime(wtx, fun_args)
                w1 = (X * gwtx).mean(axis=1) - g_wtx.mean() * w
                _gs_decorrelation(w1, W, j)
                w1 /= np.sqrt((w1**2).sum())
                lim = np.abs(np.abs((w1 * w).sum()) - 1)
                w = w1
                n_iterations = n_iterations + 1
            W[j, :] = w
        return W                    # XXXX for deflation, a converged term isn't returned

    
    def _ica_par(X, tol, g, gprime, fun_args, maxit, w_init):
        """Parallel FastICA.
        Used internally by FastICA.
        2017/05/10 | edit to? 
        """
        def _sym_decorrelation(W):
            """ Symmetric decorrelation """
            K = W @  W.T
            s, u = linalg.eigh(K) 
            # u (resp. s) contains the eigenvectors (resp. square roots of 
            # the eigenvalues) of W * W.T 
            u, W = [np.asmatrix(e) for e in (u, W)]
            W = (u * np.diag(1.0/np.sqrt(s)) * u.T) * W  # W = (W * W.T) ^{-1/2} * W
            return np.asarray(W)
    
        n, p = X.shape
        W = _sym_decorrelation(w_init)
        # we set lim to tol+1 to be sure to enter at least once in next while
        lim = tol + 1 
        it = 0
        hist_lim = np.zeros((1, maxit))                                   #initiate array for history of change of W
        hist_W = np.zeros((w_init.size, maxit))                           # and for what W actually is
        while ((lim > tol) and (it < (maxit-1))):                           # and done less than the maximum iterations
            wtx = W @ X
            gwtx = g(wtx, fun_args)
            g_wtx = gprime(wtx, fun_args)
            W1 = (gwtx @ X.T)/float(p) - ((np.diag(g_wtx.mean(axis=1))) @ W)
            W1 = _sym_decorrelation(W1)
            lim = max(abs(abs(np.diag(W1 @ W.T)) - 1))
            W = W1
            it += 1
            hist_lim[0,it] = lim                                        # recond the measure of how much W changes
            hist_W[:,it] = np.ravel(W)                                  # and what W is
        hist_lim = hist_lim[:, 0:it]                                    # crop the 0 if we finish before the max number of iterations
        hist_W = hist_W[:, 0:it]                                        # ditto
        if it < maxit-1:
            if verbose:
                print('FastICA algorithm converged in ' + str(it) + ' iterations.  ')
            converged = True
        else:
            if verbose:
                print("FastICA algorithm didn't converge in " + str(it) + " iterations.  ")
            converged = False
        return W, hist_lim, hist_W, converged
    

  
    
    algorithm_funcs = {'parallel': _ica_par,   'deflation': _ica_def}

    alpha = fun_args.get('alpha',1.0)
    if (alpha < 1) or (alpha > 2):
        raise ValueError("alpha must be in [1,2]")

    if type(fun) is str:
        # Some standard nonlinear functions
        # XXX: these should be optimized, as they can be a bottleneck.
        if fun == 'logcosh':
            def g(x, fun_args):
                alpha = fun_args.get('alpha', 1.0)
                return np.tanh(alpha * x)
            def gprime(x, fun_args):
                alpha = fun_args.get('alpha', 1.0)
                return alpha * (1 - (np.tanh(alpha * x))**2)
        elif fun == 'exp':
            def g(x, fun_args):
                return x * np.exp(-(x**2)/2)
            def gprime(x, fun_args):
                return (1 - x**2) * np.exp(-(x**2)/2)
        elif fun == 'cube':
            def g(x, fun_args):
                return x**3
            def gprime(x, fun_args):
                return 3*x**2
        else:
            raise ValueError(
                        'fun argument should be one of logcosh, exp or cube')
    elif callable(fun):
        raise ValueError('fun argument should be either a string '
                         '(one of logcosh, exp or cube) or a function') 
    else:
        def g(x, fun_args):
            return fun(x, **fun_args)
        def gprime(x, fun_args):
            return fun_prime(x, **fun_args)
#    if whiten is False:
#        print('Data must be whitened if whitening is being skipped. ')
    p, n = X.shape

    if n_comp is None:
        n_comp = min(n, p)
    if (n_comp > min(n, p)):
        n_comp = min(n, p)
        print("n_comp is too large: it will be set to %s" % n_comp)


    # whiten the data
    if whiten:
        vecs, vals, whiten_mat, dewhiten_mat, x_mc, x_decorrelate, x_white  = PCA_meg2(X)           # function determines whether to use compact trick or not
        X1 = x_white[0:n_comp, :]                                                                    # if more mixtures than components to recover, use only first few dimensions
    else:
        X1 = np.copy(X[0:n_comp, :])
        

    if w_init is None:
        w_init = np.random.normal(size=(n_comp, n_comp))
    else:
        w_init = np.asarray(w_init)
        if w_init.shape != (n_comp,n_comp):
            raise ValueError("w_init has invalid shape -- should be %(shape)s"
                             % {'shape': (n_comp,n_comp)})

    kwargs = {'tol': tol,
              'g': g,
              'gprime': gprime,
              'fun_args': fun_args,
              'maxit': maxit,
              'w_init': w_init}

    func = algorithm_funcs.get(algorithm, 'parallel')

    W, hist_lim, hist_W, converged = func(X1, **kwargs)                                    #  W unmixes the whitened data
    #del X1

    if whiten:
        S = W @ whiten_mat[0:n_comp,:] @ x_mc
        A = np.linalg.inv(W)
        A_dewhite = dewhiten_mat[:,0:n_comp] @ A    
        #S = np.dot(np.dot(W, K), X)
        return W, S, A, A_dewhite, hist_lim, hist_W, vecs, vals, x_mc, x_decorrelate, x_white, converged
    else:
        S = W @ X1
        A = np.linalg.inv(W)
        return W, S, A, hist_lim, hist_W, converged

#%%

def PCA_meg2(X, verbose = False, return_dewhiten = True):
    """
    Input:
        X | array | rows are dimensions (e.g. 2 x 1000 normally for 2 sound recordings, or eg 12x225 for 12 15x15 pixel images)
                    Doesn't have to be mean centered
        verbose | boolean | if true, prints some info to the screen.  
        return_dewhiten | boolean | if False, doesn't return the dewhitening matrix as the pseudo inverse needed to calculate this can fail with very large matrices (e.g. 1e6)
        
    Output:
        vecs | array | eigenvectors as columns, most important first
        vals | 1d array | eigenvalues, most important first
        whiten_mat | 2d array | whitens the mean centered data
        x_mc | 2d array | mean centered data
        x_newbasis | 2d array | mean centered and decorrelated
        x_white | 2d | mean centered and decorellated and unit variance in each dimension (ie whitened)
        
    
    2016/12/16 | MEG |  updated to python 3.5
    2016/03/29 | MEG |  overhaul to include whitening
    2017/07/19 | MEG |  include compact trick PCA and automatically determine which to use 
    2017/11/16 | MEG | fix bug in the order of eigenvectors and values - descending order now
    2018/01/16 | MEG | fix bug in how covariance matrix is caluculated for compact trick case (now dividing by samples, which
                       gives the same results as doing it with np.cov [which is used in the normal case])
    2018/01/17 | MEG | Swith to eigh (from eig) as this doesn't give complex results (such as (1.2i + 1.2e17j))
                       Take abs value of eigenvalues as some of the tiny ones can become negative (floating point effect)
    2018/02/12 | MEG | fix a bug in the compact trick PCA so that vectors are now unit length and outputs correct.  
    2018/02/23 | MEG | add option to no return the dewhitening matrix as the pseudo inverse needed for this can fail with 
                       very large matrices.  
    2020/06/09 | MEG | Add a raise Exception so that data cannot have nans in it.  
    2021_10_07 | MEG | Add raise Exception for the case that negative eignevalues are returned.  
    """
    
    import numpy as np

    # Check if the data are suitable
    if np.max(np.isnan(X)):
        raise Exception("Unable to proceed as the data ('X') contains Nans.  ")
    
    if not return_dewhiten:
        print('Will not return the dewhitening matrix.  ')
    dims, samples = X.shape
    X = X - X.mean(axis=1)[:,np.newaxis]                # mean center each row (ie dimension)

    if samples < dims and dims > 100:                   # do PCA using the compact trick (i.e. if there are more dimensions than samples, there will only ever be sample -1 PC [imagine a 3D space with 2 points.  There is a vector joining the points, one orthogonal to that, but then there isn't a third one])
        if verbose:
            print('There are more samples than dimensions and more than 100 dimension so using the compact trick.')
        M = (1/samples) * X.T @ X                                        # maximum liklehood covariance matrix.  See blog post for details on (samples) or (samples -1): https://lazyprogrammer.me/covariance-matrix-divide-by-n-or-n-1/
        e, EV = np.linalg.eigh(M)                                         # eigenvalues and eigenvectors.  Note that in some cases this function can return negative eigenvalues (e)    
        if np.min(e) < 0:
            print(f"There are negative values in the eigenvalues.  This is a tricky problem, but is usually only caused by poor approximations to zero by floating point arithmetic.  "
                  f"Trying to set these to a better approxiation of zero to contiue.  ")
            e = np.where(e < 0, np.abs(e), e)
        tmp = (X @ EV)                                                   # this is the compact trick
        vecs = tmp[:,::-1]                                               # vectors are columns, make first (left hand ones) the important onces
        vals = np.sqrt(e)[::-1]                                          # also reverse the eigenvectors.  Note that with the negative eigen values that can be encoutered here, this produces nans.  
        vals = np.nan_to_num(vals, nan = 0.0)               
        for i in range(vecs.shape[0]):        # normalise each eigenvector (ie change length to 1)
            vecs[i,:] /= vals
        vecs = vecs[:, 0:-1]                       # drop the last eigenvecto and value as it's not defined.            
        vecs = np.divide(vecs, np.linalg.norm(vecs, axis = 0)[np.newaxis, :])       # make unit length (columns)
        vals = vals[0:-1]                          # also drop the last eigenvealue
        X_pca_basis = vecs.T @ X                     # whitening using the compact trick is a bit of a fudge
        covs = np.diag(np.cov(X_pca_basis))
        covs_recip = np.reciprocal(np.sqrt(covs))
        covs_recip_mat = np.diag(covs_recip)
        whiten_mat = (covs_recip_mat @ vecs.T)
        if return_dewhiten:
            dewhiten_mat = np.linalg.pinv(whiten_mat)                           # as always loose a dimension with compact trick, have to use pseudoinverse

    else:                                                                       # or do PCA normally
        cov_mat = np.cov(X)                                                     # dims by dims covariance matrix
        vals_noOrder, vecs_noOrder = np.linalg.eigh(cov_mat)                    # vectors (vecs) are columns, not not ordered
        order = np.argsort(vals_noOrder)[::-1]                                  # get order of eigenvalues descending
        vals = vals_noOrder[order]                                              # reorder eigenvalues
        vals = np.abs(vals)                                                        # do to floatint point arithmetic some tiny ones can be nagative which is problematic with the later square rooting
        vecs = vecs_noOrder[:,order]                                            # reorder eigenvectors
        vals_sqrt_mat = np.diag(np.reciprocal(np.sqrt(vals)))          # square roots of eigenvalues on diagonal of square matrix
        whiten_mat = vals_sqrt_mat @ vecs.T                            # eigenvectors scaled by 1/values to make variance same in all directions
        if return_dewhiten:
            dewhiten_mat = np.linalg.inv(whiten_mat)
    # use the vectors and values to decorrelate and whiten
    x_mc = np.copy(X)                       # data mean centered
    x_decorrelate =  vecs.T @ X             # data decorrelated
    x_white = whiten_mat @ X                # data whitened
  
    if return_dewhiten:
        return vecs, vals, whiten_mat, dewhiten_mat, x_mc, x_decorrelate, x_white
    else:
        return vecs, vals, whiten_mat, x_mc, x_decorrelate, x_white
