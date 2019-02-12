import autograd.numpy as np
from sklearn.decomposition import PCA

def pca_with_imputation(D, datas, masks, num_iters=20):
    if isinstance(datas, (list, tuple)) and isinstance(masks, (list, tuple)):
        data = np.concatenate(datas)
        mask = np.concatenate(masks)
    
    if np.any(~mask):
        # Fill in missing data with mean to start
        fulldata = data.copy()
        for n in range(fulldata.shape[1]):
            fulldata[~mask[:,n], n] = fulldata[mask[:,n], n].mean()

        for itr in range(num_iters):
            # Run PCA on imputed data
            pca = PCA(D)
            x = pca.fit_transform(fulldata)
            
            # Fill in missing data with PCA predictions
            pred = pca.inverse_transform(x)
            fulldata[~mask] = pred[~mask]
    else:
        pca = PCA(D)
        x = pca.fit_transform(data)
        
    # Unpack xs
    xs = np.split(x, np.cumsum([len(data) for data in datas])[:-1])
    assert len(xs) == len(datas)
    assert all([x.shape[0] == data.shape[0] for x, data in zip(xs, datas)])

    return pca, xs


def interpolate_data(data, mask):
    """
    Interpolate over missing entries
    """
    assert data.shape == mask.shape and mask.dtype == bool
    T, N = data.shape
    interp_data = data.copy()
    if np.any(~mask):
        for n in range(N):
            if np.sum(mask[:,n]) >= 2:
                t_missing = np.arange(T)[~mask[:,n]]
                t_given = np.arange(T)[mask[:,n]]
                y_given = data[mask[:,n], n]
                interp_data[~mask[:,n], n] = np.interp(t_missing, t_given, y_given)
            else:
                # Can't do much if we don't see anything... just set it to zero
                interp_data[~mask[:,n], n] = 0
    return interp_data


def trend_filter(data):
    """
    Subtract a linear trend from the data
    """
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    T = data.shape[0]
    lr.fit(np.arange(T)[:, None], data)
    trend = lr.predict(np.arange(T)[:, None])
    return data - trend

def standardize(data, mask): 
    data2 = data.copy()
    data2[~mask] = np.nan
    m = np.nanmean(data2, axis=0)
    s = np.nanstd(data2, axis=0)
    s[~np.any(mask, axis=0)] = 1
    y = (data - m) / s
    y[~mask] = 0
    assert np.all(np.isfinite(y))
    return y

def mixture_of_gaussian_em(data, Q, init_params=None, weights=None, num_iters=100):
    """
    Use expectation-maximization (EM) to compute the maximum likelihood
    estimate of the parameters of a Gaussian mixture model.  The datapoints
    x_i are assumed to come from the following model:
        
        z_i ~ Cate(pi) 
        x_i | z_i ~ N(mu_{z_i}, Sigma_{z_i})
        
    the parameters are {pi_q, mu_q, Sigma_q} for q = 1...Q 
    
    Assume:
        - data x_i are vectors in R^M
        - covariance is diagonal S_q = diag([S_{q1}, .., S_{qm}])
    """
    N, M = data.shape  ### concatenate all marks; N = # of spikes, M = # of mark dim
    
    if init_params is not None:
        pi, mus, inv_sigmas = init_params
        assert pi.shape == (Q,)
        assert np.all(pi >= 0) and np.allclose(pi.sum(), 1)
        assert mus.shape == (M, Q)
        assert inv_sigmas.shape == (M, Q)
    else:
        pi = np.ones(Q) / Q
        mus = npr.randn(M,Q)
        inv_sigmas = -2 + npr.randn(M,Q)
        
    if weights is not None:
        assert weights.shape == (N,) and np.all(weights >= 0)
    else:
        weights = np.ones(N)
        
    for itr in range(num_iters):
        ## E-step:
        ## output: number of spikes by number of mixture
        ## attribute spikes to each Q element
        sigmas = np.exp(inv_sigmas)
        responsibilities = np.zeros((N, Q))
        responsibilities += np.log(pi)
        for q in range(Q):
            responsibilities[:, q] = np.sum(-0.5 * (data - mus[None, :, q])**2 / sigmas[None, :, q] - 0.5 * np.log(2 * np.pi * sigmas[None, :, q]), axis=1) 
            # norm.logpdf(...)
            
        responsibilities -= logsumexp(responsibilities, axis=1, keepdims=True)
        responsibilities = np.exp(responsibilities)
        
        ## M-step:
        ## take in responsibilities (output of e-step)
        ## compute MLE of Gaussian parameters
        ## mean/std is weighted means/std of mix
        for q in range(Q):
            pi[q] = np.average(responsibilities[:, q])
            mus[:, q] = np.average(data, weights=responsibilities[:, q] * weights, axis=0)
            sqerr = (data - mus[None, :, q])**2
            inv_sigmas[:, q] = np.log(1e-8 + np.average(sqerr, weights=responsibilities[:, q] * weights, axis=0))
            
    return mus, inv_sigmas, pi
