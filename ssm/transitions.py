from functools import partial
from warnings import warn

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd.scipy.stats import dirichlet
from autograd.misc.optimizers import sgd, adam
from autograd import grad

from ssm.util import ensure_args_are_lists, ensure_args_not_none, \
    ensure_elbo_args_are_lists, adam_with_convergence_check, one_hot, \
    logistic, relu

class _Transitions(object):
    def __init__(self, K, D, M=0):
        self.K, self.D, self.M = K, D, M

    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, value):
        raise NotImplementedError

    def initialize(self, datas, inputs, masks, tags):
        pass

    def permute(self, perm):
        pass

    def log_prior(self):
        return 0

    def log_transition_matrices(self, data, input, mask, tag):
        raise NotImplementedError

    def m_step(self, expectations, datas, inputs, masks, tags, optimizer="adam", num_iters=100, **kwargs):
        """
        If M-step cannot be done in closed form for the transitions, default to SGD.
        """
        optimizer = dict(sgd=sgd, adam=adam, adam_with_convergence_check=adam_with_convergence_check)[optimizer]

        # expected log joint
        def _expected_log_joint(expectations):
            elbo = self.log_prior()
            for data, input, mask, tag, (expected_states, expected_joints, _) \
                in zip(datas, inputs, masks, tags, expectations):
                log_Ps = self.log_transition_matrices(data, input, mask, tag)
                elbo += np.sum(expected_joints * log_Ps)
            return elbo

        # define optimization target
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = _expected_log_joint(expectations)
            return -obj / T

        self.params = optimizer(grad(_objective), self.params, num_iters=num_iters, **kwargs)


class StationaryTransitions(_Transitions):
    """
    Standard Hidden Markov Model with fixed initial distribution and transition matrix.
    """
    def __init__(self, K, D, M=0):
        super(StationaryTransitions, self).__init__(K, D, M=M)
        Ps = .95 * np.eye(K) + .05 * npr.rand(K, K)
        Ps /= Ps.sum(axis=1, keepdims=True)
        self.log_Ps = np.log(Ps)

    @property
    def params(self):
        return (self.log_Ps,)

    @params.setter
    def params(self, value):
        self.log_Ps = value[0]

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.log_Ps = self.log_Ps[np.ix_(perm, perm)]

    @property
    def transition_matrix(self):
        return np.exp(self.log_Ps - logsumexp(self.log_Ps, axis=1, keepdims=True))

    def log_transition_matrices(self, data, input, mask, tag):
        T = data.shape[0]
        log_Ps = np.tile(self.log_Ps[None, :, :], (T-1, 1, 1))
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        P = sum([np.sum(Ezzp1, axis=0) for _, Ezzp1, _ in expectations]) + 1e-16
        P /= P.sum(axis=-1, keepdims=True)
        self.log_Ps = np.log(P)


class StickyTransitions(StationaryTransitions):
    """
    Upweight the self transition prior.

    pi_k ~ Dir(alpha + kappa * e_k)
    """
    def __init__(self, K, D, M=0, alpha=1, kappa=100):
        super(StickyTransitions, self).__init__(K, D, M=M)
        self.alpha = alpha
        self.kappa = kappa

    def log_prior(self):
        K = self.K
        Ps = np.exp(self.log_Ps - logsumexp(self.log_Ps, axis=1, keepdims=True))

        lp = 0
        for k in range(K):
            alpha = self.alpha * np.ones(K) + self.kappa * (np.arange(K) == k)
            lp += dirichlet.logpdf(Ps[k], alpha)
        return lp

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        expected_joints = sum([np.sum(Ezzp1, axis=0) for _, Ezzp1, _ in expectations]) + 1e-8
        expected_joints += self.kappa * np.eye(self.K)
        P = expected_joints / expected_joints.sum(axis=1, keepdims=True)
        self.log_Ps = np.log(P)


class InputDrivenTransitions(_Transitions):
    """
    Hidden Markov Model whose transition probabilities are
    determined by a generalized linear model applied to the
    exogenous input.
    """
    def __init__(self, K, D, M):
        super(InputDrivenTransitions, self).__init__(K, D, M=M)

        # Baseline transition probabilities
        Ps = .95 * np.eye(K) + .05 * npr.rand(K, K)
        Ps /= Ps.sum(axis=1, keepdims=True)
        self.log_Ps = np.log(Ps)

        # Parameters linking input to state distribution
        self.Ws = npr.randn(K, M)

    @property
    def params(self):
        return self.log_Ps, self.Ws

    @params.setter
    def params(self, value):
        self.log_Ps, self.Ws = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.log_Ps = self.log_Ps[np.ix_(perm, perm)]
        self.Ws = self.Ws[perm]

    def log_transition_matrices(self, data, input, mask, tag):
        T = data.shape[0]
        assert input.shape[0] == T
        # Previous state effect
        log_Ps = np.tile(self.log_Ps[None, :, :], (T-1, 1, 1))
        # Input effect
        log_Ps = log_Ps + np.dot(input[1:], self.Ws.T)[:, None, :]
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)


class RecurrentTransitions(InputDrivenTransitions):
    """
    Generalization of the input driven HMM in which the observations serve as future inputs
    """
    def __init__(self, K, D, M=0, solver="lbfgs"):
        super(RecurrentTransitions, self).__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.Rs = npr.randn(K, D)

        # Store a scikit learn logistic regression object for warm starting
        from sklearn.linear_model import LogisticRegression
        self._lr = LogisticRegression(
            fit_intercept=False, multi_class="multinomial", solver=solver, warm_start=True)

    @property
    def params(self):
        return super(RecurrentTransitions, self).params + (self.Rs,)

    @params.setter
    def params(self, value):
        self.Rs = value[-1]
        super(RecurrentTransitions, self.__class__).params.fset(self, value[:-1])

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        super(RecurrentTransitions, self).permute(perm)
        self.Rs = self.Rs[perm]

    def log_transition_matrices(self, data, input, mask, tag):
        T, D = data.shape
        # Previous state effect
        log_Ps = np.tile(self.log_Ps[None, :, :], (T-1, 1, 1))
        # Input effect
        log_Ps = log_Ps + np.dot(input[1:], self.Ws.T)[:, None, :]
        # Past observations effect
        log_Ps = log_Ps + np.dot(data[:-1], self.Rs.T)[:, None, :]
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        """
        Fit a logistic regression for the transitions.

        Technically, this is a stochastic M-step since the states
        are sampled from their posterior marginals.
        """

        K, M, D = self.K, self.M, self.D

        zps, zns = [], []
        for Ez, _, _ in expectations:
            z = np.array([np.random.choice(K, p=p) for p in Ez])
            zps.append(z[:-1])
            zns.append(z[1:])

        X = np.vstack([np.hstack((one_hot(zp, K), input[1:], data[:-1]))
                       for zp, input, data in zip(zps, inputs, datas)])
        y = np.concatenate(zns)

        # Determine the number of states used
        used = np.unique(y)
        K_used = len(used)
        unused = np.setdiff1d(np.arange(K), used)

        # Reset parameters before filling in
        self.log_Ps = np.zeros((K, K))
        self.Ws = np.zeros((K, M))
        self.Rs = np.zeros((K, D))

        if K_used == 1:
            warn("RecurrentTransitions: Only using 1 state in expectation. "
                 "M-step cannot proceed. Resetting transition parameters.")
            return

        # Fit the logistic regression
        self._lr.fit(X, y)

        # Extract the coefficients
        assert self._lr.coef_.shape[0] == (K_used if K_used > 2 else 1)
        log_P = self._lr.coef_[:, :K]
        W = self._lr.coef_[:, K:K+M]
        R = self._lr.coef_[:, K+M:]

        if K_used == 2:
            # lr thought there were only two classes
            self.log_Ps[:,used[1]] = self._lr.coef_[0, :K]
            self.Ws[used[1]] = self._lr.coef_[0,K:K+M]
            self.Rs[used[1]] = self._lr.coef_[0,K+M:]
        else:
            self.log_Ps[:, used] = log_P.T
            self.Ws[used] = W
            self.Rs[used] = R


class RecurrentOnlyTransitions(_Transitions):
    """
    Only allow the past observations and inputs to influence the
    next state.  Get rid of the transition matrix and replace it
    with a constant bias r.
    """
    def __init__(self, K, D, M=0,  solver="lbfgs"):
        super(RecurrentOnlyTransitions, self).__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.Ws = npr.randn(K, M)
        self.Rs = npr.randn(K, D)
        self.r = npr.randn(K)

        # Store a scikit learn logistic regression object for warm starting
        from sklearn.linear_model import LogisticRegression
        self._lr = LogisticRegression(
            fit_intercept=False, multi_class="multinomial", solver=solver, warm_start=True)

    @property
    def params(self):
        return self.Ws, self.Rs, self.r

    @params.setter
    def params(self, value):
        self.Ws, self.Rs, self.r = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.Ws = self.Ws[perm]
        self.Rs = self.Rs[perm]
        self.r = self.r[perm]

    def log_transition_matrices(self, data, input, mask, tag):
        T, D = data.shape
        log_Ps = np.dot(input[1:], self.Ws.T)[:, None, :]              # inputs
        log_Ps = log_Ps + np.dot(data[:-1], self.Rs.T)[:, None, :]     # past observations
        log_Ps = log_Ps + self.r                                       # bias
        log_Ps = np.tile(log_Ps, (1, self.K, 1))                       # expand
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)       # normalize


    def m_step(self, expectations, datas, inputs, masks, tags, optimizer="adam", num_iters=10, **kwargs):
        """
        Fit a logistic regression for the transitions.

        Technically, this is a stochastic M-step since the states
        are sampled from their posterior marginals.
        """
        K, M, D = self.K, self.M, self.D

        zps, zns = [], []
        for Ez, _, _ in expectations:
            z = np.array([np.random.choice(K, p=p) for p in Ez])
            zps.append(z[:-1])
            zns.append(z[1:])


        X = np.vstack([np.hstack((input[1:], data[:-1]))
                       for input, data in zip(inputs, datas)])
        y = np.concatenate(zns)

        # Identify used states
        used = np.unique(y)
        K_used = len(used)
        unused = np.setdiff1d(np.arange(K), used)

        # Reset parameters before filling in
        self.Ws = np.zeros((K, M))
        self.Rs = np.zeros((K, D))
        self.r = np.zeros((K,))

        if K_used == 1:
            warn("RecurrentOnlyTransitions: Only using 1 state in expectation. "
                 "M-step cannot proceed. Resetting transition parameters.")
            return

        # Fit the logistic regression
        self._lr.fit(X, y)

        # Extract the coefficients
        assert self._lr.coef_.shape[0] == (K_used if K_used > 2 else 1)
        if K_used == 2:
            # lr thought there were only two classes
            self.Ws[used[1]] = self._lr.coef_[0, :M]
            self.Rs[used[1]] = self._lr.coef_[0, M:]
        else:
            self.Ws[used] = self._lr.coef_[:, :M]
            self.Rs[used] = self._lr.coef_[:, M:]

        # Set the intercept
        self.r[used] = self._lr.intercept_



# Allow general nonlinear emission models with neural networks
class NeuralNetworkRecurrentTransitions(_Transitions):
    def __init__(self, K, D, M=0, hidden_layer_sizes=(50,), nonlinearity="relu"):
        super(NeuralNetworkRecurrentTransitions, self).__init__(K, D, M=M)

        # Baseline transition probabilities
        Ps = .95 * np.eye(K) + .05 * npr.rand(K, K)
        Ps /= Ps.sum(axis=1, keepdims=True)
        self.log_Ps = np.log(Ps)

        # Initialize the NN weights
        layer_sizes = (D + M,) + hidden_layer_sizes + (K,)
        self.weights = [npr.randn(m, n) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [npr.randn(n) for n in layer_sizes[1:]]

        nonlinearities = dict(
            relu=relu,
            tanh=np.tanh,
            sigmoid=logistic)
        self.nonlinearity = nonlinearities[nonlinearity]

    @property
    def params(self):
        return self.log_Ps, self.weights, self.biases

    @params.setter
    def params(self, value):
        self.log_Ps, self.weights, self.biases = value

    def permute(self, perm):
        self.log_Ps = self.log_Ps[np.ix_(perm, perm)]
        self.weights[-1] = self.weights[-1][:,perm]
        self.biases[-1] = self.biases[-1][perm]

    def log_transition_matrices(self, data, input, mask, tag):
        # Pass the data and inputs through the neural network
        x = np.hstack((data[:-1], input[1:]))
        for W, b in zip(self.weights, self.biases):
            y = np.dot(x, W) + b
            x = self.nonlinearity(y)

        # Add the baseline transition biases
        log_Ps = self.log_Ps[None, :, :] + y[:, None, :]

        # Normalize
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)


class DistanceDependentTransitions(_Transitions):
    """
    In this model the transition probability depends on the distance
    between the latent locations associated with each discrete state.

    Specifically, each state k has a location, \ell_k \in R^J, and
    the probability of transitioning is

        Pr(z_t = k | z_{t-1} = k')
             \propto exp(-||\ell_k - \ell_{k'}||/L)   if k \neq k'
             \propto p_kk                             if k = k'

    where L is a length scale that we treat as a hyperparameter.
    """
    def __init__(self, K, D, M=0, L=1.0, J=2):
        """
        K: number of discrete states (integer)
        D: data dimension (unused)
        M: input dimension (unused)
        L: length scale (positive real)
        J: latent embedding dimension (integer)
        """
        super(DistanceDependentTransitions, self).__init__(K, D, M=M)
        self.L = L

#        """
#        Upweight the self transition prior.
#
#        pi_k ~ Dir(alpha + kappa * e_k)
#        """
#        self.alpha = alpha
#        self.kappa = kappa        
        
        ### Initialize the parameters for custom simulation
#        ell = np.zeros((K, J))
#        for k in range(K):
#            ell[k, 0] = 2 * np.cos(k * 2 * np.pi / K)
#            ell[k, 1] = 2 * np.sin(k * 2 * np.pi / K)
#        self.ell = ell
#        self.log_p = npr.uniform(low=-1, high=0, size=(K,))
        
        ### Initialize a Y topology
#        ell = np.zeros((K, J))
#        for k in [0, 1, 2, 3, 4]:
#            ell[k, 0] = 0
#            ell[k, 1] = 4 - k
#        for k in [5, 6, 7, 8, 9, 10]:
#            ell[k, 0] = (4 - k) * 0.5 * np.sqrt(2)
#            ell[k, 1] = (4 - k) * 0.5 * np.sqrt(2)
#        for k in [11, 12, 13, 14, 15, 16]:
#            ell[k, 0] = (k - 10) * 0.5 * np.sqrt(2)
#            ell[k, 1] = (10 - k) * 0.5 * np.sqrt(2)
#        
#        self.ell = ell
#        self.log_p = np.zeros(K)
        
        ### random initialization
        self.ell = npr.randn(K, J)
        self.log_p = np.zeros(K)

    @property
    def params(self):
        return self.ell, self.log_p

    @params.setter
    def params(self, value):
        self.ell, self.log_p = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.ell = self.ell[perm]
        self.log_p = self.log_p[perm]
    
    @property
    def log_transition_matrix(self):
        Ps_dist = np.sum((self.ell[None, :, :] - self.ell[:, None, :]) ** 2,
                         axis = 2)        
        log_Ps = -Ps_dist / self.L
        log_Ps += np.diag(self.log_p)
        assert np.all(np.isfinite(log_Ps))
        # Normalize and return
        return log_Ps - logsumexp(log_Ps, axis=1, keepdims=True)
    
    @property
    def transition_matrix(self):
        return np.exp(self.log_transition_matrix)

    def log_transition_matrices(self, data, input, mask, tag):
        T = data.shape[0]
        # Get the normalized transition matrix
        log_Ps = self.log_transition_matrix
        # Tile the transition matrix for each time step
        return np.tile(log_Ps[None, :, :], (T-1, 1, 1))
    
#    def log_prior(self):
#        K = self.K
#        Ps = np.exp(self.log_transition_matrix)
#
#        lp = 0
#        for k in range(K):
#            alpha = self.alpha * np.ones(K) + self.kappa * (np.arange(K) == k)
#            lp += dirichlet.logpdf(Ps[k], alpha)
#        return lp
        
class DistanceDependentMazeTransitions(_Transitions):
    """
    In this model the transition probability depends on the _empirical_
    distance between two locations on a _known_ maze environment.

    Specifically, each state k has a location on a known maze \in R^J, 
    with label_k, and the probability of transitioning is

        Pr(z_t = k | z_{t-1} = k')
             \propto exp[-d(label_k, label_k')/L]   if k \neq k'
             \propto p_kk                             if k = k'

    where L is a length scale that we treat as a hyperparameter.
    """
#    pass
    
    def __init__(self, K, D, M=0, L=1.0, J=2):
        """
        K: number of discrete states (integer)
        D: data dimension (unused)
        M: input dimension (unused)
        L: length scale (positive real)
        J: latent embedding dimension (integer)
        """
        super(DistanceDependentMazeTransitions, self).__init__(K, D, M=M)
        self.L = L

        # random initialization
        self.ell_labels = np.arange(K)
        npr.shuffle(self.ell_labels)
        
        self.log_p = np.ones(K)
        
        ### Initialize with a known maze environment
        ell = np.zeros((K, J))
        ell[0, :] = (214.5, 142.9); ell[1, :] = (214.5, 120.25); ell[2, :] = (214.5, 105.15)
        ell[3, :] = (214.5, 90.05); ell[4, :] = (214.5, 69.75); ell[5, :] = (194.7, 69.75)
        ell[6, :] = (234.3, 69.75); ell[7, :] = (174.9, 69.75); ell[8, :] = (254.1, 69.75);
        ell[9, :] = (254.1, 90.05); ell[10, :] = (254.1, 105.15); ell[11, :] = (254.1, 120.25)
        ell[12, :] = (254.1, 142.9); ell[13, :] = (174.9, 90.05); ell[14, :] = (174.9, 105.15)
        ell[15, :] = (174.9, 120.25); ell[16, :] = (174.9, 142.9)
        
        ### look-up table for pairwise empirical distance on maze
        ell_dist = np.zeros((K, K))
        for i in (0, 1, 2, 3, 4):
            for j in (0, 1, 2, 3, 4):
                if i != j:
                    ell_dist[i, j] = np.abs(ell[j, 1] - ell[i, 1])
        for i in (8, 9, 10, 11, 12):
            for j in (8, 9, 10, 11, 12):
                if i != j:
                    ell_dist[i, j] = np.abs(ell[j, 1] - ell[i, 1])        
        for i in (7, 13, 14, 15, 16):
            for j in (7, 13, 14, 15, 16):
                if i != j:
                    ell_dist[i, j] = np.abs(ell[j, 1] - ell[i, 1])      
        for i in (4, 5, 6, 7, 8):
            for j in (4, 5, 6, 7, 8):
                if i != j:
                    ell_dist[i, j] = np.abs(ell[j, 0] - ell[i, 0])  
        for i in (5, 6):
            ell_dist[0, i] = ell_dist[0, 4] + ell_dist[4, i]
        for i in (7, 8):
            ell_dist[0, i] = ell_dist[0, i-2] + ell_dist[i-2, i]
        for i in (9, 10, 11, 12):
            ell_dist[0, i] = ell_dist[0, 8] + ell_dist[8, i]
        for i in (13, 14, 15, 16):
            ell_dist[0, i] = ell_dist[0, 7] + ell_dist[7, i]
        for i in (1, 2, 3, 4):
            for j in (8, 9, 10, 11, 12):
                ell_dist[i, j] = ell_dist[i, 4] + ell_dist[4, 8] + ell_dist[8, j]
        for i in (1, 2, 3, 4):
            for j in (7, 13, 14, 15, 16):
                ell_dist[i, j] = ell_dist[i, 4] + ell_dist[4, 7] + ell_dist[7, j]  
        for i in (1, 2, 3):
            for j in (5, 6):
                ell_dist[i, j] = ell_dist[i, 4] + ell_dist[4, j]   
        for i in (5, 6, 7):
            for j in (9, 10, 11, 12):
                ell_dist[i, j] = ell_dist[i, 8] + ell_dist[8, j]
        for i in (5, 6, 8):
            for j in (13, 14, 15, 16):
                ell_dist[i, j] = ell_dist[i, 7] + ell_dist[7, j]
        for i in (9, 10, 11, 12):
            for j in (13, 14, 15, 16):
                ell_dist[i, j] = ell_dist[i, 7] + ell_dist[7, j]
                
        ell_dist[np.tril_indices(K, -1)] = ell_dist.T[np.tril_indices(K, -1)]  # make the matrix symmetric
        ell_dist_sum = np.sum(ell_dist, axis=1)
        # normalized empirical distances so that comparable with diagonal log_p (~1)
        dist_norm = ell_dist / ell_dist_sum[None, :]
        self.dist_norm = dist_norm

    @property
    def params(self):
        return self.ell_labels, self.log_p

    @params.setter
    def params(self, value):
        self.ell_labels, self.log_p = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.ell_labels = self.ell_labels[perm]
        self.log_p = self.log_p[perm]
    
    @property
    def log_transition_matrix(self):
        
        K, dist_norm = self.K, self.dist_norm
        
        ### given labels, look up pairwise distance from dist_norm
        dist_labeled = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                dist_labeled[i, j] = dist_norm[self.ell_labels[i],  self.ell_labels[j]]
                
        log_Ps = -dist_labeled / self.L
        log_Ps += np.diag(self.log_p)
        assert np.all(np.isfinite(log_Ps))
        # Normalize and return
        return log_Ps - logsumexp(log_Ps, axis=1, keepdims=True)
    
    @property
    def transition_matrix(self):
        return np.exp(self.log_transition_matrix)

    def log_transition_matrices(self, data, input, mask, tag):
        T = data.shape[0]
        # Get the normalized transition matrix
        log_Ps = self.log_transition_matrix
        # Tile the transition matrix for each time step
        return np.tile(log_Ps[None, :, :], (T-1, 1, 1))
    
    def m_step(self, expectations, datas, inputs, masks, tags, optimizer="adam", num_iters=10, **kwargs):
        """
        to find most likely labels, ell_labels; coordinate descent
        
            {\label_k} = argmax E_{z~p(z|x)}[log p(z)]
            \likelihood(\theta) = E_{z~p(z|x)}[\sum_{t=1}^T-1 log p(z_{t+1} | z_t; \theta)]

        weights entries are E[z_t = k], E[z_t = k, z_{t+1}=k'], log p(x_{1:T})
        """
        K = self.K
        
        zzps = np.concatenate([Ezzp1 for _, Ezzp1, _ in expectations]) # T by K by K
        
        ell_labels, dist_norm, L, log_p = self.ell_labels, self.dist_norm, self.L, self.log_p
        
        for itr in range(num_iters):

            for k in range(K): 
                ### create null matrix with all possible values for the k-th label
                ###    while fixing all other k-1 labels the same
                ell_labels_new = np.array([ell_labels] * K)
                ell_labels_new[:, k] = np.arange(K) 
                dist_labeled = np.zeros((K, K))
                
                ### for every possible values for the k-th label
                for l in range(K):
                    ell_labels_new_eg = ell_labels_new[l, :]
                    
                    ### compute log_transition matrix
                    log_L = np.zeros(K)
                    for i in range(K):
                        for j in range(K):
                            dist_labeled[i, j] = dist_norm[ell_labels_new_eg[i],  ell_labels_new_eg[j]]
                    log_Ps = -dist_labeled / L
                    log_Ps += np.diag(log_p)
                    log_Ps -= logsumexp(log_Ps, axis=1, keepdims=True)   
                    
                    ### compuate log_likelihood
                    log_L[l] = np.sum(zzps * log_Ps[None, :, :])
                
                ### update k-the label with mle
                ell_labels[k] = np.argmax(log_L)
        
        self.ell_labels = ell_labels