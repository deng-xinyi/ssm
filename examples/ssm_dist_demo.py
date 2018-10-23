### distance dependent state transition demo
### to-do
### 1. observation: mark
### 2. observation: pseudolikelihood

import autograd.numpy as np
import autograd.numpy.random as npr

import matplotlib.pyplot as plt
#matplotlib inline

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

from ssm.models import HMM
from ssm.util import find_permutation

# Set the parameters of the HMM
T = 200    # number of time bins
K = 10       # number of discrete states
D = 4       # data dimension
npr.seed(1)


##### #############################
#### benchmarking 1
##### #############################

# Make an HMM
true_hmm = HMM(K, D, observations="gaussian")

# Sample some data from the HMM
z, y = true_hmm.sample(T)
true_ll = true_hmm.log_probability(y)

# Plot it
plt.figure(figsize=(8, 4))
plt.subplot(211)
plt.imshow(z[None,:], aspect="auto")
plt.xlim(0, 200)
plt.ylabel("$z$")
plt.yticks([])

plt.subplot(212)
plt.plot(y + 3 * np.arange(D), '-k')
plt.xlim(0, 200)
plt.ylabel("$y$")
# plt.yticks([])
plt.xlabel("time")


## ######
# Fit an HMM to this synthetic data
N_iters = 200
hmm = HMM(K, D, transitions="distance", observations="gaussian")
hmm_lls = hmm.fit(y, method="em", num_em_iters=N_iters)

plt.figure(figsize=(8, 4))
plt.plot(hmm_lls, label="EM")
plt.plot([0, N_iters], true_ll * np.ones(2), ':k', label="True")
plt.xlabel("EM Iteration")
plt.ylabel("Log Probability")
plt.legend(loc="lower right")

# Find a permutation of the states that best matches the true and inferred states
hmm.permute(find_permutation(z, hmm.most_likely_states(y)))
### AssertionError ssm\util\line 41

# Plot the true and inferred discrete states
hmm_z = hmm.most_likely_states(y)

plt.figure(figsize=(8, 8))
plt.subplot(211)
plt.imshow(z[None,:], aspect="auto")
plt.xlim(0, 200)
plt.ylabel("$z_{\\mathrm{true}}$")
plt.yticks([])

plt.subplot(212)
plt.imshow(hmm_z[None,:], aspect="auto")
plt.xlim(0, 200)
plt.ylabel("$z_{\\mathrm{inferred}}$")
plt.yticks([])
plt.xlabel("time")

### fitting L
L_hmm = hmm.params[1][0]  # length scale/smoothing
ell_hmm = hmm.params[1][1] # latent 2D space
log_p_hmm = hmm.params[1][2]

prr_hmm = np.exp(log_p_hmm) # diagonal prob
Ps_dist_hmm = np.sum((ell_hmm[:, :, None] - ell_hmm[:, :, None].T) ** 2, axis = 1)
log_P_hmm = -Ps_dist_hmm / L_hmm
log_P_hmm += np.diag(log_p_hmm)
Ps_hmm = np.exp(log_P_hmm)
Ps_hmm /= Ps_hmm.sum(axis=1, keepdims=True)

# Plot it
plt.figure(figsize=(5, 5))
plt.scatter(ell_hmm[:,0], ell_hmm[:,1], c=range(K), cmap="summer")
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title("latent 2D space, L = %.2f" % L_hmm, fontsize=20)
for spine in plt.gca().spines.values():
    spine.set_visible(False)

##### #############################
#### benchmarking 2
##### #############################

# Make an HMM
true_hmm = HMM(K, D, transitions="distance", observations="poisson")

# Sample some data from the HMM
z, y = true_hmm.sample(T)
true_ll = true_hmm.log_probability(y)

## get L, ell, log_p, Ps
### fitting L
L = true_hmm.params[1][0]  # length scale/smoothing
ell = true_hmm.params[1][1] # latent 2D space
log_p = true_hmm.params[1][2]

### fixed L
#L = 0.1
#ell= true_hmm.params[1][0]
#log_p = true_hmm.params[1][1]

prr = np.exp(log_p) # diagonal prob
Ps_dist = np.sqrt(np.sum((ell[:, :, None] - ell[:, :, None].T) ** 2, axis = 1))
log_P = -Ps_dist / L
log_P += np.diag(log_p)
Ps = np.exp(log_P)
Ps /= Ps.sum(axis=1, keepdims=True)
#
#
## Plot it
#fig = plt.figure(figsize=(10,10))
#plt.scatter(ell[:,0], ell[:,1], c=range(K), cmap="summer", s=300)
#plt.xlim(-3,3)
#plt.ylim(-3,3)
#plt.xticks(fontsize=24)
#plt.yticks(fontsize=24)
#plt.title("latent 2D space",fontsize=32)
#for spine in plt.gca().spines.values():
#    spine.set_visible(False)
#plt.savefig('dist_map.jpg',dpi=100)
#
#
#fig = plt.figure(figsize=(20, 10))
#
#plt.subplot(121)
#plt.imshow(Ps, cmap="spring_r", aspect="equal")
#plt.xlim(0,K-1)
#plt.ylim(0,K-1)
#plt.yticks([0,2,4,6,8], fontsize=24)
#plt.xticks([0,2,4,6,8], fontsize=24)
#plt.title("state transition", fontsize=32)
#plt.xlabel("$z_{\\mathrm{true},t+1}$", fontsize=32)
#plt.ylabel("$z_{\\mathrm{true},t}$", fontsize=32)
#for spine in plt.gca().spines.values():
#    spine.set_visible(False)
#
#plt.subplot(222)
#plt.imshow(z[None,:], cmap="summer", aspect="auto")
#plt.xlim(0, 200)
#plt.yticks(fontsize=24)
#plt.xticks(fontsize=24)
#plt.title("state sequence", fontsize=32)
#plt.ylabel("$z_{\\mathrm{true}}$", fontsize=32)
#plt.yticks([])
#for spine in plt.gca().spines.values():
#    spine.set_visible(False)
#
#plt.subplot(224)
#plt.plot(y + 3 * np.arange(D), '-')
#plt.xlim(0, 200)
#plt.yticks(fontsize=24)
#plt.xticks(fontsize=24)
#plt.ylabel("spiking activity", fontsize=32)
#plt.xlabel("Time (sec)", fontsize=32)
#for spine in plt.gca().spines.values():
#    spine.set_visible(False)
#
#plt.savefig('dist_L=1.jpg',dpi=100)



### ### ### ### ### ###
## ## fit ## ##
### ### ### ### ### ###


# fit an HMM to this synthetic data
N_iters = 200
hmm = HMM(K, D, transitions="distance", observations="poisson")
hmm_lls = hmm.fit(y, method="em", num_em_iters=N_iters)
#hmm_lls = hmm.fit(y)

plt.figure(figsize=(8, 4))
plt.plot(hmm_lls, label="EM")
plt.plot([0, N_iters], true_ll * np.ones(2), ':k', label="True")
plt.xlim(0, N_iters)
plt.xlabel("EM Iteration")
plt.ylabel("Log Probability")
plt.legend(loc="lower right")

# Find a permutation of the states that best matches the true and inferred states
hmm.permute(find_permutation(z, hmm.most_likely_states(y)))
### AssertionError ssm\util\line 41

# Plot the true and inferred discrete states
z_hmm = hmm.most_likely_states(y)

plt.figure(figsize=(8, 4))
plt.subplot(211)
plt.imshow(z[None,:], aspect="auto", cmap="summer")
plt.xlim(0, 200)
plt.ylabel("$z_{\\mathrm{true}}$")
plt.yticks([])

plt.subplot(212)
plt.imshow(z_hmm[None,:], aspect="auto", cmap="summer")
plt.xlim(0, 200)
plt.ylabel("$z_{\\mathrm{inferred}}$")
plt.yticks([])
plt.xlabel("time")


# get L, ell, log_p, Ps
### fitting L
L_hmm = hmm.params[1][0]  # length scale/smoothing
ell_hmm = hmm.params[1][1] # latent 2D space
log_p_hmm = hmm.params[1][2]

### fixed L
#L_hmm = 0.1
#ell_hmm = hmm.params[1][0]
#log_p_hmm = hmm.params[1][1]


prr_hmm = np.exp(log_p_hmm) # diagonal prob
Ps_dist_hmm = np.sum((ell_hmm[:, :, None] - ell_hmm[:, :, None].T) ** 2, axis = 1)
log_P_hmm = -Ps_dist_hmm / L_hmm
log_P_hmm += np.diag(log_p_hmm)
Ps_hmm = np.exp(log_P_hmm)
Ps_hmm /= Ps_hmm.sum(axis=1, keepdims=True)


# Plot it
plt.figure(figsize=(5, 5))
plt.scatter(ell_hmm[:,0], ell_hmm[:,1], c=range(K), cmap="summer")
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title("latent 2D space, L = %.2f" % L_hmm, fontsize=20)
for spine in plt.gca().spines.values():
    spine.set_visible(False)


plt.figure(figsize=(10, 8))
plt.subplot(221)
plt.imshow(Ps, aspect="equal", cmap="spring_r")
plt.xlim(0,K-1)
plt.ylim(0,K-1)
plt.title("true",fontsize=20)
for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.subplot(222)
plt.imshow(Ps_hmm, aspect="equal", cmap="spring_r")
plt.xlim(0,K-1)
plt.ylim(0,K-1)
plt.title("inferred",fontsize=20)
for spine in plt.gca().spines.values():
    spine.set_visible(False)

# compare transition matrix
def transition_matrix(transitions):
    n = 1+ max(transitions) #number of states

    A = [[0]*n for _ in range(n)]

    for (i,j) in zip(transitions,transitions[1:]):
        A[i][j] += 1

    #now convert to probabilities:
    for row in A:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return A

A = transition_matrix(z)
A_hmm = transition_matrix(z_hmm)

plt.subplot(223)
plt.title("true empirical",fontsize=20)
plt.imshow(A, aspect="equal", interpolation="none", cmap="spring_r", vmin=0, vmax=0.2)
plt.xlim(0, K-1); plt.ylim(0, K-1)
plt.xlabel("$z_{\\mathrm{true},t+1}$",fontsize=20)
plt.ylabel("$z_{\\mathrm{true},t}$",fontsize=20)
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    
plt.subplot(224)
plt.title("inferred empirical",fontsize=20)
plt.imshow(A_hmm, aspect="equal", interpolation="none", cmap="spring_r", vmin=0, vmax=0.2)
plt.xlim(0, K-1); plt.ylim(0,K-1)
plt.xlabel("$z_{\\mathrm{inferred},t+1}$",fontsize=20)
plt.ylabel("$z_{\\mathrm{inferred},t}$",fontsize=20)
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    