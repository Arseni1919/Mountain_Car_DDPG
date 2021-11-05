from GLOBALS import *


def moving_average(x, w):
    w = int(w)
    return np.convolve(x, np.ones(w), 'valid') / w

"""
# H(P, Q) = H(P) + KL(P || Q)
# Where H(P, Q) is the cross-entropy of Q from P, 
# H(P) is the entropy of P and 
# KL(P || Q) is the divergence of Q from P.
"""


# calculate entropy H(P)
def entropy(p):
    return -sum([p[i] * log(p[i]) for i in range(len(p))])


# calculate cross entropy
def cross_entropy(p, q):
    return -sum(pp * log(qq) for pp, qq in zip(p, q))


# calculate the kl divergence KL(P || Q)
def kl_divergence(p, q):
    return sum(p[i] * log(p[i]/q[i]) for i in range(len(p)))
