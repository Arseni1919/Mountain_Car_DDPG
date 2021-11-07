from GLOBALS import *


def moving_average(x, w):
    w = int(w)
    if w > 0:
        return np.convolve(x, np.ones(w), 'valid') / w
    return [0]

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
    if len(p.shape) and len(q.shape):
        return sum(p[i] * log(p[i]/q[i]) for i in range(len(p)))
    return 0


def soft_update(target, source, tau, plotter=None):
    # for target_param, param in zip(target_critic.parameters(), critic.parameters()):
    #     target_param.data.copy_(POLYAK * target_param.data + (1.0 - POLYAK) * param.data)
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    if plotter:
        # plotter.plots_update_entropy(source, target, 'critic')
        pass
