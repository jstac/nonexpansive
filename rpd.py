"""

RPD applied to correlated shock model
Author: John Stachurski     

"""

from __future__ import division
import numpy as np
from scipy.stats import beta
from scipy.optimize import fminbound



class PWC:

    def __init__(self, xgrid, ygrid, vals):
        self.xgrid, self.ygrid, self.vals = xgrid, ygrid, vals

    def __call__(self, xpoints, ypoints):
        xindices = self.xgrid.searchsorted(xpoints, side='right') - 1
        yindices = self.ygrid.searchsorted(ypoints, side='right') - 1
        return self.vals[xindices, yindices]


######################### Primitives ######################################

RHO = 0.9     # Discount rate
LAMBDA = 0.7  # Carryover depreciation parameter
THETA = 0.3   # Parameter in the shock process
ALPHA = 0.2   # Utility parameter

# Innovation is a + b W, where W is beta(5,5)
a = 1; b = 2
B = beta(5, 5, loc=a, scale=b)
G = B.cdf    # G is the cdf of this distribution
W = B.rvs(1000)
TW = (1 - THETA) * W

########################### Grid #########################################

# Next we set out a grid on the state space S x H, where S is values for
# the first state variable (supply), and H is values for the shock.

SHOCK_UB = a + b                   # Upper bound of the innovation
SHOCK_LB = a                       # Lower bound of the innovation
J = 200                            # Number of elements in S-grid
K = 10                             # Number of elements in H-grid
S_UPPER = SHOCK_UB / (1 - LAMBDA)  # Upper bound of S
S_LOWER = SHOCK_LB                 # Lower bound of S
S = np.linspace(S_LOWER, S_UPPER, J)
H = np.linspace(SHOCK_LB, SHOCK_UB, K)


def T(v):
    Tv = np.empty((J-1, K-1))
    for j in range(J-1):
        for k in range(K-1):
            nh = THETA * H[k] + TW
            ob = lambda q: - (S[j] - q)**ALPHA - \
                RHO * np.mean(v(LAMBDA * q + nh, nh))
            Tv[j, k] = ob(fminbound(ob, 0, S[j]))
    return PWC(S, H, Tv)

########################## main loop ######################################

v = PWC(S, H, np.zeros((J-1, K-1)))

