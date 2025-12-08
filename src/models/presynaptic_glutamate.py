import numpy as np

class GlutamateDynamics:
    """
    Tewari & Majumdar (2012) – Glutamate Release (Equations 6, 7, 8)
    Fully paper-accurate implementation. NO MATLAB assumptions.
    """

    def __init__(self, params):
        self.p = params

        # State variables for Ca2+ sensor (Eq. 6)
        self.s0 = 1.0
        self.s1 = 0.0
        self.s2 = 0.0
        self.s3 = 0.0
        self.s4 = 0.0
        self.s5 = 0.0
        self.s_star = 0.0    # X(ci)5* – isomer ready for release

        # Vesicle fractions (Eq. 8)
        self.R = 1.0     # releasable
        self.E = 0.0     # effective (released into cleft)
        self.g = 0.0     # glutamate in cleft (µM)

    # ----------------------------------------------------------------------
    # Main update step
    # ----------------------------------------------------------------------
    def step(self, dt, c_i):
        p = self.p

        # ------------- A) Ca2+ Sensor Kinetics (Eq. 6) ---------------------
        # NO UNIT CLIPPING! c_i MUST BE µM.
        c = max(c_i, 0.0)

        # Forward and backward fluxes
        j01 = 5 * p['alpha'] * c * self.s0
        j10 = 1 * p['beta']  * self.s1

        j12 = 4 * p['alpha'] * c * self.s1
        j21 = 2 * p['beta']  * self.s2

        j23 = 3 * p['alpha'] * c * self.s2
        j32 = 3 * p['beta']  * self.s3

        j34 = 2 * p['alpha'] * c * self.s3
        j43 = 4 * p['beta']  * self.s4

        j45 = 1 * p['alpha'] * c * self.s4
        j54 = 5 * p['beta']  * self.s5

        # Isomerization (γ forward, δ backward)
        j_f_star = p['gamma'] * self.s5
        j_b_star = p['delta'] * self.s_star

        # Sensor state derivatives
        ds0 = j10 - j01
        ds1 = j01 + j21 - j10 - j12
        ds2 = j12 + j32 - j21 - j23
        ds3 = j23 + j43 - j32 - j34
        ds4 = j34 + j54 - j43 - j45
        ds5 = j45 + j_b_star - j54 - j_f_star
        ds_star = j_f_star - j_b_star

        # --------------------- B) Spontaneous Release Rate (Eq. 7) -------------------
        # λ(ci) = a3 * (1 + exp((a1 - c_i)/a2))^-1
        exponent = (p['a1'] - c) / p['a2']
        lambda_spont = p['a3'] / (1.0 + np.exp(exponent))

        # --------------------- C) Evoked Release Rate (from Eq. 6) -------------------
        # Evoked release depends ONLY on γ * s_star — NO 2000 factor!!
        rate_evoked = p['gamma'] * self.s_star

        # Total release rate
        f_r = lambda_spont + rate_evoked

        # --------------------- D) Vesicle Cycle (Eq. 8) ---------------------------
        I = 1.0 - self.R - self.E

        dR = (I / p['tau_rec']) - (f_r * self.R)
        dE = -(self.E / p['tau_inac']) + (f_r * self.R)

        # --------------------- E) Glutamate in cleft -----------------------------
        # dg/dt = n_v * g_v * E - g_c * g
        dg = (p['n_v'] * p['g_v'] * self.E) - (p['g_c'] * self.g)

        # --------------------- F) Integrate and clip ----------------------------
        self.s0 += dt * ds0
        self.s1 += dt * ds1
        self.s2 += dt * ds2
        self.s3 += dt * ds3
        self.s4 += dt * ds4
        self.s5 += dt * ds5
        self.s_star += dt * ds_star

        # Probabilities must remain [0,1]
        for attr in ['s0', 's1', 's2', 's3', 's4', 's5', 's_star']:
            setattr(self, attr, np.clip(getattr(self, attr), 0.0, 1.0))

        self.R += dt * dR
        self.E += dt * dE
        self.g += dt * dg

        # Physical bounds
        self.R = np.clip(self.R, 0.0, 1.0)
        self.E = np.clip(self.E, 0.0, 1.0)
        self.g = max(0.0, self.g)

        return self.g
