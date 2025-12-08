# File: src/models/gliatransmitter.py

import numpy as np

class GliatransmitterDynamics:
    """
    Tewari & Majumdar (2012) – Astrocyte Gliotransmitter Release
    Implements Equations (13), (14), (15) EXACTLY as written.
    """

    def __init__(self, params):
        self.p = params

        # --- Gate opening probabilities (Eq. 13) ---
        self.O1 = 0.0
        self.O2 = 0.0
        self.O3 = 0.0

        # --- Vesicle pools (Eq. 15) ---
        self.R_a = 1.0   # releasable
        self.E_a = 0.0   # effective

        # --- Extracellular glutamate (not in paper; analogue of Eq. 9) ---
        self.G_a = 0.0

    def step(self, dt, c_a):
        p = self.p

        # =============================================================
        # 1. GATE DYNAMICS (Eq. 13)
        # dOj/dt = kj+ * ca - (kj+ * ca + kj-) * Oj
        # =============================================================
        dO1 = p['k1_plus'] * c_a - (p['k1_plus'] * c_a + p['k1_minus']) * self.O1
        dO2 = p['k2_plus'] * c_a - (p['k2_plus'] * c_a + p['k2_minus']) * self.O2
        dO3 = p['k3_plus'] * c_a - (p['k3_plus'] * c_a + p['k3_minus']) * self.O3

        # Euler update
        self.O1 = np.clip(self.O1 + dt * dO1, 0, 1)
        self.O2 = np.clip(self.O2 + dt * dO2, 0, 1)
        self.O3 = np.clip(self.O3 + dt * dO3, 0, 1)

        # =============================================================
        # 2. RELEASE PROBABILITY (Eq. 14)
        # fr_a = O1 * O2 * O3
        # =============================================================
        f_r_a = self.O1 * self.O2 * self.O3

        # =============================================================
        # 3. VESICLE CYCLE (Eq. 15)
        # =============================================================
        I_a = 1.0 - self.R_a - self.E_a

        Theta = 1.0 if c_a > p['C_a_thresh'] else 0.0

        dRa = (I_a / p['tau_rec_a']) - Theta * f_r_a * self.R_a
        dEa = -(self.E_a / p['tau_inac_a']) + Theta * f_r_a * self.R_a

        # Update vesicle pools
        self.R_a = np.clip(self.R_a + dt * dRa, 0, 1)
        self.E_a = np.clip(self.E_a + dt * dEa, 0, 1)

        # =============================================================
        # 4. EXTRACELLULAR GLUTAMATE (Analogue of neuronal Eq. 9)
        # =============================================================
        dGa = (p['n_a_v'] * p['g_a_v'] * self.E_a) - (p['g_a_c'] * self.G_a)
        self.G_a = max(self.G_a + dt * dGa, 0.0)

        return self.G_a
