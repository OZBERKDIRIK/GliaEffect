# Dosya: src/models/post_synaptic.py

import numpy as np

class PostSynapticDynamics:
    """
    Tewari & Majumdar (2012)
    Post-synaptic membrane potential
    Eq (17), (18), (19)
    """

    def __init__(self, params):
        self.p = params
        self.V_post = params["V_post_rest"]
        self.m_AMPA = 0.0
        self.I_AMPA = 0.0

    def step(self, dt, glu_pre, I_soma):
        p = self.p

        # --- AMPA Gating (Eq 19) ---
        d_m = p["alpha_AMPA"] * glu_pre * (1 - self.m_AMPA) - p["beta_AMPA"] * self.m_AMPA
        self.m_AMPA += dt * d_m
        self.m_AMPA = np.clip(self.m_AMPA, 0, 1)

        # --- AMPA Current ---
        self.I_AMPA = p["g_AMPA"] * self.m_AMPA * (self.V_post - p["V_AMPA"])

        # --- Membrane ODE (Eq 18) ---
        dV = ( -(self.V_post - p["V_post_rest"]) 
               - p["R_m"] * (I_soma + self.I_AMPA) ) / p["tau_post"]

        self.V_post += dt * dV
        return self.V_post
