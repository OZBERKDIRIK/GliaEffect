import numpy as np

class PostSynapticDynamics:
    """
    Tewari & Majumdar (2012) – Post-Synaptic Membrane Potential
    Implements Equations 17, 18, 19
    
    UNIT PROTOCOL: SI UNITS
    - Voltage: Volts (V)
    - Current: Amperes (A)
    - Resistance: Ohms (Ω)
    - Time: Seconds (s)
    - Conductance: Siemens (S)
    """

    def __init__(self, params):
        self.p = params
        
        # Initial Conditions
        self.V_post = -70.0e-3  # Resting Potential (-70 mV -> Volts)
        self.m_AMPA = 0.0       # AMPA gating variable (0-1)
        self.I_AMPA = 0.0       # Recorded current for Calcium model

    def step(self, dt, g_syn_uM, I_soma_injected=0.0):
        """
        dt: Time step in SECONDS (s)
        g_syn_uM: Synaptic Glutamate in MicroMolar (µM)
        I_soma_injected: External current in Amperes (A)
        """
        p = self.p
        V = self.V_post
        m = self.m_AMPA
        
        # ---------------------------------------------------------
        # 1. AMPA Receptor Dynamics (Destexhe et al., 1998)
        # Denklem 19'a benzer kinetik
        # dm/dt = alpha * g * (1-m) - beta * m
        # ---------------------------------------------------------
        
        # Glutamate Concentration:
        # Destexhe modeli genelde mM (MiliMolar) glutamat ile çalışır.
        # Gelen g_syn_uM (µM) -> mM'a çevirelim.
        g_conc_mM = g_syn_uM * 1e-3
        
        # Alpha/Beta birimleri (mM^-1 ms^-1 veya M^-1 s^-1) çok kritiktir.
        # Parametre dosyasındaki değerlere göre (Destexhe):
        # alpha ~ 1.1e6 M^-1 s^-1 = 1100 mM^-1 s^-1
        # beta ~ 190 s^-1
        # Biz burada parametrelerin SI (Molar, Saniye) olduğunu varsayıyoruz.
        
        # Eğer parametreler SI ise g_conc Molar olmalı:
        g_conc_M = g_syn_uM * 1e-6 
        
        dm_dt = p['alpha_AMPA'] * g_conc_M * (1.0 - m) - p['beta_AMPA'] * m
        
        # Update gating variable
        m += dt * dm_dt
        m = np.clip(m, 0.0, 1.0)
        self.m_AMPA = m
        I_AMPA = p['g_AMPA'] * m * (V - p['V_AMPA'])
        self.I_AMPA = I_AMPA # Save for calcium model
        term_leak = -(V - p['V_rest'])
        term_current = -p['R_m'] * (I_soma_injected + I_AMPA) 
        
        dV_dt = (term_leak + term_current) / p['tau_post']
        
        # Update Voltage
        self.V_post += dt * dV_dt
        
        return self.V_post