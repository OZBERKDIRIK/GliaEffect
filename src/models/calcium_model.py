import numpy as np

class PresynapticCalciumDynamics:
    """
    Tewari & Majumdar (2012) Implementation
    Units Strategy: INTERNAL CALCS IN SI (Molar, Ampere, Volt, Meter, Second).
    Inputs: mV, ms (OR detected Volts/Seconds)
    Outputs: uM (micromolar)
    """
    def __init__(self, p):
        self.p = p
        
        # --------------------------------------------------------
        # Pre-calculate Constant Factors (SI Units)
        # --------------------------------------------------------
        # Nernst Potansiyeli (Sabit varsayılmış)
        # V_Ca = (RT / zF) * ln(c_ext / c_rest)
        RT_zF = (p["R"] * p["T"]) / (p["z_Ca"] * p["F"])
        self.V_Ca = RT_zF * np.log(p["c_ext"] / p["c_i_rest"]) # Sonuç: Volt

        # Akım (Amper) -> Konsantrasyon Hızı (Molar/s) Dönüşüm Faktörü
        # Denklem: dc/dt = - I / (z * F * Vol_Liter)
        vol_liter = p["V_btn"] * 1000.0 # m3 -> Liter
        self.inv_zFV = 1.0 / (p["z_Ca"] * p["F"] * vol_liter)

        # --------------------------------------------------------
        # Initial Conditions (in Molar)
        # --------------------------------------------------------
        self.c_fast = 0.0              # Fast component starts at 0
        self.c_slow = p["c_i_rest"]    # Slow component starts at rest
        self.c_ER   = 400.0e-6         # ER starts filled (~400 uM)
        self.p_ip3  = p["p0"]          # IP3 rest
        self.m_Ca   = 0.0              # Gate closed
        self.q      = 0.5              # IP3R gating

    def step(self, dt_input, V_pre_input, glu=0.0):
        """
        dt_input: Time step (Expected ms, but robust to seconds)
        V_pre_input: Membrane voltage (Expected mV, but robust to Volts)
        glu: Glutamate concentration in uM
        """
        p = self.p
        
        # --- UNIT AUTO-DETECTION & CORRECTION ---
        # Voltaj: Eğer -1 ile 1 arasındaysa muhtemelen VOLT verilmiştir, mV'ye gerek yok.
        # Eğer -100, -60 gibiyse mV verilmiştir.
        if abs(V_pre_input) < 1.0: 
            V_pre = V_pre_input      # Zaten Volt verilmiş
        else:
            V_pre = V_pre_input * 1e-3 # mV verilmiş, Volt'a çevir
            
        # Zaman: Eğer dt < 1e-4 ise muhtemelen SANİYE verilmiştir.
        if dt_input < 1e-4:
            dt = dt_input            # Zaten Saniye verilmiş
        else:
            dt = dt_input * 1e-3     # ms verilmiş, Saniye'ye çevir

        # Glutamate: uM -> Molar
        glu_molar = glu * 1e-6        
        
        # Current Cytosolic Calcium (Molar)
        c_i = self.c_fast + self.c_slow
        c_i = max(c_i, 1e-9) # Safety floor (1 nM)

        # ========================================================
        # 1. FAST DYNAMICS (VGCC & PMCA)
        # ========================================================
        
        # --- VGCC (N-Type) Current ---
        # m_inf (Boltzmann)
        m_inf = 1.0 / (1.0 + np.exp((p["V_mCa"] - V_pre) / p["k_mCa"]))
        
        # dm/dt
        dm_dt = (m_inf - self.m_Ca) / p["tau_mCa"]
        self.m_Ca += dm_dt * dt
        
        # I_Ca Calculation (Current Density: A/m^2)
        # I = rho * m^2 * g * (V - V_Ca)
        g_total = p["rho_Ca"] * (self.m_Ca**2) * p["g_Ca"]
        I_Ca_density = g_total * (V_pre - self.V_Ca) 
        
        # Convert Density (A/m2) to Current (A)
        I_Ca_amp = I_Ca_density * p["A_btn"]

        # --- PMCA Pump Current (Destruction) ---
        I_PMCA_density = p["v_PMCA_max"] * (c_i**2) / (c_i**2 + p["K_PMCA"]**2)
        I_PMCA_amp = I_PMCA_density * p["A_btn"]

        # --- Membrane Leak (Construction) ---
        J_leak = p["v_leak"] * (p["c_ext"] - c_i)

        # --- TOTAL FAST DERIVATIVE (Molar/s) ---
        flux_membrane = -(I_Ca_amp + I_PMCA_amp) * self.inv_zFV
        dc_fast_dt = flux_membrane + J_leak

        # ========================================================
        # 2. SLOW DYNAMICS (ER & IP3)
        # ========================================================
        
        # IP3 Gating
        m_inf_ip3 = self.p_ip3 / (self.p_ip3 + p["d1"])
        n_inf_ip3 = c_i / (c_i + p["d5"])
        
        alpha_q = p["a2"] * p["d2"] * (self.p_ip3 + p["d1"]) / (self.p_ip3 + p["d3"])
        beta_q  = p["a2"] * c_i
        dq_dt = alpha_q * (1.0 - self.q) - beta_q * self.q
        self.q += dq_dt * dt

        # Fluxes (Molar/s)
        prob = (m_inf_ip3**3) * (n_inf_ip3**3) * (self.q**3)
        J_IP3R = p["c1"] * p["v1"] * prob * (self.c_ER - c_i)
        
        J_SERCA = p["v3"] * (c_i**2) / (c_i**2 + p["k3"]**2)
        J_ER_Leak = p["c1"] * p["v2"] * (self.c_ER - c_i)

        # dc_slow/dt
        dc_slow_dt = J_IP3R + J_ER_Leak - J_SERCA
        
        # dc_ER/dt
        dc_ER_dt = -(1.0 / p["c1"]) * dc_slow_dt

        # IP3 Dynamics (dp/dt)
        term_prod = p["v_g"] * (glu_molar**0.7) / (p["k_g"]**0.7 + glu_molar**0.7)
        term_deg  = p["tau_p"] * (self.p_ip3 - p["p0"])
        dp_dt = term_prod - term_deg

        # ========================================================
        # 3. UPDATE STATES
        # ========================================================
        self.c_fast += dc_fast_dt * dt
        self.c_slow += dc_slow_dt * dt
        self.c_ER   += dc_ER_dt   * dt
        self.p_ip3  += dp_dt      * dt

        # Sanity Checks
        self.c_fast = max(self.c_fast, 0.0)
        self.c_slow = max(self.c_slow, 1e-10)
        self.c_ER   = max(self.c_ER, 1e-10)

        # --- OUTPUT CONVERSION (SI -> uM) ---
        return (self.c_fast + self.c_slow) * 1e6
    
    def get_states(self):
        return {
            "c_total": (self.c_fast + self.c_slow) * 1e6,
            "c_fast": self.c_fast * 1e6,
            "c_slow": self.c_slow * 1e6,
            "c_ER": self.c_ER * 1e6,
            "IP3": self.p_ip3 * 1e6
        }