# ==============================================================================
# FINAL VERIFIED PARAMETERS - TEWARI & MAJUMDAR (2012)
# ==============================================================================

# --------------------------
# 1. Calcium Parameters (SI Units for Physics Stability)
# --------------------------
CA_PARAMS = {
    # Constants
    "F": 96487.0,        "R": 8.314,          "T": 293.15,         "z_Ca": 2,

    # Geometry (SI: Meters)
    "A_btn": 1.24e-12,   # m^2 (Table 3)
    "V_btn": 0.13e-18,   # m^3 (Table 3)

    # Concentrations (SI: Molar)
    "c_ext": 2000.0e-6,  # 2 mM -> Molar
    "c_i_rest": 0.1e-6,  # 0.1 µM -> Molar

    # Channel & Pump Properties (SI Units)
    "rho_Ca": 3.2e12,    # channels/m^2
    "g_Ca": 2.3e-12,     # Siemens (2.3 pS)
    "V_mCa": -0.017,     # Volts
    "k_mCa": 0.0084,     # Volts
    "tau_mCa": 0.010,    # Seconds (10 ms)

    "v_PMCA_max": 0.004, # A/m^2 (Derived from 0.4 uA/cm^2)
    "K_PMCA": 0.1e-6,    # Molar

    "v_leak": 2.66e-3,   # 1/s (Derived from 2.66e-6/ms)

    # ER & IP3 Dynamics (SI: Molar, Seconds)
    "c1": 0.185,
    "v1": 30.0,          # 1/s
    "v2": 0.2374,        # 1/s
    "v3": 90.0e-6,       # Molar/s (90 uM/s)
    "k3": 0.1e-6,        # Molar

    "d1": 0.13e-6,       "d2": 1.049e-6,
    "d3": 0.9434e-6,     "d5": 0.08234e-6,
    "a2": 0.2e6,         # 1/(Molar*s)

    "v_g": 0.062e-6,     # Molar/s
    "k_g": 0.78e-6,      # Molar
    "tau_p": 0.14,       # 1/s
    "p0": 0.16e-6        # Molar
}

# --------------------------
# 2. Glutamate Parameters (Table 4)
# --------------------------
GLUTAMATE_PARAMS = {
    # Sensor Kinetics
    "alpha": 0.3,      # µM^-1 ms^-1
    "beta": 3.0,       # ms^-1
    "gamma": 30.0,     # ms^-1
    "delta": 8.0,      # ms^-1

    # Spontaneous Release
    "a1": 50.0,        # µM
    "a2": 5.0,         # µM
    "a3": 0.85,        # ms^-1

    # Vesicle Cycle
    "tau_rec": 800.0,  # ms
    "tau_inac": 3.0,   # ms
    
    # Cleft Dynamics
    "n_v": 2.0,        # Number of vesicles
    "g_v": 60000.0,    # µM (60 mM)
    "g_c": 10.0        # ms^-1
}