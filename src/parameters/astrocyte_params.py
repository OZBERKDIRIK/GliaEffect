# ==============================================================================
# ASTROCYTE PARAMETERS - TEWARI & MAJUMDAR (2012) / DE PITTA (2009)
# ALL UNITS CONVERTED TO SI (Molar, Seconds, Amperes)
# ==============================================================================

ASTROCYTE_PARAMS = {
    # --------------------------------------------------------------------------
    # 1. Flux Rates (Table 5 - Parametre1.jpg)
    # --------------------------------------------------------------------------
    "r_c": 6.0,          # 1/s (Max IP3R flux)
    "r_L": 0.11,         # 1/s (Leak rate)
    "v_ER": 0.9e-6,      # Molar/s (0.9 uM/s -> SERCA pump)
    
    # --------------------------------------------------------------------------
    # 2. Concentrations & Ratios (Table 5)
    # --------------------------------------------------------------------------
    "c_0": 2.0e-6,       # Molar (Total free Ca)
    "c1_a": 0.185,       # Dimensionless (ER/Cytosol volume ratio)
    
    # --------------------------------------------------------------------------
    # 3. Dissociation Constants (Table 5 - Converted to Molar)
    # --------------------------------------------------------------------------
    "K_ER": 0.1e-6,      # Molar
    "d1": 0.13e-6,       # Molar
    "d2": 1.049e-6,      # Molar
    "d3": 0.9434e-6,     # Molar
    "d5": 0.08234e-6,    # Molar

    # --------------------------------------------------------------------------
    # 4. IP3R Binding Rate (CRITICAL FIX)
    # --------------------------------------------------------------------------
    # Makale tablosunda "2 / s" yazıyor ancak denklem (alpha_h) gereği 
    # bu birim 1/(uM*s) olmalı. SI biriminde: 0.2 * 1e6.
    "a2": 0.2e6,         # Molar^-1 s^-1 (0.2 uM^-1 s^-1)

    # --------------------------------------------------------------------------
    # 5. IP3 Production - Glutamate Dependent (PLC_beta)
    # --------------------------------------------------------------------------
    "v_beta": 0.5e-6,    # Molar/s (Max production)
    "K_R": 1.3e-6,       # Molar (Glutamate Affinity)
    "K_p": 10.0e-6,      # Molar (Ca inhibition)
    "K_pi": 0.6e-6,      # Molar (Ca affinity of PKC)

    # --------------------------------------------------------------------------
    # 6. IP3 Production - Ca Dependent (PLC_delta)
    # --------------------------------------------------------------------------
    "v_delta": 0.05e-6,     # Molar/s
    "K_PLC_delta": 0.1e-6,  # Molar (Half-max concentration)
    "k_delta": 1.5e-6,      # Molar (Inhibition constant)

    # --------------------------------------------------------------------------
    # 7. IP3 Degradation (Table 5 - Parametre2.png)
    # --------------------------------------------------------------------------
    "r_5p": 0.05,        # 1/s (Degradation by IP-5P)
    "v_3k": 2.0e-6,      # Molar/s (Degradation by IP3-3K)
    "K_D": 0.7e-6,       # Molar (Ca affinity)
    "K_3": 1.0e-6        # Molar (IP3 affinity)
}