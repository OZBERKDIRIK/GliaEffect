CAMKII_PARAMS = {
    # ============================================================
    # 1) CaMKII TOTAL CONCENTRATION
    # ============================================================
    "e_k": 80e-6,          # Total CaMKII (M)

    # ============================================================
    # 2) CaMKII AUTOPHOSPHORYLATION (Zhabotinsky 2000)
    # ============================================================
#    "K1": 0.5,             # Catalytic const. for autophosphorylation (1/s)
#      "K1" : 0.05,
      "K1" : 0.005,
#    "k_h": 4.0e-6,         # Hill constant for CaM-KII activation (M)
#    "k_h" : 60e-6,
     "k_h" : 150.0e-6,
    "n_h": 3.0,            # Hill coefficient

    # Statistical weights w1–w9 (Zhabotinsky 2000, Fig. 3)
    "w": {
        1: 1.0,
        2: 1.8,
        3: 2.3,
        4: 2.7,
        5: 2.8,
        6: 2.7,
        7: 2.3,
        8: 1.8,
        9: 1.0
    },

    # ============================================================
    # 3) PP1 – CaMKII dephosphorylation
    # ============================================================
#    "K2": 10.0,            # PP1 catalytic constant (1/s)
    "K2" : 50.0,
    "K_M": 20e-6,          # MM constant of PP1 (M)
    "ep_0": 0.1e-6,        # Total PP1 concentration (M)

    # ============================================================
    # 4) I1 / I1P MODULE  (PKA–CaN regulatory loop)
    # ============================================================
    "I_0": 0.1e-6,         # Free I1 concentration (M)

    "k_F": 1e6,            # Association rate PP1–I1P complex (1/(M*s))
    "k_B": 1e-3,           # Dissociation rate (1/s)
    "k_I": 1.0,            # I1-dependent PP1 regulation rate (1/s)

    "v_PKA": 0.45e-6,      # PKA phosphorylation rate of I1 (M/s)
    "K_PKA": 0.0059e-6,    # MM constant of PKA (M)

    "v_CaN": 2.0,          # CaN-dependent I1P dephosphorylation (1/s)
    "k_h2": 0.7e-6,        # CaN activation Hill const (M)

    # ============================================================
    # 5) RETROGRADE SIGNALLING (NO → presynaptic α modulation)
    # ============================================================
    #"P_half": 40e-6,       # Threshold CaMKII-P for NO production (M)
   # "P_half" : 50e-6,
   "P_half" : 25e-6,
    "k_half": 0.4e-6,      # Slope factor (M)
    "k_syt": 0.005         # 0.5% synaptotagmin increase
}
