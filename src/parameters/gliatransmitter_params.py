# Dosya Yolu: src/parameters/gliatransmitter_params.py

GLIATRANSMITTER_PARAMS = {
    # ------------------------------------------------------------
    # Astrosit Kalsiyum Sensörü Kinetiği (Table 6)
    # UNITS: MicroMolar (uM) and Milliseconds (ms)
    # ------------------------------------------------------------
    
    # Association Rates (Table 6: / (uM * ms))
    "k1_plus": 3.75e-3,  # 3.75 x 10^-3
    "k2_plus": 2.5e-3,   # 2.5 x 10^-3
    "k3_plus": 1.25e-2,  # 1.25 x 10^-2
    
    # Dissociation Rates (Table 6: / ms)
    "k1_minus": 4e-4,    # 4 x 10^-4
    "k2_minus": 1e-3,    # 1 x 10^-3
    "k3_minus": 1e-3,    # 1 x 10^-3

    # ------------------------------------------------------------
    # Vesicle Cycle (Table 6)
    # ------------------------------------------------------------
    "tau_rec_a": 800.0,   # ms
    "tau_inac_a": 3.0,    # ms
    
    # Threshold: 196.69 nM = 0.19669 uM
    "C_a_thresh": 0.19669, # uM

    # ------------------------------------------------------------
    # Extra-Synaptic Cleft Dynamics (Table 6)
    # ------------------------------------------------------------
    "n_a_v": 12.0,        # Number of SLMVs
    
    # Glutamate Concentration: 20 mM = 20000 uM
    "g_a_v": 20000.0,     # uM
    
    # Clearance Rate: 10 / ms
    "g_a_c": 10.0         # 1/ms
}