# Dosya Yolu: src/parameters/pre_synaptic_params.py

PRE_SYNAPTIC_PARAMS = {
    # ------------------------------------------------------------
    # İletkenlikler (Conductances) - [mS/cm^2]
    # ------------------------------------------------------------
    "g_K": 36.0,    # Potasyum
    "g_Na": 120.0,  # Sodyum
    "g_L": 0.3,     # Sızıntı (Leak)

    # ------------------------------------------------------------
    # Denge Potansiyelleri (Reversal Potentials) - [mV]
    # ------------------------------------------------------------
    "V_K": -82.0,   
    "V_Na": 45.0,   
    "V_L": -59.4,   

    # ------------------------------------------------------------
    # Membran Kapasitansı - [uF/cm^2]
    # ------------------------------------------------------------
    "C_m": 1.0,

    # ------------------------------------------------------------
    # Dışarıdan Uygulanan Akım (Stimülasyon)
    # Makalede: 10 uA/cm^2, 5 Hz
    # ------------------------------------------------------------
    "I_app_amp": 10.0,   # Genlik (uA/cm^2)
    "freq": 5.0,         # Frekans (Hz)
    "pulse_width": 5.0,  # Sinyal genişliği (ms)

    # ------------------------------------------------------------
    # Başlangıç Değerleri (Initial Conditions)
    # ------------------------------------------------------------
    "V_init": -70.0,  # Dinlenme Potansiyeli (mV)
    "m_init": 0.05,   # Na aktivasyon kapısı
    "h_init": 0.6,    # Na inaktivasyon kapısı
    "n_init": 0.32,   # K aktivasyon kapısı
}