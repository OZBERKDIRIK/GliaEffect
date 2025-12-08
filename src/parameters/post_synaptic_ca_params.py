# Dosya Yolu: src/parameters/post_synaptic_ca_params.py

POST_SYNAPTIC_CA_PARAMS = {
    # ------------------------------------------------------------
    # Tablo 8: Post-Sinaptik Kalsiyum Parametreleri
    # ------------------------------------------------------------
    
    # eta: AMPA akımının kalsiyum taşıyan fraksiyonu
    "eta": 0.012,        # Boyutsuz (Bollman et al. 1998)

    # VGCC (R-Tipi) Parametreleri
    "P_open": 0.52,      # Kanal açık kalma olasılığı (Sabatini & Svoboda 2000)
    "g_R": 15e-12,       # İletkenlik (15 pS -> Siemens)
    "N_R": 12.0,         # Kanal sayısı (Number of R-type channels)
    
    # *** DÜZELTME BURADA ***
    # Model kodu "V_R" arıyor. Tablo 8'deki adı da V_R'dir.
    "V_R": 0.0274,       # Kalsiyum Reversal Potansiyeli (27.4 mV -> Volt)
    
    # Endojen Tampon (Endogenous Buffer)
    "K_endo": 10e-6,     # Afinite (10 uM -> Molar)
    "b_t": 200e-6,       # Toplam tampon konsantrasyonu (200 uM -> Molar)

    # Fiziksel Özellikler
    "V_spine": 0.9048e-18, # Spine Hacmi (0.9048 um^3 -> m^3)
    "c_post_rest": 100e-9, # Dinlenme Kalsiyumu (100 nM -> Molar)

    # PMCA Pompası (Efflux)
    "k_s": 100.0,        # 1/s (Maximum PMCa efflux rate constant)
    
    # Fiziksel Sabitler
    "F": 96487.0,        # C/mol
    "z_Ca": 2            # Değerlik
}