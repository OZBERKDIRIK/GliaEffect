# Dosya Yolu: src/parameters/gliatransmitter_params.py

GLIATRANSMITTER_PARAMS = {
    # ------------------------------------------------------------
    # Astrosit Kalsiyum Sensörü Kinetiği (Bertram et al. 1996)
    # Tablo 6
    # Birimler: ms -> s, uM -> M, mM -> M
    # ------------------------------------------------------------
    
    # Kademeli Bağlanma Hızları (Association Rates)
    # k1_plus: 3.75e-3 / (uM * ms)
    # Dönüşüm: 3.75e-3 * 1e6 (uM->M) * 1e3 (ms->s) = 3.75e6
    "k1_plus": 3.75e6,  # 1/(M*s)
    
    # k2_plus: 2.5e-3 / (uM * ms) -> 2.5e6
    "k2_plus": 2.5e6,   # 1/(M*s)
    
    # k3_plus: 1.25e-2 / (uM * ms) -> 1.25e7
    "k3_plus": 1.25e7,  # 1/(M*s)
    
    # Ayrılma Hızları (Dissociation Rates)
    # k1_minus: 4e-4 / ms -> 0.4 / s
    "k1_minus": 0.4,    # 1/s
    
    # k2_minus: 1e-3 / ms -> 1.0 / s
    "k2_minus": 1.0,    # 1/s
    
    # k3_minus: 1e-3 / ms -> 1.0 / s
    "k3_minus": 1.0,    # 1/s

    # ------------------------------------------------------------
    # Vezikül Döngüsü (TMM - Astrosit Versiyonu)
    # ------------------------------------------------------------
    "tau_rec_a": 0.8,     # İyileşme süresi (800 ms -> 0.8 s)
    "tau_inac_a": 0.003,  # İnaktivasyon süresi (3 ms -> 0.003 s)
    
    # Salınım Eşiği (Heaviside Fonksiyonu için)
    "C_a_thresh": 196.69e-9, # 196.69 nM -> Molar

    # ------------------------------------------------------------
    # Ekstra-Sinaptik Aralık Dinamikleri
    # ------------------------------------------------------------
    "n_a_v": 12.0,       # Salınıma hazır SLMV sayısı
    "g_a_v": 0.02,       # Vezikül içi glutamat (20 mM -> 0.02 M)
    "g_a_c": 10000.0,    # Temizlenme hızı (10 / ms -> 10000 / s)
}