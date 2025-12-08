# Dosya Yolu: src/models/post_synaptic_ca.py

import numpy as np

class PostSynapticCalciumDynamics:
    """
    Tewari & Majumdar (2012) - Post-Sinaptik Kalsiyum (Section 2.9)
    Denklemler: 20 - 27
    """
    def __init__(self, params):
        self.p = params
        
        # Başlangıç Değeri (Resting Calcium)
        self.c_post = self.p['c_post_rest']
        
        # Sabitler (Hesaplama yükünü azaltmak için)
        # alpha = 1 / (z * F * V_spine)
        self.alpha_conv = 1.0 / (self.p['z_Ca'] * self.p['F'] * self.p['V_spine'])

    def step(self, dt, V_post, I_AMPA):
        """
        dt: Zaman adımı (saniye)
        V_post: Post-sinaptik voltaj (Volt)
        I_AMPA: AMPA akımı (Amper)
        """
        p = self.p
        
        # =================================================================
        # 1. R-Tipi VGCC Akımı (i_R) - Denklem 25 & Metin
        # =================================================================
        # Metin: "...whenever V_post is greater than activation threshold... -30 mV"
        # Metin: "...number of channels... governed by binomial distribution"
        
        activation_thresh = -0.030 # -30 mV
        
        if V_post > activation_thresh:
            # Binomial dağılım: B(N, P_open)
            N_open = np.random.binomial(p['N_R'], p['P_open'])
        else:
            N_open = 0
            
        # i_R = g_R * N_open * (V - V_R)
        i_R = p['g_R'] * N_open * (V_post - p['V_R'])

        # =================================================================
        # 2. PMCA Pompası (S_pump) - Denklem 26
        # =================================================================
        # S_pump = k_s * (c_post - c_rest)
        S_pump = p['k_s'] * (self.c_post - p['c_post_rest'])

        # =================================================================
        # 3. Toplam Kalsiyum Akısı f(c_post) - Denklem 27
        # =================================================================
        # f(c) = - (eta * I_AMPA + i_R) / (zFV) - S_pump
        # Not: I_AMPA ve i_R inward akımlardır (negatiftir).
        # Başındaki eksi işareti ile pozitife (influx) dönerler.
        
        influx_term = - (p['eta'] * I_AMPA + i_R) * self.alpha_conv
        f_c = influx_term - S_pump

        # =================================================================
        # 4. Tamponlama Faktörü (Theta) - Denklem 23
        # =================================================================
        # Theta = (b_t * K_endo) / (K_endo + c_post)^2
        
        numerator = p['b_t'] * p['K_endo']
        denominator = (p['K_endo'] + self.c_post)**2
        theta = numerator / denominator

        # =================================================================
        # 5. Türev ve Güncelleme - Denklem 24
        # =================================================================
        # dc/dt = f(c) / (1 + theta)
        
        dc_dt = f_c / (1.0 + theta)
        
        self.c_post += dt * dc_dt
        
        # Negatiflik kontrolü (Kalsiyum 0'ın altına inemez)
        self.c_post = max(self.c_post, 1e-9) # min 1 nM
        
        return self.c_post