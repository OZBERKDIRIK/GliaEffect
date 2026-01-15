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
        
        # [EKLENDİ] R-tipi akım kaydı için başlangıç değeri
        self.i_R = 0.0
        
        # -------------------------------------------------------------------
        # Birim Dönüşümü
        # -------------------------------------------------------------------
        # SI Birimleri: I(Amper), F(C/mol), V(m^3) -> Sonuç: mol/m^3/s (mM/s)
        # Ancak c_post ve Pompa "Molar" (mol/L) cinsinden çalışıyor.
        # 1 mM = 1e-3 Molar.
        # Bu yüzden sonucu Molar/s yapmak için 1e-3 ile çarpıyoruz.
        
        self.alpha_conv = (1.0 / (self.p['z_Ca'] * self.p['F'] * self.p['V_spine'])) * 1e-3

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
        
        activation_thresh = -0.030 # -30 mV
        
        if V_post > activation_thresh:
            # Binomial dağılım: B(N, P_open)
            N_open = np.random.binomial(p['N_R'], p['P_open'])
        else:
            N_open = 0
            
        # i_R = g_R * N_open * (V - V_R)
        i_R = p['g_R'] * N_open * (V_post - p['V_R'])
        
        # Grafik çizimi için akımı kaydet
        self.i_R = i_R 

        # =================================================================
        # 2. PMCA Pompası (S_pump) - Denklem 26
        # =================================================================
        # S_pump birimi: Molar/s (Çünkü c_post Molar, k_s 1/s)
        S_pump = p['k_s'] * (self.c_post - p['c_post_rest'])

        # =================================================================
        # 3. Toplam Kalsiyum Akısı f(c_post) - Denklem 27
        # =================================================================
        # f(c) birimi: Molar/s (Influx - Efflux)
        
        influx_term = - (p['eta'] * I_AMPA + i_R) * self.alpha_conv
        f_c = influx_term - S_pump

        # =================================================================
        # 4. Tamponlama Faktörü (Theta) - Denklem 23
        # =================================================================
        
        numerator = p['b_t'] * p['K_endo']
        denominator = (p['K_endo'] + self.c_post)**2
        theta = numerator / denominator

        # =================================================================
        # 5. Türev ve Güncelleme - Denklem 24
        # =================================================================
        
        dc_dt = f_c / (1.0 + theta)
        
        self.c_post += dt * dc_dt
        
        # Negatiflik kontrolü
        self.c_post = max(self.c_post, 1e-9) 
        
        return self.c_post