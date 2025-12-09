import numpy as np

class AstrocyteDynamics:
    """
    Tewari & Majumdar (2012) - Astrocyte Dynamics
    Implements Equations 10, 11, 12 from the paper.
    
    UNITS: Internal calculations are strictly in SI (Molar, Seconds).
    Input 'g_syn_molar' must be in Molar!
    """

    def __init__(self, params):
        self.p = params
        
        # Initial Conditions (Steady State Yaklaşımı)
        self.c_a = 0.1e-6  # Ca2+ (Molar) - Resting ~100 nM
        self.p_a = 0.1e-6  # IP3 (Molar)
        self.h_a = 0.8     # Gating variable (Dimensionless)

    def hill(self, x, K, n):
        """
        Generic Hill function: x^n / (x^n + K^n)
        Safe against negative x and division by zero.
        """
        x = max(x, 0.0)
        xn = x ** n
        Kn = K ** n
        if xn + Kn == 0: return 0.0
        return xn / (xn + Kn)

    def compute_derivatives(self, dt, g_syn_molar):
        """
        dt: Zaman adımı (Saniye)
        g_syn_molar: Sinaptik Glutamat (Molar)
        Returns: Yeni Ca_astro (Molar)
        """
        p = self.p
        c_a = self.c_a
        p_a = self.p_a
        h_a = self.h_a

        # ---------------------------------------------------------------------
        # 0. Parametre Güvenliği (c1 vs c1_a karmaşasını önlemek için)
        # ---------------------------------------------------------------------
        # Eğer sözlükte 'c1_a' yoksa 'c1' kullan, o da yoksa varsayılan 0.185 al.
        c1_val = p.get('c1_a', p.get('c1', 0.185))

        # ---------------------------------------------------------------------
        # 1. Gating Variables (m_inf, n_inf) - Denklem 4 & 10
        # ---------------------------------------------------------------------
        # m_inf = Hill(p_a, d1) -> n=1
        m_inf = self.hill(p_a, p['d1'], 1.0)
        
        # n_inf = Hill(c_a, d5) -> n=1
        n_inf = self.hill(c_a, p['d5'], 1.0)

        # ---------------------------------------------------------------------
        # 2. Ca2+ Dynamics (Denklem 10)
        # ---------------------------------------------------------------------
        # Driving force (Konsantrasyon farkı: ER - Sitoplazma)
        driving = p['c_0'] - (1.0 + c1_val) * c_a
        
        # Fluxes
        # J_IP3R: ER'den sitoplazmaya akış (+)
        J_IP3R = p['r_c'] * (m_inf**3) * (n_inf**3) * (h_a**3) * driving
        
        # J_SERCA: Sitoplazmadan ER'ye pompalama (-)
        J_SERCA = p['v_ER'] * (c_a**2) / (c_a**2 + p['K_ER']**2)
        
        # J_Leak: ER'den sitoplazmaya sızıntı (+)
        J_Leak = p['r_L'] * driving
        
        # Fiziksel Denklem: Değişim = Girişler - Çıkışlar
        dc_a_dt = J_IP3R - J_SERCA + J_Leak 

        # ---------------------------------------------------------------------
        # 3. IP3 Dynamics (Denklem 11)
        # ---------------------------------------------------------------------
        
        # --- Production by PLC_beta (Glutamate Dependent) ---
        # Term: v_beta * Hill(g^0.7, K_R)
        # Not: Hill fonksiyonuna g_syn_molar veriyoruz, K_R de Molar.
        # Fonksiyon içeride kuvvetlerini alıyor (0.7).
        prod_beta = p['v_beta'] * self.hill(g_syn_molar, p['K_R'], 0.7)
        
        # Inhibition factor: 1 + (Kp/KR)*Hill(Ca, K_pi)
        inhib = 1.0 + (p['K_p'] / p['K_R']) * self.hill(c_a, p['K_pi'], 1.0)
        
        term_PLC_beta = prod_beta / inhib

        # --- Production by PLC_delta (Ca Dependent) ---
        # Term: v_delta / (1 + p/k_delta) * Hill(c^2, K_PLC_delta)
        term_delta_1 = p['v_delta'] / (1.0 + p_a / p['k_delta'])
        
        # Hill(c^2, K) -> c^2 / (c^2 + K) mantığı için n=2.0 kullanıyoruz.
        term_delta_2 = self.hill(c_a, p['K_PLC_delta'], 2.0) 
        
        term_PLC_delta = term_delta_1 * term_delta_2

        # --- Degradation (3K and 5P) ---
        # 3K: v_3K * Hill(c^4, K_D) * Hill(p, K_3)
        deg_3K = p['v_3k'] * self.hill(c_a, p['K_D'], 4.0) * self.hill(p_a, p['K_3'], 1.0)
        
        # 5P: r_5p * p
        deg_5P = p['r_5p'] * p_a

        dp_a_dt = term_PLC_beta + term_PLC_delta - deg_3K - deg_5P

        # ---------------------------------------------------------------------
        # 4. h-Gate Dynamics (Denklem 12)
        # ---------------------------------------------------------------------
        # alpha_h, beta_h
        alpha_h = p['a2'] * p['d2'] * (p_a + p['d1']) / (p_a + p['d3'])
        beta_h  = p['a2'] * c_a
        
        dh_a_dt = alpha_h * (1.0 - h_a) - beta_h * h_a

        # ---------------------------------------------------------------------
        # Euler Update
        # ---------------------------------------------------------------------
        self.c_a += dt * dc_a_dt
        self.p_a += dt * dp_a_dt
        self.h_a += dt * dh_a_dt
        
        # Sınırlandırmalar
        self.c_a = max(self.c_a, 1e-12) # 0 yerine çok küçük bir sayı (Nan önler)
        self.p_a = max(self.p_a, 0.0)
        self.h_a = np.clip(self.h_a, 0.0, 1.0)

        return self.c_a