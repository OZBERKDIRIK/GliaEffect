import numpy as np

class AstrocyteDynamics:
    """
    Tewari & Majumdar (2012) – Section 2.5
    De Pitta et al. (2009) – G-ChI astrocyte model

    Tamamen makale denklemlerine sadık versiyon.
    (Denklem 10, 11, 12)
    """

    def __init__(self, params):
        self.p = params

        # Başlangıç değerleri (makale tipik steady state değerleri)
        self.c_a = 0.1e-6   # Ca (M)
        self.p_a = 0.1e-6   # IP3 (M)
        self.h_a = 0.8      # h-gate (boyutsuz)

    # -------------------------------------------------------------------------
    # Hill fonksiyonu (makaledeki tanım: x^n / (x^n + K^n))
    # -------------------------------------------------------------------------
    def hill(self, x, K, n):
        x = max(x, 0.0)
        xn = x ** n
        Kn = K ** n
        return xn / (xn + Kn)

    # -------------------------------------------------------------------------
    # Tüm türevlerin hesaplanması (denklem 10–12)
    # -------------------------------------------------------------------------
    def compute_derivatives(self, dt, g_syn):
        p = self.p
        c_a = self.c_a
        p_a = self.p_a
        h_a = self.h_a

        # =====================================================================
        # GATING DEĞİŞKENLERİ (m_inf, n_inf)
        # =====================================================================
        # m_inf = Hill(p_a, d1)
        m_inf = self.hill(p_a, p['d1'], 1.0)

        # n_inf = Hill(c_a, d5)
        n_inf = self.hill(c_a, p['d5'], 1.0)

        # =====================================================================
        # driving_term = (c0 - (1 + c1a) * ca)
        # =====================================================================
        driving = p['c_0'] - (1.0 + p['c1_a']) * c_a

        # =====================================================================
        # α_h ve β_h (denklem açıklaması)
        # =====================================================================
        alpha_h = p['a2'] * p['d2'] * (p_a + p['d1']) / (p_a + p['d3'])
        beta_h  = p['a2'] * c_a

        # =====================================================================
        # ➤ DENKLEM (10): dc/dt
        # =====================================================================
        # dc/dt = -r_c m^3 n^3 h^3 * driving
        #        - v_ER * ca^2 / (ca^2 + K_ER^2)
        #        - r_L * driving

        J_IP3R = p['r_c'] * (m_inf**3) * (n_inf**3) * (h_a**3) * driving
        J_SERCA = p['v_ER'] * (c_a**2) / (c_a**2 + p['K_ER']**2)
        J_Leak = p['r_L'] * driving

        dc_a_dt = -(J_IP3R + J_SERCA + J_Leak)  # İŞARETLER %100 MAKALEYLE AYNI

        # =====================================================================
        # ➤ DENKLEM (11): dp/dt
        # =====================================================================

        # PLCβ: vβ * Hill(g^0.7, K_R) * (1 + (Kp/KR)*Hill(ca, Kπ))^-1
        inhibition_factor = 1.0 + (p['K_p'] / p['K_R']) * self.hill(c_a, p['K_pi'], 1.0)
        prod_PLC_beta = p['v_beta'] * self.hill(g_syn, p['K_R'], 0.7) / inhibition_factor

        # PLCδ: vδ / (1 + pa/kδ) * Hill(ca^2, K_PLCδ)
        prod_PLC_delta = (p['v_delta'] / (1.0 + p_a / p['k_delta'])) * \
                         self.hill(c_a, p['K_PLC_delta'], 2.0)

        # IP3-3K (degradation): v_3K * Hill(ca^4, KD) * Hill(pa, K3)
        deg_3K = p['v_3k'] * self.hill(c_a, p['K_D'], 4.0) * self.hill(p_a, p['K_3'], 1.0)

        # IP-5P (degradation): r5p * pa
        deg_5P = p['r_5p'] * p_a

        dp_a_dt = prod_PLC_beta + prod_PLC_delta - deg_3K - deg_5P

        # =====================================================================
        # ➤ DENKLEM (12): dh/dt
        # =====================================================================
        # dh/dt = α_h (1 - h) - β_h h
        # Gürültü terimi G_h(t) şimdilik eklenmedi.

        dh_a_dt = alpha_h * (1.0 - h_a) - beta_h * h_a

        # =====================================================================
        # EULER GÜNCELLEME
        # =====================================================================
        self.c_a += dt * dc_a_dt
        self.p_a += dt * dp_a_dt
        self.h_a += dt * dh_a_dt

        # Fiziksel sınırlar
        self.c_a = max(self.c_a, 1e-12)
        self.p_a = max(self.p_a, 0.0)
        self.h_a = np.clip(self.h_a, 0.0, 1.0)

        return self.c_a
