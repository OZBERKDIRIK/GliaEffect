import numpy as np

class CaMKIIDynamics:
    """
    Tewari & Majumdar (2012) — CaMKII phosphorylation model
    Implements equations (28)–(39) exactly as defined in the paper.
    """

    def __init__(self, params):
        self.p = params

        # --------------------------------------------------------------
        # P0..P10 (phosphorylation states of CaMKII holoenzyme)
        # Makale: başlangıçta hepsi P0 (unphosphorylated)
        # --------------------------------------------------------------
        self.P = np.zeros(11)
        self.P[0] = 1.0  # total CaMKII normalized

        # --------------------------------------------------------------
        # PP1 and Inhibitor-1 states (e_p, I)
        # --------------------------------------------------------------
        self.ep = self.p["ep_0"]   # active PP1
        self.I = 0.0               # free Inhibitor-1 (I)

        # --------------------------------------------------------------
        # CaMKII effective subunit count (w_i coefficients)
        # Makalede verilen değerler:
        # w2 = w8 = 1.8
        # w3 = w7 = 2.3
        # w4 = w6 = 2.7
        # w5 = 2.8
        # Diğerleri 1.0 alınır
        # --------------------------------------------------------------
        self.w = np.ones(11)
        self.w[2] = self.w[8] = 1.8
        self.w[3] = self.w[7] = 2.3
        self.w[4] = self.w[6] = 2.7
        self.w[5] = 2.8
        # w0, w1, w9, w10 = 1.0 (zaten öyle)

    # ==================================================================
    # CAUTION: c_post burada Molar (M) biriminde olmalı (post-sinaptik Ca²⁺)
    # ==================================================================
    def step(self, dt, c_post):
        p = self.p
        P = self.P

        # --------------------------------------------------------------
        # (Hill term) — kullanıldığı yerler: v_phos ve v_a-phos
        # --------------------------------------------------------------
        cn = c_post ** p["n_h"]
        kn = p["k_h"] ** p["n_h"]
        hill = cn / (kn + cn)

        # --------------------------------------------------------------
        # (32) Initiation phosphorylation v_phos
        # --------------------------------------------------------------
        v_phos = 10.0 * p["K1"] * (hill ** 2) * P[0]

        # --------------------------------------------------------------
        # (35) Autophosphorylation rate v_a-phos
        # --------------------------------------------------------------
        v_a = p["K1"] * hill

        # --------------------------------------------------------------
        # (37) Dephosphorylation rate v_d-phos
        # denominator = KM + sum(i * P_i)
        # --------------------------------------------------------------
        total_phos = np.sum([i * P[i] for i in range(1, 11)])
        v_d = (p["K2"] * self.ep) / (p["K_M"] + total_phos)

        # --------------------------------------------------------------
        # (38) dP_i/dt ODE system
        # --------------------------------------------------------------
        dP = np.zeros(11)

        # P0
        dP[0] = -v_phos + v_d * P[1]

        # P1
        dP[1] = (
            v_phos
            - v_d * P[1]
            - v_a * self.w[1] * P[1]
            + 2.0 * v_d * P[2]
        )

        # Pi (i = 2..9)
        for i in range(2, 10):
            in_autophos = v_a * self.w[i - 1] * P[i - 1]
            out_autophos = v_a * self.w[i] * P[i]
            out_dephos = v_d * i * P[i]
            in_dephos = v_d * (i + 1) * P[i + 1]
            dP[i] = in_autophos - out_autophos - out_dephos + in_dephos

        # P10
        dP[10] = v_a * self.w[9] * P[9] - v_d * 10.0 * P[10]

        # --------------------------------------------------------------
        # PP1 ODE (38)
        # dep/dt = -kF * I * ep + kB * (ep0 - ep) + kI * I0
        # --------------------------------------------------------------
        assoc = p["k_F"] * self.I * self.ep
        dissoc = p["k_B"] * (p["ep_0"] - self.ep)
        dep_dt = -assoc + dissoc + p["k_I"] * p["I_0"]

        # --------------------------------------------------------------
        # I ODE (38) — includes CaN & PKA contributions
        # dI/dt = -kF I ep + kB(ep0 - ep) 
        #         + v_PKA * I0/(I0 + K_PKA)
        #         - v_CaN * I * (c^3 / (k_h2^3 + c^3))
        # --------------------------------------------------------------
        hill_can = (c_post ** 3) / (p["k_h2"] ** 3 + c_post ** 3)

        term_PKA = p["v_PKA"] * (p["I_0"] / (p["I_0"] + p["K_PKA"]))
        term_CaN = p["v_CaN"] * self.I * hill_can

        dI_dt = -assoc + dissoc + term_PKA - term_CaN

        # --------------------------------------------------------------
        # State update
        # --------------------------------------------------------------
        self.P += dt * dP
        self.ep += dt * dep_dt
        self.I += dt * dI_dt

        # Clamp to prevent negative concentrations
        self.P = np.clip(self.P, 0, None)
        self.ep = np.clip(self.ep, 0, p["ep_0"])
        self.I = max(self.I, 0)

    # ==================================================================
    # (39) Sigmoidal α-modulation (NO effect on presynaptic α increase)
    # ==================================================================
    def get_alpha_modulation(self):
        p = self.p

        # 1. Adım: Fosforile olmuş alt birimlerin KESRİNİ topla (0.0 ile 1.0 arası bir sayı çıkar)
        fraction_P = np.sum(self.P[1:])

        # 2. Adım: Bunu Molariteye çevir (ÇÜNKÜ P_half PARAMETRESİ MOLAR CİNSİNDEN!)
        # Kesir * Toplam Konsantrasyon = Anlık Molar Değer
        total_P_molar = fraction_P * p["e_k"]

        # 3. Adım: Sigmoid hesabını artık Molar vs Molar olarak yapabiliriz
        # (Eski hatan: fraction_P ile P_half'ı kıyaslıyordun, o yüzden hep tavan yapıyordu)
        exponent = -((total_P_molar - p["P_half"]) / p["k_half"])
        
        # Matematiksel hata (overflow) olmasın diye sınırla
        exponent = np.clip(exponent, -50, 50)

        k_syt_eff = p["k_syt"] / (1.0 + np.exp(exponent))

        return k_syt_eff