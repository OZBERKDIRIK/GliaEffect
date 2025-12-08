# Dosya Yolu: glia effect/src/models/hh.py

import numpy as np

class PresynapticHH:
    """
    Tewari & Majumdar (2012) - Hodgkin-Huxley Modeli.
    Voltaj değişimini (V_pre) hesaplar.
    """
    def __init__(self, params):
        self.p = params
        
        # Başlangıç Değerlerini Yükle
        self.V = self.p["V_init"]
        self.m = self.p["m_init"]
        self.h = self.p["h_init"]
        self.n = self.p["n_init"]

    # --- Yardımcı Gating Fonksiyonları (Alpha/Beta) ---
    def alpha_n(self, V): return (0.01 * (10 - (V + 65))) / (np.exp((10 - (V + 65)) / 10) - 1)
    def beta_n(self, V):  return 0.125 * np.exp(-(V + 65) / 80)

    def alpha_m(self, V): return (0.1 * (25 - (V + 65))) / (np.exp((25 - (V + 65)) / 10) - 1)
    def beta_m(self, V):  return 4.0 * np.exp(-(V + 65) / 18)

    def alpha_h(self, V): return 0.07 * np.exp(-(V + 65) / 20)
    def beta_h(self, V):  return 1.0 / (np.exp((30 - (V + 65)) / 10) + 1)

    # --- Uygulanan Akım (Pulse) ---
    def get_applied_current(self, t):
        """Belirtilen frekansta (5Hz) kare dalga akım üretir."""
        period = 1000.0 / self.p["freq"] # ms cinsinden periyot
        if (t % period) <= self.p["pulse_width"]:
            return self.p["I_app_amp"]
        return 0.0

    # --- Ana Adım Fonksiyonu (Step) ---
    def step(self, dt, t):
        V = self.V
        
        # 1. Gating Değişkenlerini Güncelle
        dm = (self.alpha_m(V) * (1 - self.m)) - (self.beta_m(V) * self.m)
        dh = (self.alpha_h(V) * (1 - self.h)) - (self.beta_h(V) * self.h)
        dn = (self.alpha_n(V) * (1 - self.n)) - (self.beta_n(V) * self.n)
        
        self.m += dt * dm
        self.h += dt * dh
        self.n += dt * dn

        # 2. Akımları Hesapla
        I_Na = self.p["g_Na"] * (self.m**3) * self.h * (V - self.p["V_Na"])
        I_K  = self.p["g_K"]  * (self.n**4) * (V - self.p["V_K"])
        I_L  = self.p["g_L"]  * (V - self.p["V_L"])
        I_app = self.get_applied_current(t)

        # 3. Voltajı Güncelle
        dV = (I_app - I_Na - I_K - I_L) / self.p["C_m"]
        self.V += dt * dV
        
        return self.V