import numpy as np

class PresynapticHH:
    """
    Tewari & Majumdar (2012) - Hodgkin-Huxley Modeli.
    Voltaj değişimini (V_pre) hesaplar.
    
    GÜNCELLEME NOTU: 
    Makale ile birebir uyum (Spike genliğinin +40 mV olması) için 
    alpha/beta fonksiyonlarındaki voltaj shift değeri +65'ten +70'e çekilmiştir.
    """
    def __init__(self, params):
        self.p = params
        
        # Başlangıç Değerlerini Yükle
        self.V = self.p.get("V_init", -70.0) # Varsayılan -70 yapıldı
        self.m = self.p.get("m_init", 0.05)
        self.h = self.p.get("h_init", 0.6)
        self.n = self.p.get("n_init", 0.32)

    # --- Yardımcı Gating Fonksiyonları (Alpha/Beta) ---
    # Not: Tewari (2012) -70 mV bazlı çalıştığı için V + 70 kullanıldı.
    
    def alpha_n(self, V): 
        # Denklem: 0.01 * (10 - u) / (exp((10-u)/10) - 1)  where u = V + 70
        shift = 70.0
        num = 0.01 * (10 - (V + shift))
        denom = np.exp((10 - (V + shift)) / 10) - 1
        if abs(denom) < 1e-9: return 0.1 # Limit değeri
        return num / denom

    def beta_n(self, V):
        shift = 70.0
        return 0.125 * np.exp(-(V + shift) / 80)

    def alpha_m(self, V): 
        # Denklem: 0.1 * (25 - u) / (exp((25-u)/10) - 1)
        shift = 70.0
        num = 0.1 * (25 - (V + shift))
        denom = np.exp((25 - (V + shift)) / 10) - 1
        if abs(denom) < 1e-9: return 1.0 # Limit değeri
        return num / denom

    def beta_m(self, V):  
        shift = 70.0
        return 4.0 * np.exp(-(V + shift) / 18)

    def alpha_h(self, V): 
        shift = 70.0
        return 0.07 * np.exp(-(V + shift) / 20)
        
    def beta_h(self, V):  
        shift = 70.0
        return 1.0 / (np.exp((30 - (V + shift)) / 10) + 1)

    # --- Uygulanan Akım (Pulse) ---
    def get_applied_current(self, t):
        """Belirtilen frekansta kare dalga akım üretir."""
        # Eğer parametreler yoksa varsayılanları kullan (Güvenlik)
        freq = self.p.get("freq", 5.0)
        width = self.p.get("pulse_width", 10.0) # Makalede 10ms daha uygun
        amp = self.p.get("I_app_amp", 10.0)
        
        if freq <= 0: return 0.0 # Bölme hatası önlemi
        
        period = 1000.0 / freq # ms cinsinden periyot
        if (t % period) <= width:
            return amp
        return 0.0

    # --- Ana Adım Fonksiyonu (Step) ---
    def step(self, dt, t, I_inj=0.0):
        """
        dt: Zaman adımı (ms)
        t: Şu anki zaman (ms)
        I_inj: Dışarıdan enjekte edilen ek akım (uA/cm2) [HFS için]
        """
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
        
        # Kendi iç akımı + Dışarıdan gelen HFS akımı
        I_app_total = self.get_applied_current(t) + I_inj

        # 3. Voltajı Güncelle
        dV = (I_app_total - I_Na - I_K - I_L) / self.p["C_m"]
        self.V += dt * dV
        
        return self.V