import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. PARAMETRELER (SI BİRİMLERİ - ELLE KONTROL EDİLDİ)
# =============================================================================
p = {
    # Fiziksel Sabitler
    'F': 96487.0, 'R': 8.314, 'T': 293.15, 'z_Ca': 2,
    
    # Geometri (LİTRE CİNSİNDEN HACİM ŞART!)
    'A_btn': 1.24e-12,       # m^2
    'V_btn_L': 0.13e-15,     # Litre (0.13 um^3)
    
    # Konsantrasyonlar (Molar)
    'c_ext': 2.0e-3, 'c_i_rest': 0.1e-6,
    
    # N-Tipi Kanal (I_Ca)
    'rho_Ca': 3.2e12,        # /m^2
    'g_Ca': 2.3e-12,         # Siemens
    'V_mCa': -0.017,         # Volt (-17 mV)
    'k_mCa': 0.0084,         # Volt
    'tau_mCa': 0.01,         # Saniye
    
    # Pompa ve Sızıntı
    'v_PMCA_max': 0.004,     # A/m^2
    'K_PMCA': 0.1e-6,        # Molar
    'v_leak': 2.66e-3,       # /s (Hız sabiti)
    
    # ER (Yavaş)
    'c1': 0.185, 'v1': 30.0, 'v2': 0.2374, 'v3': 90.0e-6, 'k3': 0.1e-6,
    
    # IP3
    'd1': 0.13e-6, 'd2': 1.049e-6, 'd3': 0.9434e-6, 'd5': 0.08234e-6, 'a2': 0.2e6,
    'v_g': 0.062e-6, 'k_g': 0.78e-6, 'tau_p': 0.14, 'p0': 0.16e-6
}

# =============================================================================
# 2. MODEL SINIFI (KORUMALI)
# =============================================================================
class CalciumModelFixed:
    def __init__(self, params):
        self.p = params
        
        # Dönüşüm Faktörleri
        # Akım (A) -> Molar Hız (M/s)
        self.k_flux = self.p['A_btn'] / (self.p['z_Ca'] * self.p['F'] * self.p['V_btn_L'])
        
        # Nernst Potansiyeli
        rt_zf = (self.p['R'] * self.p['T']) / (self.p['z_Ca'] * self.p['F'])
        self.V_Ca = rt_zf * np.log(self.p['c_ext'] / self.p['c_i_rest'])
        print(f"Nernst Potansiyeli (V_Ca): {self.V_Ca:.4f} V")
        
        # Başlangıç Değerleri
        self.c_fast = 0.0
        self.c_slow = self.p['c_i_rest']
        self.c_ER = 400e-6
        self.m = 0.0
        self.p_ip3 = self.p['p0']
        self.q = 0.8

    def step(self, dt, V_pre_volts, glu_conc):
        # Toplam Kalsiyum
        c_i = self.c_fast + self.c_slow
        c_i = max(c_i, 1e-9) # Güvenlik

        # --- FAST DYNAMICS (VGCC) ---
        # 1. Gating
        m_inf = 1.0 / (1.0 + np.exp((self.p['V_mCa'] - V_pre_volts) / self.p['k_mCa']))
        self.m += dt * (m_inf - self.m) / self.p['tau_mCa']
        
        # 2. Akımlar
        # I_Ca (Amper)
        conductance = self.p['rho_Ca'] * self.p['A_btn'] * (self.m**2) * self.p['g_Ca']
        I_Ca = conductance * (V_pre_volts - self.V_Ca)
        
        # I_pump (Amper)
        I_pump_dens = self.p['v_PMCA_max'] * (c_i**2) / (c_i**2 + self.p['K_PMCA']**2)
        I_pump = I_pump_dens * self.p['A_btn']
        
        # J_leak (M/s)
        # Sızıntı formülünü basitleştirdik: Hedef dinlenme değerine çekmek
        # J_leak = v_leak * (c_rest - c_i) (Daha stabil)
        # Ama makalede (c_ext - c_i) denmiş. Biz dengeyi sağlamak için şunu kullanacağız:
        # Denge anında: J_in = J_out. 
        # Dinlenmede I_Ca ~ 0, I_pump ~ 0. O yüzden J_leak ~ 0 olmalı.
        J_leak = self.p['v_leak'] * (self.p['c_ext'] - c_i)
        
        # 3. Türev (dc_fast)
        # I_Ca (inward) negatiftir -> -I_Ca pozitiftir.
        dc_fast = (-I_Ca * self.k_flux) - (I_pump * self.k_flux) + J_leak
        
        # --- SLOW DYNAMICS (ER) ---
        # Gating
        m_inf_ip3 = self.p_ip3 / (self.p_ip3 + self.p['d1'])
        n_inf_ip3 = c_i / (c_i + self.p['d5'])
        alpha_q = self.p['a2'] * self.p['d2'] * (self.p_ip3 + self.p['d1']) / (self.p_ip3 + self.p['d3'])
        beta_q = self.p['a2'] * c_i
        
        self.q += dt * (alpha_q * (1 - self.q) - beta_q * self.q)
        
        # Fluxlar
        prob = (m_inf_ip3**3) * (n_inf_ip3**3) * (self.q**3)
        J_IP3R = self.p['c1'] * self.p['v1'] * prob * (self.c_ER - c_i)
        J_SERCA = self.p['v3'] * (c_i**2) / (self.p['k3']**2 + c_i**2)
        J_Leak_ER = self.p['c1'] * self.p['v2'] * (self.c_ER - c_i)
        
        dc_slow = J_IP3R + J_Leak_ER - J_SERCA
        dc_ER = -(1/self.p['c1']) * dc_slow
        
        # IP3
        prod = self.p['v_g'] * (glu_conc**0.7) / (self.p['k_g']**0.7 + glu_conc**0.7)
        deg = self.p['tau_p'] * (self.p_ip3 - self.p['p0'])
        self.p_ip3 += dt * (prod - deg)
        
        # --- GÜNCELLEME ---
        self.c_fast += dt * dc_fast
        self.c_slow += dt * dc_slow
        self.c_ER += dt * dc_ER
        
        # Yüksek sızıntıyı engellemek için drift correction (opsiyonel ama iyi)
        self.c_fast = max(self.c_fast, 0.0)
        
        return self.c_fast + self.c_slow

# =============================================================================
# 3. TEST VE GÖRSELLEŞTİRME
# =============================================================================
def run_test():
    model = CalciumModelFixed(p)
    
    # 5 Hz Voltaj Sinyali Üret (Yapay)
    T = 1000 # ms
    dt = 0.01
    steps = int(T/dt)
    time = np.linspace(0, T, steps)
    
    V_trace = np.ones(steps) * -0.070 # -70 mV
    # Her 200 ms'de bir spike (2 ms sürer, +30 mV)
    for i in range(0, steps, int(200/dt)):
        V_trace[i:i+int(2/dt)] = 0.030 
        
    # Glutamat (400-800 ms arası)
    Glu_trace = np.zeros(steps)
    Glu_trace[int(400/dt):int(800/dt)] = 1e-3
    
    # Kayıt
    rec_Ca = []
    rec_ER = []
    
    for i in range(steps):
        c = model.step(dt*1e-3, V_trace[i], Glu_trace[i])
        rec_Ca.append(c * 1e6) # uM
        rec_ER.append(model.c_ER * 1e6) # uM
        
    # Çiz
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(time, rec_Ca, 'r')
    plt.title('Hızlı Kalsiyum (uM) - BEKLENEN: 5-10 uM arası Spike')
    plt.ylabel('[Ca] uM')
    
    plt.subplot(2,1,2)
    plt.plot(time, rec_ER, 'purple')
    plt.title('ER Deposu (uM) - BEKLENEN: Glutamat gelince azalmalı')
    plt.xlabel('Zaman (ms)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_test()