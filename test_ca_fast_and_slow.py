# Dosya Yolu: test_ca_fast_slow.py

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# src klasörünü yola ekle
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from parameters.pre_synaptic_params import PRE_SYNAPTIC_PARAMS
    from parameters.ca_params import CA_PARAMS
    from models.hh import PresynapticHH
    from models.calcium_model import PresynapticCalciumDynamics
except ImportError as e:
    print(f"Modül Hatası: {e}")
    sys.exit(1)

def test_calcium_dynamics():
    print("--- 2. BÖLÜM DOĞRULAMA: Pre-Sinaptik Kalsiyum (Hızlı + Yavaş) ---")
    
    # Modelleri Başlat
    hh = PresynapticHH(PRE_SYNAPTIC_PARAMS)
    ca = PresynapticCalciumDynamics(CA_PARAMS)
    
    # Ayarlar
    T_total = 1000.0 # 1000 ms (1 saniye)
    dt = 0.005       # Hassas adım
    steps = int(T_total / dt)
    time = np.linspace(0, T_total, steps)
    
    # Kayıt Dizileri
    rec_V = np.zeros(steps)
    rec_Ca_Fast = np.zeros(steps)
    rec_Ca_Slow = np.zeros(steps)
    rec_Ca_Total = np.zeros(steps)
    rec_Ca_ER = np.zeros(steps)
    rec_IP3 = np.zeros(steps)
    rec_Glu_Input = np.zeros(steps)
    
    print("Simülasyon Başlıyor... (Yapay Glutamat Enjeksiyonu ile)")
    
    for i in range(steps):
        t_ms = time[i]
        dt_s = dt * 1e-3
        
        # 1. HH Modeli (Voltaj Üret)
        V_pre = hh.step(dt, t_ms)
        
        # 2. YAPAY GLUTAMAT SİNYALİ (TEST İÇİN)
        # Amaç: IP3 üretimini ve Slow Ca salınımını zorlamak.
        # 400. ms ile 800. ms arasında 1 mM glutamat verelim.
        if 400 < t_ms < 800:
            glu_input = 1.0e-3 # 1 mM (Yüksek doz)
        else:
            glu_input = 0.0
            
        # 3. Kalsiyum Modeli
        # Not: V_pre mV cinsinden geliyor, modele Volt (V_pre * 1e-3) olarak girmeli.
        c_total = ca.step(dt_s, V_pre * 1e-3, glu=glu_input)
        
        # 4. Kayıt (uM cinsinden)
        rec_V[i] = V_pre
        rec_Ca_Total[i] = c_total * 1e6
        rec_Ca_Fast[i]  = ca.c_fast * 1e6
        rec_Ca_Slow[i]  = ca.c_slow * 1e6
        rec_Ca_ER[i]    = ca.c_ER * 1e6
        rec_IP3[i]      = ca.p_ip3 * 1e6
        rec_Glu_Input[i] = glu_input * 1e3 # mM

    print("Çiziliyor...")
    
    # --- GÖRSELLEŞTİRME ---
    fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    
    # 1. Voltaj
    axes[0].plot(time, rec_V, 'k', lw=1)
    axes[0].set_title('A. Pre-Sinaptik Voltaj (HH)')
    axes[0].set_ylabel('mV')
    
    # 2. Fast Ca
    axes[1].plot(time, rec_Ca_Fast, 'r', lw=1)
    axes[1].set_title('B. Hızlı Kalsiyum (VGCC Kanalları)')
    axes[1].set_ylabel('[Ca] (uM)')
    
    # 3. Glutamat Girişi (Test Sinyali)
    axes[2].plot(time, rec_Glu_Input, 'g', lw=2)
    axes[2].set_title('C. Yapay Glutamat Girişi (Test Sinyali)')
    axes[2].set_ylabel('mM')
    
    # 4. IP3
    axes[3].plot(time, rec_IP3, 'brown', lw=2)
    axes[3].set_title('D. IP3 Üretimi (Glutamat Varken Artmalı)')
    axes[3].set_ylabel('uM')
    
    # 5. Slow Ca ve ER
    axes[4].plot(time, rec_Ca_Slow, 'orange', label='Slow Ca (Sitozol)')
    axes[4].plot(time, rec_Ca_ER, 'purple', label='ER Ca (Depo)')
    axes[4].set_title('E. Yavaş Kalsiyum ve ER Deposu')
    axes[4].set_ylabel('uM')
    axes[4].legend()
    axes[4].set_xlabel('Zaman (ms)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_calcium_dynamics()