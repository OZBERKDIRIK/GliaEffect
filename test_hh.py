# Dosya Yolu: test_hh.py

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# src klasörünü yola ekle
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.parameters.pre_synaptic_params import PRE_SYNAPTIC_PARAMS
    from src.models.hh import PresynapticHH
except ImportError as e:
    # Alternatif import (Klasör yapısına göre)
    from parameters.pre_synaptic_params import PRE_SYNAPTIC_PARAMS
    from models.hh import PresynapticHH

def test_action_potential():
    print("--- 1. BÖLÜM DOĞRULAMA: Pre-Sinaptik HH Modeli ---")
    
    # Model Başlatma
    hh = PresynapticHH(PRE_SYNAPTIC_PARAMS)
    
    # Süre Ayarları
    # 5 Hz uyarım demek, saniyede 5 spike demektir.
    # 1000 ms (1 saniye) simüle edelim.
    T_total = 1000.0 
    dt = 0.01 # Hassas adım
    steps = int(T_total / dt)
    time = np.linspace(0, T_total, steps)
    
    # Kayıt
    V_trace = np.zeros(steps)
    I_app_trace = np.zeros(steps) # Uygulanan akımı da görelim
    
    print(f"Simülasyon Başlıyor: {T_total} ms...")
    
    for i in range(steps):
        t_ms = time[i]
        
        # Voltajı güncelle
        v = hh.step(dt, t_ms)
        
        # Uygulanan akımı al (Kontrol için)
        i_app = hh.get_applied_current(t_ms)
        
        V_trace[i] = v
        I_app_trace[i] = i_app

    print("Simülasyon Bitti. Grafik çiziliyor...")
    
    # Analiz
    spikes = np.where(V_trace > 0)[0] # 0 mV'yi geçen noktalar
    # Basit bir spike sayımı (Ardışık noktaları filtrelemeden kaba taslak)
    # Daha doğru sayım için peak detection gerekir ama görsel kontrol yeterli.
    
    plt.figure(figsize=(10, 8))
    
    # Grafik 1: Voltaj
    plt.subplot(2, 1, 1)
    plt.plot(time, V_trace, 'k', linewidth=1)
    plt.title('Pre-Sinaptik Voltaj ($V_{pre}$)')
    plt.ylabel('Voltaj (mV)')
    plt.grid(True)
    
    # Beklenen aralık çizgileri
    plt.axhline(y=-70, color='r', linestyle='--', label='Dinlenme (-70 mV)')
    plt.axhline(y=30, color='g', linestyle='--', label='Spike Zirvesi (~30 mV)')
    plt.legend(loc='upper right')
    
    # Grafik 2: Uygulanan Akım
    plt.subplot(2, 1, 2)
    plt.plot(time, I_app_trace, 'b')
    plt.title('Uygulanan Akım ($I_{app}$) - 5 Hz Olmalı')
    plt.ylabel('Akım ($\mu A/cm^2$)')
    plt.xlabel('Zaman (ms)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_action_potential()