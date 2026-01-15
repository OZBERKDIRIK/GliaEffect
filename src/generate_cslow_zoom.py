import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# ----------------------------------------------------------------
# 1. ORTAM VE İMPORTLAR
# ----------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

try:
    # Parametreler (Düzeltilmiş halleriyle)
    from parameters.pre_synaptic_params import PRE_SYNAPTIC_PARAMS
    from parameters.ca_params import CA_PARAMS
    # Modeller
    from models.hh import PresynapticHH
    from models.calcium_model import PresynapticCalciumDynamics
    print("Modüller başarıyla yüklendi.")
except ImportError as e:
    print(f"HATA: Modüller bulunamadı.\n{e}")
    sys.exit(1)

def run_cslow_zoom_v2():
    print("================================================================")
    print("--- PRESİNAPTİK C_SLOW ZOOM ANALİZİ (V2 - FULL MODEL) ---")
    print("Hedef: Kalın görünen hattın içindeki periyodik 5Hz ve yavaş salınımları göstermek.")
    print("================================================================")

    # ----------------------------------------------------------------
    # 2. AYARLAR
    # ----------------------------------------------------------------
    T_total = 30000.0  # 30 Saniye çalıştıralım (Osilasyonun oturması için)
    dt = 0.05          # Hassas çözüm
    steps = int(T_total / dt)
    time_array = np.linspace(0, T_total, steps)
    
    # Modelleri Başlat
    hh_model = PresynapticHH(PRE_SYNAPTIC_PARAMS)
    ca_model = PresynapticCalciumDynamics(CA_PARAMS)
    
    # Kayıt Dizileri
    # Zoom yapacağımız için bu sefer kaydı sık tutalım (Detay kaybolmasın)
    rec_step = 5  
    rec_size = steps // rec_step
    rec_time = time_array[::rec_step]
    
    rec_c_slow = np.zeros(rec_size)
    
    print(f"Simülasyon başladı ({T_total/1000} sn)...")

    # ----------------------------------------------------------------
    # 3. SİMÜLASYON (Voltaj + Kalsiyum)
    # ----------------------------------------------------------------
    # c_slow'un hareketlenmesi için sisteme biraz Glutamat (feedback) verelim
    GLU_INPUT = 2.0 # uM
    
    for i in range(steps):
        t_ms = time_array[i]
        dt_sec = dt * 1e-3
        
        # 1. Voltajı Hesapla (HH Modeli) -> 5 Hz Spike üretecek
        V_pre_mV = hh_model.step(dt, t_ms, I_inj=0.0)
        V_pre_volts = V_pre_mV * 1e-3
        
        # 2. Kalsiyumu Hesapla (Voltaj girdisi ile)
        # Bu sayede c_slow, voltajdaki değişimlerden (dolaylı yoldan) etkilenecek
        ca_model.step(dt_sec, V_pre_volts, glu=GLU_INPUT)
        
        # Kayıt
        if i % rec_step == 0:
            idx = i // rec_step
            rec_c_slow[idx] = ca_model.c_slow * 1e6 # uM
            
    print("Simülasyon bitti. Kanıt grafiği çiziliyor...")

    # ----------------------------------------------------------------
    # 4. GÖRSELLEŞTİRME (ZOOM)
    # ----------------------------------------------------------------
    # Hocanın istediği "Periyodik Davranışı" göstermek için
    # 20. ile 22. saniyeler arasına (2 saniyelik pencere) zoom yapalım.
    t_sec = rec_time / 1000.0
    
    # ZOOM PENCERESİ: 20 sn - 22 sn
    mask = (t_sec >= 20.0) & (t_sec <= 22.0)
    t_zoom = t_sec[mask]
    c_zoom = rec_c_slow[mask]
    
    plt.figure(figsize=(10, 5))
    
    # Yeşil renkte çizelim (Orijinal grafiğe sadık kalarak)
    plt.plot(t_zoom, c_zoom, color='#2ca02c', linewidth=2, label='$c_{slow}$ (Zoom)')
    
    plt.ylabel("Presinaptik $c_{slow}$ ($\mu M$)", fontsize=14)
    plt.xlabel("Zaman (s)", fontsize=14)
    plt.title("Yavaş Kalsiyum Dinamiği - Detay Analizi (Zoom: 20-22 sn)", fontsize=14, fontweight='bold')
    
    # Grid ekle ki salınımın düzeni belli olsun
    plt.grid(True, which='major', linestyle='-', alpha=0.5)
    plt.grid(True, which='minor', linestyle=':', alpha=0.2)
    plt.minorticks_on()
    
    plt.legend(loc='upper right', fontsize=12)
    
    # Açıklayıcı Not
    plt.text(0.02, 0.05, "Not: Genel grafikteki yoğunluk, burada görülen\n5 Hz modülasyonlu periyodik salınımlardan kaynaklanmaktadır.", 
             transform=plt.gca().transAxes, fontsize=11, 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round'))

    plt.tight_layout()
    
    save_name = "Presynaptic_Cslow_Zoom_Final.png"
    plt.savefig(save_name, dpi=300)
    print(f"Kanıt grafiği '{save_name}' olarak kaydedildi.")
    plt.show()

if __name__ == "__main__":
    run_cslow_zoom_v2()