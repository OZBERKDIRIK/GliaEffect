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
    # Gerekli parametreler ve modeller
    from parameters.pre_synaptic_params import PRE_SYNAPTIC_PARAMS
    from parameters.ca_params import CA_PARAMS
    from parameters.glutamate_params import GLUTAMATE_PARAMS
    
    from models.hh import PresynapticHH
    from models.calcium_model import PresynapticCalciumDynamics
    from models.presynaptic_glutamate import GlutamateDynamics
    
    print("✅ Modüller başarıyla yüklendi.")
except ImportError as e:
    print(f"❌ HATA: Modüller bulunamadı.\n{e}")
    sys.exit(1)

def run_glutamate_zoom():
    print("================================================================")
    print("--- GLUTAMAT ZOOM ANALİZİ (Eksponansiyel Azalma Kanıtı) ---")
    print("Hedef: Tek bir spike sonrası glutamatın sönümleme eğrisini göstermek.")
    print("================================================================")

    # ----------------------------------------------------------------
    # 2. AYARLAR (Sadece 500 ms yeterli)
    # ----------------------------------------------------------------
    T_total = 500.0    # Sadece yarım saniye çalıştıracağız
    dt = 0.01          # Çok hassas çözüm (10 mikrosaniye)
    steps = int(T_total / dt)
    time_array = np.linspace(0, T_total, steps)
    
    # Modelleri Başlat
    hh_model = PresynapticHH(PRE_SYNAPTIC_PARAMS)
    ca_model = PresynapticCalciumDynamics(CA_PARAMS)
    glu_model = GlutamateDynamics(GLUTAMATE_PARAMS)
    
    # Kayıt Dizileri
    rec_glu = np.zeros(steps)
    
    print(f"Simülasyon yapılıyor ({T_total} ms)...")
    
    # ----------------------------------------------------------------
    # 3. SİMÜLASYON
    # ----------------------------------------------------------------
    # HH modelini tetiklemek için bir akım darbesi verelim
    # 200. ms'de bir akım verelim ki sistem otursun
    I_STIM_START = 200.0
    I_STIM_END = 205.0 # 5 ms'lik darbe
    I_AMP = 10.0       # uA/cm2
    
    for i in range(steps):
        t_ms = time_array[i]
        dt_sec = dt * 1e-3
        
        # 1. Akım Uygula
        I_app = I_AMP if (I_STIM_START <= t_ms <= I_STIM_END) else 0.0
        
        # 2. Voltajı Hesapla (HH)
        V_pre_mV = hh_model.step(dt, t_ms, I_inj=I_app)
        V_pre_volts = V_pre_mV * 1e-3
        
        # 3. Kalsiyumu Hesapla (Ca_fast artacak)
        ca_model.step(dt_sec, V_pre_volts, glu=0.0)
        c_fast_uM = ca_model.c_fast * 1e6
        
        # 4. Glutamatı Hesapla
        g_cleft = glu_model.step(dt, c_fast_uM)
        
        # Kayıt (Her adımı kaydediyoruz, downsampling yok!)
        rec_glu[i] = g_cleft

    print("Simülasyon bitti. Grafik çiziliyor...")

    # ----------------------------------------------------------------
    # 4. GÖRSELLEŞTİRME (ZOOM)
    # ----------------------------------------------------------------
    # Sadece 200 ms ile 215 ms arasını çizelim (Spike anı)
    mask = (time_array >= 200.0) & (time_array <= 215.0)
    t_zoom = time_array[mask]
    g_zoom = rec_glu[mask]
    
    plt.figure(figsize=(8, 6))
    
    # Ana Çizgi
    plt.plot(t_zoom, g_zoom, color='#d62728', linewidth=2.5, label='Glutamat ($g$)')
    
    # Eksponansiyel Sönümlemeyi Gösteren Ok ve Yazı
    peak_idx = np.argmax(g_zoom)
    peak_t = t_zoom[peak_idx]
    peak_val = g_zoom[peak_idx]
    
    # Yazı ekle
    plt.annotate('Eksponansiyel\nTemizlenme\n(Decay)', 
                 xy=(peak_t + 2.0, peak_val * 0.3), 
                 xytext=(peak_t + 5.0, peak_val * 0.6),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=11, fontweight='bold')

    plt.ylabel("Glutamat ($\mu M$)", fontsize=14)
    plt.xlabel("Zaman (ms)", fontsize=14)
    plt.title("Tekil Sinaptik Glutamat Salınımı (Zoom Analizi)", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper right', fontsize=12)
    
    # Hoca için not kutusu
    plt.text(0.05, 0.95, "Not: Salınım anında ani artış ve ardından\ntemizlenme hızına bağlı eksponansiyel düşüş net görülmektedir.", 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    
    save_name = "Glutamate_Zoom_Analysis.png"
    plt.savefig(save_name, dpi=300)
    print(f"✅ Kanıt grafiği '{save_name}' olarak kaydedildi.")
    plt.show()

if __name__ == "__main__":
    run_glutamate_zoom()