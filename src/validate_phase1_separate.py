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
    from parameters.pre_synaptic_params import PRE_SYNAPTIC_PARAMS
    from parameters.ca_params import CA_PARAMS
    from models.hh import PresynapticHH
    from models.calcium_model import PresynapticCalciumDynamics
    print("Modüller başarıyla yüklendi.")
except ImportError as e:
    print(f"HATA: Modüller bulunamadı. Lütfen 'src' klasörünü kontrol edin.\n{e}")
    sys.exit(1)

def run_validation_separate():
    print("================================================================")
    print("--- TEWARI & MAJUMDAR (2012) - FIG 3 AYRIK VALIDASYON ---")
    print("Hedef: 60 saniye, 5Hz aktivite. Grafikler ayrı ayrı kaydedilecek.")
    print("================================================================")

    # ----------------------------------------------------------------
    # 2. AYARLAR
    # ----------------------------------------------------------------
    T_total = 60000.0  # 60 Saniye
    dt = 0.05          # Hassas çözüm
    steps = int(T_total / dt)
    time_array = np.linspace(0, T_total, steps)
    
    # Modelleri Başlat
    hh_model = PresynapticHH(PRE_SYNAPTIC_PARAMS)
    ca_pre_model = PresynapticCalciumDynamics(CA_PARAMS)
    
    # Kayıt Dizileri
    rec_step = 20
    rec_size = steps // rec_step
    rec_time = time_array[::rec_step]
    
    rec_V_pre = np.zeros(rec_size)
    rec_Ca_Fast_nM = np.zeros(rec_size)
    
    print(f"Toplam Adım: {steps} (Tahmini süre: 10-15 sn)...")

    # ----------------------------------------------------------------
    # 3. SİMÜLASYON DÖNGÜSÜ
    # ----------------------------------------------------------------
    for i in range(steps):
        t_ms = time_array[i]
        dt_sec = dt * 1e-3
        
        # A. Voltaj (HH) - I_inj=0.0 (Sadece bazal 5Hz)
        V_pre_mV = hh_model.step(dt, t_ms, I_inj=0.0)
        
        # B. Kalsiyum (Ca_fast) - Glu=0.0 (Feedback yok)
        V_pre_volts = V_pre_mV * 1e-3
        ca_pre_model.step(dt_sec, V_pre_volts, glu=0.0) 
        
        # Kayıt
        if i % rec_step == 0:
            idx = i // rec_step
            rec_V_pre[idx] = V_pre_mV
            # Birim Dönüşümü: Molar -> nM
            rec_Ca_Fast_nM[idx] = ca_pre_model.c_fast * 1e9 
            
    print("Simülasyon bitti. Grafikler oluşturuluyor...")

    # Zaman eksenini saniyeye çevir
    t_sec = rec_time / 1000.0

    # ----------------------------------------------------------------
    # 4. GRAFİK 1: VOLTAJ (V_pre)
    # ----------------------------------------------------------------
    plt.figure(figsize=(10, 4)) # Geniş ve kısa (Makale formatı)
    plt.plot(t_sec, rec_V_pre, 'k', linewidth=0.5)
    plt.ylabel("$V_{pre}$ (mV)", fontsize=12)
    plt.xlabel("Zaman (sn)", fontsize=12) # İstersen kaldırabilirsin
    plt.title("Pre-synaptic Aksiyon Potansiyeli (Model Sonucu)", loc='left', fontsize=12, fontweight='bold')
    plt.ylim(-90, 50)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_name_v = "Validation_Fig3A_Voltage.png"
    plt.savefig(save_name_v, dpi=300)
    print(f"-> {save_name_v} kaydedildi.")
    plt.close() # Belleği temizle

    # ----------------------------------------------------------------
    # 5. GRAFİK 2: KALSİYUM (Ca_fast)
    # ----------------------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(t_sec, rec_Ca_Fast_nM, 'k', linewidth=0.5)
    plt.ylabel("$Ca^{2+}$ (nM)", fontsize=12) # Birim nM
    plt.xlabel("Zaman (sn)", fontsize=12)
    plt.title("Kalsiyum Konsantrasyon Sonucu (Model Sonucu)", loc='left', fontsize=12, fontweight='bold')
    plt.ylim(0, 6000) # Makale sınırları
    plt.grid(True, alpha=0.3)
    
    plt.text(0.02, 0.85, "Birim Notu: Çıktı nM'a dönüştürülmüştür\n(5.5 uM = 5500 nM)", 
             transform=plt.gca().transAxes, fontsize=9, 
             bbox=dict(facecolor='yellow', alpha=0.2, edgecolor='none'))

    plt.tight_layout()
    
    save_name_ca = "Validation_Fig3B_Calcium.png"
    plt.savefig(save_name_ca, dpi=300)
    print(f"-> {save_name_ca} kaydedildi.")
    plt.close()

    print("Tüm işlemler tamamlandı.")

if __name__ == "__main__":
    run_validation_separate()