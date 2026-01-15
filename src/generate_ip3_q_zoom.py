import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker # Eksen formatı için gerekli
import sys
import os

# ----------------------------------------------------------------
# 1. ORTAM VE İMPORTLAR
# ----------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

try:
    # Parametreler ve Modeller
    from parameters.pre_synaptic_params import PRE_SYNAPTIC_PARAMS
    from parameters.ca_params import CA_PARAMS
    from parameters.glutamate_params import GLUTAMATE_PARAMS
    from parameters.astrocyte_params import ASTROCYTE_PARAMS
    from parameters.gliatransmitter_params import GLIATRANSMITTER_PARAMS
    from parameters.post_synaptic_params import POST_SYNAPTIC_PARAMS
    from parameters.post_synaptic_ca_params import POST_SYNAPTIC_CA_PARAMS
    from parameters.camkii_params import CAMKII_PARAMS

    from models.hh import PresynapticHH
    from models.calcium_model import PresynapticCalciumDynamics
    from models.presynaptic_glutamate import GlutamateDynamics
    from models.astrocyte import AstrocyteDynamics
    from models.gliatransmitter import GliatransmitterDynamics
    from models.post_synaptic import PostSynapticDynamics
    from models.post_synaptic_ca import PostSynapticCalciumDynamics
    from models.camkii import CaMKIIDynamics

    print("✅ Modüller başarıyla yüklendi.")
except ImportError as e:
    print(f"❌ HATA: Modüller bulunamadı. Lütfen dosya konumunu kontrol et.\n{e}")
    sys.exit(1)

def run_ip3_q_zoom():
    print("================================================================")
    print("--- IP3 ve Q (GATING) ZOOM ANALİZİ (DÜZELTİLMİŞ) ---")
    print("Hedef: Eksen kayması (Offset) olmadan net grafik çizmek.")
    print("================================================================")

    # ----------------------------------------------------------------
    # 2. AYARLAR
    # ----------------------------------------------------------------
    T_total = 30000.0  # 30 Saniye
    dt = 0.05          # Zaman adımı
    steps = int(T_total / dt)
    time_array = np.linspace(0, T_total, steps)
    
    # Modelleri Başlat
    hh_model = PresynapticHH(PRE_SYNAPTIC_PARAMS)
    ca_pre_model = PresynapticCalciumDynamics(CA_PARAMS)
    glu_pre_model = GlutamateDynamics(GLUTAMATE_PARAMS)
    astro_model = AstrocyteDynamics(ASTROCYTE_PARAMS)
    glia_trans_model = GliatransmitterDynamics(GLIATRANSMITTER_PARAMS)
    
    # Kayıt Dizileri
    rec_ip3 = np.zeros(steps, dtype=np.float32)
    rec_q   = np.zeros(steps, dtype=np.float32)
    
    # Değişkenler
    current_glu_extra = 0.0
    current_glu_syn = 0.0
    current_alpha = GLUTAMATE_PARAMS['alpha']
    
    print(f"Simülasyon yapılıyor ({T_total/1000} sn)...")
    
    # ----------------------------------------------------------------
    # 3. SİMÜLASYON DÖNGÜSÜ
    # ----------------------------------------------------------------
    for i in range(steps):
        t_ms = time_array[i]
        dt_sec = dt * 1e-3
        
        # 5 Hz gibi davranması için akım
        I_stim = 10.0 
        
        # --- Pre-Sinaptik Zincir ---
        V_pre_mV = hh_model.step(dt, t_ms, I_stim)
        V_pre_volts = V_pre_mV * 1e-3
        
        # Calcium Model Step
        ca_pre_model.step(dt_sec, V_pre_volts, glu=current_glu_extra * 1e-6)
        
        # --- KAYIT ---
        rec_ip3[i] = ca_pre_model.p_ip3 * 1e6 # M -> uM
        rec_q[i]   = ca_pre_model.q           # 0-1 arası
        
        # Döngünün kalanı
        Ca_pre_uM = ca_pre_model.c_fast * 1e6
        current_glu_syn = glu_pre_model.step(dt, Ca_pre_uM)
        Ca_astro = astro_model.compute_derivatives(dt_sec, current_glu_syn * 1e-6)
        current_glu_extra = glia_trans_model.step(dt, Ca_astro * 1e6)

    print("Simülasyon bitti. Grafik oluşturuluyor...")

    # ----------------------------------------------------------------
    # 4. GÖRSELLEŞTİRME (ZOOM: 20.0 - 25.0 sn)
    # ----------------------------------------------------------------
    t_sec = time_array / 1000.0
    
    # Zoom Aralığı (Biraz daha geniş tuttum ki dalga tam görünsün)
    t_start = 20.0
    t_end = 25.0
    mask = (t_sec >= t_start) & (t_sec <= t_end)
    
    t_zoom = t_sec[mask]
    ip3_zoom = rec_ip3[mask]
    q_zoom = rec_q[mask]
    
    # Grafik Ayarları
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # --- Sol Eksen: IP3 (Mavi) ---
    color1 = '#1f77b4' 
    ax1.set_xlabel('Zaman (s)', fontsize=12)
    ax1.set_ylabel('Presinaptik $IP_3$ ($\mu M$)', color=color1, fontsize=12, fontweight='bold')
    ax1.plot(t_zoom, ip3_zoom, color=color1, linewidth=2.5, label='$IP_3$ Konsantrasyonu')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # --- DÜZELTME BURADA: Offset (Bilimsel Gösterim) Kapatılıyor ---
    y_formatter = ticker.ScalarFormatter(useOffset=False)
    ax1.yaxis.set_major_formatter(y_formatter)
    
    # --- Sağ Eksen: q (Kırmızı) ---
    ax2 = ax1.twinx()
    color2 = '#d62728' 
    ax2.set_ylabel('IP$_3$R Açık Kalma Olasılığı ($q$)', color=color2, fontsize=12, fontweight='bold')
    ax2.plot(t_zoom, q_zoom, color=color2, linewidth=2.5, linestyle='--', label='$q$ (Gating)')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(-0.05, 1.05) # 0-1 arasını net görelim
    
    # Başlık
    plt.title("Mekanizma Analizi: $IP_3$ Üretimi ve Kanal Gating ($q$) İlişkisi", fontsize=14, fontweight='bold')
    
    # Not Kutusu
    text_str = "Mekanizma:\n1. $IP_3$ (Mavi) artışı kanalı açmaya zorlar.\n2. Ca$^{2+}$ artınca (İnhibisyon) $q$ (Kırmızı) düşer."
    ax1.text(0.02, 0.95, text_str, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    
    save_name = "IP3_q_Zoom_Analysis.png"
    plt.savefig(save_name, dpi=300)
    print(f"✅ Kanıt grafiği '{save_name}' olarak kaydedildi.")
    # plt.show() # İstersen açabilirsin

if __name__ == "__main__":
    run_ip3_q_zoom()