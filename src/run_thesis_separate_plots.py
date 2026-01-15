import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# -------------------------------------------------------------------------
# 1. ORTAM VE İMPORTLAR
# -------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

try:
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
except ImportError as e:
    print(f"Hata: {e}")
    sys.exit(1)

# =========================================================================
# ⚙️ KALİBRE EDİLMİŞ PARAMETRELER (Senin Manuel Ayarların)
# =========================================================================
print(">>> Parametreler yükleniyor (Kalibre edilmiş)...")
CAMKII_PARAMS['K1'] = 0.005       # Yavaşlatılmış üretim
CAMKII_PARAMS['k_h'] = 150.0e-6   # Duyarsızlaştırılmış Hill sabiti
CAMKII_PARAMS['K2'] = 50.0        # Hızlandırılmış yıkım
CAMKII_PARAMS['P_half'] = 25e-6   # Düşürülmüş Eşik (LTP için)
POST_SYNAPTIC_CA_PARAMS['k_s'] = 450.0 # Hızlı pompa

# =========================================================================
# 🧪 DENEY LİSTESİ
# =========================================================================
EXPERIMENTS = [
    {"label": "10Hz",  "current": 5.0},
    {"label": "30Hz",  "current": 8.0},
    {"label": "50Hz",  "current": 10.0},
    {"label": "70Hz",  "current": 15.0},
    {"label": "90Hz",  "current": 20.0},
    {"label": "100Hz", "current": 22.0}
]

# Ana Klasör
MAIN_FOLDER = "Tez_Ayri_Grafikler"

def plot_single(time, data, title, ylabel, color, folder, filename, hline=None):
    """Tek bir grafiği çizip kaydeden yardımcı fonksiyon"""
    plt.figure(figsize=(8, 5)) # Tez için ideal en/boy oranı
    plt.plot(time, data, color=color, linewidth=2)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("Zaman (s)", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Eşik çizgisi varsa ekle (Örn: CaMKII için)
    if hline is not None:
        plt.axhline(y=hline, color='green', linestyle='--', linewidth=2, label=f'Eşik ({hline} uM)')
        plt.legend()
        
    plt.tight_layout()
    plt.savefig(f"{folder}/{filename}", dpi=300) # 300 DPI (Baskı Kalitesi)
    plt.close()

def run_simulation(exp):
    freq_label = exp["label"]
    current_amp = exp["current"]
    
    # Her frekans için alt klasör oluştur
    exp_folder = os.path.join(MAIN_FOLDER, freq_label)
    if not os.path.exists(exp_folder): os.makedirs(exp_folder)

    print(f"\n>>> HESAPLANIYOR: {freq_label} (Akım: {current_amp} uA)...")
    
    # Zaman Ayarları
    T_total = 30000.0
    dt = 0.05
    steps = int(T_total / dt)
    time_array = np.linspace(0, T_total, steps)
    
    # Modelleri Başlat
    hh_model = PresynapticHH(PRE_SYNAPTIC_PARAMS)
    ca_pre_model = PresynapticCalciumDynamics(CA_PARAMS)
    glu_pre_model = GlutamateDynamics(GLUTAMATE_PARAMS)
    astro_model = AstrocyteDynamics(ASTROCYTE_PARAMS)
    glia_trans_model = GliatransmitterDynamics(GLIATRANSMITTER_PARAMS)
    post_neuron_model = PostSynapticDynamics(POST_SYNAPTIC_PARAMS)
    post_ca_model = PostSynapticCalciumDynamics(POST_SYNAPTIC_CA_PARAMS)
    camkii_model = CaMKIIDynamics(CAMKII_PARAMS)

    # Kayıt Dizileri
    rec_step = 20
    rec_size = steps // rec_step
    rec_time = time_array[::rec_step] / 1000.0 
    
    rec_V_pre = np.zeros(rec_size, dtype=np.float32)
    rec_Ca_Fast = np.zeros(rec_size, dtype=np.float32)
    rec_Glu_Syn = np.zeros(rec_size, dtype=np.float32)
    
    rec_IP3_Astro = np.zeros(rec_size, dtype=np.float32)
    rec_Glu_Extra = np.zeros(rec_size, dtype=np.float32)
    
    rec_V_post = np.zeros(rec_size, dtype=np.float32)
    rec_Ca_Post = np.zeros(rec_size, dtype=np.float32)
    
    rec_CaMKII_P = np.zeros(rec_size, dtype=np.float32)
    rec_Alpha = np.zeros(rec_size, dtype=np.float32)

    current_glu_syn = 0.0
    current_glu_extra = 0.0
    base_alpha = GLUTAMATE_PARAMS['alpha']
    current_alpha = base_alpha

    # SİMÜLASYON DÖNGÜSÜ
    for i in range(steps):
        t_ms = time_array[i]
        dt_sec = dt * 1e-3

        if 10000 <= t_ms <= 20000:
            I_stim = current_amp
        else:
            I_stim = 0.0

        # Model Adımları
        V_pre_mV = hh_model.step(dt, t_ms, I_stim)
        Ca_pre = ca_pre_model.step(dt_sec, V_pre_mV*1e-3, glu=current_glu_extra*1e-6)
        
        glu_pre_model.p['alpha'] = current_alpha
        current_glu_syn = glu_pre_model.step(dt, ca_pre_model.c_fast * 1e6)
        
        Ca_astro = astro_model.compute_derivatives(dt_sec, current_glu_syn * 1e-6)
        current_glu_extra = glia_trans_model.step(dt, Ca_astro * 1e6)

        V_post_volts = post_neuron_model.step(dt_sec, current_glu_syn, I_soma_injected=0.0)
        I_AMPA = post_neuron_model.I_AMPA
        Ca_post = post_ca_model.step(dt_sec, V_post_volts, I_AMPA)

        camkii_model.step(dt_sec, Ca_post)
        alpha_mod = camkii_model.get_alpha_modulation()
        current_alpha = base_alpha * (1.0 + alpha_mod)

        # Kayıt
        if i % rec_step == 0:
            idx = i // rec_step
            rec_V_pre[idx] = V_pre_mV
            rec_Ca_Fast[idx] = ca_pre_model.c_fast * 1e6
            rec_Glu_Syn[idx] = current_glu_syn
            rec_IP3_Astro[idx] = astro_model.p_a * 1e6
            rec_Glu_Extra[idx] = current_glu_extra
            rec_V_post[idx] = V_post_volts * 1e3
            rec_Ca_Post[idx] = Ca_post * 1e6
            rec_CaMKII_P[idx] = np.sum(camkii_model.P[1:]) * CAMKII_PARAMS["e_k"] * 1e6
            rec_Alpha[idx] = current_alpha

    print(f"   -> Veriler işlendi. {freq_label} klasörüne kaydediliyor...")

    # --- AYRI AYRI GRAFİK KAYDETME ---

    # 1. Presinaptik Voltaj
    plot_single(rec_time, rec_V_pre, f"Presinaptik Voltaj ({freq_label})", "Voltaj (mV)", "black", exp_folder, "1_Pre_Voltaj.png")

    # 2. Presinaptik Kalsiyum
    plot_single(rec_time, rec_Ca_Fast, f"Presinaptik Kalsiyum ({freq_label})", "Kalsiyum (uM)", "blue", exp_folder, "2_Pre_Kalsiyum.png")

    # 3. Astrosit IP3
    plot_single(rec_time, rec_IP3_Astro, f"Astrosit IP3 Seviyesi ({freq_label})", "IP3 (uM)", "purple", exp_folder, "3_Astro_IP3.png")

    # 4. Gliotransmitter (Salınan Glutamat)
    plot_single(rec_time, rec_Glu_Extra, f"Gliotransmitter Salınımı ({freq_label})", "Glutamat (uM)", "green", exp_folder, "4_Astro_Glutamat.png")

    # 5. Post-Sinaptik Voltaj
    plot_single(rec_time, rec_V_post, f"Post-Sinaptik Voltaj ({freq_label})", "Voltaj (mV)", "grey", exp_folder, "5_Post_Voltaj.png")

    # 6. Post-Sinaptik Kalsiyum (Dendritik)
    plot_single(rec_time, rec_Ca_Post, f"Post-Sinaptik Kalsiyum ({freq_label})", "Kalsiyum (uM)", "orange", exp_folder, "6_Post_Kalsiyum.png")

    # 7. CaMKII (Hafıza Enzimi)
    plot_single(rec_time, rec_CaMKII_P, f"CaMKII Aktivasyonu ({freq_label})", "CaMKII-P (uM)", "magenta", exp_folder, "7_CaMKII_Enzimi.png", hline=25)

    # 8. LTP (Sonuç) - Alpha
    # Eğer LTP yoksa y eksenini sabitliyoruz ki fark anlaşılsın
    plt.figure(figsize=(8, 5))
    plt.plot(rec_time, rec_Alpha, color='black', linewidth=2)
    plt.title(f"Sinaptik Plastisite / LTP ({freq_label})", fontsize=14, fontweight='bold')
    plt.ylabel("Alpha (Salınım Olasılığı)", fontsize=12)
    plt.xlabel("Zaman (s)", fontsize=12)
    plt.grid(True, alpha=0.3)
    if np.max(rec_Alpha) < 0.3001:
        plt.ylim(0.2999, 0.3020) # Düz çizgiyi ortala
    plt.tight_layout()
    plt.savefig(f"{exp_folder}/8_LTP_Sonuc.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    print("======================================================")
    print("   TEZ MODU: AYRIŞTIRILMIŞ GRAFİK ÜRETİCİSİ")
    print("======================================================")
    
    if not os.path.exists(MAIN_FOLDER): os.makedirs(MAIN_FOLDER)

    for exp in EXPERIMENTS:
        run_simulation(exp)
        
    print(f"\n✅ İŞLEM TAMAM! '{MAIN_FOLDER}' klasörüne bak.")