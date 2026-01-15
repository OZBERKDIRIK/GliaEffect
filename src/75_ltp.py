import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# =================================================================================
# SİMÜLASYON AYARLARI (75 Hz SENARYOSU)
# =================================================================================
# 50 Hz için I_stim ~ 10.0 idi.
# 100 Hz için I_stim ~ 22.0 idi.
# 75 Hz (Ara Değer) için I_stim = 16.0 olarak ayarlıyoruz.
CURRENT_INJECTION = 5.0 
FREQ_LABEL = "10Hz"
SAVE_FOLDER = "results_separate" 

# -------------------------------------------------------------------------
# ORTAM VE İMPORTLAR
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
    print("✅ Modüller yüklendi.")
except ImportError as e:
    print(f"❌ Hata: {e}")
    sys.exit(1)

def run_75hz_simulation():
    print(f"🚀 {FREQ_LABEL} Simülasyonu Başlatılıyor (I_stim={CURRENT_INJECTION})...")
    
    # Süre Ayarları (30 Saniye)
    T_total = 30000.0  
    dt = 0.05          
    steps = int(T_total / dt)
    time_array = np.linspace(0, T_total, steps)

    if not os.path.exists(SAVE_FOLDER): os.makedirs(SAVE_FOLDER)

    # Modeller
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
    rec_time = time_array[::rec_step] / 1000.0 # Saniye cinsinden

    rec_V_pre = np.zeros(rec_size, dtype=np.float32)
    rec_Ca_Fast = np.zeros(rec_size, dtype=np.float32)
    rec_IP3_Astro = np.zeros(rec_size, dtype=np.float32)
    rec_Glu_Extra = np.zeros(rec_size, dtype=np.float32)
    rec_Ca_Post = np.zeros(rec_size, dtype=np.float32)
    rec_CaMKII_P = np.zeros(rec_size, dtype=np.float32)
    rec_Alpha = np.zeros(rec_size, dtype=np.float32)

    # Döngü Değişkenleri
    current_glu_syn = 0.0
    current_glu_extra = 0.0
    base_alpha = GLUTAMATE_PARAMS['alpha']
    current_alpha = base_alpha

    print("Simülasyon koşuyor...")
    for i in range(steps):
        t_ms = time_array[i]
        dt_sec = dt * 1e-3
        
        # Uyarım Protokolü: 10. ve 20. saniyeler arası aktif
        if 10000 <= t_ms <= 20000:
            I_stim = CURRENT_INJECTION
        else:
            I_stim = 0.0

        # --- MODÜL İŞLEMLERİ ---
        # 1. Presinaptik
        V_pre_mV = hh_model.step(dt, t_ms, I_stim)
        V_pre_volts = V_pre_mV * 1e-3
        Ca_pre = ca_pre_model.step(dt_sec, V_pre_volts, glu=current_glu_extra * 1e-6)
        Ca_pre_uM = ca_pre_model.c_fast * 1e6
        
        glu_pre_model.p['alpha'] = current_alpha
        current_glu_syn = glu_pre_model.step(dt, Ca_pre_uM)

        # 2. Astrosit
        Ca_astro = astro_model.compute_derivatives(dt_sec, current_glu_syn * 1e-6)
        current_glu_extra = glia_trans_model.step(dt, Ca_astro * 1e6)

        # 3. Post-Sinaptik
        V_post_volts = post_neuron_model.step(dt_sec, current_glu_syn, I_soma_injected=0.0)
        I_AMPA = post_neuron_model.I_AMPA
        Ca_post = post_ca_model.step(dt_sec, V_post_volts, I_AMPA)

        # 4. LTP
        camkii_model.step(dt_sec, Ca_post)
        alpha_mod_factor = camkii_model.get_alpha_modulation()
        current_alpha = base_alpha * (1.0 + alpha_mod_factor)

        # --- KAYIT ---
        if i % rec_step == 0:
            idx = i // rec_step
            rec_V_pre[idx] = V_pre_mV
            rec_Ca_Fast[idx] = Ca_pre_uM
            rec_IP3_Astro[idx] = astro_model.p_a * 1e6
            rec_Glu_Extra[idx] = current_glu_extra
            rec_Ca_Post[idx] = Ca_post * 1e6
            rec_CaMKII_P[idx] = np.sum(camkii_model.P[1:]) * CAMKII_PARAMS["e_k"] * 1e6
            rec_Alpha[idx] = current_alpha

    print("Grafikler çiziliyor...")

    # --- GRAFİK FONKSİYONU ---
    def plot_and_save(data, title, y_label, filename, color):
        plt.figure(figsize=(10, 4))
        plt.plot(rec_time, data, color=color, linewidth=1)
        plt.title(f"{title} ({FREQ_LABEL})", fontsize=12, fontweight='bold')
        plt.xlabel("Zaman (s)")
        plt.ylabel(y_label)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{SAVE_FOLDER}/{filename}", dpi=300)
        plt.close()
        print(f"-> {filename} kaydedildi.")

    # 1. Presinaptik (V_pre ve Ca_fast) - Tek figürde subplot
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(rec_time, rec_V_pre, 'k', lw=0.5)
    ax[0].set_ylabel("V_pre (mV)")
    ax[0].set_title(f"Presinaptik Aktivite ({FREQ_LABEL})")
    ax[0].grid(True, alpha=0.3)
    
    ax[1].plot(rec_time, rec_Ca_Fast, 'b', lw=1)
    ax[1].set_ylabel("Ca_fast (uM)")
    ax[1].grid(True, alpha=0.3)
    ax[1].set_xlabel("Zaman (s)")
    plt.tight_layout()
    plt.savefig(f"{SAVE_FOLDER}/Pre_{FREQ_LABEL}.png", dpi=300)
    plt.close()

    # 2. Astrosit (IP3 ve Glu_Extra)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(rec_time, rec_IP3_Astro, 'purple', lw=1)
    ax[0].set_ylabel("IP3 (uM)")
    ax[0].set_title(f"Astrositik Modülasyon ({FREQ_LABEL})")
    ax[0].grid(True, alpha=0.3)
    
    ax[1].plot(rec_time, rec_Glu_Extra, 'g', lw=1)
    ax[1].set_ylabel("Gliotransmitter (uM)")
    ax[1].grid(True, alpha=0.3)
    ax[1].set_xlabel("Zaman (s)")
    plt.tight_layout()
    plt.savefig(f"{SAVE_FOLDER}/Astro_{FREQ_LABEL}.png", dpi=300)
    plt.close()

    # 3. Post-Sinaptik (Ca_Post)
    plot_and_save(rec_Ca_Post, "Post-Sinaptik Kalsiyum", "Ca_post (uM)", f"Post_{FREQ_LABEL}.png", "orange")

    # 4. LTP (CaMKII ve Alpha)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(rec_time, rec_CaMKII_P, 'm', lw=1.5)
    ax[0].set_ylabel("CaMKII-P (uM)")
    ax[0].set_title(f"LTP ve Hafıza ({FREQ_LABEL})")
    ax[0].grid(True, alpha=0.3)
    
    ax[1].plot(rec_time, rec_Alpha, 'k', lw=1.5)
    ax[1].set_ylabel("Alpha (Pre-Syn)")
    ax[1].grid(True, alpha=0.3)
    ax[1].set_xlabel("Zaman (s)")
    plt.tight_layout()
    plt.savefig(f"{SAVE_FOLDER}/LTP_{FREQ_LABEL}.png", dpi=300)
    plt.close()

    print("\n✅ 75 Hz Simülasyonu Tamamlandı.")

if __name__ == "__main__":
    run_75hz_simulation()