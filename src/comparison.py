import numpy as np
import matplotlib.pyplot as plt
import sys
import os

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
except ImportError as e:
    print(f"Hata: {e}")
    sys.exit(1)

# =========================================================================
# ⚙️ SENARYO AYARLARI (METİNLE UYUMLU)
# =========================================================================
CAMKII_PARAMS['P_half'] = 55e-6   # EŞİK
CAMKII_PARAMS['K1'] = 0.012       # Üretim Hızı
CAMKII_PARAMS['K2'] = 50.0        # Yıkım Hızı
CAMKII_PARAMS['k_h'] = 150.0e-6 

SCENARIOS = [
    {"label": "50Hz",  "current": 6.0},
    {"label": "75Hz",  "current": 14.0},
    {"label": "100Hz", "current": 22.0}
]

SAVE_FOLDER = "Tez_Full_Bilesenler"

def plot_save(time, data, title, ylabel, color, folder, filename, threshold=None):
    plt.figure(figsize=(8, 4))
    plt.plot(time, data, color=color, linewidth=1.2)
    plt.title(title, fontweight='bold')
    plt.ylabel(ylabel)
    plt.xlabel("Zaman (s)")
    plt.grid(True, alpha=0.3)
    if threshold:
        plt.axhline(y=threshold, color='green', linestyle='--', label=f'Eşik ({threshold})')
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"{folder}/{filename}", dpi=300)
    plt.close()

def run_simulation(scn):
    label = scn["label"]
    curr = scn["current"]
    print(f"\n>>> KOŞTURULUYOR: {label} (Akım: {curr} uA)...")
    
    # Klasör
    freq_folder = os.path.join(SAVE_FOLDER, label)
    if not os.path.exists(freq_folder): os.makedirs(freq_folder)

    # Zaman
    T_total = 30000.0
    dt = 0.05
    steps = int(T_total / dt)
    time_array = np.linspace(0, T_total, steps)
    
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
    rec_time = time_array[::rec_step] / 1000.0
    
    rec_V_pre = np.zeros(rec_size, dtype=np.float32)
    rec_Ca_Pre = np.zeros(rec_size, dtype=np.float32)
    rec_IP3 = np.zeros(rec_size, dtype=np.float32)
    rec_Glu_Ast = np.zeros(rec_size, dtype=np.float32)
    
    # YENİ EKLENEN: Post Voltaj
    rec_V_post = np.zeros(rec_size, dtype=np.float32)
    
    rec_Ca_Post = np.zeros(rec_size, dtype=np.float32)
    rec_CaMKII_P = np.zeros(rec_size, dtype=np.float32)
    rec_Alpha = np.zeros(rec_size, dtype=np.float32)

    current_glu_syn = 0.0
    current_glu_extra = 0.0
    base_alpha = GLUTAMATE_PARAMS['alpha']
    current_alpha = base_alpha

    for i in range(steps):
        t_ms = time_array[i]
        dt_sec = dt * 1e-3
        I_stim = curr if 10000 <= t_ms <= 20000 else 0.0

        # Zincir
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

        if i % rec_step == 0:
            idx = i // rec_step
            rec_V_pre[idx] = V_pre_mV
            rec_Ca_Pre[idx] = ca_pre_model.c_fast * 1e6
            rec_IP3[idx] = astro_model.p_a * 1e6
            rec_Glu_Ast[idx] = current_glu_extra
            
            # Post Voltajı mV cinsinden kaydediyoruz (x1000)
            rec_V_post[idx] = V_post_volts * 1000.0 
            
            rec_Ca_Post[idx] = Ca_post * 1e6
            rec_CaMKII_P[idx] = np.sum(camkii_model.P[1:]) * CAMKII_PARAMS["e_k"] * 1e6
            rec_Alpha[idx] = current_alpha

    print(f"   -> Veriler işlendi. Grafikler çiziliyor...")

    # --- 1. PRESİNAPTİK ---
    plot_save(rec_time, rec_V_pre, f"Presinaptik Voltaj ({label})", "V_pre (mV)", "black", freq_folder, "1_Pre_Voltaj.png")
    plot_save(rec_time, rec_Ca_Pre, f"Presinaptik Kalsiyum ({label})", "Ca_pre (uM)", "blue", freq_folder, "2_Pre_Ca.png")

    # --- 2. ASTROSİT ---
    plot_save(rec_time, rec_IP3, f"Astrosit IP3 ({label})", "IP3 (uM)", "purple", freq_folder, "3_Astro_IP3.png")
    plot_save(rec_time, rec_Glu_Ast, f"Gliotransmitter ({label})", "Glu (uM)", "green", freq_folder, "4_Astro_Glu.png")

    # --- 3. POST-SİNAPTİK (VOLTAJ + KALSİYUM) ---
    # Bu ikisi birbirine bağlı olduğu için ardışık numaralandırdık
    plot_save(rec_time, rec_V_post, f"Post-Sinaptik Potansiyel ({label})", "V_post (mV)", "grey", freq_folder, "5_Post_Voltaj.png")
    plot_save(rec_time, rec_Ca_Post, f"Post-Sinaptik Kalsiyum ({label})", "Ca_post (uM)", "orange", freq_folder, "6_Post_Ca.png")

    # --- 4. LTP (SONUÇ) ---
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(rec_time, rec_CaMKII_P, 'm', lw=2)
    ax[0].set_title(f"Moleküler Hafıza ({label})", fontweight='bold')
    ax[0].set_ylabel("CaMKII-P")
    ax[0].axhline(y=55, color='green', linestyle='--', label='Eşik (55)')
    ax[0].legend(loc="upper left")
    ax[0].grid(True, alpha=0.3)
    
    ax[1].plot(rec_time, rec_Alpha, 'k', lw=2)
    ax[1].set_ylabel("Alpha")
    ax[1].set_xlabel("Zaman (s)")
    ax[1].grid(True, alpha=0.3)
    if np.max(rec_Alpha) < 0.3001: ax[1].set_ylim(0.2999, 0.3020)
    
    plt.tight_layout()
    plt.savefig(f"{freq_folder}/7_LTP_Sonuc.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    print("======================================================")
    print("   TAM BİLEŞEN ANALİZİ (Post-Voltaj Dahil)")
    print("======================================================")
    for scn in SCENARIOS:
        run_simulation(scn)
    print(f"\n✅ GRAFİKLER HAZIR: '{SAVE_FOLDER}' klasörüne bak.")