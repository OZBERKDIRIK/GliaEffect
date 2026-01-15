import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# -------------------------------------------------------------------------
# 1. ORTAM AYARLARI
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
    print(f"Hata: Modüller bulunamadı.\n{e}")
    sys.exit(1)

def run_simulation(mode_name, current_amp):
    print(f"\n>>> HESAPLANIYOR: {mode_name} (Akım: {current_amp} uA/cm2)...")
    
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

    # Kayıt
    rec_step = 20
    rec_size = steps // rec_step
    rec_time = time_array[::rec_step]
    
    rec_V_pre = np.zeros(rec_size, dtype=np.float32)
    rec_Ca_Fast = np.zeros(rec_size, dtype=np.float32)
    rec_Glu_Syn = np.zeros(rec_size, dtype=np.float32)
    rec_Ca_Astro = np.zeros(rec_size, dtype=np.float32)
    rec_V_post = np.zeros(rec_size, dtype=np.float32)
    rec_Ca_Post = np.zeros(rec_size, dtype=np.float32)
    rec_CaMKII_P = np.zeros(rec_size, dtype=np.float32)
    rec_Alpha_Mod = np.zeros(rec_size, dtype=np.float32)

    current_glu_syn = 0.0
    current_glu_extra = 0.0
    base_alpha = GLUTAMATE_PARAMS['alpha']
    current_alpha = base_alpha

    # --- DÖNGÜ ---
    for i in range(steps):
        t_ms = time_array[i]
        dt_sec = dt * 1e-3

        # UYARI (10-20 sn)
        if 10000 <= t_ms <= 20000:
            I_stim = current_amp
        else:
            I_stim = 0.0

        # Pre
        V_pre_mV = hh_model.step(dt, t_ms, I_stim)
        V_pre_volts = V_pre_mV * 1e-3
        Ca_pre = ca_pre_model.step(dt_sec, V_pre_volts, glu=current_glu_extra * 1e-6)
        Ca_pre_uM = ca_pre_model.c_fast * 1e6
        
        glu_pre_model.p['alpha'] = current_alpha
        current_glu_syn = glu_pre_model.step(dt, Ca_pre_uM)
        if t_ms < 100: current_glu_syn = 0.0

        # Astro
        Ca_astro = astro_model.compute_derivatives(dt_sec, current_glu_syn * 1e-6)
        current_glu_extra = glia_trans_model.step(dt, Ca_astro * 1e6)

        # Post
        V_post_volts = post_neuron_model.step(dt_sec, current_glu_syn, I_soma_injected=0.0)
        I_AMPA = post_neuron_model.I_AMPA
        Ca_post = post_ca_model.step(dt_sec, V_post_volts, I_AMPA)

        # LTP
        camkii_model.step(dt_sec, Ca_post)
        alpha_mod = camkii_model.get_alpha_modulation()
        current_alpha = base_alpha * (1.0 + alpha_mod)

        # Kayıt
        if i % rec_step == 0:
            idx = i // rec_step
            rec_V_pre[idx] = V_pre_mV
            rec_Ca_Fast[idx] = Ca_pre_uM
            rec_Glu_Syn[idx] = current_glu_syn
            rec_Ca_Astro[idx] = Ca_astro * 1e6
            rec_V_post[idx] = V_post_volts * 1e3
            rec_Ca_Post[idx] = Ca_post * 1e6
            rec_CaMKII_P[idx] = np.sum(camkii_model.P[1:]) * CAMKII_PARAMS["e_k"] * 1e6
            rec_Alpha_Mod[idx] = current_alpha

    # --- KAYDETME ---
    folder = "final_results"
    if not os.path.exists(folder): os.makedirs(folder)
    t_sec = rec_time / 1000.0

    # PRE
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1); plt.plot(t_sec, rec_V_pre, 'k', lw=0.5); plt.title(f"Pre Voltaj ({mode_name})")
    plt.subplot(2,1,2); plt.plot(t_sec, rec_Ca_Fast, 'b'); plt.title(f"Pre Kalsiyum - MAX: {np.max(rec_Ca_Fast):.0f}")
    plt.tight_layout(); plt.savefig(f"{folder}/Pre_{mode_name}.jpg")
    plt.close()

    # POST (Sadece burası kafa karıştırıyordu, şimdi düzelecek)
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1); plt.plot(t_sec, rec_V_post, 'b', lw=0.5); plt.title(f"Post Voltaj ({mode_name})")
    plt.subplot(2,1,2); plt.plot(t_sec, rec_Ca_Post, 'orange'); plt.title(f"Post Kalsiyum ({mode_name})")
    plt.tight_layout(); plt.savefig(f"{folder}/Post_{mode_name}.jpg")
    plt.close()

    # LTP (En önemlisi)
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1); plt.plot(t_sec, rec_CaMKII_P, 'm'); plt.title(f"CaMKII ({mode_name})")
    plt.subplot(2,1,2); plt.plot(t_sec, rec_Alpha_Mod, 'k'); plt.title(f"LTP Alpha ({mode_name})")
    plt.ylim(0.2999, 0.3020)
    plt.tight_layout(); plt.savefig(f"{folder}/LTP_{mode_name}.png")
    plt.close()

if __name__ == "__main__":
    # 50 Hz (Akım 10) -> LTP Olmamalı
    run_simulation('50Hz', 10.0)
    
    # 75 Hz (Akım 16) -> LTP Olmamalı (Geçiş)
    run_simulation('75Hz', 16.0)
    
    # 100 Hz (Akım 35 - ARTIRILDI!) -> LTP OLMALI
    run_simulation('100Hz', 35.0)
    
    print("\n✅ BİTTİ. 'final_results' klasörüne bak.")