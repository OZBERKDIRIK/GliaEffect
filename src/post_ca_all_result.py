import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# =================================================================================
# KULLANICI AYARLARI
# =================================================================================
SIMULATION_MODE = '50Hz' 
SAVE_FOLDER = "results_separate" 

# -------------------------------------------------------------------------
# 1. ORTAM AYARLARI VE İMPORTLAR
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

    print("✅ Tüm modüller başarıyla yüklendi.")

except ImportError as e:
    print(f"❌ Kritik Hata: Modüller yüklenemedi. 'src' yapısını kontrol et.\n{e}")
    sys.exit(1)

def run_simulation_separate():
    print("======================================================================")
    print(f"   TEWARI & MAJUMDAR (2012) - GRAFİK ÜRETİMİ (AKADEMİK)")
    print("======================================================================")

    T_total = 60000.0  
    dt = 0.05          
    steps = int(T_total / dt)
    time_array = np.linspace(0, T_total, steps)

    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

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
    rec_time = time_array[::rec_step]

    # Presinaptik (Kütle dengesi için gerekli)
    rec_Ca_Slow = np.zeros(rec_size, dtype=np.float32) 
    rec_Ca_ER = np.zeros(rec_size, dtype=np.float32)   
    rec_Glu_Syn = np.zeros(rec_size, dtype=np.float32)
    
    # Postsinaptik
    rec_V_post = np.zeros(rec_size, dtype=np.float32)
    rec_Ca_Post = np.zeros(rec_size, dtype=np.float32)
    rec_I_AMPA = np.zeros(rec_size, dtype=np.float32)
    rec_i_R = np.zeros(rec_size, dtype=np.float32) # R-type current

    # Simülasyon Değişkenleri
    current_glu_syn = 0.0
    current_glu_extra = 0.0
    base_alpha = GLUTAMATE_PARAMS['alpha']
    current_alpha = base_alpha

    print("Simülasyon başlıyor...")
    start_time = time.time()

    for i in range(steps):
        t_ms = time_array[i]
        dt_sec = dt * 1e-3
        I_stim = 0.0 

        # 1. Pre-Sinaptik
        V_pre_mV = hh_model.step(dt, t_ms, I_stim)
        V_pre_volts = V_pre_mV * 1e-3
        Ca_pre = ca_pre_model.step(dt_sec, V_pre_volts, glu=current_glu_extra * 1e-6)
        Ca_pre_uM = ca_pre_model.c_fast * 1e6
        
        glu_pre_model.p['alpha'] = current_alpha
        current_glu_syn = glu_pre_model.step(dt, Ca_pre_uM)

        # 2. Astrocyte
        Ca_astro = astro_model.compute_derivatives(dt_sec, current_glu_syn * 1e-6)
        current_glu_extra = glia_trans_model.step(dt, Ca_astro * 1e6)

        # 3. Post-Synaptic
        V_post_volts = post_neuron_model.step(dt_sec, current_glu_syn, I_soma_injected=0.0)
        I_AMPA = post_neuron_model.I_AMPA
        Ca_post = post_ca_model.step(dt_sec, V_post_volts, I_AMPA)

        # 4. LTP
        camkii_model.step(dt_sec, Ca_post)
        alpha_mod_factor = camkii_model.get_alpha_modulation()
        current_alpha = base_alpha * (1.0 + alpha_mod_factor)

        # Kayıt
        if i % rec_step == 0:
            idx = i // rec_step
            # Kütle dengesi (Equation 5 türevi için) - Presinaptik modelden gelir
            rec_Ca_Slow[idx] = ca_pre_model.c_slow * 1e6
            rec_Ca_ER[idx] = ca_pre_model.c_ER * 1e6
            rec_Glu_Syn[idx] = current_glu_syn
            
            # Postsinaptik Kayıtlar
            rec_V_post[idx] = V_post_volts * 1e3
            rec_Ca_Post[idx] = Ca_post * 1e6
            rec_I_AMPA[idx] = I_AMPA * 1e9
            
            try:
                rec_i_R[idx] = post_ca_model.i_R * 1e12 
            except AttributeError:
                rec_i_R[idx] = 0.0

        if i % (steps // 10) == 0:
            print(f"%{(i / steps) * 100:.0f} tamamlandı.")

    print(f"Simülasyon Bitti: {time.time() - start_time:.2f} sn")

    # ---------------------------------------------------------------------
    # GRAFİK ÇİZİM VE KAYIT
    # ---------------------------------------------------------------------
    t_sec = rec_time / 1000.0 

    def save_plot(data, ylabel, title, filename, color='k', ylim=None):
        plt.figure(figsize=(10, 4))
        plt.plot(t_sec, data, color=color, linewidth=0.8)
        plt.ylabel(ylabel, fontsize=12)
        plt.xlabel("Zaman (s)", fontsize=12)
        plt.title(title, loc='left', fontweight='bold', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Y ekseni formatını bilimsel gösterimden (1e7) kurtarmak için:
        plt.ticklabel_format(style='plain', axis='y', useOffset=False)
        
        if ylim: plt.ylim(ylim)
        plt.tight_layout()
        plt.savefig(f"{SAVE_FOLDER}/{filename}", dpi=300)
        plt.close()
        print(f"-> {filename} kaydedildi.")

    # 1. PostSynapticAmparCurrent.png (I_AMPA)
    save_plot(rec_I_AMPA, "I_AMPA (nA)", "Post-sinaptik AMPAR Akımı", "PostSynapticAmparCurrent.png", 'cyan')

    # 2. SinaptikGlutamate.png (Glu_Syn)
    save_plot(rec_Glu_Syn, "Glu (uM)", "Sinaptik Glutamat Derişimi", "SinaptikGlutamate.png", 'green')

    # 3. PostSynapticMembranPotansiyeli.png (V_post)
    save_plot(rec_V_post, "V_post (mV)", "Post-sinaptik Membran Potansiyeli", "PostSynapticMembranPotansiyeli.png", 'blue')

    # 4. RtypeVGCCcurrent.png (i_R)
    save_plot(rec_i_R, "i_R (pA)", "R-tipi VGCC Akımı", "RtypeVGCCcurrent.png", 'firebrick')

    # 5. PostSynapticCA.png (Ca_Post)
    save_plot(rec_Ca_Post, "Ca_post (uM)", "Post-sinaptik Ca2+ Derişimi", "PostSynapticCA.png", 'orange')

    # 6. ER_Ca.png (Ca_ER)
    save_plot(rec_Ca_ER, "Ca_ER (uM)", "ER İçi Kalsiyum Deposu (c_ER)", "ER_Ca.png", 'purple')

    # 7. KutleDengesi.png (c_tot)
    c1 = CA_PARAMS.get('c1', 0.185)
    rec_c_tot = rec_Ca_Slow + c1 * rec_Ca_ER
    save_plot(rec_c_tot, "c_tot (uM)", "Toplam Kalsiyum Kütle Dengesi (c_tot)", "KutleDengesi.png", 'black')

    # 8. dC_dt_post.png (Net Kalsiyum Akı Hızı)
    # DÜZELTME: uM/s çok büyük çıktığı için mM/s'ye çeviriyoruz (1000'e bölerek)
    dC_dt = np.gradient(rec_Ca_Post, t_sec) 
    dC_dt_mM = dC_dt / 1000.0  # uM/s -> mM/s
    save_plot(dC_dt_mM, r"$\nu_{net}$ (mM/s)", "Net Kalsiyum Akı Hızı (Türev)", "dC_dt_post.png", 'darkgreen')

    print(f"\n✅ Tüm grafikler '{SAVE_FOLDER}' klasörüne başarıyla kaydedildi.")

if __name__ == "__main__":
    run_simulation_separate()