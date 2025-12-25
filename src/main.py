import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# =================================================================================
# KULLANICI AYARLARI (BURAYI DEĞİŞTİREREK 50HZ / 100HZ GEÇİŞİ YAPABİLİRSİN)
# =================================================================================
# Seçenekler: '50Hz' veya '100Hz'
# HOCANIN İSTEĞİ ÜZERİNE ÖNCE 100HZ ALALIM, SONRA BURAYI '50Hz' YAPIP TEKRAR ÇALIŞTIR.
SIMULATION_MODE = '100Hz' 

if SIMULATION_MODE == '50Hz':
    CURRENT_AMPLITUDE = 10.0  # uA/cm2 -> Yaklaşık 50-60 Hz ateşleme sağlar (Eşik altı/sınırda LTP)
elif SIMULATION_MODE == '100Hz':
    CURRENT_AMPLITUDE = 22.0  # uA/cm2 -> Yaklaşık 100 Hz ateşleme sağlar (Güçlü LTP)
else:
    raise ValueError("Lütfen '50Hz' veya '100Hz' seçiniz.")

print(f"--- SİMÜLASYON MODU: {SIMULATION_MODE} (Akım: {CURRENT_AMPLITUDE} uA/cm2) ---")
# =================================================================================

# -------------------------------------------------------------------------
# 1. ORTAM AYARLARI VE İMPORTLAR
# -------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

try:
    # PARAMETRELER
    from parameters.pre_synaptic_params import PRE_SYNAPTIC_PARAMS
    from parameters.ca_params import CA_PARAMS
    from parameters.glutamate_params import GLUTAMATE_PARAMS
    from parameters.astrocyte_params import ASTROCYTE_PARAMS
    from parameters.gliatransmitter_params import GLIATRANSMITTER_PARAMS
    from parameters.post_synaptic_params import POST_SYNAPTIC_PARAMS
    from parameters.post_synaptic_ca_params import POST_SYNAPTIC_CA_PARAMS
    from parameters.camkii_params import CAMKII_PARAMS  # [YENİ]

    # MODELLER
    from models.hh import PresynapticHH
    from models.calcium_model import PresynapticCalciumDynamics
    from models.presynaptic_glutamate import GlutamateDynamics
    from models.astrocyte import AstrocyteDynamics
    from models.gliatransmitter import GliatransmitterDynamics
    from models.post_synaptic import PostSynapticDynamics
    from models.post_synaptic_ca import PostSynapticCalciumDynamics
    from models.camkii import CaMKIIDynamics            # [YENİ]

except ImportError as e:
    print(f"Kritik Hata: Modüller yüklenemedi. 'src' yapısını kontrol et.\n{e}")
    sys.exit(1)

def run():
    print("======================================================================")
    print(f"    TEWARI & MAJUMDAR (2012) – TRIPARTITE SYNAPSE & LTP ({SIMULATION_MODE})")
    print("======================================================================")

    # ---------------------------------------------------------------------
    # 2. SİMÜLASYON ZAMANI
    # ---------------------------------------------------------------------
    T_total = 30000.0  # 30 Saniye
    dt = 0.05          # 0.05 ms
    steps = int(T_total / dt)
    time_array = np.linspace(0, T_total, steps)

    print(f"Toplam Süre: {T_total/1000} saniye")
    print(f"Zaman adımı: {dt} ms")
    print(f"Adım sayısı: {steps}")
    print("----------------------------------------------------------------------")

    # ---------------------------------------------------------------------
    # 3. MODELLERİN OLUŞTURULMASI
    # ---------------------------------------------------------------------
    hh_model = PresynapticHH(PRE_SYNAPTIC_PARAMS)
    ca_pre_model = PresynapticCalciumDynamics(CA_PARAMS)
    glu_pre_model = GlutamateDynamics(GLUTAMATE_PARAMS)
    astro_model = AstrocyteDynamics(ASTROCYTE_PARAMS)
    glia_trans_model = GliatransmitterDynamics(GLIATRANSMITTER_PARAMS)
    post_neuron_model = PostSynapticDynamics(POST_SYNAPTIC_PARAMS)
    post_ca_model = PostSynapticCalciumDynamics(POST_SYNAPTIC_CA_PARAMS)
    camkii_model = CaMKIIDynamics(CAMKII_PARAMS) # [YENİ]

    # ---------------------------------------------------------------------
    # 4. VERİ KAYIT DİZİLERİ (Genişletilmiş)
    # ---------------------------------------------------------------------
    rec_step = 20  # Her 20 adımda bir kayıt (Downsampling)
    rec_size = steps // rec_step
    rec_time = time_array[::rec_step]

    # --- PRE-SYNAPTIC ---
    rec_V_pre = np.zeros(rec_size, dtype=np.float32)
    rec_Ca_Fast = np.zeros(rec_size, dtype=np.float32)
    rec_Ca_Slow = np.zeros(rec_size, dtype=np.float32)
    rec_Ca_ER = np.zeros(rec_size, dtype=np.float32)
    rec_IP3_Pre = np.zeros(rec_size, dtype=np.float32)
    rec_Glu_Syn = np.zeros(rec_size, dtype=np.float32)

    # --- ASTROCYTE ---
    rec_Ca_Astro = np.zeros(rec_size, dtype=np.float32)
    rec_IP3_Astro = np.zeros(rec_size, dtype=np.float32)
    rec_h_Gate = np.zeros(rec_size, dtype=np.float32)
    rec_Glu_Extra = np.zeros(rec_size, dtype=np.float32)

    # --- POST-SYNAPTIC ---
    rec_V_post = np.zeros(rec_size, dtype=np.float32)
    rec_Ca_Post = np.zeros(rec_size, dtype=np.float32)
    rec_I_AMPA = np.zeros(rec_size, dtype=np.float32)

    # --- LTP / CaMKII [YENİ] ---
    rec_CaMKII_P = np.zeros(rec_size, dtype=np.float32) # Toplam Fosforile
    rec_Alpha_Mod = np.zeros(rec_size, dtype=np.float32) # Değişen Alpha Değeri

    # ---------------------------------------------------------------------
    # 5. SİMÜLASYON DÖNGÜSÜ
    # ---------------------------------------------------------------------
    current_glu_syn = 0.0   # uM
    current_glu_extra = 0.0 # uM
    
    # Orijinal Alpha değerini sakla (Referans)
    base_alpha = GLUTAMATE_PARAMS['alpha']
    current_alpha = base_alpha

    start_time = time.time()

    for i in range(steps):
        t_ms = time_array[i]
        dt_sec = dt * 1e-3

        # A. UYARI (HFS - 10-20 sn arası)
        # BURADA CURRENT_AMPLITUDE KULLANIYORUZ (Otomatik)
        if 10000 <= t_ms <= 20000:
            I_stim = CURRENT_AMPLITUDE 
        else:
            I_stim = 0.0

        # B. PRE-SINAPTIK
        V_pre_mV = hh_model.step(dt, t_ms, I_stim)
        V_pre_volts = V_pre_mV * 1e-3

        # Feedback: Glu_extra (uM -> Molar)
        Ca_pre = ca_pre_model.step(dt_sec, V_pre_volts, glu=current_glu_extra * 1e-6)
        
        # Ca_pre uM kullanımı
        Ca_pre_uM = ca_pre_model.c_fast * 1e6
        
        # [MODÜLASYON] Alpha değerini LTP'ye göre güncelle
        glu_pre_model.p['alpha'] = current_alpha
        current_glu_syn = glu_pre_model.step(dt, Ca_pre_uM)

        # C. ASTROCYTE
        Ca_astro = astro_model.compute_derivatives(dt_sec, current_glu_syn * 1e-6)
        current_glu_extra = glia_trans_model.step(dt, Ca_astro * 1e6)

        # D. POST-SINAPTIK
        V_post_volts = post_neuron_model.step(dt_sec, current_glu_syn, I_soma_injected=0.0)
        I_AMPA = post_neuron_model.I_AMPA
        Ca_post = post_ca_model.step(dt_sec, V_post_volts, I_AMPA)

        # E. CaMKII & RETROGRADE SIGNALING [YENİ]
        # 1. CaMKII'yi güncelle (Girdi: Molar Ca_post)
        # Ca_post zaten Molar dönüyor.
        camkii_model.step(dt_sec, Ca_post)

        # 2. NO Modülasyonunu hesapla (0.0 ile k_syt arasında bir değer döner)
        alpha_mod_factor = camkii_model.get_alpha_modulation()

        # 3. Alpha parametresini güncelle (Retrograde Feedback)
        # alpha = alpha_base * (1 + modulation)
        current_alpha = base_alpha * (1.0 + alpha_mod_factor)

        # -------------------------
        # KAYIT (Downsampling)
        # -------------------------
        if i % rec_step == 0:
            idx = i // rec_step
            
            # Pre
            rec_V_pre[idx] = V_pre_mV
            rec_Ca_Fast[idx] = ca_pre_model.c_fast * 1e6 
            rec_Ca_Slow[idx] = ca_pre_model.c_slow * 1e6 
            rec_Ca_ER[idx] = ca_pre_model.c_ER * 1e6     
            rec_IP3_Pre[idx] = ca_pre_model.p_ip3 * 1e6  
            rec_Glu_Syn[idx] = current_glu_syn
            
            # Astro
            rec_Ca_Astro[idx] = Ca_astro * 1e6           
            rec_IP3_Astro[idx] = astro_model.p_a * 1e6   
            rec_h_Gate[idx] = astro_model.h_a            
            rec_Glu_Extra[idx] = current_glu_extra
            
            # Post
            rec_V_post[idx] = V_post_volts * 1e3         
            rec_Ca_Post[idx] = Ca_post * 1e6             
            rec_I_AMPA[idx] = I_AMPA * 1e9               

            # LTP [YENİ]
            # P[1:] tüm fosforile alt birimlerin toplamı
            rec_CaMKII_P[idx] = np.sum(camkii_model.P[1:]) * CAMKII_PARAMS["e_k"] * 1e6
            rec_Alpha_Mod[idx] = current_alpha

        # İLERLEME
        if i % (steps // 10) == 0:
            percent = (i / steps) * 100
            elapsed = time.time() - start_time
            print(f"%{percent:.0f} tamamlandı. ({t_ms:.0f} ms)")

    print(f"\nSimülasyon Bitti. Süre: {time.time() - start_time:.2f} sn")
    print("Grafikler oluşturuluyor ve kaydediliyor...")

    # ---------------------------------------------------------------------
    # 6. GÖRSELLEŞTİRME VE KAYIT (OTOMATİK DOSYALAMA)
    # ---------------------------------------------------------------------
    t_axis = rec_time / 1000 # Saniye
    
    # Klasör oluşturma (yoksa)
    if not os.path.exists("results_comparison"):
        os.makedirs("results_comparison")

    # --- FIG 1: PRE-SYNAPTIC BÖLÜM ---
    fig1, ax1 = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
    fig1.suptitle(f"1. Pre-Synaptic Dynamics ({SIMULATION_MODE})", fontsize=14)
    ax1[0].plot(t_axis, rec_V_pre, 'k', lw=0.5); ax1[0].set_ylabel("V_pre (mV)"); ax1[0].set_title("Action Potentials")
    ax1[1].plot(t_axis, rec_Ca_Fast, 'b', label="Ca_fast"); ax1[1].plot(t_axis, rec_Ca_Slow, 'orange', label="Ca_slow")
    ax1[1].set_ylabel("Ca (uM)"); ax1[1].legend(loc="upper right"); ax1[1].set_title("Cytosolic Calcium")
    ax1[2].plot(t_axis, rec_Ca_ER, 'purple'); ax1[2].set_ylabel("Ca_ER (uM)"); ax1[2].set_title("ER Calcium Store")
    ax1[3].plot(t_axis, rec_IP3_Pre, 'brown'); ax1[3].set_ylabel("IP3 (uM)"); ax1[3].set_title("Presynaptic IP3")
    ax1[4].plot(t_axis, rec_Glu_Syn, 'g', lw=0.8); ax1[4].set_ylabel("Glu (uM)"); ax1[4].set_title("Glutamate Release"); ax1[4].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(f"results_comparison/Pre_{SIMULATION_MODE}.png")

    # --- FIG 2: ASTROCYTE BÖLÜMÜ ---
    fig2, ax2 = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig2.suptitle(f"2. Astrocyte Dynamics ({SIMULATION_MODE})", fontsize=14)
    ax2[0].plot(t_axis, rec_Ca_Astro, 'r', lw=1.5); ax2[0].set_ylabel("Ca_astro (uM)"); ax2[0].set_title("Calcium Oscillations")
    ax2[1].plot(t_axis, rec_IP3_Astro, 'm'); ax2[1].set_ylabel("IP3 (uM)"); ax2[1].set_title("IP3 Dynamics")
    ax2[2].plot(t_axis, rec_h_Gate, 'gray'); ax2[2].set_ylabel("h gate"); ax2[2].set_ylim(0, 1); ax2[2].set_title("IP3 Receptor Gating (h)")
    ax2[3].plot(t_axis, rec_Glu_Extra, 'purple', lw=1); ax2[3].set_ylabel("Glu_extra (uM)"); ax2[3].set_title("Gliotransmitter Release"); ax2[3].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(f"results_comparison/Astro_{SIMULATION_MODE}.png")

    # --- FIG 3: POST-SYNAPTIC BÖLÜM ---
    fig3, ax3 = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig3.suptitle(f"3. Post-Synaptic Dynamics ({SIMULATION_MODE})", fontsize=14)
    ax3[0].plot(t_axis, rec_V_post, 'b', lw=0.8); ax3[0].set_ylabel("V_post (mV)"); ax3[0].set_title("Membrane Potential")
    ax3[1].plot(t_axis, rec_Ca_Post, 'orange'); ax3[1].set_ylabel("Ca_post (uM)"); ax3[1].set_title("Spine Calcium Concentration")
    ax3[2].plot(t_axis, rec_I_AMPA, 'cyan', lw=0.8); ax3[2].set_ylabel("I_AMPA (nA)"); ax3[2].set_title("AMPA Receptor Current"); ax3[2].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(f"results_comparison/Post_{SIMULATION_MODE}.png")

    # --- FIG 4: RETROGRADE SIGNALING (LTP) ---
    fig4, ax4 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig4.suptitle(f"4. Retrograde Signaling (LTP & NO) - {SIMULATION_MODE}", fontsize=14)
    
    ax4[0].plot(t_axis, rec_CaMKII_P, 'm', lw=1.5)
    ax4[0].set_ylabel("CaMKII-P (uM)")
    ax4[0].set_title("Phosphorylated CaMKII (Memory Molecule)")
    
    ax4[1].plot(t_axis, rec_Alpha_Mod, 'k', lw=1.5)
    ax4[1].set_ylabel("Alpha (Pre-Synaptic)")
    ax4[1].set_title("Vesicle Release Probability Modulation (LTP)")
    ax4[1].set_xlabel("Time (s)")
    
    plt.tight_layout()
    plt.savefig(f"results_comparison/LTP_{SIMULATION_MODE}.png")
    
    print(f"Tüm grafikler 'results_comparison' klasörüne başarıyla kaydedildi.")
    plt.show()

if __name__ == "__main__":
    run()