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
    # PARAMETRELER
    from parameters.pre_synaptic_params import PRE_SYNAPTIC_PARAMS
    from parameters.ca_params import CA_PARAMS
    from parameters.glutamate_params import GLUTAMATE_PARAMS
    from parameters.astrocyte_params import ASTROCYTE_PARAMS
    from parameters.gliatransmitter_params import GLIATRANSMITTER_PARAMS
    from parameters.post_synaptic_params import POST_SYNAPTIC_PARAMS
    from parameters.post_synaptic_ca_params import POST_SYNAPTIC_CA_PARAMS
    from parameters.camkii_params import CAMKII_PARAMS

    # MODELLER
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
    print(f"   TEWARI & MAJUMDAR (2012) - TÜM GRAFİKLER (AYRI AYRI)")
    print("======================================================================")

    # ---------------------------------------------------------------------
    # 2. SİMÜLASYON ZAMANI 
    # ---------------------------------------------------------------------
    T_total = 60000.0  
    dt = 0.05          
    steps = int(T_total / dt)
    time_array = np.linspace(0, T_total, steps)

    print(f"Toplam Süre: {T_total/1000} saniye")
    
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

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
    camkii_model = CaMKIIDynamics(CAMKII_PARAMS)

    # ---------------------------------------------------------------------
    # 4. VERİ KAYIT DİZİLERİ
    # ---------------------------------------------------------------------
    rec_step = 20  
    rec_size = steps // rec_step
    rec_time = time_array[::rec_step]

    # Standart Kayıtlar
    rec_V_pre = np.zeros(rec_size, dtype=np.float32)
    rec_Ca_Fast = np.zeros(rec_size, dtype=np.float32)
    rec_Ca_Slow = np.zeros(rec_size, dtype=np.float32)
    rec_Ca_Total = np.zeros(rec_size, dtype=np.float32) 
    rec_Ca_ER = np.zeros(rec_size, dtype=np.float32)
    rec_IP3_Pre = np.zeros(rec_size, dtype=np.float32)
    rec_q_Pre = np.zeros(rec_size, dtype=np.float32)
    rec_Glu_Syn = np.zeros(rec_size, dtype=np.float32)
    rec_Ca_Astro = np.zeros(rec_size, dtype=np.float32)
    rec_IP3_Astro = np.zeros(rec_size, dtype=np.float32)
    rec_h_Gate = np.zeros(rec_size, dtype=np.float32)
    rec_Glu_Extra = np.zeros(rec_size, dtype=np.float32)
    rec_V_post = np.zeros(rec_size, dtype=np.float32)
    rec_Ca_Post = np.zeros(rec_size, dtype=np.float32)
    rec_I_AMPA = np.zeros(rec_size, dtype=np.float32)
    rec_CaMKII_P = np.zeros(rec_size, dtype=np.float32)
    rec_Alpha_Mod = np.zeros(rec_size, dtype=np.float32)

    # Astro Detay Kayıtları
    rec_R_a = np.zeros(rec_size, dtype=np.float32)
    rec_E_a = np.zeros(rec_size, dtype=np.float32)
    rec_I_a = np.zeros(rec_size, dtype=np.float32)
    rec_O1  = np.zeros(rec_size, dtype=np.float32)
    rec_O2  = np.zeros(rec_size, dtype=np.float32)
    rec_O3  = np.zeros(rec_size, dtype=np.float32)
    rec_G_a = np.zeros(rec_size, dtype=np.float32)

    # AMPA Gate
    rec_m_AMPA = np.zeros(rec_size, dtype=np.float32)

    # [YENİ] R-Type VGCC Akımı Kaydı
    rec_i_R = np.zeros(rec_size, dtype=np.float32)

    # ---------------------------------------------------------------------
    # 5. SİMÜLASYON DÖNGÜSÜ
    # ---------------------------------------------------------------------
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

        # Pre-Sinaptik
        V_pre_mV = hh_model.step(dt, t_ms, I_stim)
        V_pre_volts = V_pre_mV * 1e-3

        Ca_pre = ca_pre_model.step(dt_sec, V_pre_volts, glu=current_glu_extra * 1e-6)
        Ca_pre_uM = ca_pre_model.c_fast * 1e6
        
        glu_pre_model.p['alpha'] = current_alpha
        current_glu_syn = glu_pre_model.step(dt, Ca_pre_uM)

        # Astrocyte
        Ca_astro = astro_model.compute_derivatives(dt_sec, current_glu_syn * 1e-6)
        current_glu_extra = glia_trans_model.step(dt, Ca_astro * 1e6)

        # Post-Synaptic
        V_post_volts = post_neuron_model.step(dt_sec, current_glu_syn, I_soma_injected=0.0)
        I_AMPA = post_neuron_model.I_AMPA
        
        # Post-Ca Step (Burada i_R hesaplanıyor olmalı)
        Ca_post = post_ca_model.step(dt_sec, V_post_volts, I_AMPA)

        # LTP
        camkii_model.step(dt_sec, Ca_post)
        alpha_mod_factor = camkii_model.get_alpha_modulation()
        current_alpha = base_alpha * (1.0 + alpha_mod_factor)

        # -------------------------
        # KAYIT
        # -------------------------
        if i % rec_step == 0:
            idx = i // rec_step
            
            # Pre
            rec_V_pre[idx] = V_pre_mV
            rec_Ca_Fast[idx] = ca_pre_model.c_fast * 1e6 
            rec_Ca_Slow[idx] = ca_pre_model.c_slow * 1e6 
            rec_Ca_Total[idx] = (ca_pre_model.c_fast + ca_pre_model.c_slow) * 1e6 
            rec_Ca_ER[idx] = ca_pre_model.c_ER * 1e6     
            rec_IP3_Pre[idx] = ca_pre_model.p_ip3 * 1e6  
            rec_q_Pre[idx] = ca_pre_model.q 
            rec_Glu_Syn[idx] = current_glu_syn
            
            # Astro
            rec_Ca_Astro[idx] = Ca_astro * 1e6           
            rec_IP3_Astro[idx] = astro_model.p_a * 1e6   
            rec_h_Gate[idx] = astro_model.h_a            
            rec_Glu_Extra[idx] = current_glu_extra

            # Astro Detayları
            rec_R_a[idx] = glia_trans_model.R_a
            rec_E_a[idx] = glia_trans_model.E_a
            rec_I_a[idx] = 1.0 - glia_trans_model.R_a - glia_trans_model.E_a
            
            rec_O1[idx] = glia_trans_model.O1
            rec_O2[idx] = glia_trans_model.O2
            rec_O3[idx] = glia_trans_model.O3
            
            rec_G_a[idx] = glia_trans_model.G_a
            
            # Post
            rec_V_post[idx] = V_post_volts * 1e3         
            rec_Ca_Post[idx] = Ca_post * 1e6             
            rec_I_AMPA[idx] = I_AMPA * 1e9
            
            # [YENİ] R-Type VGCC Akımı Kaydı
            # post_ca_model içinde i_R değişkeninin olması gerekir.
            # Birim pA olsun diye 1e12 ile çarpıyoruz.
            try:
                rec_i_R[idx] = post_ca_model.i_R * 1e12 
            except AttributeError:
                rec_i_R[idx] = 0.0 # Eğer modelde yoksa hata vermesin, 0 bassın
            
            # AMPA Gate
            rec_m_AMPA[idx] = post_neuron_model.m_AMPA

            # LTP
            rec_CaMKII_P[idx] = np.sum(camkii_model.P[1:]) * CAMKII_PARAMS["e_k"] * 1e6
            rec_Alpha_Mod[idx] = current_alpha

        # İlerleme
        if i % (steps // 10) == 0:
            percent = (i / steps) * 100
            print(f"%{percent:.0f} tamamlandı.")

    print(f"\nSimülasyon Bitti. Süre: {time.time() - start_time:.2f} sn")
    print("Grafikler oluşturuluyor...")

    # ---------------------------------------------------------------------
    # 6. GÖRSELLEŞTİRME
    # ---------------------------------------------------------------------
    t_sec = rec_time / 1000.0 

    def save_plot(data, ylabel, title, filename, color='k', ylim=None):
        plt.figure(figsize=(10, 4))
        plt.plot(t_sec, data, color=color, linewidth=0.8)
        plt.ylabel(ylabel, fontsize=12)
        plt.xlabel("Zaman (s)", fontsize=12)
        plt.title(title, loc='left', fontweight='bold', fontsize=12)
        plt.grid(True, alpha=0.3)
        if ylim:
            plt.ylim(ylim)
        plt.tight_layout()
        plt.savefig(f"{SAVE_FOLDER}/{filename}", dpi=300)
        plt.close()
        print(f"-> {filename} kaydedildi.")

    # --- MEVCUT GRAFİKLER ---
    save_plot(rec_V_pre, "V_pre (mV)", "Pre-sinaptik Membran Potansiyeli", "V_pre.png", 'k', ylim=(-90, 50))
    save_plot(rec_Ca_Fast, "Ca_fast (uM)", "Hızlı Kalsiyum Dinamiği (Bouton)", "Ca_fast.png", 'b')
    save_plot(rec_Ca_Slow, "Ca_slow (uM)", "Yavaş Kalsiyum Dinamiği (ER Kaynaklı)", "Ca_slow.png", 'g')
    save_plot(rec_Ca_Total, "Ca_total (uM)", "Toplam Pre-sinaptik Kalsiyum (Fast + Slow)", "Ca_total.png", 'teal')
    save_plot(rec_Ca_ER, "Ca_ER (uM)", "Endoplazmik Retikulum Kalsiyum Deposu", "Ca_ER.png", 'purple')
    save_plot(rec_IP3_Pre, "IP3 (uM)", "Presinaptik IP3 Konsantrasyonu", "IP3_Pre.png", 'brown')
    save_plot(rec_q_Pre, "q (Olasılık)", "IP3 Reseptör Gating Değişkeni (q)", "q_Pre.png", 'orange', ylim=(0, 1))
    save_plot(rec_Glu_Syn, "Glu (uM)", "Sinaptik Aralıktaki Glutamat", "Glu_Syn.png", 'green')
    
    save_plot(rec_Ca_Astro, "Ca_astro (uM)", "Astrositik Kalsiyum Salınımları", "Ca_Astro.png", 'r')
    save_plot(rec_IP3_Astro, "IP3 (uM)", "Astrositik IP3 Dinamiği", "IP3_Astro.png", 'm')
    save_plot(rec_h_Gate, "h (kapı)", "IP3 Reseptör İnaktivasyon Kapısı (h)", "h_Gate.png", 'gray', ylim=(0, 1))
    save_plot(rec_Glu_Extra, "Glu_extra (uM)", "Ekstra-sinaptik (Glial) Glutamat", "Glu_Extra.png", 'purple')
    
    save_plot(rec_V_post, "V_post (mV)", "Post-sinaptik Membran Potansiyeli", "V_post.png", 'b')
    save_plot(rec_Ca_Post, "Ca_post (uM)", "Post-sinaptik Spine Kalsiyumu", "Ca_Post.png", 'orange')
    save_plot(rec_I_AMPA, "I_AMPA (nA)", "AMPA Reseptör Akımı", "I_AMPA.png", 'cyan')
    
    save_plot(rec_m_AMPA, "m_AMPA (Olasılık)", "AMPA Reseptör Aktivasyonu (m_AMPA)", "AMPARGATE.png", 'darkcyan', ylim=(0, 1.05))

    # --- [YENİ] R-TYPE VGCC CURRENT GRAFİĞİ ---
    save_plot(rec_i_R, "i_R (pA)", "R-Tipi Voltaj Kapılı Ca2+ Kanalı Akımı", "RtypeVGCCcurrent.png", 'firebrick')

    save_plot(rec_CaMKII_P, "CaMKII-P (uM)", "Fosforile CaMKII (Hafıza Molekülü)", "CaMKII_P.png", 'magenta')
    save_plot(rec_Alpha_Mod, "Alpha", "Vezikül Salınım Olasılığı (LTP Modülasyonu)", "Alpha_Mod.png", 'k')

    # --- VEZİKÜL HAVUZLARI ---
    plt.figure(figsize=(10, 5))
    plt.plot(t_sec, rec_R_a, label=r'$R_a$ (Hazır)', color='blue', linewidth=1.5)
    plt.plot(t_sec, rec_E_a, label=r'$E_a$ (Etkin)', color='green', linewidth=1.5)
    plt.plot(t_sec, rec_I_a, label=r'$I_a$ (İnaktif)', color='red', linestyle='--', linewidth=1.5)
    plt.ylabel("Vezikül Fraksiyonu", fontsize=12)
    plt.xlabel("Zaman (s)", fontsize=12)
    plt.title("Astrosit Vezikül Havuzu Dinamikleri", loc='left', fontweight='bold', fontsize=12)
    plt.legend(loc="right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{SAVE_FOLDER}/AstroVezikulHavuzu.png", dpi=300)
    plt.close()
    print(f"-> AstroVezikulHavuzu.png kaydedildi.")

    # --- GATES ---
    plt.figure(figsize=(10, 5))
    plt.plot(t_sec, rec_O1, label=r'$O_1$', color='blue', linewidth=1.5)
    plt.plot(t_sec, rec_O2, label=r'$O_2$', color='orange', linewidth=1.5)
    plt.plot(t_sec, rec_O3, label=r'$O_3$', color='green', linewidth=1.5)
    plt.ylabel("Açılma Olasılığı", fontsize=12)
    plt.xlabel("Zaman (s)", fontsize=12)
    plt.title("Astrosit Ca2+ Kapıları (O1, O2, O3)", loc='left', fontweight='bold', fontsize=12)
    plt.legend(loc="lower right")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{SAVE_FOLDER}/AstrositOGate.png", dpi=300)
    plt.close()
    print(f"-> AstrositOGate.png kaydedildi.")

    # --- ASTROSİT GLUTAMAT (Ga) ---
    plt.figure(figsize=(10, 5))
    plt.plot(t_sec, rec_G_a, color='rebeccapurple', linewidth=1.5, label=r'$G_a$ (Glutamat)')
    plt.ylabel(r'Astrositik Glutamat ($\mu M$)', fontsize=12)
    plt.xlabel("Zaman (s)", fontsize=12)
    plt.title("Astrositik Glutamat Salınımı ($G_a$)", loc='left', fontweight='bold', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{SAVE_FOLDER}/AstrositGlutamate.png", dpi=300)
    plt.close()
    print(f"-> AstrositGlutamate.png kaydedildi.")


    print(f"\n✅ Tüm grafikler '{SAVE_FOLDER}' klasörüne başarıyla kaydedildi.")

if __name__ == "__main__":
    run_simulation_separate()