# Dosya Yolu: main.py

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

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

except ImportError as e:
    print(f"Kritik Hata: Modüller yüklenemedi. 'src' yapısını kontrol et.\n{e}")
    sys.exit(1)

def run():
    print("======================================================================")
    print("     TEWARI & MAJUMDAR (2012) – TRIPARTITE SYNAPSE LTP SIMULATION")
    print("======================================================================")

    # ---------------------------------------------------------------------
    # 2. SİMÜLASYON ZAMANI
    # ---------------------------------------------------------------------
    T_total = 10000.0  # ms
    dt = 0.01          # ms
    steps = int(T_total / dt)
    time_array = np.linspace(0, T_total, steps)

    print(f"Toplam Süre: {T_total} ms")
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
    camkii_model = CaMKIIDynamics(CAMKII_PARAMS)

    # ---------------------------------------------------------------------
    # 4. VERİ KAYIT DİZİLERİ (Buffer)
    # ---------------------------------------------------------------------
    rec_V_pre = np.zeros(steps, dtype=np.float32)
    rec_Ca_Total_Pre = np.zeros(steps, dtype=np.float32)
    rec_Ca_Fast = np.zeros(steps, dtype=np.float32)
    rec_Ca_Slow = np.zeros(steps, dtype=np.float32)
    rec_Ca_ER = np.zeros(steps, dtype=np.float32)
    rec_IP3_Pre = np.zeros(steps, dtype=np.float32)
    rec_Glu_Syn = np.zeros(steps, dtype=np.float32)
    rec_Ca_Astro = np.zeros(steps, dtype=np.float32)
    rec_IP3_Astro = np.zeros(steps, dtype=np.float32)
    rec_Glu_Extra = np.zeros(steps, dtype=np.float32)
    rec_V_post = np.zeros(steps, dtype=np.float32)
    rec_Ca_Post = np.zeros(steps, dtype=np.float32)
    rec_CaMKII_P = np.zeros(steps, dtype=np.float32)
    rec_Alpha_Mod = np.zeros(steps, dtype=np.float32)

    # ---------------------------------------------------------------------
    # 5. BAŞLANGIÇ DEĞERLERİ
    # ---------------------------------------------------------------------
    current_glu_syn = 0.0
    current_glu_extra = 0.0
    base_alpha = GLUTAMATE_PARAMS["alpha"]

    start_time = time.time()

    # ---------------------------------------------------------------------
    # 6. SİMÜLASYON DÖNGÜSÜ
    # ---------------------------------------------------------------------
    for i in range(steps):

        t_ms = time_array[i]
        dt_sec = dt * 1e-3

        # (1) PRESYNAPTIC HH VOLTAJ
        V_pre_mV = hh_model.step(dt, t_ms)
        V_pre_volts = V_pre_mV * 1e-3

        # (2) PRESYNAPTIC Ca2+ (IP3 DÖNGÜSÜ)
        # glu=current_glu_extra -> Astrositten gelen glutamat IP3 üretimini tetikler.
        Ca_pre = ca_pre_model.step(dt_sec, V_pre_volts, glu=current_glu_extra)

        # (3) PRESYNAPTIC GLUTAMATE RELEASE
        # Çıktı birimi: uM
        current_glu_syn = glu_pre_model.step(dt_sec, Ca_pre)

        # (4) ASTROCYTE Ca2+ + IP3 [KRİTİK DÜZELTME BURADA]
        # Astrocyte Molar bekliyor, Glutamate uM geliyor.
        # uM -> Molar çevrimi için 1e-6 ile çarpıyoruz.
        Ca_astro = astro_model.compute_derivatives(dt_sec, current_glu_syn * 1e-6)

        # (5) GLIOTRANSMITTER FEEDBACK
        # Gliatransmitter muhtemelen uM bazlı çalışıyor.
        # Ca_astro (Molar) -> uM çevirip veriyoruz (1e6)
        current_glu_extra = glia_trans_model.step(dt_sec, Ca_astro * 1e6)

        # (6) POSTSYNAPTIC – I_soma pulse
        if 100 < t_ms < 120:
            I_soma = 0.5e-9   # 20 ms external stimulation
        else:
            I_soma = 0.0

        V_post_volts = post_neuron_model.step(dt_sec, current_glu_syn, I_soma)

        # (7) POST-SYNAPTIC Ca2+
        I_AMPA = post_neuron_model.I_AMPA
        Ca_post = post_ca_model.step(dt_sec, V_post_volts, I_AMPA)

        # (8) CaMKII (P0-P10)
        camkii_model.step(dt_sec, Ca_post)

        # Retrograde modulation
        alpha_mod = camkii_model.get_alpha_modulation()
        glu_pre_model.p["alpha"] = base_alpha * (1.0 + alpha_mod)

        # -------------------------
        # KAYIT (DOĞRU BİRİMLER)
        # -------------------------
        rec_V_pre[i] = V_pre_mV
        
        # Ca_pre zaten uM dönüyor
        rec_Ca_Total_Pre[i] = Ca_pre  
        
        # State'ler Molar -> uM
        rec_Ca_Fast[i] = ca_pre_model.c_fast * 1e6
        rec_Ca_Slow[i] = ca_pre_model.c_slow * 1e6
        rec_Ca_ER[i] = ca_pre_model.c_ER * 1e6
        rec_IP3_Pre[i] = ca_pre_model.p_ip3 * 1e6
        
        # Glu uM -> mM (Grafik için)
        rec_Glu_Syn[i] = current_glu_syn * 1e-3
        
        # Astrocyte State Molar -> uM
        rec_Ca_Astro[i] = Ca_astro * 1e6
        rec_IP3_Astro[i] = astro_model.p_a * 1e6
        
        rec_Glu_Extra[i] = current_glu_extra * 1e-3
        
        rec_V_post[i] = V_post_volts * 1e3
        rec_Ca_Post[i] = Ca_post * 1e6
        rec_CaMKII_P[i] = np.sum(camkii_model.P[1:]) * 1e6
        rec_Alpha_Mod[i] = glu_pre_model.p["alpha"]

        # İLERLEME ÇUBUĞU
        if i % (steps // 20) == 0:
            percent = (i / steps) * 100
            elapsed = time.time() - start_time
            print(f"%{percent:.0f} tamamlandı. Geçen süre: {elapsed:.1f} sn")

    print(f"\nSimülasyon tamamlandı. Süre: {time.time() - start_time:.2f} sn")
    print("Grafikler hazırlanıyor...")

    # ---------------------------------------------------------------------
    # 7. GRAFİKLER
    # ---------------------------------------------------------------------

    # --- PRE-SYNAPTIC ---
    fig1, ax1 = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig1.suptitle("1. Pre-Sinaptik Dinamikler", fontsize=14)
    ax1[0].plot(time_array, rec_V_pre); ax1[0].set_ylabel("V_pre (mV)")
    ax1[1].plot(time_array, rec_Ca_Fast, label="Ca_fast")
    ax1[1].plot(time_array, rec_Ca_Slow, label="Ca_slow")
    ax1[1].legend(); ax1[1].set_ylabel("[Ca] (uM)")
    ax1[2].plot(time_array, rec_Ca_ER); ax1[2].set_ylabel("Ca_ER (uM)")
    ax1[3].plot(time_array, rec_IP3_Pre); ax1[3].set_ylabel("IP3 (uM)"); ax1[3].set_xlabel("Time (ms)")

    # --- ASTROCYTE ---
    fig2, ax2 = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig2.suptitle("2. Astrosit & Feedback", fontsize=14)
    ax2[0].plot(time_array, rec_Glu_Syn); ax2[0].set_ylabel("Glu_syn (mM)")
    ax2[1].plot(time_array, rec_IP3_Astro); ax2[1].set_ylabel("IP3_astro (uM)")
    ax2[2].plot(time_array, rec_Ca_Astro); ax2[2].set_ylabel("Ca_astro (uM)")
    ax2[3].plot(time_array, rec_Glu_Extra); ax2[3].set_ylabel("Glu_extra (mM)"); ax2[3].set_xlabel("Time (ms)")

    # --- POST-SYNAPTIC & CaMKII ---
    fig3, ax3 = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig3.suptitle("3. Post-Sinaptik ve CaMKII / LTP", fontsize=14)
    ax3[0].plot(time_array, rec_V_post); ax3[0].set_ylabel("V_post (mV)")
    ax3[1].plot(time_array, rec_Ca_Post); ax3[1].set_ylabel("Ca_post (uM)")
    ax3[2].plot(time_array, rec_CaMKII_P); ax3[2].set_ylabel("CaMKII-P (uM)")
    ax3[3].plot(time_array, rec_Alpha_Mod); ax3[3].set_ylabel("alpha (mod)"); ax3[3].set_xlabel("Time (ms)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()