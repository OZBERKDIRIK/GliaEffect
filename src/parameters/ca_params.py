CA_PARAMS = {
    # ============================================================
    # 1. Fiziksel Sabitler (Physical Constants)
    # ============================================================
    "F": 96487.0,        # C/mol (Tablo 3)
    "R": 8.314,          # J/(mol K)
    "T": 293.15,         # K (Tablo 3)
    "z_Ca": 2,           # Değerlik

    # ============================================================
    # 2. Geometri (SI: Metre ve Metre Küp)
    # ============================================================
    # Tablo: 1.24 µm² -> 1.24e-12 m²
    "A_btn": 1.24e-12,   
    # Tablo: 0.13 µm³ -> 0.13e-18 m³
    "V_btn": 0.13e-18,   

    # ============================================================
    # 3. Konsantrasyonlar (SI: Molar [mol/L] veya [mol/m³] değil, Molar kullanacağız)
    # 1 Molar = 1 mol/L
    # ============================================================
    "c_ext": 2000.0e-6,  # 2 mM -> 0.002 Molar (Tablo: 2 mM)
    "c_i_rest": 0.1e-6,  # 0.1 µM -> 1e-7 Molar (Tablo: 0.1 µM)

    # ============================================================
    # 4. Hızlı (Fast) Dinamikler: Kanallar ve Pompalar
    # ============================================================
    # Kanal Yoğunluğu: Tablo 3.2 / µm² -> 3.2e12 / m²
    "rho_Ca": 3.2e12,    

    # Kanal İletkenliği: Tablo 2.3 pS -> 2.3e-12 Siemens
    "g_Ca": 2.3e-12,     

    # Voltaj Parametreleri (Volt cinsinden)
    "V_mCa": -0.017,     # -17 mV -> -0.017 V (Tablo)
    "k_mCa": 0.0084,     # 8.4 mV -> 0.0084 V (Tablo)
    "tau_mCa": 0.010,    # 10 ms -> 0.010 s (SI: Saniye)

    # PMCA Pompası
    # Tablo: 0.4 µA/cm².
    # Çevrim: (0.4e-6 A) / (1e-4 m²) = 0.4e-2 A/m² = 0.004 A/m²
    "v_PMCA_max": 0.004, 
    "K_PMCA": 0.1e-6,    # 0.1 µM -> 1e-7 Molar

    # Sızıntı (Leak)
    # Tablo: 2.66e-6 / ms. SI (Saniye): 2.66e-6 * 1000 = 2.66e-3 / s
    "v_leak": 2.66e-3,   # 1/s

    # ============================================================
    # 5. Yavaş (Slow) Dinamikler: ER ve IP3 (Tüm birimler Molar ve Saniye)
    # ============================================================
    "c1": 0.185,         # Hacim oranı (boyutsuz)

    "v1": 30.0,          # 30 / s (Tablo: 30 s^-1)
    "v2": 0.2374,        # 0.2374 / s
    
    # SERCA: Tablo 90 µM/s. SI: 90e-6 Molar/s
    "v3": 90.0e-6,       
    "k3": 0.1e-6,        # 0.1 µM -> 1e-7 Molar

    # Dissociation Constants (µM -> Molar)
    "d1": 0.13e-6,       
    "d2": 1.049e-6,      
    "d3": 0.9434e-6,     # Tablo 943.4 nM = 0.9434 µM
    "d5": 0.08234e-6,    # Tablo 82.34 nM = 0.08234 µM

    # Bağlanma Sabiti: Tablo 0.2 / (µM s). 
    # SI: 0.2 / (1e-6 M * s) = 0.2e6 / (M s) = 200,000 M^-1 s^-1
    "a2": 0.2e6,         

    # IP3 Üretimi
    # Tablo: 0.062 µM/s -> 0.062e-6 Molar/s
    "v_g": 0.062e-6,     
    "k_g": 0.78e-6,      # 0.78 µM -> Molar
    "tau_p": 0.14,       # 0.14 / s (Tablo)
    "p0": 0.16e-6        # 0.16 µM -> Molar
}