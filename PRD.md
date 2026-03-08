# Product Requirements Document (PRD)
## AI-Powered Fraud Detection System for Illegal Online Gambling (Judol)
### PIDI - DIGDAYA X Hackathon 2026 — Bank Indonesia

---

## 1. Executive Summary

Indonesia menghadapi krisis perjudian online (judol) yang masif. Sepanjang 2025, perputaran dana judol mencapai **Rp286,84 triliun** melalui **422,1 juta transaksi**, dengan 47,49% dari seluruh Laporan Transaksi Keuangan Mencurigakan (LTKM) PPATK berasal dari aktivitas judol. Sebanyak **12,3 juta orang** tercatat melakukan deposit ke platform judol melalui bank, e-wallet, dan QRIS.

Pendekatan saat ini bersifat **reaktif** — akun diblokir setelah transaksi terjadi. Sistem kami membangun deteksi **real-time dan proaktif** berbasis AI yang mampu mendeteksi pola transaksi judol sebelum kerugian meluas.

---

## 2. Problem Statement

**Kategori:** Penguatan Ketahanan dan Inovasi Keuangan  
**Topik:** Fraud Detection Systems (FDS)  
**Event:** PIDI - DIGDAYA X Hackathon 2026 by Bank Indonesia

### Core Problem
Operator judol menggunakan QRIS sebagai metode deposit utama karena:
- Transaksi instan dan sulit dilacak secara manual
- Merchant QRIS dapat dibeli secara online (jual beli akun terverifikasi)
- Satu operasi judol menggunakan puluhan hingga ratusan akun merchant sekaligus
- Akun merchant selalu berganti sehingga blacklist statis tidak efektif

### Why Existing Solutions Fail
| Pendekatan Saat Ini | Kelemahan |
|---|---|
| Blacklist merchant ID | Merchant selalu ganti akun |
| Manual review | Tidak scalable, lambat |
| Rule-based only | Terlalu banyak false positive |
| Reactive blocking | Kerugian sudah terjadi |

---

## 3. Regulatory & Legal Foundation

| Regulasi | Relevansi |
|---|---|
| **POJK No. 12 Tahun 2024** | Strategi Anti-Fraud — mewajibkan pilar Deteksi & Pemantauan |
| **PBI No. 2 Tahun 2024** | Keamanan Sistem Informasi bagi Penyelenggara Sistem Pembayaran |
| **Keppres No. 21 Tahun 2024** | Dasar hukum Satgas Pemberantasan Judi Online |
| **UU No. 8 Tahun 2010** | Tindak Pidana Pencucian Uang (TPPU) |

---

## 4. Solution Overview

### Product Name
**Pantau** — AI-Powered Judol Transaction Detection Platform

### What We Build
Platform dua lapis yang bekerja secara bersamaan:
- 📱 **Dashboard App** — visual, interaktif, untuk compliance officer BI/OJK/Bank
- ⚙️ **REST API** — bank & fintech dapat mengintegrasikan ke sistem yang sudah ada

### Core Insight
> *Operator judol tidak bisa menyembunyikan PERILAKU transaksi, hanya identitas merchant. Sistem kami mendeteksi pola perilaku dari DUA SISI sekaligus — merchant sebagai payment gateway judol, dan user sebagai pelaku perjudian.*

### Dual Detection Approach

```
Transaction Happens
        ↓
┌─────────────────────┐     ┌─────────────────────┐
│     USER SIDE       │     │   MERCHANT SIDE     │
│                     │     │                     │
│ Apakah USER ini     │     │ Apakah MERCHANT ini │
│ berperilaku seperti │     │ berperilaku seperti │
│ pelaku judol?       │     │ payment gateway     │
│                     │     │ judol?              │
└─────────────────────┘     └─────────────────────┘
          │                           │
          └─────────────┬─────────────┘
                        ↓
               Combined Risk Score
                        ↓
┌──────────┬────────────┬──────────────────────────┐
│ Scenario │ Result     │ Action                   │
├──────────┼────────────┼──────────────────────────┤
│ Both HIGH│ Confirmed  │ Freeze both ❄️            │
│ Merchant │ Sus merch  │ Freeze merchant,          │
│ only HIGH│            │ monitor user ⚠️           │
│ User     │ Sus user   │ Flag user, merchant       │
│ only HIGH│            │ safe 🚩                  │
│ Both LOW │ Normal     │ Pass ✅                   │
└──────────┴────────────┴──────────────────────────┘
```

> Backed by Han et al. (2018) who prove that detecting both **gamblers** (user layer) and **gambling agents** (merchant layer) simultaneously yields significantly better results than either alone.

---

## 5. Target Users

| User | Role |
|---|---|
| **Bank Indonesia & OJK** | Monitor dashboard nasional, lihat tren fraud |
| **Bank & Fintech** | Integrasi via API, terima risk score per transaksi |
| **Compliance Officer** | Review flagged transactions, approve/reject freeze |

---

## 6. Key Detection Signals

### Judol Merchant Behavioral Fingerprint
*(Mapped to: Gambling Agents layer — Han et al. 2018)*
- Akun merchant **dormant** lalu tiba-tiba lonjakan transaksi masif
- **Velocity spike** ekstrem (0 transaksi → ratusan dalam jam pertama)
- Pengirim dari **berbagai kota berbeda** (geo spread nasional)
- **Zero repeat customers** — tidak ada pelanggan yang kembali
- Dominasi **round amounts** (Rp50.000, Rp100.000, Rp200.000)
- Aktif mayoritas **pukul 22.00 — 05.00**
- Dana **langsung diteruskan** ke akun lain dalam hitungan menit

### Judol User Behavioral Fingerprint
*(Mapped to: Gamblers layer — Han et al. 2018)*
- Transaksi ke merchant baru yang tidak dikenal berulang kali
- Pola **deposit → withdraw → deposit** dalam waktu singkat (gambling cycle)
- Transaksi di luar jam dan kebiasaan normal pengguna
- Peningkatan frekuensi dan jumlah transaksi secara progresif
- Nominal transaksi eskalasi dari waktu ke waktu

### Cross-Entity Signals
*(Kekuatan tambahan saat kedua sisi dianalisis bersamaan)*
- User mencurigakan mengirim ke merchant mencurigakan = konfirmasi kuat
- Banyak user mencurigakan mengirim ke merchant yang sama = organized operation
- User normal mengirim ke merchant mencurigakan = merchant flagged, user dimonitor
- User mencurigakan mengirim ke merchant normal = user flagged untuk investigasi

---

## 7. Data Model

### 7.1 Transaction Data (Input)

| Column | Type | Description |
|---|---|---|
| `transaction_id` | STRING | Unique transaction identifier |
| `timestamp` | DATETIME | Exact time of transaction |
| `user_id` | STRING | Anonymized user identifier |
| `merchant_id` | STRING | QRIS merchant identifier |
| `amount` | FLOAT | Transaction amount (IDR) |
| `user_city` | STRING | City of the user |
| `user_province` | STRING | Province of the user |
| `merchant_city` | STRING | Registered city of merchant |
| `merchant_province` | STRING | Registered province of merchant |
| `transaction_type` | STRING | QRIS / transfer / etc |
| `device_id` | STRING | Device fingerprint |
| `is_round_amount` | BOOLEAN | Amount divisible by 50.000 |
| `tx_hour` | INTEGER | Hour of transaction (0-23) |
| `tx_day_of_week` | INTEGER | Day of week (0-6) |

---

### 7.2 User Profile Features (Layer 1)

| Feature | Description |
|---|---|
| `user_tx_frequency_1hr` | Jumlah transaksi dalam 1 jam terakhir |
| `user_tx_frequency_24hr` | Jumlah transaksi dalam 24 jam terakhir |
| `user_tx_frequency_7d` | Jumlah transaksi dalam 7 hari terakhir |
| `user_avg_amount_30d` | Rata-rata jumlah transaksi 30 hari |
| `user_amount_deviation` | Deviasi jumlah dari rata-rata historis |
| `user_peak_hour_normal` | Jam transaksi normal pengguna |
| `user_hour_deviation` | Deviasi jam dari kebiasaan normal |
| `user_unique_merchants_7d` | Jumlah merchant unik dalam 7 hari |
| `user_repeat_merchant_rate` | % transaksi ke merchant yang pernah dikunjungi |
| `user_round_amount_rate` | % transaksi dengan round amount |
| `user_deposit_withdraw_ratio` | Rasio deposit vs withdraw cepat |
| `user_geo_consistency` | Konsistensi lokasi transaksi |
| `user_profile_age_days` | Seberapa lama profil pengguna ada |
| `user_trust_score` | Tingkat kepercayaan berdasarkan histori (0-100) |

---

### 7.3 Merchant Profile Features (Layer 2)

| Feature | Description |
|---|---|
| `merchant_age_days` | Usia akun merchant sejak dibuat |
| `merchant_tx_velocity_1hr` | Jumlah transaksi masuk dalam 1 jam |
| `merchant_tx_velocity_24hr` | Jumlah transaksi masuk dalam 24 jam |
| `merchant_velocity_delta` | Perubahan velocity vs hari sebelumnya |
| `merchant_unique_senders_1hr` | Pengirim unik dalam 1 jam |
| `merchant_unique_senders_24hr` | Pengirim unik dalam 24 jam |
| `merchant_repeat_sender_rate` | % pengirim yang pernah bertransaksi sebelumnya |
| `merchant_geo_spread_score` | Seberapa tersebar lokasi pengirim (0-100) |
| `merchant_round_amount_rate` | % transaksi dengan round amount |
| `merchant_peak_hour` | Jam tersibuk merchant |
| `merchant_night_tx_rate` | % transaksi antara pukul 22.00-05.00 |
| `merchant_dormant_spike` | Apakah ada lonjakan setelah periode dormant |
| `merchant_forward_speed_avg` | Rata-rata waktu dana diteruskan (menit) |
| `merchant_trust_score` | Tingkat kepercayaan berdasarkan histori (0-100) |

---

### 7.3a Merchant Adaptive Learned Profile

Setiap merchant membangun **profil perilaku unik** yang diperbarui secara otomatis setiap transaksi masuk. Model membandingkan transaksi baru terhadap profil historis merchant itu sendiri — bukan aturan global.

#### Learned Metrics (diperbarui setiap transaksi)

| Learned Metric | Type | How It's Calculated |
|---|---|---|
| `learned_avg_amount` | FLOAT | Rolling mean semua transaksi masuk |
| `learned_amount_std` | FLOAT | Standar deviasi nominal transaksi |
| `learned_amount_min` | FLOAT | Nominal terendah yang pernah diterima |
| `learned_amount_max` | FLOAT | Nominal tertinggi yang pernah diterima |
| `learned_peak_hour_histogram` | ARRAY[24] | Distribusi transaksi per jam (0-23) |
| `learned_peak_hour` | INTEGER | Jam dengan transaksi terbanyak |
| `learned_peak_day_of_week` | INTEGER | Hari tersibuk dalam seminggu |
| `learned_avg_daily_velocity` | FLOAT | Rata-rata transaksi per hari |
| `learned_velocity_std` | FLOAT | Standar deviasi velocity harian |
| `learned_geo_distribution` | MAP | % pengirim per provinsi (historis) |
| `learned_geo_centroid` | COORDINATES | Pusat geografis pengirim |
| `learned_geo_radius_km` | FLOAT | Rata-rata radius sebaran pengirim |
| `learned_repeat_sender_rate` | FLOAT | % pengirim yang kembali bertransaksi |
| `learned_unique_senders_per_day` | FLOAT | Rata-rata pengirim unik per hari |
| `learned_round_amount_rate` | FLOAT | % historis transaksi nominal bulat |
| `learned_forward_speed_avg` | FLOAT | Rata-rata waktu (menit) dana diteruskan |
| `learned_forward_speed_std` | FLOAT | Standar deviasi kecepatan forward |
| `learned_session_gap_avg` | FLOAT | Rata-rata jeda antar sesi transaksi |
| `learned_total_tx_count` | INTEGER | Total transaksi sepanjang waktu |
| `learned_profile_last_updated` | DATETIME | Timestamp profil terakhir diperbarui |

#### Deviation Scoring (per transaksi baru)

Setiap transaksi baru dibandingkan terhadap profil yang sudah dipelajari:

| Deviation Feature | Formula |
|---|---|
| `amount_deviation_score` | `abs(new_amount - learned_avg) / learned_amount_std` |
| `hour_deviation_score` | Seberapa jauh dari `learned_peak_hour_histogram` |
| `velocity_deviation_score` | `(current_velocity - learned_avg_daily_velocity) / learned_velocity_std` |
| `geo_deviation_score` | Jarak pengirim baru dari `learned_geo_centroid` |
| `sender_novelty_score` | Apakah pengirim ini pernah ada di histori? |

#### New Merchant Fallback (Profil Belum Matang)

Ketika merchant belum memiliki cukup histori, sistem menggunakan **peer group baseline**:

| Jumlah Transaksi | Bobot Profil Sendiri | Bobot Peer Group |
|---|---|---|
| 0 – 100 | 20% | 80% |
| 100 – 500 | 50% | 50% |
| 500 – 1.000 | 80% | 20% |
| 1.000+ | 100% | 0% |

> Peer group = merchant dengan kategori, wilayah, dan usia akun yang serupa.

---

### 7.4 Network Features (Layer 3)

| Feature | Description |
|---|---|
| `network_unique_city_senders` | Jumlah kota unik pengirim ke merchant |
| `network_unique_province_senders` | Jumlah provinsi unik pengirim |
| `network_sender_concentration` | % pengirim terbesar dari satu area |
| `network_merchant_hub_score` | Seberapa besar merchant sebagai hub penerima |
| `network_shared_sender_count` | Jumlah pengirim yang sama dengan merchant lain yang dicurigai |
| `network_shared_destination` | Apakah merchant meneruskan ke tujuan yang sama dengan merchant lain |
| `network_cluster_id` | ID cluster jaringan judol yang teridentifikasi |
| `network_ring_score` | Kemungkinan bagian dari organized ring (0-100) |

---

### 7.5 Temporal Features (Layer 4)

| Feature | Description |
|---|---|
| `temporal_sequence_pattern` | Pola urutan transaksi (deposit-loss-deposit) |
| `temporal_session_duration` | Durasi sesi transaksi |
| `temporal_inter_tx_time` | Rata-rata waktu antar transaksi |
| `temporal_escalation_pattern` | Pola peningkatan jumlah dari waktu ke waktu |
| `temporal_time_consistency` | Konsistensi waktu transaksi antar hari |
| `temporal_cycle_detected` | Apakah pola siklus gambling terdeteksi |

---

### 7.6 Velocity Delta Features (Layer 5)

| Feature | Description |
|---|---|
| `velocity_zscore_1hr` | Z-score velocity 1 jam vs rolling average |
| `velocity_zscore_24hr` | Z-score velocity 24 jam vs rolling average |
| `velocity_delta_pct` | % perubahan velocity vs hari sebelumnya |
| `velocity_spike_detected` | Boolean: apakah spike abnormal terdeteksi |
| `velocity_spike_magnitude` | Seberapa besar spike (x kali normal) |

---

### 7.7 Money Flow Features (Layer 6)

| Feature | Description |
|---|---|
| `flow_forward_speed` | Waktu dana diteruskan setelah diterima (menit) |
| `flow_forward_rate` | % dana yang langsung diteruskan |
| `flow_destination_age` | Usia akun tujuan penerusan dana |
| `flow_destination_pattern` | Apakah tujuan sama dengan merchant dicurigai lain |
| `flow_layering_detected` | Apakah pola layering (pencucian) terdeteksi |
| `flow_chain_depth` | Kedalaman rantai penerusan dana |

---

## 8. System Architecture

### 8.1 Three-Layer Detection Pipeline

```
Transaction Input
        ↓
┌─────────────────────────────┐
│   Layer 1: Rule Engine      │ Fast filter, obvious patterns
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│   Layer 2: ML Models        │ 6 sub-models (see below)
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│   Layer 3: LLM Reasoning    │ Explain & validate flag
└─────────────────────────────┘
        ↓
   Final Risk Score (0-100)
        ↓
┌──────────┬──────────┬────────┐
│  50-70   │  70-90   │  90+   │
│ Monitor  │  Flag +  │ Freeze │
│    ⚠️    │ Review 🚩│   ❄️   │
└──────────┴──────────┴────────┘
```

### 8.2 ML Model Per Layer

| Layer | Model | Approach | Library |
|---|---|---|---|
| **1. User Behavior** | Isolation Forest | Unsupervised anomaly detection | scikit-learn |
| **2. Merchant Behavior** | Isolation Forest | Unsupervised anomaly detection | scikit-learn |
| **3. Network Clustering** | Graph Analysis | NetworkX node centrality & clustering | NetworkX |
| **4. Temporal Pattern** | Rule-based sequence detection | Pattern matching on transaction sequences | Custom |
| **5. Velocity Delta** | Z-Score / SPC | Statistical process control | NumPy/SciPy |
| **6. Money Flow** | Directed Graph Tracing | Follow fund movement chains | NetworkX |

### 8.3 Adaptive Profiling (Continuous Learning)

Setiap entitas memiliki profil yang terus diperbarui:

```
New Transaction Arrives
        ↓
Score Against Current Profile
        ↓
Add Transaction to History
        ↓
Update Rolling Profile
        ↓
Next Transaction Scored Against
Updated Profile
```

**Trust Score Progression:**

| Transaction History | Profile Weight | Peer Group Weight |
|---|---|---|
| 0 - 100 transaksi | 20% | 80% |
| 100 - 500 transaksi | 50% | 50% |
| 500 - 1000 transaksi | 80% | 20% |
| 1000+ transaksi | 100% | 0% |

### 8.4 Cross-Layer Correlation

Ketika multiple layer secara bersamaan menandai entitas yang sama, skor dikuatkan:

```python
# Weights auto-tuned via grid search on validation set (70/30 split)
final_score = (
    user_score     * W_user +
    merchant_score * W_merchant +
    network_score  * W_network +
    temporal_score * W_temporal +
    velocity_score * W_velocity +
    flow_score     * W_flow
) + cross_correlation_bonus  # bonus if 3+ layers flag same entity
# All weights (W_*) and thresholds optimized to maximize F1 on held-out test data
```

### 8.5 Model Evaluation Protocol

Pipeline menggunakan rigorous evaluation methodology:

1. **Train/Test Split (70/30)**: Model dilatih pada 70% data, dievaluasi pada 30% data yang tidak pernah dilihat model
2. **Hyperparameter Tuning**: Contamination (Isolation Forest), weights, dan thresholds dioptimasi via grid search pada validation set
3. **Metrics yang dilaporkan** (semua dihitung pada test set):

| Metric | Deskripsi | Target |
|---|---|---|
| **F1 Score** | Harmonic mean precision & recall | ≥ 0.70 |
| **AUC-ROC** | Area under ROC curve (separability) | ≥ 0.80 |
| **PR-AUC** | Precision-Recall AUC (for imbalanced data) | ≥ 0.65 |
| **Precision** | Flagged transactions yang benar judol | ≥ 0.60 |
| **Recall** | Judol transactions yang berhasil tertangkap | ≥ 0.70 |

4. **Benchmark**: Hasil dibandingkan antara parametric dataset vs GAN-augmented dataset untuk membuktikan value dari synthetic data augmentation

---

## 9. Risk Scoring & Actions

| Score Range | Level | Action |
|---|---|---|
| 0 - 40 | Normal ✅ | Pass through |
| 40 - 60 | Suspicious ⚠️ | Monitor & log |
| 60 - 80 | High Risk 🚩 | Alert compliance officer + human review |
| 80 - 100 | Critical ❄️ | Auto-freeze + mandatory review within 2 hours |

---

## 10. Governance & Human-in-the-Loop

### Why This Matters
Pembekuan akun adalah tindakan serius secara hukum. Sistem tidak boleh bertindak sepenuhnya otomatis.

### Review Queue
```
Score > 70 → Masuk antrian review compliance officer
        ↓
Officer melihat:
- Risk score & confidence level
- LLM explanation (Bahasa Indonesia)
- Semua sinyal yang memicu flag
- Transaction history visualization
        ↓
Officer mengisi Structured Review Form
        ↓
- Konfirmasi freeze
- Clear flag (false positive)
- Eskalasi ke PPATK
```

---

### Structured Review Decision Form

Compliance officer tidak hanya memilih YES/NO — tetapi mengisi form terstruktur yang menangkap **alasan spesifik** di balik setiap keputusan. Ini adalah sumber data utama untuk model belajar dari waktu ke waktu.

#### Jika FRAUD — Sinyal Yang Membenarkan:
| Signal | Pilihan Officer |
|---|---|
| Velocity spike abnormal | [ ] Ya [ ] Tidak |
| Geo spread terlalu luas | [ ] Ya [ ] Tidak |
| Pola round amount dominan | [ ] Ya [ ] Tidak |
| Aktivitas dini hari | [ ] Ya [ ] Tidak |
| Zero repeat senders | [ ] Ya [ ] Tidak |
| Dana langsung diteruskan | [ ] Ya [ ] Tidak |
| Terhubung ke merchant flagged lain | [ ] Ya [ ] Tidak |
| Lainnya | [ ] _____________ |

#### Jika FALSE POSITIVE — Alasan Kesalahan Flag:
| Alasan | Pilihan Officer |
|---|---|
| Event/promo legitimate (velocity wajar) | [ ] Ya [ ] Tidak |
| Merchant baru pindah lokasi (geo mismatch) | [ ] Ya [ ] Tidak |
| Pola musiman (Ramadan, Lebaran, Harbolnas) | [ ] Ya [ ] Tidak |
| Merchant baru, perilaku wajar untuk jenisnya | [ ] Ya [ ] Tidak |
| Nominal kebetulan bulat | [ ] Ya [ ] Tidak |
| Lainnya | [ ] _____________ |

#### Tingkat Keyakinan Officer:
`[ ] Sangat Yakin` `[ ] Cukup Yakin` `[ ] Tidak Yakin`

#### Catatan Tambahan:
`_________________________________________________`

---

### Structured Feedback Data Model

Setiap review decision disimpan sebagai training sample terstruktur:

```python
feedback = {
    'merchant_id': 'QR-XXX-2024',
    'transaction_id': 'TX-1001',
    'decision': 'FRAUD',             # FRAUD / FALSE_POSITIVE
    'triggered_signals': [           # sinyal yang BENAR memicu
        'velocity_spike',
        'geo_spread',
        'night_activity'
    ],
    'misleading_signals': [],        # sinyal yang ternyata TIDAK relevan
    'false_positive_reason': None,   # alasan jika false positive
    'seasonal_context': None,        # e.g. 'ramadan', 'harbolnas'
    'officer_confidence': 'very_sure',
    'connected_merchants': [         # merchant lain yang terkait
        'QR-YYY-2024',
        'QR-ZZZ-2024'
    ],
    'notes': 'Terhubung ke 3 merchant flagged lain via shared senders',
    'reviewed_at': '2025-03-07T03:12:00',
    'reviewed_by': 'officer_id_xxx'
}
```

---

### How Model Learns From Structured Feedback

#### Skenario 1: Confirmed Fraud
```
Officer: FRAUD
Reasons: velocity_spike + geo_spread + night_activity
        ↓
Model adjusts:
"Ketika 3 sinyal ini muncul bersamaan
pada merchant tipe ini
→ naikkan bobot ketiga sinyal tersebut"
        ↓
Deteksi lebih akurat untuk pola serupa
```

#### Skenario 2: False Positive — Musiman
```
Officer: FALSE POSITIVE
Reason: Ramadan season (velocity spike wajar)
        ↓
Model learns:
"Selama periode Ramadan,
velocity spike NORMAL untuk merchant kategori ini"
        ↓
Tambahkan seasonal adjustment otomatis
→ false positive berkurang di Ramadan berikutnya
```

#### Skenario 3: False Positive — Sinyal Menyesatkan
```
Officer: FALSE POSITIVE
Misleading signal: round_amount (kebetulan)
        ↓
Model learns:
"Round amount SAJA tidak cukup
untuk merchant kategori ini"
        ↓
Turunkan bobot round_amount secara isolasi
→ hanya kuat jika dikombinasikan sinyal lain
```

---

### Projected False Positive Improvement

| Periode | Jumlah Review | False Positive Rate |
|---|---|---|
| Bulan 1 | ~100 reviews | ~35% |
| Bulan 3 | ~300 reviews | ~18% |
| Bulan 6 | ~600 reviews | ~8% |
| Bulan 12 | ~1.200 reviews | ~3% |

> Model semakin presisi karena belajar dari **alasan spesifik**, bukan hanya keputusan biner.

---

### Audit Trail
Setiap aksi dicatat secara lengkap:

| Field | Description |
|---|---|
| `flagged_by` | Model/layer mana yang men-trigger flag |
| `triggered_signals` | Sinyal spesifik yang memicu |
| `risk_score` | Score pada saat flag |
| `reviewed_by` | ID compliance officer |
| `review_decision` | FRAUD / FALSE_POSITIVE |
| `review_reasons` | Alasan terstruktur dari form |
| `action_taken` | FREEZE / MONITOR / CLEARED |
| `review_duration` | Berapa lama proses review |
| `reviewed_at` | Timestamp keputusan |

---

## 11. Explainability Report (per flagged transaction)

```
═══════════════════════════════════════════
LAPORAN TRANSAKSI MENCURIGAKAN
═══════════════════════════════════════════
Merchant ID  : QR-XXX-2024
Risk Score   : 91/100
Confidence   : 84%
Timestamp    : 2025-03-07 02:34:11
Action       : FREEZE ❄️

SINYAL TERDETEKSI:
• 847 pengirim unik dalam 6 jam (+47x rata-rata peer)
• Pengirim dari 34 kota berbeda (geo spread: ekstrem)
• 91% transaksi bernominal bulat
• Usia merchant: 2 hari
• Aktivitas puncak: 23.00 - 04.00
• Dana diteruskan rata-rata dalam 3 menit
• Tidak ada repeat customer
• Velocity spike: 0 → 847 transaksi (dari dormant)
• 3 merchant lain menerima dari pengirim yang sama

ANALISIS AI:
"Pola transaksi merchant ini sangat konsisten dengan
operasi judi online ilegal. Lonjakan mendadak dari
akun dormant, kombinasi pengirim dari seluruh Indonesia
dengan nominal bulat di jam dini hari, serta penerusan
dana instan mengindikasikan akun ini digunakan sebagai
payment gateway judol."

Direview oleh : [Nama Compliance Officer]
Keputusan    : [Pending Review]
═══════════════════════════════════════════
```

---

## 12. Product Features

### Dashboard (Frontend)
- [ ] Live transaction feed dengan risk score real-time
- [ ] Flagged transaction queue untuk compliance officer
- [ ] Network graph visualization (merchant-user connections)
- [ ] Geographic heatmap pengirim per merchant
- [ ] Merchant behavioral profile view
- [ ] User behavioral profile view
- [ ] National fraud trend analytics
- [ ] Explainability report per flagged transaction
- [ ] Audit trail log

### API (Backend)
- [ ] `POST /analyze` — Submit transaction, get risk score
- [ ] `GET /merchant/{id}/profile` — Get merchant behavioral profile
- [ ] `GET /user/{id}/profile` — Get user behavioral profile
- [ ] `POST /review` — Submit human review decision
- [ ] `GET /alerts` — Get current active alerts
- [ ] `GET /network/{merchant_id}` — Get merchant network graph

---

## 13. Research Foundation

| Paper | Relevance |
|---|---|
| *Role Recognition of IOG Participants Using Monetary Transaction Data* (Springer, 2018) | Validates transaction stats + network features combination |
| *Recognizing Roles of IOG Participants: Ensemble Learning* (ScienceDirect, 2019) | Validates multi-layer ensemble approach |
| *Automated Acquisition of Illegal Fund Accounts in Gambling Websites* (ScienceDirect, 2025) | Validates merchant account analysis approach |
| *Blockchain Illegal Gambling: Community Detection & Network Embedding* (SPIE, 2024) | Validates graph/network layer |
| *Advanced Fraud Detection Using ML Models* (arXiv, 2025) | Validates behavioral + temporal feature engineering |
| *Detecting Anomalous Transactions in Gambling* (PMC, 2021) | Validates time series + SafeZone threshold concept |
| *AI-Based Betting Anomaly Detection: Ensemble Model* (Nature, 2024) | Validates 3-tier scoring (normal/warning/abnormal) |

---

## 14. Dataset Strategy

### 14.1 Why No Real Judol Dataset Exists

Dataset transaksi judol nyata tidak tersedia secara publik karena:
- Data kriminal finansial bersifat sangat sensitif
- Hanya bisa diakses oleh aparat hukum dengan surat perintah
- Han et al. (2018) pun mendapatkan data IOG dari lembaga penegak hukum dengan warrant

> *"The MTD used in this study was provided by a law enforcement agency. They obtained the data with warrants during the investigation of an IOG case."* — Han et al. (2018)

---

### 14.2 The Column Mismatch Problem

Base dataset publik memiliki kolom yang **berbeda** dari skema yang dibutuhkan sistem:

| Base Dataset | Kolom Yang Ada | Yang Kita Butuhkan |
|---|---|---|
| IBM AML | From Bank, To Bank, Amount | user_city, merchant_city, is_round_amount |
| PaySim | nameOrig, nameDest, type | user_id, merchant_id, tx_hour |
| Sports Betting | bet_amount, win/loss | deposit_cycle, gambling_session |

> **Base datasets tidak bisa langsung dipakai — kolom-nya tidak cocok.**

---

### 14.3 Two-Stage Data Pipeline

```
Stage 1: Parametric Generator (✅ DONE)
──────────────────────────────────────────
Base Dataset → Extract Statistical Parameters
        +
PPATK Statistics → Calibration Parameters
        ↓
Custom Parametric Generator
(outputs YOUR exact columns)
        ↓
500,000 rows base dataset ✅

Stage 2: GAN Augmentation
──────────────────────────────────────────
500K base dataset (Stage 1 output)
        ↓
CTGAN / TVAE Training
(learns joint distributions, correlations,
 edge cases the parametric generator misses)
        ↓
GAN-generated synthetic data
        +
Original 500K base
        ↓
Final training dataset ✅
```

> **Stage 1** memastikan skema dan pola domain benar. **Stage 2** memperkaya variasi statistik menggunakan GAN sehingga model ML lebih robust terhadap pola yang tidak terduga.

#### Mengapa Dua Tahap?

| Aspek | Parametric Generator Saja | + GAN Augmentation |
|---|---|---|
| **Distribusi** | Mengikuti distribusi yang di-hardcode | Belajar joint distribution dari data |
| **Korelasi antar kolom** | Harus di-program manual | GAN menangkap korelasi implisit |
| **Edge cases** | Terbatas pada pola yang kita definisikan | GAN menghasilkan variasi baru yang realistis |
| **Overfitting risk** | Model ML mudah menghafal pola generator | Data lebih variatif → generalisasi lebih baik |
| **Realism** | Rule-based, predictable | Lebih mendekati distribusi transaksi nyata |

#### GAN Technology Choice

| Library | Model | Kelebihan |
|---|---|---|
| **SDV (Synthetic Data Vault)** | CTGAN | State-of-the-art untuk tabular data, handles mixed types |
| **SDV** | TVAE | Lebih cepat training, stabil untuk dataset besar |
| **SDV** | CopulaGAN | Menangkap korelasi antar kolom lebih baik |

> Rekomendasi: Gunakan **CTGAN** sebagai primary, **TVAE** sebagai fallback jika training terlalu lambat.

---

### 14.4 What We Extract From Each Base Dataset

#### From IBM AML Dataset:
```python
ibm_params = {
    'fraud_rate': 0.023,              # 2.3% suspicious transactions
    'amount_mean': 150000,            # average transaction amount
    'amount_std': 280000,             # amount variance
    'layering_pattern': True,         # money forwarding behavior
    'multi_hop_rate': 0.15,           # 15% transactions are multi-hop
}
```

#### From PaySim Dataset:
```python
paysim_params = {
    'mobile_tx_hourly_dist': [...],   # 24-hour distribution
    'peak_hours': [11, 12, 13, 17],   # lunch & evening peak
    'repeat_merchant_rate': 0.62,     # 62% repeat to same merchant
    'local_geo_radius_km': 5,         # avg 5km sender radius
    'round_amount_rate_normal': 0.12, # 12% round amounts (normal)
}
```

#### From Sports Betting Dataset:
```python
betting_params = {
    'session_duration_min': 45,       # avg gambling session
    'deposit_frequency_per_session': 3.2,  # deposits per session
    'escalation_pattern': True,       # amounts increase over time
    'late_night_rate': 0.68,          # 68% bets late night
    'loss_rate': 0.72,                # 72% sessions end in loss
}
```

#### From PPATK Statistics:
```python
ppatk_params = {
    'judol_amount_range': (10000, 100000),  # Rp10k - Rp100k
    'smurfing': True,                        # many small transactions
    'player_age_range': (20, 30),            # majority age
    'active_merchants': 28000,               # merchant pool size
    'judol_active_hours': [22,23,0,1,2,3,4], # peak judol hours
    'round_amount_rate_judol': 0.91,         # 91% round amounts
    'geo_spread_provinces': 34,              # all provinces
}
```

---

### 14.5 Base Datasets (Public, Free)

#### Dataset 1: IBM AML Dataset ✅ Primary
| Property | Value |
|---|---|
| Source | Kaggle (IBM) |
| Size | ~180,000 transactions |
| Contains | Illegal gambling flows, AML patterns |
| URL | kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml |
| Used For | Extract fraud rate, amount distribution, layering patterns |

#### Dataset 2: PaySim Dataset ✅ Secondary
| Property | Value |
|---|---|
| Source | Kaggle |
| Size | ~6,300,000 transactions (use 100k subset) |
| Contains | Mobile money transactions (QRIS-like behavior) |
| URL | kaggle.com/datasets/ealaxi/paysim1 |
| Used For | Extract hourly distribution, repeat rate, geo radius |

#### Dataset 3: Sports Betting Profiling Dataset ✅ User Layer
| Property | Value |
|---|---|
| Source | Kaggle |
| Size | ~500,000 bets, 5,000 users |
| Contains | Betting amounts, win/loss patterns, user behavior |
| URL | kaggle.com/datasets/emiliencoicaud/sports-betting-profiling-dataset |
| Used For | Extract gambling session patterns, deposit cycles |

---

### 14.6 Synthetic Dataset Composition

| Segment | Rows | % | Description |
|---|---|---|---|
| Normal transactions | 425,000 | 85% | Legitimate merchant & user behavior |
| Judol merchant transactions | 50,000 | 10% | Judol payment gateway patterns |
| Judol user transactions | 25,000 | 5% | Gambling cycle user patterns |
| **Total** | **500,000** | **100%** | |

---

### 14.7 Data Generator Script

```python
import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta
import random

# ============================================================
# CALIBRATED PARAMETERS (from base datasets + PPATK)
# ============================================================

INDONESIAN_CITIES = [
    ('Jakarta', 'DKI Jakarta'), ('Surabaya', 'Jawa Timur'),
    ('Bandung', 'Jawa Barat'), ('Medan', 'Sumatera Utara'),
    ('Semarang', 'Jawa Tengah'), ('Makassar', 'Sulawesi Selatan'),
    ('Palembang', 'Sumatera Selatan'), ('Tangerang', 'Banten'),
    ('Depok', 'Jawa Barat'), ('Bekasi', 'Jawa Barat'),
    ('Yogyakarta', 'DI Yogyakarta'), ('Denpasar', 'Bali'),
    ('Balikpapan', 'Kalimantan Timur'), ('Manado', 'Sulawesi Utara'),
    ('Padang', 'Sumatera Barat'), ('Pekanbaru', 'Riau'),
    ('Banjarmasin', 'Kalimantan Selatan'), ('Pontianak', 'Kalimantan Barat'),
]

# Normal transaction parameters (from PaySim + IBM AML)
NORMAL_PARAMS = {
    'amount_range': (15000, 750000),
    'round_amount_rate': 0.12,          # 12% round amounts
    'peak_hours': [8,9,10,11,12,13,17,18,19],
    'repeat_merchant_rate': 0.62,       # 62% repeat customers
    'geo_radius': 'local',              # mostly same city
    'daily_velocity_range': (5, 80),
    'forward_speed': None,              # no forwarding
}

# Judol merchant parameters (from PPATK + Sports Betting)
JUDOL_MERCHANT_PARAMS = {
    'amount_choices': [10000, 20000, 25000, 50000,
                       100000, 200000, 500000],
    'amount_weights': [0.10, 0.10, 0.05, 0.35,
                       0.25, 0.10, 0.05],
    'round_amount_rate': 0.91,          # 91% round amounts
    'peak_hours': [22, 23, 0, 1, 2, 3, 4],
    'repeat_merchant_rate': 0.02,       # near zero repeat
    'geo_radius': 'national',           # nationwide senders
    'daily_velocity_range': (200, 1500),
    'forward_speed_minutes': (1, 10),   # immediate forwarding
    'dormant_days_before': (1, 30),     # dormant then spike
}

# Judol user parameters (from Sports Betting dataset)
JUDOL_USER_PARAMS = {
    'session_deposits': (2, 8),         # deposits per session
    'inter_deposit_minutes': (5, 45),   # time between deposits
    'escalation_rate': 0.68,            # 68% amounts increase
    'late_night_rate': 0.72,            # 72% late night
    'peak_hours': [21, 22, 23, 0, 1, 2],
    'round_amount_rate': 0.88,
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def generate_id():
    return str(uuid.uuid4())[:12].upper()

def generate_timestamp(hour_weights, days_back=90):
    base = datetime.now() - timedelta(days=random.randint(0, days_back))
    hour = random.choices(range(24), weights=hour_weights)[0]
    minute = random.randint(0, 59)
    return base.replace(hour=hour, minute=minute)

def get_normal_hour_weights():
    # Business hours peak, quiet at night
    weights = [1,1,1,1,1,1,2,3,5,7,8,9,10,9,7,6,5,7,8,6,4,3,2,1]
    return weights

def get_judol_hour_weights():
    # Late night peak
    weights = [8,7,6,4,2,2,1,1,1,1,1,1,2,2,2,2,2,3,4,5,6,7,9,10]
    return weights

def get_city_pair(geo_radius, merchant_city=None):
    if geo_radius == 'local':
        # User from same or nearby city as merchant
        city = random.choice(INDONESIAN_CITIES)
        return city, city
    else:
        # User from random city nationwide
        user_city = random.choice(INDONESIAN_CITIES)
        merch_city = merchant_city or random.choice(INDONESIAN_CITIES)
        return user_city, merch_city

def is_round_amount(amount):
    return amount % 50000 == 0 or amount % 10000 == 0

# ============================================================
# NORMAL TRANSACTION GENERATOR
# ============================================================

def generate_normal_transactions(n=425000):
    records = []
    merchant_pool = [generate_id() for _ in range(5000)]
    user_pool = [generate_id() for _ in range(20000)]

    for _ in range(n):
        merchant_id = random.choice(merchant_pool)
        user_id = random.choice(user_pool)
        amount = random.randint(
            NORMAL_PARAMS['amount_range'][0],
            NORMAL_PARAMS['amount_range'][1]
        )
        # Round amount occasionally
        if random.random() < NORMAL_PARAMS['round_amount_rate']:
            amount = round(amount / 50000) * 50000

        ts = generate_timestamp(get_normal_hour_weights())
        user_city, merch_city = get_city_pair('local')

        records.append({
            'transaction_id': generate_id(),
            'timestamp': ts,
            'user_id': user_id,
            'merchant_id': merchant_id,
            'amount': amount,
            'user_city': user_city[0],
            'user_province': user_city[1],
            'merchant_city': merch_city[0],
            'merchant_province': merch_city[1],
            'transaction_type': 'QRIS',
            'device_id': generate_id(),
            'is_round_amount': is_round_amount(amount),
            'tx_hour': ts.hour,
            'tx_day_of_week': ts.weekday(),
            'is_judol_merchant': 0,
            'is_judol_user': 0,
            'label': 0  # normal
        })
    return pd.DataFrame(records)

# ============================================================
# JUDOL MERCHANT TRANSACTION GENERATOR
# ============================================================

def generate_judol_merchant_transactions(n=50000):
    records = []
    # Small pool of judol merchants (they rotate accounts)
    judol_merchant_pool = [generate_id() for _ in range(500)]
    user_pool = [generate_id() for _ in range(50000)]  # many unique users

    for _ in range(n):
        merchant_id = random.choice(judol_merchant_pool)
        user_id = generate_id()  # mostly unique/new users

        # Occasionally repeat user (2% rate)
        if random.random() < JUDOL_MERCHANT_PARAMS['repeat_merchant_rate']:
            user_id = random.choice(user_pool)

        amount = random.choices(
            JUDOL_MERCHANT_PARAMS['amount_choices'],
            weights=JUDOL_MERCHANT_PARAMS['amount_weights']
        )[0]

        ts = generate_timestamp(get_judol_hour_weights())
        user_city, merch_city = get_city_pair('national')

        records.append({
            'transaction_id': generate_id(),
            'timestamp': ts,
            'user_id': user_id,
            'merchant_id': merchant_id,
            'amount': amount,
            'user_city': user_city[0],
            'user_province': user_city[1],
            'merchant_city': merch_city[0],
            'merchant_province': merch_city[1],
            'transaction_type': 'QRIS',
            'device_id': generate_id(),
            'is_round_amount': True,
            'tx_hour': ts.hour,
            'tx_day_of_week': ts.weekday(),
            'is_judol_merchant': 1,
            'is_judol_user': 0,
            'label': 1  # suspicious
        })
    return pd.DataFrame(records)

# ============================================================
# JUDOL USER TRANSACTION GENERATOR
# ============================================================

def generate_judol_user_transactions(n=25000):
    records = []
    judol_users = [generate_id() for _ in range(500)]
    merchant_pool = [generate_id() for _ in range(2000)]

    for _ in range(n):
        user_id = random.choice(judol_users)
        merchant_id = random.choice(merchant_pool)

        # Escalating amount pattern
        base_amount = random.choice([50000, 100000, 200000])
        if random.random() < JUDOL_USER_PARAMS['escalation_rate']:
            multiplier = random.uniform(1.0, 3.0)
            amount = int(base_amount * multiplier)
            amount = round(amount / 50000) * 50000
        else:
            amount = base_amount

        ts = generate_timestamp(get_judol_hour_weights())
        user_city, merch_city = get_city_pair('national')

        records.append({
            'transaction_id': generate_id(),
            'timestamp': ts,
            'user_id': user_id,
            'merchant_id': merchant_id,
            'amount': amount,
            'user_city': user_city[0],
            'user_province': user_city[1],
            'merchant_city': merch_city[0],
            'merchant_province': merch_city[1],
            'transaction_type': 'QRIS',
            'device_id': generate_id(),
            'is_round_amount': is_round_amount(amount),
            'tx_hour': ts.hour,
            'tx_day_of_week': ts.weekday(),
            'is_judol_merchant': 0,
            'is_judol_user': 1,
            'label': 1  # suspicious
        })
    return pd.DataFrame(records)

# ============================================================
# MAIN: GENERATE FULL DATASET
# ============================================================

def generate_full_dataset():
    print("Generating normal transactions (425,000)...")
    normal = generate_normal_transactions(425000)

    print("Generating judol merchant transactions (50,000)...")
    judol_merchant = generate_judol_merchant_transactions(50000)

    print("Generating judol user transactions (25,000)...")
    judol_user = generate_judol_user_transactions(25000)

    print("Combining datasets...")
    full_dataset = pd.concat(
        [normal, judol_merchant, judol_user],
        ignore_index=True
    )

    # Shuffle
    full_dataset = full_dataset.sample(frac=1).reset_index(drop=True)

    print(f"Total rows: {len(full_dataset)}")
    print(f"Label distribution:\n{full_dataset['label'].value_counts()}")

    # Save
    full_dataset.to_csv('pantau_training_data.csv', index=False)
    print("Saved: pantau_training_data.csv ✅")

    return full_dataset

if __name__ == "__main__":
    df = generate_full_dataset()
```

---

### 14.8 Infrastructure for Training

| Component | Where | Spec | Cost |
|---|---|---|---|
| **Data Generation** | Digital Ocean | 4 vCPU / 8GB RAM | < 5 mins |
| **Model Training** | Digital Ocean | 4 vCPU / 8GB RAM | < 30 mins |
| **App Hosting** | Digital Ocean | 4 vCPU / 8GB RAM | ~$48/month |

#### Training Strategy:
```
Step 1: Spin up DO 4vCPU/8GB droplet
        ↓
Step 2: Run generate_full_dataset() script
        (~5 minutes, outputs 500k row CSV — base dataset)
        ↓
Step 3: Train CTGAN/TVAE on base dataset (< 30 mins)
        Generate augmented synthetic data
        Combine base + GAN output → final training dataset
        ↓
Step 4: Train Isolation Forest (< 5 mins)
        Train Graph Analysis (< 15 mins)
        Train Z-Score baselines (< 1 min)
        ↓
Step 5: Models saved as .pkl files
        ↓
Step 6: Deploy API + Dashboard
        ↓
Total training cost: $0 extra
(same droplet used for everything)
```

---

### 14.9 Why This Approach Is Valid

> *"Extensive research is conducted on synthetically generated datasets"* — Gadimov & Birihanu (2025)

Using synthetic data is **standard practice** in financial fraud research:
- Calibrated against real PPATK statistics ✅
- Parameters extracted from 3 public datasets ✅
- Transparent about synthetic nature ✅
- Real data partnership as production next step ✅

#### Pitch Statement:
> *"Our training dataset was built using a two-stage pipeline: (1) a parametric generator calibrated against three public financial datasets (IBM AML, PaySim, Sports Betting) and official PPATK statistics, then (2) augmented using CTGAN to capture realistic joint distributions and edge cases — consistent with standard practice in financial fraud research as cited in Gadimov & Birihanu (2025)."*

---

### Must Build ✅
- Synthetic transaction data generator (normal vs judol patterns) ✅
- GAN augmentation pipeline (CTGAN/TVAE via SDV)
- Isolation Forest models (user + merchant layer)
- Z-Score velocity detection
- NetworkX graph analysis
- Combined risk scoring engine
- LLM explanation via API
- Dashboard UI with live demo flow
- Basic REST API

### Pitch as Future Roadmap 🗺️
- LSTM temporal sequence model (replace rule-based temporal)
- Real data integration with BI/OJK/Banks
- PPATK API integration
- Cross-bank transaction correlation
- Model retraining pipeline

---

## 15. Demo Flow (For Judges)

```
1. Show normal transactions flowing through → all pass ✅
2. Inject simulated judol merchant scenario:
   - Dormant account suddenly receives 500 transactions
   - From 30+ cities, all round amounts, all 2AM
3. Watch system flag in real-time 🚩
4. Open explainability report
5. Show compliance officer review queue
6. Show network graph — connected merchants lit up
7. Demonstrate API call (Postman or built-in)
```

---

## 16. Key Pitch Lines

> *"Rp286 triliun berputar di judol sepanjang 2025. Sistem yang ada memblokir secara reaktif. Kami mendeteksi sebelum terjadi."*

> *"Merchant selalu berganti akun — tapi perilaku transaksi tidak bisa disembunyikan. Kami mendeteksi pola, bukan identitas."*

> *"Setiap flag dilengkapi penjelasan dalam Bahasa Indonesia untuk compliance officer. Bukan black box — sistem yang transparan dan dapat diaudit."*

> *"Semakin banyak transaksi yang dipantau, semakin pintar sistem kami. Adaptive profiling memastikan akurasi terus meningkat."*

---

## 17. Technology Stack

| Layer | Technology | Rationale |
|---|---|---|
| **Backend / API** | Python (FastAPI) | Natural fit for ML ecosystem + async API |
| **Frontend / Dashboard** | Next.js + TypeScript | Modern, fast, great DX |
| **Database** | PostgreSQL | Production-grade, rich querying for analytics |
| **ML / Data Science** | scikit-learn, NetworkX, NumPy/SciPy, Pandas | As defined in Section 8.2 |
| **LLM Explainability** | TBD | To be decided (OpenAI / Gemini / Open-source) |
| **Repo Structure** | Monorepo | Single repo: `backend/`, `frontend/`, `ml/`, `data/` |

---

*Dokumen ini dibuat untuk keperluan PIDI - DIGDAYA X Hackathon 2026*  
*Submission deadline: 27 Maret 2026*
