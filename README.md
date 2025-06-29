# Model Hybrid Transformer–BiLSTM + XGBoost  
**Studi Kasus:** Peramalan Harga Minyak Brent

**Deskripsi**  
Metode hybrid ini menggabungkan:

- **Transformer–Bidirectional LSTM** untuk menangkap pola sekuensial jangka panjang, multi‑musiman, dan efek kejadian eksogen.  
- **XGBoost** untuk mengoreksi residual (galat) sehingga model mampu menyesuaikan anomali dan perubahan volatilitas.

---

## 1. Abstrak  
Repositori ini memperkenalkan arsitektur hibrida yang menggabungkan kekuatan self‑attention dan LSTM ganda arah pada Transformer–BiLSTM dengan kemampuan gradient boosting pada XGBoost. Tujuannya adalah meningkatkan akurasi peramalan harga Brent dengan menurunkan error (MAPE 1.82 %, RMSE 1.87, MAE 1.43) dan menjaga kestabilan saat menghadapi outlier atau regime shifts.

---

## 2. Permasalahan  
Kita memiliki:
- **Target**: harga minyak \(P_t\) pada waktu ke‑\(t\).  
- **Fitur eksogen**: vektor dummy dan fitur periode (Fourier, moving average, dll.) pada jendela sebelum waktu \(t\).

---

## 3. Fitur dan Pra‑proses

1. **Lag Features**  
   - Ambil nilai harga pada \(t-\tau\) sebagai fitur tunggal, dengan \(\tau\) misalnya 1, 7, atau 22 hari bisnis.  
   - Contoh: jika \(\tau=22\), maka fitur lag adalah harga 22 hari sebelumnya.

2. **Moving Average (MA)**  
   - Rata‑rata harga pada window berukuran \(w\).  
   - Misal MA(2) = rata‑rata harga 2 hari terakhir.

3. **Rolling Standard Deviation (RS)**  
   - Simpangan baku harga dalam window \(w\).  
   - Mengukur volatilitas lokal.

4. **Fourier Terms**  
   - Sine dan cosine dengan periode \(p\) (misal 21 hari).  
   - Menangkap pola musiman.

5. **Skewness & Kurtosis**  
   - Skewness: ukuran simetri sebaran residual.  
   - Kurtosis: ukuran ketajaman ekor sebaran.

6. **Scaling**  
   - Harga distandarkan dengan RobustScaler.  
   - Fitur eksogen dengan StandardScaler.

7. **Windowing**  
   - Bentuk input sekuens: untuk setiap titik, kumpulkan data harga dan eksogen selama \( \tau \) hari sebelumnya.

---

## 4. Arsitektur Model

### 4.1 Transformer–BiLSTM (Base)  
- **Positional Encoding**  
  Menambahkan informasi posisi urutan pada setiap embedding titik waktu.  
- **Stacked Blocks**  
  - **Multi‑Head Self‑Attention**: Memungkinkan model melihat semua posisi dalam kunci window secara paralel dan memberikan bobot perhatian.  
  - **Bidirectional LSTM**: Memproses urutan dari dua arah (maju dan mundur) untuk menangkap konteks masa lalu dan masa depan lokal.  
  - **Layer Normalization** dan **Residual Connection** setelah setiap modul.  
  - **Feed‑Forward**: Lapisan padat di akhir setiap blok untuk transformasi non‑linier tambahan.  
- **Global Average Pooling**  
  Merangkum semua hidden state menjadi vektor tunggal untuk prediksi output.

### 4.2 XGBoost Residual Correction  
1. **Hitung residual**:  
   \[
     r_t = y_t - \hat y_t^{\text{base}}.
   \]
2. **Bentuk fitur residual**  
   - Harga terakhir, eksogen terakhir, moving average & rolling std residual, skewness & kurtosis residual, prediksi base.  
3. **Train XGBoost** pada data residual untuk mempelajari pola kesalahan sisa.  
4. **Hybrid Output**  
   \[
     \hat y_t^{\text{hybrid}} = \hat y_t^{\text{base}} + \hat r_t^{\text{XGB}}.
   \]

---

## 5. Proses Training & Tuning

1. **Train/Test Split**: 80 % data awal untuk train, 20 % akhir untuk test, tanpa acak (chronological).  
2. **Optuna Tuning**:  
   - Transformer–BiLSTM: hyperparameter seperti jumlah head, unit LSTM, dropout, learning rate.  
   - XGBoost: jumlah pohon, kedalaman maksimum, learning rate, subsample, colsample.  
3. **Callbacks**:  
   - EarlyStopping untuk menghentikan ketika tidak ada perbaikan loss validasi.  
   - ReduceLROnPlateau untuk menurunkan learning rate saat stagnasi.  
4. **Train Final**:  
   - Latih base model pada keseluruhan set train dengan hyperparameter terbaik.  
   - Ekstrak residual lalu latih XGBoost pada residual penuh.

---

## 6. Evaluasi & Hasil

- **MAPE** (Mean Absolute Percentage Error): 1.82
- **RMSE** (Root Mean Squared Error): 1.87  
- **MAE** (Mean Absolute Error): 1.43  

---

## 7. Kesimpulan  
Pendekatan hibrida ini berhasil:
- Menangkap pola jangka panjang & multi‑seasonal  
- Mengoreksi residual anomali dan volatilitas yang berubah  
- Memberikan prediksi stabil dengan error rendah dan interval kepercayaan sempit  

--- 
