# Model-HybridTransformer-Bidirectional-LSTM-dan-XGBoost-dengan-Koreksi-Residual-untuk-Deret-Waktu (Studi Kasus: Harga Minyak Brent)
Deskripsi: Metode hybrid dengan integrasi model deep learning Transformer-BiLSTM untuk menangkap pola temporal kompleks (dependensi jangka panjang, multi-musiman) dengan koreksi residual berbasis XGBoost guna mengatasi anomali dan heteroskedastisitas dalam peramalan deret waktu.

## Abstrak  
Repositori ini menyajikan kerangka kerja deep learning hibrida yang canggih untuk memforecast harga minyak Brent, mengkombinasikan jaringan **Transformer–Bidirectional LSTM** dengan koreksi residual menggunakan **XGBoost**. Metodologi ini dirancang untuk menangkap dinamika temporal kompleks, volatilitas heteroskedastik, dan dampak kejadian eksogen di pasar komoditas melalui pendekatan matematis yang ketat.

---

## 1. Permasalahan  
Diberikan harga minyak \(P_t \in \mathbb{R}^+\) pada waktu \(t\) dan vektor eksogen \(\mathbf{X}_t^{(1:k)}\) (misal kejadian geopolitik, anomali), kita memodelkan:
\[
P_{t+1} \;=\; f\bigl(P_{t-\tau:t},\,\mathbf{X}_{t-\tau:t}^{(1:k)}\bigr)\;+\;\varepsilon_t
\]
dengan \(\tau\) jendela look‑back (ditentukan dengan pola rataan bulanan) dan \(\varepsilon_t\) noise heteroskedastik.

---

## 2. Fondasi Matematis

### 2.1 Rekayasa Fitur Temporal  
- **Lag** \(\tau\):  
  \(\;L^{(\tau)}_t = P_{t-\tau}\).  
- **Moving Average (MA)** lebar \(w\):  
  \[
    \mathrm{MA}_t^{(w)}
    = \frac{1}{w}\sum_{i=0}^{w-1}P_{t-i}.
  \]
- **Rolling Std (RS)** lebar \(w\):  
  \[
    \mathrm{RS}_t^{(w)}
    = \sqrt{\frac{1}{w}\sum_{i=0}^{w-1}\bigl(P_{t-i} - \mathrm{MA}_t^{(w)}\bigr)^2}.
  \]
- **Fourier Terms** periode \(p\):  
  \[
    \sin^{(p)}_t = \sin\!\bigl(2\pi t/p\bigr),\quad
    \cos^{(p)}_t = \cos\!\bigl(2\pi t/p\bigr).
  \]
- **Skewness & Kurtosis** pada window \(w\) lokal:
  \[
    \gamma_t = \frac{1}{w}\sum\Bigl(\tfrac{P-\mu}{\sigma}\Bigr)^3,\quad
    \kappa_t = \frac{1}{w}\sum\Bigl(\tfrac{P-\mu}{\sigma}\Bigr)^4.
  \]

### 2.2 Arsitektur Hibrida  
\[
\hat y_t
= \underbrace{f_\theta\bigl(\text{Transformer–BiLSTM}\bigr)}_{\hat y_t^{\mathrm{base}}}
\;+\;
\underbrace{g_\phi\bigl(\mathrm{XGBoost}\,\bigl|\;r_t\bigr)}_{\hat r_t}
\]
dengan \(r_t = y_t - \hat y_t^{\mathrm{base}}\).

#### 2.2.1 Transformer–BiLSTM  
1. **Positional Encoding** citeVaswani2017:  
   \[
   \begin{aligned}
     \mathrm{PE}_{t,2i}   &= \sin\!\bigl(\tfrac{t}{10000^{2i/D}}\bigr),\\
     \mathrm{PE}_{t,2i+1} &= \cos\!\bigl(\tfrac{t}{10000^{2i/D}}\bigr).
   \end{aligned}
   \]
2. **Multi‑Head Self‑Attention**:
   \[
     \mathrm{Attention}(Q,K,V)
     = \mathrm{softmax}\!\bigl(QK^\top/\sqrt{d_k}\bigr)\,V.
   \]
3. **Feed‑Forward Network**:
   \[
     \mathrm{FFN}(x)
     = W_2\bigl(\phi(W_1x + b_1)\bigr) + b_2.
   \]
4. **Bidirectional LSTM** citeHochreiter1997:
   \[
   \begin{aligned}
     f_t &= \sigma(W_fx_t+U_fh_{t-1}+b_f),\\
     i_t &= \sigma(W_ix_t+U_ih_{t-1}+b_i),\\
     \tilde c_t &= \tanh(W_cx_t+U_ch_{t-1}+b_c),\\
     c_t &= f_t\odot c_{t-1}+i_t\odot\tilde c_t,\\
     o_t &= \sigma(W_ox_t+U_oh_{t-1}+b_o),\\
     h_t &= o_t\odot\tanh(c_t).
   \end{aligned}
   \]
5. **Output Base**:
   \[
     \hat y_t^{\mathrm{base}}
     = W_o\;\mathrm{MeanPool}(H_L) + b_o.
   \]

#### 2.2.2 XGBoost Residual Correction citeChen2016  
Boosting round \(m\):
\[
\begin{aligned}
  g_i^{(m)} &= -\frac{\partial \ell(y_i,\,F^{(m-1)}(x_i))}{\partial F(x_i)},\\
  F^{(m)}(x) &= F^{(m-1)}(x) + \eta\,\mathrm{Tree}_m(x).
\end{aligned}
\]
Objektif total:
\[
  \mathcal{L}(\phi) = \sum_i \ell\bigl(r_i,\,g_\phi(Z_i)\bigr)
  + \Omega(\phi),\quad
  \Omega(\phi)=\alpha\|w\|_1+\tfrac12\lambda\|w\|_2^2.
\]

### 2.3 Back‑Propagation & Optimasi  
- **Loss**: Huber (\(\delta=1\)) citeHuber1964  
  \[
  L_\delta =
  \begin{cases}
    \tfrac12(y-\hat y)^2, & |y-\hat y|\le\delta,\\
    \delta|y-\hat y|-\tfrac12\delta^2, & \text{lainnya}.
  \end{cases}
  \]
- **Gradien Transformer–BiLSTM** dihitung via BPTT (chain‑rule).
- **Optimasi**: Adam, EarlyStopping, ReduceLROnPlateau.
- **Residual** dipelajari terpisah oleh XGBoost (gradient boosting).

---

## 3. Proses Implementasi  
1. **ETL & EDA**  
2. **Feature Engineering** (Lag, MA, RS, sin/cos, skew, kurt)  
3. **Train/Test Split** (80%/20%, chronological)  
4. **Preprocessor**: RobustScaler (Price), StandardScaler (exog)  
5. **Optuna Tuning** untuk Transformer–BiLSTM dan XGBoost citeAkiba2019  
6. **Cross‑Validation** (TimeSeriesSplit, 5 folds)  
7. **Final Training** pada data train penuh  
8. **Eval** pada data test 

---

## 4. Hasil & Evaluasi  
| Metrik               | Nilai        |
|---------------------:|-------------:|
| **MAPE**             | 1.82 %       |
| **RMSE**             | 1.87         |
| **MAE**              | 1.43         |

---
