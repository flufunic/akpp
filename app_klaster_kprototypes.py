import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from kmodes.kprototypes import KPrototypes # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import davies_bouldin_score
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import warnings
import io
import os

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Aplikasi Klasterisasi (K-Prototypes)", layout="wide")

# -------------------------
# normalisasi nama kolom & deteksi otomatis
# -------------------------
def norm_name(c):
    return str(c).strip().lower().replace(" ", "").replace("_", "") if c is not None else ""

def detect_columns(df):
    cols = df.columns.tolist()
    lc = [norm_name(c) for c in cols]

    candidates = {
        "usia": ["usia", "umur", "age"],
        "jenis_kelamin": ["jeniskelamin", "jk", "gender", "sex", "jeniskelamin"],
        "tempat_lahir": ["tempatlahir", "tempat_lahir", "birthplace", "birth_place"],
        "status": ["status", "status_pengambilan", "statusambil", "status_pengambilan"]
    }

    mapping = {}
    for key, tries in candidates.items():
        found = None
        for t in tries:
            if t in lc:
                found = cols[lc.index(t)]
                break
        mapping[key] = found
    return mapping

def humanize_mapping(mapping):
    lines = []
    for dst, src in mapping.items():
        if src:
            lines.append(f"Kolom `{src}` digunakan sebagai *{dst}*")
        else:
            lines.append(f"Kolom untuk *{dst}* TIDAK DITEMUKAN")
    return lines

def preprocess_full(df, mapping, perform_reduction=True):
    report = {}
    df_work = df.copy()
    report['original_rows'] = int(len(df_work))

    # ============================
    # AMBIL KOLOM DARI MAPPING
    # ============================
    usia_col = mapping['usia']
    jk_col   = mapping['jenis_kelamin']
    tp_col   = mapping['tempat_lahir']
    st_col   = mapping['status']

    cols_needed = [usia_col, jk_col, tp_col, st_col]

    # ============================
    # MISSING VALUES
    # ============================

    # Ringkas missing sebelum
    missing_before = df_work[cols_needed].isna().sum()
    report['missing_before'] = missing_before.to_dict()

    # Pastikan usia numerik
    df_work[usia_col] = pd.to_numeric(df_work[usia_col], errors='coerce')

    # Simpan jumlah baris sebelum missing
    rows_before_missing = len(df_work)
    report['rows_before_missing'] = rows_before_missing

    # Hapus semua baris yang memiliki missing di kolom penting
    df_work = df_work.dropna(subset=cols_needed)

    # Simpan hasil setelah missing
    report['rows_after_missing'] = len(df_work)
    report['dropped_missing_total'] = rows_before_missing - len(df_work)

    # Ringkas missing setelah
    missing_after = df_work[cols_needed].isna().sum()
    report['missing_after'] = missing_after.to_dict()



    # ============================
    # DATA TIDAK VALID
    # ============================

    invalid = {}

    # Jumlah data sebelum pembersihan invalid
    rows_before_invalid = len(df_work)
    report['rows_before_invalid'] = rows_before_invalid

    # ----------------------------
    # Validasi USIA (0–120)
    # ----------------------------
    invalid_usia = df_work[(df_work[usia_col] < 0) | (df_work[usia_col] > 120)]
    invalid['usia'] = len(invalid_usia)

    df_work = df_work[(df_work[usia_col] >= 0) & (df_work[usia_col] <= 120)]

    # ----------------------------
    # Normalisasi & validasi JENIS KELAMIN
    # ----------------------------
    def norm_jk(x):
        s = str(x).lower().strip()
        if s in ['l','laki','laki-laki','pria','male','m']:
            return 'L'
        if s in ['p','perempuan','wanita','female','f']:
            return 'P'
        return 'Unknown'

    df_work[jk_col] = df_work[jk_col].apply(norm_jk)

    invalid_jk = df_work[df_work[jk_col] == 'Unknown']
    invalid['jenis_kelamin'] = len(invalid_jk)

    df_work = df_work[df_work[jk_col].isin(['L', 'P'])]

    # ----------------------------
    # Validasi TEMPAT LAHIR
    # ----------------------------
    df_work[tp_col] = df_work[tp_col].astype(str).str.strip()

    invalid_tp_numeric = df_work[df_work[tp_col].str.match(r'^\d+$')]
    invalid_tp_anynum  = df_work[df_work[tp_col].str.contains(r'\d')]
    invalid_tp_short   = df_work[df_work[tp_col].str.len() < 2]
    invalid_tp_symbol  = df_work[df_work[tp_col].str.match(r'^[^a-zA-Z]+$')]

    invalid_tp_all = pd.concat([
        invalid_tp_numeric,
        invalid_tp_anynum,
        invalid_tp_short,
        invalid_tp_symbol
    ]).drop_duplicates()

    invalid['tempat_lahir'] = {
        'count': int(len(invalid_tp_all)),
        'examples': invalid_tp_all.to_dict(orient='records')  # Semua baris
    }

    df_work = df_work[~df_work.index.isin(invalid_tp_all.index)]

    # ----------------------------
    # Validasi STATUS 
    # ----------------------------
    df_work[st_col] = df_work[st_col].astype(str).str.strip()
    invalid_st = df_work[df_work[st_col].str.len() == 0]
    invalid['status'] = len(invalid_st)

    df_work = df_work[df_work[st_col].str.len() > 0]

    # ----------------------------
    # Ringkasan hasil invalid
    # ----------------------------
    report['invalid_details'] = invalid
    report['rows_after_invalid'] = len(df_work)
    report['dropped_invalid_total'] = rows_before_invalid - len(df_work)


    # ============================
    # DUPLIKASI ASLI
    # ============================
    dup_count = df_work.duplicated().sum()
    report['duplicates_found'] = int(dup_count)

    before_dup = len(df_work)
    df_work = df_work.drop_duplicates()
    report['dropped_duplicates'] = int(before_dup - len(df_work))
    report['after_dup_rows'] = int(len(df_work))

    # ============================
    # TRANSFORMASI
    # ============================

    df_work[usia_col + "_orig"] = df_work[usia_col].astype(float).copy()
    # Z-score usia
    scaler = StandardScaler()
    df_work[usia_col] = scaler.fit_transform(df_work[[usia_col]])

    # Label Encoding
    le_jk = LabelEncoder()
    le_tp = LabelEncoder()
    le_st = LabelEncoder()

    df_work[jk_col + "_label"] = le_jk.fit_transform(df_work[jk_col])
    df_work[tp_col + "_label"] = le_tp.fit_transform(df_work[tp_col])
    df_work[st_col + "_label"] = le_st.fit_transform(df_work[st_col])

    report['after_transform_rows'] = len(df_work)

    # ============================
    # REDUKSI
    # ============================
    rep_cols = [usia_col, jk_col, tp_col, st_col]
    df_reduced = df_work.drop_duplicates(subset=rep_cols).reset_index(drop=True)

    report['reduced_rows'] = len(df_reduced)
    report['final_rows'] = len(df_work)

    # ============================
    # STANDARDISASI NAMA KOLOM
    # ============================
    rename_map = {
        usia_col: 'usia',
        usia_col + "_orig": 'usia_tahun', 
        jk_col: 'jenis_kelamin',
        tp_col: 'tempat_lahir',
        st_col: 'status'
    }

    df_work = df_work.rename(columns=rename_map)
    df_reduced = df_reduced.rename(columns=rename_map)

    df_work = df_work.rename(columns={
        jk_col + "_label": "jenis_kelamin_label",
        tp_col + "_label": "tempat_lahir_label",
        st_col + "_label": "status_label"
    })

    df_reduced = df_reduced.rename(columns={
        jk_col + "_label": "jenis_kelamin_label",
        tp_col + "_label": "tempat_lahir_label",
        st_col + "_label": "status_label"
    })

    # ============================
    # AMBIL WILAYAH UNIK (REFERENSI)
    # ============================
    wilayah_unik = (
        df_work["tempat_lahir"]
        .dropna()
        .str.title()
        .unique()
    )

    report["jumlah_wilayah_unik"] = len(wilayah_unik)
    report["contoh_wilayah"] = wilayah_unik[:10].tolist()

    # (opsional) wilayah dominan
    wilayah_dominan = (
        df_work["tempat_lahir"]
        .value_counts()
        .head(20)
        .index
        .tolist()
    )

    report["wilayah_dominan"] = wilayah_dominan


    return df_work.reset_index(drop=True), df_reduced.reset_index(drop=True), report

# --------------------------
# Fungsi K-Prototypes & DBI
# --------------------------
def run_kprototypes_on_array(df_for_cluster, k, random_state=42):
    num_cols = ['usia']  
    cat_cols = ['jenis_kelamin_label', 'tempat_lahir_label', 'status_label']
    arr_num = df_for_cluster[num_cols].to_numpy()
    arr_cat = df_for_cluster[cat_cols].to_numpy(dtype=int)
    arr = np.hstack([arr_num, arr_cat]).astype(object)
    cat_idx = list(range(1, 1 + len(cat_cols)))
    kp = KPrototypes(n_clusters=k, init='Huang', n_init=5, verbose=0, random_state=random_state)
    labels = kp.fit_predict(arr, categorical=cat_idx)
    return kp, labels, arr, cat_idx

def compute_dbi_mixed(df_for_cluster, labels, gamma=1):
    num_cols = ['usia']
    cat_cols = ['jenis_kelamin_label', 'tempat_lahir_label', 'status_label']
    arr_num = df_for_cluster[num_cols].to_numpy(dtype=float)
    arr_cat = df_for_cluster[cat_cols].to_numpy(dtype=int)
    unique_labels = np.unique(labels)
    centroids_num = np.array([arr_num[labels==l].mean(axis=0) for l in unique_labels])
    centroids_cat = np.array([np.array([np.bincount(arr_cat[labels==l, idx]).argmax() 
                                        for idx in range(arr_cat.shape[1])])
                            for l in unique_labels], dtype=int)
    S = compute_dispersion(arr_num, arr_cat, labels, centroids_num, centroids_cat, gamma)
    M = compute_Mij_with_sqrt(centroids_num, centroids_cat, gamma)
    _, _, DBI = compute_Rij_and_DBI(S, M)
    return DBI

# ==============================
# DBI campuran (numerik + kategorik)
# ==============================
def compute_dispersion(arr_num, arr_cat, labels, centroids_num, centroids_cat, gamma=1):
    import numpy as np
    k = centroids_num.shape[0]
    S = []
    for i in range(k):
        mask = labels == i
        cluster_num = arr_num[mask]
        cluster_cat = arr_cat[mask]
        dist_list = []
        for j in range(cluster_num.shape[0]):
            dist = np.sum((cluster_num[j] - centroids_num[i])**2)
            dist += gamma * np.sum(cluster_cat[j] != centroids_cat[i])
            dist_list.append(np.sqrt(dist))
        S.append(np.mean(dist_list))
    return np.array(S)

def compute_Mij_with_sqrt(centroids_num, centroids_cat, gamma=1):
    import numpy as np
    k = centroids_num.shape[0]
    M = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                num_dist = np.sum((centroids_num[i] - centroids_num[j])**2)
                cat_dist = np.sum(centroids_cat[i] != centroids_cat[j])
                M[i, j] = np.sqrt(num_dist + gamma * cat_dist)
    return M

def compute_Rij_and_DBI(S, M):
    import numpy as np
    k = len(S)
    R = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j and M[i,j] != 0:
                R[i,j] = (S[i]+S[j])/M[i,j]
    R_max = np.max(R, axis=1)
    DBI = np.mean(R_max)
    return R, R_max, DBI

def compute_dbi_mixed(df_for_cluster, labels, gamma=1):
    import numpy as np
    num_cols = ['usia']
    cat_cols = ['jenis_kelamin_label', 'tempat_lahir_label', 'status_label']

    arr_num = df_for_cluster[num_cols].to_numpy(dtype=float)
    arr_cat = df_for_cluster[cat_cols].to_numpy(dtype=int)

    unique_labels = np.unique(labels)
    centroids_num = np.array([arr_num[labels==l].mean(axis=0) for l in unique_labels])
    centroids_cat = np.array([np.array([np.bincount(arr_cat[labels==l, idx]).argmax() 
                                        for idx in range(arr_cat.shape[1])])
                            for l in unique_labels], dtype=int)

    S = compute_dispersion(arr_num, arr_cat, labels, centroids_num, centroids_cat, gamma)
    M = compute_Mij_with_sqrt(centroids_num, centroids_cat, gamma)
    _, _, DBI = compute_Rij_and_DBI(S, M)
    return DBI

def make_merge_key(df):
    return (
        df['usia'].round(6).astype(str) + '||' +
        df['jenis_kelamin'].astype(str) + '||' +
        df['tempat_lahir'].astype(str) + '||' +
        df['status'].astype(str)
    )

# -------------------------
# Save outputs helpers
# -------------------------
def save_excel_with_clusters(original_df, df_representative, labels, now, best_k):
    import io
    rep = df_representative.copy()
    rep['cluster'] = labels.astype(int)

    # merge key unik
    rep['__merge_key'] = rep['usia'].astype(str) + '_' + \
                         rep['jenis_kelamin'].astype(str) + '_' + \
                         rep['tempat_lahir'].astype(str) + '_' + \
                         rep['status'].astype(str)

    orig = original_df.copy()
    orig['__merge_key'] = orig['usia'].astype(str) + '_' + \
                          orig['jenis_kelamin'].astype(str) + '_' + \
                          orig['tempat_lahir'].astype(str) + '_' + \
                          orig['status'].astype(str)

    merged = orig.merge(rep[['__merge_key','cluster']], on='__merge_key', how='left')
    merged['cluster'] = merged['cluster'].fillna(-1).astype(int)
    merged.drop(columns='__merge_key', inplace=True)

    # hapus kolom label yang tidak ingin ditampilkan
    cols_to_drop = ['usia', 'jenis_kelamin_label', 'tempat_lahir_label', 'status_label']
    merged = merged.drop(columns=[c for c in cols_to_drop if c in merged.columns])

    # simpan ke buffer
    buffer = io.BytesIO()
    filename = f"hasil_klaster_k{best_k}_{now}.xlsx"
    merged.to_excel(buffer, index=False, engine='openpyxl')
    buffer.seek(0)
    return filename, buffer


def buat_rekomendasi_pelayanan(data_k, wilayah_dominan, df_wilayah_ref):
    rekomendasi = []

    # =====================================================
    # 1. PROFIL USIA KLASTER
    # =====================================================
    prop_anak = data_k['usia_tahun'].between(0, 16).mean()
    prop_produktif = data_k['usia_tahun'].between(17, 45).mean()
    prop_dewasa_akhir = data_k['usia_tahun'].between(46, 65).mean()
    prop_lansia = (data_k['usia_tahun'] > 65).mean()

    prop_diambil = (
        data_k['status']
        .value_counts(normalize=True)
        .get("diambil", 0)
    )

    # =====================================================
    # 2. ARAH LAYANAN UTAMA KLASTER (KOMBINASI)
    # =====================================================
    if prop_lansia >= 0.20:
        arah_layanan = (
            "Layanan Prioritas Lansia dengan pendampingan administratif "
            "dan optimalisasi pengiriman paspor melalui pos"
        )
    elif prop_produktif >= 0.50 and prop_diambil >= 0.90:
        arah_layanan = (
            "Layanan Reguler Berbasis Digital dengan penguatan sistem antrean "
            "dan efisiensi waktu layanan"
        )
    else:
        arah_layanan = (
            "Pelayanan Reguler dengan pendampingan administratif "
            "dan penguatan sosialisasi tahapan layanan"
        )

    rekomendasi.append(
        f"Arah layanan utama klaster ini adalah {arah_layanan}, "
        f"yang ditetapkan berdasarkan distribusi usia pemohon, "
        f"pola penyelesaian layanan, serta kebutuhan akses layanan."
    )

    # =====================================================
    # 3. REKOMENDASI BERDASARKAN KELOMPOK USIA
    # =====================================================

    # Anak & Remaja Awal
    if prop_anak > 0:
        rekomendasi.append(
            "Keberadaan pemohon anak dan remaja awal menunjukkan perlunya "
            "pendekatan pelayanan berbasis keluarga, dengan penyederhanaan "
            "alur administrasi serta penguatan verifikasi dokumen orang tua atau wali."
        )

    # Usia Produktif (17–45)
    if prop_produktif >= 0.30:
        rekomendasi.append(
            "Dominasi pemohon usia produktif mengindikasikan kebutuhan terhadap "
            "pelayanan yang cepat, fleksibel, dan terintegrasi secara digital, "
            "terutama pada tahap pendaftaran dan pengaturan jadwal layanan."
        )

    # Dewasa Akhir (46–65)
    if prop_dewasa_akhir >= 0.25:
        rekomendasi.append(
            "Proporsi pemohon usia dewasa akhir memerlukan penyampaian informasi "
            "yang lebih jelas dan terstruktur, khususnya terkait persyaratan "
            "dan tahapan administrasi pelayanan paspor."
        )

    # Lansia (>65)
    if prop_lansia > 0:
        rekomendasi.append(
            "Keberadaan pemohon lansia menuntut penyediaan layanan prioritas "
            "melalui pendampingan langsung, penyederhanaan prosedur, layanan jemput bola, "
            "serta pemanfaatan pengiriman paspor melalui pos untuk "
            "mengurangi kebutuhan kunjungan ulang."
        )

    # =====================================================
    # 4. JENIS KELAMIN DOMINAN
    # =====================================================
    jk_dominan = data_k['jenis_kelamin'].mode()[0]
    jk_prop = (data_k['jenis_kelamin'] == jk_dominan).mean()

    if jk_dominan.lower().startswith('p') and jk_prop >= 0.55:
        rekomendasi.append(
            "Dominasi pemohon perempuan menjadi pertimbangan dalam "
            "penyediaan lingkungan layanan yang aman, nyaman, dan ramah perempuan."
        )

    # =====================================================
    # 5. POLA PENYELESAIAN LAYANAN
    # =====================================================
    if prop_diambil < 0.90:
        rekomendasi.append(
            "Masih terdapat pemohon yang belum menyelesaikan proses pengambilan paspor, "
            "sehingga diperlukan penguatan mekanisme pengingat serta pendampingan layanan."
        )

    # =====================================================
    # 6. AKSES LAYANAN BERDASARKAN WILAYAH DOMINAN
    # =====================================================
    from collections import defaultdict
    kelompok = defaultdict(list)

    for w in wilayah_dominan:
        row = df_wilayah_ref[df_wilayah_ref["wilayah"] == w]

        if row.empty:
            kelompok["data_tidak_tersedia"].append(w)
            continue

        ada_mpp = row.iloc[0]["ada_mpp"]
        jarak = row.iloc[0]["jarak_km"]

        if jarak == 0:
            kelompok["lokasi_kanim"].append(w)
        elif ada_mpp and jarak <= 50:
            kelompok["mpp_dekat"].append(w)
        elif ada_mpp:
            kelompok["mpp_jauh"].append(w)
        else:
            kelompok["tanpa_mpp"].append(w)

    def gabung_wilayah(w):
        return ", ".join(w[:-1]) + " dan " + w[-1] if len(w) > 1 else w[0]

    if kelompok["lokasi_kanim"]:
        rekomendasi.append(
            f"Wilayah {gabung_wilayah(kelompok['lokasi_kanim'])} merupakan lokasi "
            f"Kantor Imigrasi Cilacap, sehingga pelayanan dapat dipusatkan langsung di kantor."
        )

    if kelompok["mpp_dekat"]:
        rekomendasi.append(
            f"Wilayah {gabung_wilayah(kelompok['mpp_dekat'])} memiliki MPP dengan jarak dekat, "
            f"sehingga pemohon diarahkan memanfaatkan layanan keimigrasian di MPP."
        )

    if kelompok["mpp_jauh"]:
        rekomendasi.append(
            f"Wilayah {gabung_wilayah(kelompok['mpp_jauh'])} memiliki MPP namun berjarak jauh, "
            f"sehingga pengiriman paspor melalui pos menjadi alternatif layanan."
        )

    if kelompok["tanpa_mpp"]:
        rekomendasi.append(
            f"Wilayah {gabung_wilayah(kelompok['tanpa_mpp'])} belum memiliki MPP, "
            f"sehingga pelayanan diarahkan pada pengiriman paspor melalui pos "
            f"atau layanan jemput bola."
        )

    return rekomendasi


def rekomendasi_individu(row, df_wilayah_ref):
    alasan = []
    layanan_dasar = None
    penyesuaian_layanan = None

    # ======================
    # USIA (LAYANAN DASAR)
    # ======================
    if row['usia_tahun'] < 17:
        layanan_dasar = "Pelayanan Berbasis Keluarga"
        alasan.append("Pemohon berusia di bawah 17 tahun")

    elif row['usia_tahun'] <= 45:
        layanan_dasar = "Layanan Reguler Berbasis Digital"
        alasan.append("Pemohon berada pada usia produktif")

    elif row['usia_tahun'] <= 65:
        layanan_dasar = "Layanan Reguler dengan Pendampingan Administrasi"
        alasan.append("Pemohon usia dewasa akhir")

    else:
        layanan_dasar = "Layanan Prioritas Lansia"
        alasan.append("Pemohon berusia lanjut")

    # ======================
    # JENIS KELAMIN (PENDEKATAN)
    # ======================
    if str(row['jenis_kelamin']).lower().startswith('p'):
        alasan.append(
            "Pendekatan pelayanan memperhatikan aspek kenyamanan dan keamanan bagi pemohon perempuan"
        )

    # ======================
    # STATUS PENGAMBILAN
    # ======================
    if str(row['status']).lower() != "diambil":
        alasan.append("Paspor belum diambil sehingga diperlukan tindak lanjut pelayanan")

    # ======================
    # WILAYAH & AKSES (PENYESUAIAN LAYANAN)
    # ======================
    ref = df_wilayah_ref[df_wilayah_ref['wilayah'] == row['tempat_lahir']]

    if not ref.empty:
        ada_mpp = ref.iloc[0]['ada_mpp']
        jarak = ref.iloc[0]['jarak_km']

        if jarak == 0:
            penyesuaian_layanan = "Pelayanan Langsung di Kantor Imigrasi Cilacap"
            alasan.append("Domisili berada di lokasi Kantor Imigrasi Cilacap")

        elif ada_mpp and jarak <= 50:
            penyesuaian_layanan = "Pelayanan melalui Mal Pelayanan Publik (MPP)"
            alasan.append("Wilayah memiliki MPP dengan jarak dekat")

        elif ada_mpp:
            penyesuaian_layanan = (
                "Pelayanan melalui MPP dan Pengiriman Paspor melalui Pos"
            )
            alasan.append("Wilayah memiliki MPP namun berjarak cukup jauh")

        else:
            penyesuaian_layanan = (
                "Pengiriman Paspor melalui Pos atau Layanan Jemput Bola"
            )
            alasan.append("Wilayah belum memiliki MPP dan berjarak jauh")

    layanan_utama = penyesuaian_layanan or layanan_dasar

    return (
        f"Layanan utama: {layanan_utama}. "
        f"Pertimbangan: {', '.join(alasan)}."
    )


   

from io import BytesIO

def save_visuals_pdf(df_final, now, best_k):
    buffer = BytesIO()

    # =========================
    # VARIABEL RINGKASAN (WAJIB)
    # =========================
    jumlah_klaster = df_final['cluster'].nunique()
    total_pemohon = len(df_final)
    klaster_terbesar = df_final['cluster'].value_counts().idxmax()

    with PdfPages(buffer) as pdf:

        # =========================
        # HALAMAN 1 – RINGKASAN & USIA
        # =========================
        fig = plt.figure(figsize=(8.27, 11.69))
        gs = fig.add_gridspec(3, 2, height_ratios=[0.7, 1.2, 1.2], hspace=0.5)

        # --- Ringkasan
        ax0 = fig.add_subplot(gs[0, :])
        ax0.axis('off')

        ax0.text(
            0.5, 0.80,
            "RINGKASAN HASIL KLASTERISASI PEMOHON PASPOR",
            ha='center',
            fontsize=16,
            fontweight='bold'
        )

        ax0.text(
            0.1, 0.10,
            f"Jumlah Klaster : {jumlah_klaster}\n"
            f"Total Pemohon : {total_pemohon:,}\n"
            f"Klaster Terbesar : Klaster {klaster_terbesar}",
            fontsize=11,
            linespacing=1.8
        )

        # --- Boxplot Usia
        ax1 = fig.add_subplot(gs[1, 0])
        df_final.boxplot(column='usia_tahun', by='cluster', ax=ax1, grid=False)
        ax1.set_title("Distribusi Usia per Klaster")
        ax1.set_xlabel("Klaster")
        ax1.set_ylabel("Usia")
        plt.suptitle("")

        # --- Histogram Usia
        ax2 = fig.add_subplot(gs[1, 1])
        for c in sorted(df_final['cluster'].unique()):
            ax2.hist(
                df_final[df_final['cluster'] == c]['usia_tahun'],
                bins=20, alpha=0.6, label=f"Klaster {c}"
            )
        ax2.set_title("Sebaran Usia Pemohon")
        ax2.legend(frameon=False, fontsize=9)

        # --- Scatter Usia
        ax3 = fig.add_subplot(gs[2, :])
        y_jitter = np.random.normal(0, 0.08, len(df_final))
        ax3.scatter(
            df_final['usia_tahun'],
            y_jitter,
            c=df_final['cluster'],
            cmap='viridis',
            alpha=0.6,
            s=20
        )
        ax3.set_title("Scatter Usia Berdasarkan Klaster")
        ax3.set_xlabel("Usia")
        ax3.set_yticks([])

        pdf.savefig(fig)
        plt.close(fig)

        # =========================
        # HALAMAN 2 – LAYANAN & WILAYAH
        # =========================
        fig = plt.figure(figsize=(8.27, 11.69))
        gs = fig.add_gridspec(3, 2, hspace=0.5)

        # --- Jenis Kelamin
        ax1 = fig.add_subplot(gs[0, 0])
        (
            df_final
            .groupby(['cluster', 'jenis_kelamin'])
            .size()
            .unstack(fill_value=0)
            .plot(kind='bar', ax=ax1)
        )
        ax1.set_title("Komposisi Jenis Kelamin")
        ax1.legend(frameon=False, fontsize=9)

        # --- Status Paspor
        ax2 = fig.add_subplot(gs[0, 1])
        (
            df_final
            .groupby(['cluster', 'status'])
            .size()
            .unstack(fill_value=0)
            .plot(kind='bar', ax=ax2)
        )
        ax2.set_title("Status Pengambilan Paspor")
        ax2.legend(frameon=False, fontsize=9)

        # --- TOP 5 WILAYAH ASAL PER KLASTER
        row = 1
        for cluster_id in sorted(df_final['cluster'].unique()):
            ax = fig.add_subplot(gs[row, cluster_id % 2])

            top5 = (
                df_final[df_final['cluster'] == cluster_id]
                ['tempat_lahir']
                .value_counts()
                .head(5)
            )

            top5.plot(kind='bar', ax=ax)
            ax.set_title(f"Top 5 Wilayah Asal – Klaster {cluster_id}")
            ax.set_ylabel("Jumlah Pemohon")
            ax.set_xlabel("Wilayah")
            ax.tick_params(axis='x', rotation=30)

            if cluster_id % 2 == 1:
                row += 1

        pdf.savefig(fig)
        plt.close(fig)


    buffer.seek(0)
    return buffer.getvalue()

# -------------------------
# Session state init (keys used)
# -------------------------
for key, default in [
    ('raw_df', None), ('mapping', None), ('clean_df', None), ('reduced_df', None),
    ('preproc_report', None), ('df_final', None), ('kproto_models', {}), ('dbi_scores', {}),
    ('labels_for_k', {}), ('best_k', None), ('now', datetime.now().strftime("%Y%m%d_%H%M%S"))
]:
    if key not in st.session_state:
        st.session_state[key] = default

# -------------------------
# Sidebar main menu
# -------------------------
import base64

with st.sidebar:
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        background-color: #0A2540;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    .sidebar-logo {
        width: 120px;
        margin-left: auto;
        margin-right: auto;
        display: block;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.4);
    }

    /* TOMBOL RESET */
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
        background-color: transparent;
        color: white;
        font-weight: 600;
        border: 1.5px solid rgba(255,255,255,0.9);
        border-radius: 8px;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
        background-color: rgba(255,255,255,0.12);
    }
    </style>
    """, unsafe_allow_html=True)

    with open("logo.png", "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode()

    st.markdown(f"""
        <img src="data:image/png;base64,{encoded}" class="sidebar-logo">
    """, unsafe_allow_html=True)
 
    st.markdown(
        """
        <div style="text-align: center; font-size: 15px; font-weight: bold;">
            Aplikasi Klaster Pemohon Paspor
        </div>
        <hr style="border: 1px solid #999; width: 80%; margin: 20px auto 40px auto;">
        """,
        unsafe_allow_html=True
    )
    # ======================================================
    # DIALOG KONFIRMASI RESET
    # ======================================================
    @st.dialog("Konfirmasi Muat Ulang Proses")
    def dialog_reset():
        st.warning(
            "⚠️ Semua data, hasil preprocessing, dan hasil klasterisasi "
            "akan dihapus. Apakah anda yakin untuk muat ulang proses?"
        )

        # Trik pusatkan tombol
        col_left, col_center, col_right = st.columns([1, 2, 1])

        with col_center:
            if st.button("Ya, Reset Proses", use_container_width=True):
                keys_to_clear = [
                    'raw_df',
                    'mapping',
                    'clean_df',
                    'reduced_df',
                    'preproc_report',
                    'df_final',
                    'kproto_models',
                    'dbi_scores',
                    'labels_for_k',
                    'best_k',
                    'merged',
                    'labels_rep',
                    'cluster_results',
                    'klaster_sudah_dijalankan',
                    'menu_utama',
                    'tahap_klasterisasi'
                ]

                for k in keys_to_clear:
                    if k in st.session_state:
                        del st.session_state[k]

                st.success("Proses berhasil direset. Silakan mulai dari awal.")
                st.rerun()

    # ======================================================
    # SIDEBAR
    # ======================================================
    with st.sidebar:

        # -------------------------
        # MENU UTAMA
        # -------------------------
        main = st.radio(
            "Menu Utama:",
            ["Beranda", "Klasterisasi"],
            key="menu_utama",
            index=0
        )

        # -------------------------
        # TAHAP KLASTERISASI
        # -------------------------
        stage = None
        if main == "Klasterisasi":
            col_space, col_radio = st.columns([1, 10])

            with col_radio:
                stage = st.radio(
                    "Tahap Klasterisasi",
                    [
                        "Unggah Data",
                        "Data Preprocessing",
                        "Klasterisasi",
                        "Visualisasi"
                    ],
                    key="tahap_klasterisasi"
                )


        # -------------------------
        # TOMBOL RESET (PEMICU POPUP)
        # -------------------------
        if st.button("🔄 Muat Ulang Proses"):
            dialog_reset()



# -------------------------
# BERANDA
# -------------------------
if main == "Beranda":

    st.markdown(
        "<h1 style='text-align:center; margin-bottom:0px'>"
        "Aplikasi Klasterisasi Data Pemohon Paspor Berdasarkan Karakteristik Pemohon Menggunakan Algoritma K-Prototypes<br>"
        "<span style='font-size:16px; color:#555;'>",
        unsafe_allow_html=True
    )

    st.markdown("<hr style='width:80%; margin:20px auto;'>", unsafe_allow_html=True)

    st.markdown(
        """
        <p style="text-align: center;">
        Aplikasi ini dirancang untuk mendukung proses analisis data pemohon paspor melalui
        metode klasterisasi menggunakan algoritma K-Prototypes. Sistem ini mengelompokkan
        pemohon ke dalam beberapa klaster berdasarkan kesamaan karakteristik numerik dan
        kategorikal sehingga dapat membantu pengambilan keputusan berbasis data.
        </p>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
    "<h5 style='text-align:center; margin-bottom:0px;'>"
    "Fitur Utama Aplikasi"
    "</h3>",
    unsafe_allow_html=True
    )
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            """
            <ul style="
                list-style-type: disc;
                list-style-position: inside;
                padding-left: 0;
                margin: 0 auto;
                width: fit-content;
                text-align: left;
            ">
                <li>Unggah dataset pemohon paspor</li>
                <li>Data preprocessing</li>
                <li>Penentuan klaster optimal (DBI)</li>
            </ul>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <ul style="
                list-style-type: disc;
                list-style-position: inside;
                padding-left: 0;
                margin: 0 auto;
                width: fit-content;
                text-align: left;
            ">
                <li>Klasterisasi K-Prototypes</li>
                <li>Visualisasi hasil klaster</li>
                <li>Unduh hasil analisis</li>
            </ul>
            """,
            unsafe_allow_html=True
        )




# -------------------------
# KLASTERISASI 
# -------------------------
elif main == "Klasterisasi":
    st.title("Tahap Klasterisasi")


    # -------------------------
    # UNGGAH DATA
    # -------------------------
    import streamlit.components.v1 as components # type: ignore
    if stage == "Unggah Data":
        st.header("Unggah Dataset")

        components.html(
            """
            <div style="
                background-color:#f8f9fa;
                padding:18px 22px;
                border-radius:10px;
                border-left:6px solid #0A2540;
                max-width:1200px;
                margin-bottom:20px;
                font-family: 'Inter', sans-serif;
                font-size:15px;
            ">
                <h4 style="margin-top:0; margin-bottom:10px; color:#0A2540;">
                    Ketentuan Dataset
                </h4>
                <ul style="margin:0; padding-left:20px; color:#444;">
                    <li>Format file: <strong>Microsoft Excel (.xlsx)</strong></li>
                    <li>Dataset wajib memuat kolom:
                        <ul style="margin-top:6px;">
                            <li>Usia</li>
                            <li>Jenis Kelamin</li>
                            <li>Status Pengambilan</li>
                            <li>Tempat Lahir</li>
                        </ul>
                    </li>
                </ul>
            </div>
            """,
            height=200
        )

        st.markdown(
            "<p style='margin-bottom:0; color:#0A2540; font-weight:600;'>Unggah Dataset Excel</p>",
            unsafe_allow_html=True
        )
        uploaded = st.file_uploader(
            "Unggah Dataset Pemohon Paspor (.xlsx)",
            type=["xlsx"],
            label_visibility="collapsed"
        )

        # -------------------------
        # JIKA FILE BARU DIUPLOAD
        # -------------------------
        if uploaded is not None:
            try:
                df = pd.read_excel(uploaded)
                st.session_state['raw_df'] = df.copy()
                mapping = detect_columns(df)
                st.session_state['mapping'] = mapping

                missing = [k for k, v in mapping.items() if v is None]
                if missing:
                    st.error("❌ Dataset ditolak! Kolom berikut tidak ditemukan:")
                    for m in missing:
                        st.write(f"- {m}")
                    st.info("Silakan perbaiki nama kolom di file Anda.")
                else:
                    st.write(f" Dataset berisi {len(df)} baris data dan {len(df.columns)} kolom")
                    st.success("✅ Semua kolom penting berhasil terdeteksi.")
                    for line in humanize_mapping(mapping):
                        st.write("- " + line)

                    st.markdown("### Preview data")
                    st.dataframe(df.head(8))

            except Exception as e:
                st.error(f"Gagal membaca file: {e}")

        # ---------------------------------------------------------
        # JIKA TIDAK ADA FILE BARU TAPI SESSION STATE MASIH ADA
        # ---------------------------------------------------------
        elif st.session_state['raw_df'] is not None:
            st.info("Menggunakan data yang sudah diunggah sebelumnya.")

            df = st.session_state['raw_df']
            mapping = st.session_state['mapping']

            # INFO dataset
            st.write(f" Dataset berisi {len(df)} baris data dan {len(df.columns)} kolom")

            # INFO mapping kolom
            if mapping is not None:
                st.success("✅ Semua kolom penting berhasil terdeteksi otomatis.")
                for line in humanize_mapping(mapping):
                    st.write("- " + line)

            # PREVIEW data
            st.markdown("### Preview data (dari sesi sebelumnya)")
            st.dataframe(df.head(8))


        # ---------------------------------------------------------
        # BELUM ADA FILE SAMA SEKALI
        # ---------------------------------------------------------
        else:
            st.warning("Belum ada file yang diunggah.")


    # ==========================
    # DATA PREPROCESSING 
    # ==========================
    elif stage == "Data Preprocessing":
        st.header("Data Preprocessing")

        if st.session_state.get('raw_df') is None or st.session_state.get('mapping') is None:
            st.warning("Unggah dataset dulu di tahap 'Unggah Data' dan pastikan mapping lengkap.")
        else:
            df_raw = st.session_state['raw_df']
            mapping = st.session_state['mapping']

            if any(v is None for v in mapping.values()):
                st.error("Mapping kolom belum lengkap. Kembali ke 'Unggah Data'.")
            else:
                # ==========================
                # Ringkasan dataset awal
                # ==========================
                st.markdown(f"""
                <div style="
                    background-color:#f8f9fa;
                    padding:14px 16px;
                    border-radius:10px;
                    border-left:6px solid #0A2540;
                    margin-bottom:12px;
                    font-family:'Inter', sans-serif;
                    font-size:15px;
                    color:#444;">
                    <p style="margin:0; font-weight:600; color:#0A2540;">Ringkasan Dataset Awal</p>
                    <p style="margin:2px 0 0 0; font-size:15px; color:#666;">
                        Baris: <strong>{len(df_raw)}</strong> — Kolom: <strong>{len(df_raw.columns)}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("Lihat 10 baris awal", expanded=False):
                    st.dataframe(df_raw.head(10))

                # ==========================
                # Jalankan Preprocessing
                # ==========================
                st.markdown("""
                <style>
                div.stButton > button:first-child {
                    background-color: #0A2540;  /* biru tua */
                    color: white;
                    height: 40px;
                    width: 220px;
                    border-radius: 8px;
                    font-weight: 600;
                    font-size: 15px;
                }
                div.stButton > button:first-child:hover {
                    background-color: #071c33;
                }
                </style>
                """, unsafe_allow_html=True)
                if st.button("Jalankan Preprocessing"):
                    with st.spinner("Menjalankan preprocessing..."):
                        df_clean, df_reduced, report = preprocess_full(df_raw, mapping, perform_reduction=True)
                        st.session_state['clean_df'] = df_clean
                        st.session_state['reduced_df'] = df_reduced
                        st.session_state['preproc_report'] = report
                        st.session_state['clean_df_original_values_for_merge'] = df_clean.copy()
                    st.success("✅ Preprocessing selesai, lihat detail di bawah.")

                rpt = st.session_state.get('preproc_report', None)

                if rpt:
                    # ==========================
                    # Missing Values
                    # ==========================
                    st.markdown(f"""
                    <div style="
                        background-color:#f8f9fa;
                        padding:14px 16px;
                        border-radius:10px;
                        border-left:6px solid #0A2540;
                        margin-bottom:12px;
                        font-family:'Inter', sans-serif;
                        font-size:15px;
                        color:#444;">
                        <p style="margin:0; font-weight:600; color:#0A2540;">Missing Values</p>
                        <p style="margin:4px 0 0 0; font-size:15px; color:#666;">
                            Jumlah data setelah pembersihan missing value: <strong>{rpt.get('rows_after_missing',0)}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.write("Jumlah missing sebelum pembersihan (per kolom):")
                    st.table(pd.DataFrame.from_dict(rpt.get('missing_before', {}), orient='index', columns=['Jumlah']).reset_index().rename(columns={'index':'Kolom'}))

                    st.write("Jumlah missing setelah pembersihan (per kolom):")
                    st.table(pd.DataFrame.from_dict(rpt.get('missing_after', {}), orient='index', columns=['Jumlah']).reset_index().rename(columns={'index':'Kolom'}))

                    # ==========================
                    # Pembersihan Data Tidak Valid (per kolom, semua)
                    # ==========================
                    st.markdown(f"""
                    <div style="
                        background-color:#f8f9fa;
                        padding:14px 16px;
                        border-radius:10px;
                        border-left:6px solid #0A2540;
                        margin-bottom:12px;
                        font-family:'Inter', sans-serif;
                        font-size:15px;
                        color:#444;">
                        <p style="margin:0; font-weight:600; color:#0A2540;">Pembersihan Data Tidak Valid</p>
                        <p style="margin:4px 0 0 0; font-size:15px; color:#666;">
                            Jumlah data setelah pembersihan: <strong>{rpt.get('rows_after_invalid',0)}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    invalids = rpt.get('invalid_details', {})

                    if isinstance(invalids, dict) and len(invalids) > 0:
                        for col, info in invalids.items():
                            # Tentukan jumlah dan data invalid
                            if isinstance(info, dict):
                                cnt = info.get('count', 0)
                                ex = info.get('examples', [])
                            else:
                                cnt = int(info)
                                ex = []
                            
                            st.markdown(f"**Kolom `{col}`** — jumlah data tidak valid: {cnt}", unsafe_allow_html=True)
                            
                            # Tampilkan semua baris invalid jika ada
                            if cnt > 0 and len(ex) > 0:
                                st.dataframe(pd.DataFrame(ex))
                    else:
                        st.write("Tidak ditemukan data tidak valid.")


                    # ==========================
                    # Duplikasi Data dengan Preview
                    # ==========================
                    dupes_count = rpt.get('duplicates_found', 0)
                    dropped_count = rpt.get('dropped_duplicates', 0)
                    after_dup_rows = rpt.get('after_dup_rows', 0)
                    st.markdown(f"""
                    <div style="
                        background-color:#f8f9fa;
                        padding:14px 16px;
                        border-radius:10px;
                        border-left:6px solid #0A2540;
                        margin-bottom:12px;
                        font-family:'Inter', sans-serif;
                        font-size:15px;
                        color:#444;">
                        <p style="margin:0; font-weight:600; color:#0A2540;">Duplikasi Data</p>
                        <p style="margin:4px 0 0 0; font-size:15px; color:#666;">
                            Duplikasi ditemukan: <strong>{dupes_count}</strong>, dibersihkan: <strong>{dropped_count}</strong>
                        </p>
                        <p style="margin:2px 0 0 0; font-size:15px; color:#666;">
                            Jumlah data setelah pembersihan duplikasi: <strong>{after_dup_rows}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Jika ada duplikasi, tampilkan preview
                    if dupes_count > 0 and rpt.get('duplicates_examples', None) is not None:
                        st.markdown("<p style='font-weight:600; color:#0A2540;'>Contoh baris duplikasi:</p>", unsafe_allow_html=True)
                        st.dataframe(pd.DataFrame(rpt['duplicates_examples']))


                    # ==========================
                    # Transformasi Data
                    # ==========================
                    st.markdown(f"""
                    <div style="
                        background-color:#f8f9fa;
                        padding:14px 16px;
                        border-radius:10px;
                        border-left:6px solid #0A2540;
                        margin-bottom:12px;
                        font-family:'Inter', sans-serif;
                        font-size:15px;
                        color:#444;">
                        <p style="margin:0; font-weight:600; color:#0A2540;">Transformasi Data</p>
                        <p style="margin:4px 0 0 0; font-size:15px; color:#666;">
                            Standarisasi numerik menggunakan Z-score.
                        </p>
                        <p style="margin:2px 0 0 0; font-size:15px; color:#666;">
                            Transformasi kategorikal menggunakan Label Encoding.
                        </p>
                        <p style="margin:2px 0 0 0; font-size:15px; color:#666;">
                            Jumlah data setelah transformasi: <strong>{rpt.get('after_transform_rows',0)}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Preview transformasi (utama)
                    if st.session_state.get('clean_df') is not None:
                        df_clean = st.session_state['clean_df']
                        cols_to_show = [c for c in ['usia','jenis_kelamin','tempat_lahir','status','jenis_kelamin_label','tempat_lahir_label','status_label'] if c in df_clean.columns]
                        if cols_to_show:
                            st.markdown("<p style='font-weight:600; color:#0A2540; font-size:15px; margin-bottom:4px;'>Preview Data Transformasi</p>", unsafe_allow_html=True)
                            st.dataframe(df_clean[cols_to_show])

                    # ==========================
                    # Reduksi Data
                    # ==========================
                    st.markdown(f"""
                    <div style="
                        background-color:#f8f9fa;
                        padding:14px 16px;
                        border-radius:10px;
                        border-left:6px solid #0A2540;
                        margin-bottom:12px;
                        font-family:'Inter', sans-serif;
                        font-size:15px;
                        color:#444;">
                        <p style="margin:0; font-weight:600; color:#0A2540;">Reduksi Data</p>
                        <p style="margin:4px 0 0 0; font-size:15px; color:#666;">
                            Jumlah baris representatif (unik): <strong>{rpt.get('reduced_rows',0)}</strong>
                        </p>
                        <p style="margin:2px 0 0 0; font-size:15px; color:#666;">
                            Jumlah data setelah preprocessing: <strong>{rpt.get('final_rows',0)}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.session_state.get('reduced_df') is not None:
                        st.markdown("<p style='font-weight:600; color:#0A2540;'>Preview Data Reduksi</p>", unsafe_allow_html=True)
                        st.dataframe(st.session_state['reduced_df'][['usia_tahun','jenis_kelamin','tempat_lahir','status']])

                    # ==========================
                    # Simpan df_wilayah untuk logic saja
                    # ==========================
                    wilayah_referensi = rpt.get("wilayah_dominan", [])
                    if wilayah_referensi:
                        df_wilayah = pd.DataFrame({
                            "wilayah": wilayah_referensi,
                            "jarak_km": [0,40,77,117,123,88,119,60,380,44,135,165,298,129,217,131,197,217,173,93],
                            "ada_mpp": [True,True,True,False,True,True,False,True,False,True,False,False,True,False,False,False,False,False,False,False]
                        })
                        st.session_state["df_wilayah"] = df_wilayah  # disimpan untuk logic, tidak ditampilkan

        
    # ==============================
    # Cari k terbaik & jalankan klaster
    # ==============================
    
    elif stage == "Klasterisasi":
        st.header("Proses Klasterisasi")

        df_clean = st.session_state.get('clean_df')
        df_reduced = st.session_state.get('reduced_df')

        if df_clean is None or df_reduced is None:
            st.warning("Lakukan preprocessing terlebih dahulu.")
            st.stop()

        # ======================================================
        # Inisialisasi sel_k 
        # ======================================================
        if 'sel_k' not in st.session_state:
            if 'best_k' in st.session_state:
                st.session_state['sel_k'] = st.session_state['best_k']
            else:
                st.session_state['sel_k'] = 3


        # ======================================================
        # PILIH RENTANG K UNTUK DBI (card-style)
        # ======================================================
        st.markdown("""
        <div style="
            background-color:#f8f9fa;
            padding:14px 16px;
            border-radius:10px;
            border-left:6px solid #0A2540;
            margin-bottom:12px;
            font-family:'Inter', sans-serif;
            font-size:15px;
            color:#444;">
            <p style="margin:0; font-weight:600; color:#0A2540;">Tentukan Rentang k</p>
            <p style="margin:4px 0 0 0; font-size:15px; color:#666;">
                Pilih rentang jumlah klaster (k) untuk menghitung DBI.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Slider Streamlit (tetap fungsional)
        kmin, kmax = st.slider(
            "",
            min_value=2,
            max_value=12,
            value=(2, 6),
            step=1
        )

        # ======================================================
        # CARI K TERBAIK (DBI)
        # ======================================================
        st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #0A2540;
            color: white;
            font-size: 15px;
            font-weight: 600;
            padding: 8px 20px;
            border-radius: 8px;
            border: none;
        }
        div.stButton > button:first-child:hover {
            background-color: #081B33;
        }
        </style>
        """, unsafe_allow_html=True)
        if st.button("Cari nilai k terbaik", key="btn_cari_k"):
            dbi_scores = {}
            labels_for_k = {}
            kproto_models = {}

            with st.spinner("Menghitung DBI untuk setiap k..."):
                for k in range(kmin, kmax + 1):
                    try:
                        kp, labels, arr, cat_idx = run_kprototypes_on_array(df_reduced, k)
                        dbi = compute_dbi_mixed(df_reduced, labels)

                        dbi_scores[k] = float(dbi)
                        labels_for_k[k] = labels
                        kproto_models[k] = kp

                        st.write(f"k = {k} → DBI = {dbi:.4f}")
                    except Exception as e:
                        st.error(f"k = {k} gagal: {e}")

            if dbi_scores:
                best_k = min(dbi_scores, key=dbi_scores.get)

                # SIMPAN & SINKRONKAN KE WIDGET
                st.session_state['dbi_scores'] = dbi_scores
                st.session_state['labels_for_k'] = labels_for_k
                st.session_state['kproto_models'] = kproto_models
                st.session_state['best_k'] = best_k
                st.session_state['sel_k'] = best_k   # ← INI YANG PENTING

                st.success(
                    f"DBI terbaik diperoleh pada k = {best_k} "
                    f"(DBI = {dbi_scores[best_k]:.4f})"
                )

                # Plot DBI
                import plotly.graph_objects as go # type: ignore
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=list(dbi_scores.keys()),
                    y=list(dbi_scores.values()),
                    mode='lines+markers',
                    line=dict(color='#0A2540', width=3),   # garis biru tua
                    marker=dict(color='#0072B5', size=10), # marker biru terang
                    hovertemplate='k=%{x}<br>DBI=%{y:.4f}<extra></extra>'
                ))

                fig.update_layout(
                    title="Davies–Bouldin Index vs k",
                    xaxis_title="k",
                    yaxis_title="DBI",
                    font=dict(family="Inter, sans-serif", size=15, color="#0A2540"),
                    plot_bgcolor="#f8f9fa",
                    paper_bgcolor="#f8f9fa",
                    margin=dict(l=40, r=40, t=60, b=40)
                )

                st.plotly_chart(fig, use_container_width=True)


        # ======================================================
        # PILIH K UNTUK DIJALANKAN
        # ======================================================
        sel_k = st.number_input(
            "Pilih k untuk dijalankan",
            min_value=2,
            max_value=50,
            key="sel_k"
        )

        # ======================================================
        # JALANKAN K-PROTOTYPES
        # ======================================================
        st.markdown("""
        <style>
        div.stButton > button[key="btn_run_kproto"] {
            background-color: #0072B5;
            color: white;
            font-size: 15px;
            font-weight: 600;
            padding: 8px 20px;
            border-radius: 8px;
            border: none;
        }
        div.stButton > button[key="btn_run_kproto"]:hover {
            background-color: #005a8c;
        }
        </style>
        """, unsafe_allow_html=True)
        if st.button("Jalankan K-Prototypes", key="btn_run_kproto"):
            try:
                kp, labels_rep, arr, cat_idx = run_kprototypes_on_array(
                    df_reduced,
                    int(st.session_state['sel_k'])
                )

                # Label ke data representatif
                df_reduced_labeled = df_reduced.copy()
                df_reduced_labeled['cluster'] = labels_rep.astype(int)
                df_reduced_labeled['__merge_key'] = make_merge_key(df_reduced_labeled)

                # Tebarkan ke data asli
                df_clean_copy = df_clean.copy()
                df_clean_copy['__merge_key'] = make_merge_key(df_clean_copy)

                merged = df_clean_copy.merge(
                    df_reduced_labeled[['__merge_key', 'cluster']],
                    on='__merge_key',
                    how='left'
                )
                merged['cluster'] = merged['cluster'].fillna(-1).astype(int)
                merged.drop(columns='__merge_key', inplace=True)



                # ============================
                # SIMPAN KE SESSION STATE
                # ============================
                st.session_state['merged'] = merged
                st.session_state['labels_rep'] = labels_rep
                st.session_state['cluster_results'] = {
                    'df_final': merged
                }
                st.session_state['df_final'] = merged
                st.session_state['klaster_sudah_dijalankan'] = True
                st.session_state['now'] = datetime.now().strftime("%Y%m%d_%H%M%S")

                st.success(f"Klasterisasi selesai untuk k = {st.session_state['sel_k']}.")

            except Exception as e:
                st.error(f"Gagal menjalankan klasterisasi: {e}")

        # ======================================================
        # TAMPILKAN INTERPRETASI + DATA PEMOHON
        # ======================================================
        if (
            st.session_state.get('klaster_sudah_dijalankan') is True
            and isinstance(st.session_state.get('cluster_results'), dict)
        ):

            st.subheader("Interpretasi Klaster Pemohon Paspor dan Rekomendasi Pelayanan Individual")

            df_final = st.session_state['cluster_results']['df_final']

            # ======================================================
            # REKOMENDASI INDIVIDU (DITURUNKAN DARI KLASTER)
            # ======================================================
            df_tampil = df_final.copy()
            df_tampil['Rekomendasi Pelayanan'] = df_tampil.apply(
                lambda row: rekomendasi_individu(row, st.session_state['df_wilayah']),
                axis=1
            )

            # ======================================================
            # PENCARIAN PEMOHON INDIVIDUAL
            # ======================================================
            st.markdown("### Cek Klaster Pemohon")

            with st.form("form_search"):
                col1, col2 = st.columns([6, 1], vertical_alignment="bottom")

                with col1:
                    keyword = st.text_input(
                        "Masukkan nama / nomor permohonan / nomor paspor",
                        placeholder="Contoh: Andi / 123456 / A1234567"
                    )
                
                with col2:
                    search = st.form_submit_button("Search", use_container_width=True)



            if keyword:
                hasil_cari = df_tampil[
                    df_tampil['nama'].str.contains(keyword, case=False, na=False) |
                    df_tampil['nopermohonan'].astype(str).str.contains(keyword, na=False) |
                    df_tampil['nopaspor'].astype(str).str.contains(keyword, na=False)
                ]

                if hasil_cari.empty:
                    st.warning("Data pemohon tidak ditemukan.")
                else:
                    jumlah = len(hasil_cari)
                    klaster_unik = hasil_cari['cluster'].unique()

                    if len(klaster_unik) == 1:
                        st.success(
                            f"Ditemukan {jumlah} data pemohon pada klaster {klaster_unik[0]}."
                        )
                    else:
                        st.success(
                            f"Ditemukan {jumlah} data pemohon pada klaster {', '.join(map(str, klaster_unik))}."
                        )

                    st.dataframe(
                        hasil_cari[
                            [
                            'id',
                            'nopermohonan',
                            'nama',
                            'tempat_lahir',
                            'tanggallahir',
                            'jenis_kelamin',
                            'notelepon',
                            'nopaspor',
                            'kodebilling',
                            'tanggalsimpan',
                            'tanggalambil',
                            'smsgateway',
                            'koderak',
                            'status',
                            'usia_tahun',
                            'cluster',
                            'Rekomendasi Pelayanan'
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True
                    )

                    st.markdown("<hr>", unsafe_allow_html=True)

            for c in sorted(df_tampil['cluster'].unique()):
                if c == -1:
                    continue

                df_klaster = df_tampil[df_tampil['cluster'] == c]

                # =========================
                # INTERPRETASI KLASTER
                # =========================
                usia_min = df_klaster['usia_tahun'].min()
                usia_max = df_klaster['usia_tahun'].max()
                usia_mean = df_klaster['usia_tahun'].mean()

                jk_prop = df_klaster['jenis_kelamin'].value_counts(normalize=True) * 100
                jk_dom = jk_prop.idxmax()

                status_prop = df_klaster['status'].value_counts(normalize=True) * 100
                status_dom = status_prop.idxmax()

                top_tempat = df_klaster['tempat_lahir'].value_counts().head(5)
                tempat_teks = ", ".join(top_tempat.index)

                st.markdown(f"### Klaster {c}")

                st.markdown(f"""
                <div style="
                    background-color:#f8f9fa;
                    padding:16px 20px;
                    border-radius:10px;
                    border-left:6px solid #0A2540;
                    margin-bottom:12px;
                ">
                    <p style="margin:0; color:#444;">
                        <strong>Jumlah pemohon:</strong> {len(df_klaster)} <br>
                        <strong>Rentang usia:</strong> {usia_min}–{usia_max} tahun <br>
                        <strong>Rata-rata usia:</strong> {usia_mean:.2f} tahun <br>
                        <strong>Jenis kelamin dominan:</strong> {jk_dom} ({jk_prop[jk_dom]:.2f}%) <br>
                        <strong>Status pengambilan dominan:</strong> {status_dom} ({status_prop[status_dom]:.2f}%) <br>
                        <strong>Wilayah asal terbanyak:</strong> {tempat_teks}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # =========================
                # DATA PEMOHON + REKOMENDASI
                # =========================
                with st.expander("Lihat data pemohon"):
                    st.dataframe(
                        df_klaster[
                            [
                            'id',
                            'nopermohonan',
                            'nama',
                            'tempat_lahir',
                            'tanggallahir',
                            'jenis_kelamin',
                            'notelepon',
                            'nopaspor',
                            'kodebilling',
                            'tanggalsimpan',
                            'tanggalambil',
                            'smsgateway',
                            'koderak',
                            'status',
                            'usia_tahun',
                            'cluster',
                            'Rekomendasi Pelayanan'
                            ]
                        ],
                        use_container_width=True,
                        height=380
                    )

                    # =========================
                    # DOWNLOAD EXCEL PER KLASTER
                    # =========================
                    excel_buffer = io.BytesIO()

                    df_klaster[
                        [
                            'id',
                            'nopermohonan',
                            'nama',
                            'tempat_lahir',
                            'tanggallahir',
                            'jenis_kelamin',
                            'notelepon',
                            'nopaspor',
                            'kodebilling',
                            'tanggalsimpan',
                            'tanggalambil',
                            'smsgateway',
                            'koderak',
                            'status',
                            'usia_tahun',
                            'cluster',
                            'Rekomendasi Pelayanan'
                        ]
                    ].to_excel(
                        excel_buffer,
                        index=False,
                        engine='openpyxl'
                    )

                    excel_buffer.seek(0)

                    st.download_button(
                        label="📥 Unduh Data Klaster (Excel)",
                        data=excel_buffer,
                        file_name=f"data_pemohon_klaster_{c}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                    st.markdown("<hr>", unsafe_allow_html=True)

            st.subheader("Rekomendasi Strategi Pelayanan Keimigrasian Berbasis Hasil Klasterisasi Pemohon Paspor")

            st.markdown("""
            <p style="text-align: justify; color:#555; max-width:900px;">
           Rekomendasi strategi pelayanan keimigrasian pada tingkat klaster yang disusun berdasarkan hasil klasterisasi data pemohon paspor. Rekomendasi bersifat analitis dan digunakan sebagai bahan pertimbangan dalam pengambilan keputusan manajerial, tanpa menggantikan kewenangan institusional yang berlaku.
            </p>
            <hr>
            """, unsafe_allow_html=True)

            for k in sorted(df_final['cluster'].unique()):
                if k == -1:
                    continue

                data_k = df_final[df_final['cluster'] == k]

                wilayah_dominan = (
                    data_k['tempat_lahir']
                    .value_counts()
                    .head(5)
                    .index
                    .tolist()
                )

                rekomendasi = buat_rekomendasi_pelayanan(
                    data_k=data_k,
                    wilayah_dominan=wilayah_dominan,
                    df_wilayah_ref=st.session_state["df_wilayah"]
                )

                with st.expander(f"Rekomendasi Strategi – Klaster {k}"):
                    st.markdown(
                        """
                        <div style="text-align: justify;">
                            <ul style="padding-left: 20px;">
                                {}
                            </ul>
                        </div>
                        """.format(
                            "".join(
                                f"<li style='margin-bottom:8px;'>{r}</li>"
                                for r in rekomendasi
                            )
                        ),
                        unsafe_allow_html=True
                    )

    # -------------------------
    # VISUALISASI
    # -------------------------
    
    elif stage == "Visualisasi":

        # =========================
        # HEADER HALAMAN (WEB FEEL)
        # =========================
        st.markdown("""
        <h2 style="text-align:center;">Visualisasi Hasil Klaster Pemohon Paspor</h2>
        <hr style="margin:30px 0;">
        """, unsafe_allow_html=True)

        if st.session_state.get('df_final') is None:
            st.warning("Data klaster belum tersedia. Silakan jalankan proses klasterisasi terlebih dahulu.")
            st.stop()

        df_final = st.session_state['df_final']

        with st.container():
            # JUDUL CARD
            st.markdown(
                "<p style='margin:0; font-weight:600; font-size:28px; color:#0A2540;'>"
                "Ringkasan Hasil Klasterisasi"
                "</p>",
                unsafe_allow_html=True
            )

            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

            # ISI CARD (SEMUA METRIC MASUK)
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Jumlah Klaster", df_final['cluster'].nunique())

            with col2:
                st.metric("Total Pemohon", f"{len(df_final):,}")

            with col3:
                st.metric(
                    "Klaster Terbesar",
                    f"Klaster {df_final['cluster'].value_counts().idxmax()}"
                )

            # TUTUP DIV CARD
            st.markdown("</div>", unsafe_allow_html=True)



        # =====================================================
        # BAGIAN 1 — KARAKTERISTIK USIA
        # =====================================================
        st.subheader("Karakteristik Usia Pemohon")

        col_left, col_right = st.columns(2)

        # =========================
        # BOX PLOT USIA PER KLASTER
        # =========================
        with col_left:
            fig, ax = plt.subplots(figsize=(6, 4))

            df_final.boxplot(
                column='usia_tahun',
                by='cluster',
                ax=ax,
                grid=False,
                boxprops=dict(color='#0A2540', linewidth=1.5),
                medianprops=dict(color='#E5533D', linewidth=2),
                whiskerprops=dict(color='#0A2540'),
                capprops=dict(color='#0A2540')
            )

            ax.set_title("Distribusi Usia per Klaster", fontsize=13, fontweight='bold')
            ax.set_xlabel("Klaster", fontsize=11)
            ax.set_ylabel("Usia (Tahun)", fontsize=11)

            ax.tick_params(axis='both', labelsize=10)
            ax.yaxis.grid(True, linestyle='--', alpha=0.4)

            plt.suptitle("")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # =========================
        # HISTOGRAM SEBARAN USIA
        # =========================
        with col_right:
            fig, ax = plt.subplots(figsize=(6, 4))

            for cluster_id in sorted(df_final['cluster'].unique()):
                subset = df_final[df_final['cluster'] == cluster_id]
                ax.hist(
                    subset['usia_tahun'],
                    bins=20,
                    alpha=0.6,
                    label=f'Klaster {cluster_id}'
                )

            ax.set_title("Pola Sebaran Usia Pemohon", fontsize=13, fontweight='bold')
            ax.set_xlabel("Usia (Tahun)", fontsize=11)
            ax.set_ylabel("Frekuensi", fontsize=11)

            ax.tick_params(axis='both', labelsize=10)
            ax.grid(axis='y', linestyle='--', alpha=0.4)

            ax.legend(
                frameon=False,
                fontsize=10,
                loc='upper right'
            )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


        # =========================
        # SEBARAN INDIVIDU (PENDUKUNG)
        # =========================
        col_pad_left, col_center, col_pad_right = st.columns([1, 2, 1])

        with col_center:
            fig, ax = plt.subplots(figsize=(6, 4))

            # jitter untuk indeks individu (dummy)
            y_jitter = np.random.normal(0, 0.08, size=len(df_final))

            ax.scatter(
                df_final['usia_tahun'],
                y_jitter,
                c=df_final['cluster'],
                cmap='viridis',
                alpha=0.6,
                s=30
            )

            ax.set_title(
                "Scatter Plot Usia Berdasarkan Klaster",
                fontsize=13,
                fontweight='bold'
            )

            ax.set_xlabel("Usia", fontsize=11)

            ax.set_ylabel(
                "Indeks Dummy (Jitter)",
                fontsize=11
            )

            # y dummy → tidak perlu angka
            ax.set_yticks([])

            ax.grid(axis='x', linestyle='--', alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)



        st.markdown("<hr style='margin-top:20px; margin-bottom:20px;'>", unsafe_allow_html=True)
        

        # =====================================================
        # BAGIAN 2 — KOMPOSISI PEMOHON
        # =====================================================
        st.subheader("Komposisi Pemohon per Klaster")

        col_left, col_right = st.columns(2)

        # =========================
        # KOMPOSISI JENIS KELAMIN
        # =========================
        with col_left:
            jk_cluster = (
                df_final
                .groupby(['cluster', 'jenis_kelamin'])
                .size()
                .unstack(fill_value=0)
            )

            fig, ax = plt.subplots(figsize=(6, 4))

            jk_cluster.plot(
                kind='bar',
                ax=ax,
                width=0.75
            )

            ax.set_title("Komposisi Jenis Kelamin", fontsize=13, fontweight='bold')
            ax.set_xlabel("Klaster", fontsize=11)
            ax.set_ylabel("Jumlah Pemohon", fontsize=11)

            ax.tick_params(axis='both', labelsize=10)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.legend(
                title="Jenis Kelamin",
                frameon=False,
                fontsize=10,
                title_fontsize=10
            )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # =========================
        # STATUS PENGAMBILAN PASPOR
        # =========================
        with col_right:
            status_cluster = (
                df_final
                .groupby(['cluster', 'status'])
                .size()
                .unstack(fill_value=0)
            )

            fig, ax = plt.subplots(figsize=(6, 4))

            status_cluster.plot(
                kind='bar',
                ax=ax,
                width=0.75
            )

            ax.set_title("Status Pengambilan Paspor", fontsize=13, fontweight='bold')
            ax.set_xlabel("Klaster", fontsize=11)
            ax.set_ylabel("Jumlah Pemohon", fontsize=11)

            ax.tick_params(axis='both', labelsize=10)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.legend(
                title="Status",
                frameon=False,
                fontsize=10,
                title_fontsize=10
            )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown(
            "<hr style='margin-top:20px; margin-bottom:20px;'>",
            unsafe_allow_html=True
        )

        # =====================================================
        # BAGIAN 3 — SEBARAN WILAYAH
        # =====================================================
        st.subheader("Sebaran Wilayah Asal Pemohon")

        top_n = 5

        for cluster_id in sorted(df_final['cluster'].unique()):

            st.markdown(
                f"<p style='margin:12px 0 6px 0; font-weight:600; color:#0A2540;'>"
                f"Klaster {cluster_id} — {top_n} Wilayah Asal Terbanyak"
                f"</p>",
                unsafe_allow_html=True
            )

            subset = df_final[df_final['cluster'] == cluster_id]
            top_tl = subset['tempat_lahir'].value_counts().head(top_n)

            fig, ax = plt.subplots(figsize=(7, 4))

            top_tl.plot(
                kind='bar',
                ax=ax,
                width=0.7
            )

            ax.set_title(
                f"Distribusi Wilayah Asal Pemohon (Klaster {cluster_id})",
                fontsize=13,
                fontweight='bold'
            )
            ax.set_xlabel("Wilayah", fontsize=11)
            ax.set_ylabel("Jumlah Pemohon", fontsize=11)

            ax.tick_params(axis='both', labelsize=10)
            ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=30,
                ha='right'
            )

            ax.grid(axis='y', linestyle='--', alpha=0.4)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown(
            "<hr style='margin-top:20px; margin-bottom:20px;'>",
            unsafe_allow_html=True
        )

        # =========================
        # DOWNLOAD PDF
        # =========================
        st.markdown("""
        <style>
        div[data-testid="stDownloadButton"] > button {
            background-color: #0A2540;
            color: white;
            border-radius: 6px;
            border: none;
            padding: 0.55em 1.2em;
            font-weight: 600;
        }

        div[data-testid="stDownloadButton"] > button:hover {
            background-color: #081E33;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

        st.download_button(
            label="📥 Unduh PDF Visualisasi",
            data=save_visuals_pdf(
                df_final,
                st.session_state['now'],
                st.session_state.get('best_k') or 'manual'
            ),
            file_name=f"visualisasi_k{st.session_state.get('best_k','manual')}_{st.session_state['now']}.pdf",
            mime="application/pdf"
        )
