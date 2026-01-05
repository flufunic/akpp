import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from kmodes.kprototypes import KPrototypes
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
        'examples': invalid_tp_all.head(5).to_dict(orient='records')
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


from io import BytesIO

def save_visuals_pdf(df_final, now, best_k):
    buffer = BytesIO()

    with PdfPages(buffer) as pdf:

        # =========================
        # BOXPLOT USIA
        # =========================
        fig, ax = plt.subplots()

        df_final.boxplot(
            column='usia_tahun',
            by='cluster',
            ax=ax
        )

        ax.set_title("Boxplot Usia per Klaster")
        ax.set_xlabel("Klaster")
        ax.set_ylabel("Usia")
        plt.suptitle("")

        pdf.savefig(fig)
        plt.close(fig)

        # =========================
        # HISTOGRAM USIA
        # =========================
        fig, ax = plt.subplots()

        for cluster_id in sorted(df_final['cluster'].unique()):
            subset = df_final[df_final['cluster'] == cluster_id]
            ax.hist(
                subset['usia_tahun'],
                bins=20,
                alpha=0.6,
                label=f'Klaster {cluster_id}'
            )
        ax.set_title("Histogram Distribusi Usia per Klaster")
        ax.set_xlabel("Usia")
        ax.set_ylabel("Frekuensi")
        ax.legend()

        pdf.savefig(fig)
        plt.close(fig)

        # =========================
        # JENIS KELAMIN
        # =========================
        jk_cluster = (
            df_final
            .groupby(['cluster', 'jenis_kelamin'])
            .size()
            .unstack(fill_value=0)
        )

        fig, ax = plt.subplots()
        jk_cluster.plot(kind='bar', ax=ax)
        ax.set_title("Distribusi Jenis Kelamin per Klaster")
        ax.set_xlabel("Klaster")
        ax.set_ylabel("Jumlah")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

        pdf.savefig(fig)
        plt.close(fig)

        # =========================
        # STATUS PENGAMBILAN
        # =========================
        status_cluster = (
            df_final
            .groupby(['cluster', 'status'])
            .size()
            .unstack(fill_value=0)
        )

        fig, ax = plt.subplots()
        status_cluster.plot(kind='bar', ax=ax)
        ax.set_title("Distribusi Status Pengambilan per Klaster")
        ax.set_xlabel("Klaster")
        ax.set_ylabel("Jumlah")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

        pdf.savefig(fig)
        plt.close(fig)

        # =========================
        # TOP 5 TEMPAT LAHIR PER KLASTER
        # =========================
        top_n = 5

        for cluster_id in sorted(df_final['cluster'].unique()):
            fig, ax = plt.subplots()

            subset = df_final[df_final['cluster'] == cluster_id]
            top_tl = subset['tempat_lahir'].value_counts().head(top_n)

            top_tl.plot(kind='bar', ax=ax)
            ax.set_title(f"Top {top_n} Tempat Lahir – Klaster {cluster_id}")
            ax.set_xlabel("Tempat Lahir")
            ax.set_ylabel("Jumlah")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            pdf.savefig(fig)
            plt.close(fig)
        # =========================
        # SCATTER USIA (JITTER)
        # =========================
        fig, ax = plt.subplots()

        y_jitter = np.random.normal(0, 0.05, size=len(df_final))

        scatter = ax.scatter(
            df_final['usia_tahun'],
            y_jitter,
            c=df_final['cluster'],
            alpha=0.6
        )
        ax.set_title("Scatter Plot Usia Berdasarkan Klaster")
        ax.set_xlabel("Usia")
        ax.set_ylabel("Indeks Dummy (Jitter)")
        ax.set_title("Scatter Plot Usia Berdasarkan Klaster")

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
            .sidebar-logo {
                width: 120px;    
                margin-left: auto;
                margin-right: auto;
                display: block;
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
            Aplikasi Klasterisasi Data Pemohon Paspor Berdasarkan Karakteristik Pemohon
            Menggunakan Algoritma K-Prototypes
        </div>
        <hr style="border: 1px solid #999; width: 80%; margin: 20px auto 40px auto;">
        """,
        unsafe_allow_html=True
    )

    main = st.radio(
        "Menu Utama:",
        ["Beranda", "Klasterisasi", "Muat Ulang Proses"]
    )


# -------------------------
# BERANDA
# -------------------------
if main == "Beranda":
    st.markdown(
    """
    <h1 style="text-align: center;">
        Aplikasi Klasterisasi Data Pemohon Paspor Berdasarkan Karakteristik Pemohon
        Menggunakan Algoritma K-Prototypes
    </h1>
    <hr style="border: 1px solid #999; width: 80%; margin: 20px auto 80px auto;">
    """,
    unsafe_allow_html=True
)


    st.markdown("""
    Aplikasi ini dirancang untuk melakukan klasterisasi menggunakan algoritma K-Prototyoes dengan mengelompokkan pemohon paspor ke dalam beberapa klaster yang memiliki karakteristik serupa

    Fitur utama dari aplikasi ini mencakup:
                
    1. Unggah dataset pemohon paspor yang akan digunakan dalam proses klasterisasi
    2. Data Preprocessing untuk menyiapkan data melalui proses pembersihan dan standarisasi data
    3. Penentuan jumlah klaster optimal menggunakan evaluasi Davies Bouldin Index
    4. Klasterisasi pemohon paspor menggunakan algoritma K-prototypes
    5. Visualisasi hasil klasterisasi
    6. Fitur unduh hasil klasterisasi dan visualisasi      
    """)

# -------------------------
# KLASTERISASI 
# -------------------------
elif main == "Klasterisasi":
    st.title("Klasterisasi")
    stage = st.selectbox("Pilih Tahap:", ["Unggah Data", "Data Preprocessing", "Klasterisasi", "Visualisasi"])

   # -------------------------
    # UNGGAH DATA
    # -------------------------
    if stage == "Unggah Data":
        st.header("📤 Unggah Dataset")

        uploaded = st.file_uploader("Unggah file Excel (.xlsx)", type=["xlsx"])

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
            st.info("📁 Menggunakan data yang sudah diunggah sebelumnya.")

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


    # -------------------------
    # DATA PREPROCESSING
    # -------------------------
    elif stage == "Data Preprocessing":
        st.header("🧹 Data Preprocessing")

        if st.session_state['raw_df'] is None or st.session_state['mapping'] is None:
            st.warning("Unggah dataset dulu di tahap 'Unggah Data' dan pastikan mapping lengkap.")
        else:
            df_raw = st.session_state['raw_df']
            mapping = st.session_state['mapping']
            if any(v is None for v in mapping.values()):
                st.error("Mapping kolom belum lengkap. Kembali ke 'Unggah Data'.")
            else:
                st.subheader("Ringkasan dataset awal")
                st.write(f"Baris: {len(df_raw)} — Kolom: {len(df_raw.columns)}")
                with st.expander("Lihat 10 baris awal"):
                    st.dataframe(df_raw.head(10))

                if st.button("Jalankan Preprocessing"):
                    with st.spinner("Menjalankan preprocessing..."):
                        df_clean, df_reduced, report = preprocess_full(df_raw, mapping, perform_reduction=True)
                        st.session_state['clean_df'] = df_clean
                        st.session_state['reduced_df'] = df_reduced
                        st.session_state['preproc_report'] = report
                        st.session_state['clean_df_original_values_for_merge'] = df_clean.copy()
                    st.success("✅ Preprocessing selesai — lihat detail di bawah.")

                if st.session_state['preproc_report'] is not None:
                    rpt = st.session_state['preproc_report']
                    st.subheader("1. Missing Values")

                    st.write("Jumlah missing sebelum pembersihan (per kolom):")
                    st.table(
                        pd.DataFrame.from_dict(
                            rpt.get('missing_before', {}),
                            orient='index',
                            columns=['Jumlah']
                        )
                        .reset_index()
                        .rename(columns={'index': 'Kolom'})
                    )

                    st.write("Jumlah missing setelah pembersihan (per kolom):")
                    st.table(
                        pd.DataFrame.from_dict(
                            rpt.get('missing_after', {}),
                            orient='index',
                            columns=['Jumlah']
                        )
                        .reset_index()
                        .rename(columns={'index': 'Kolom'})
                    )

                    st.write(
                        "Jumlah data setelah pembersihan missing value:",
                        rpt.get('rows_after_missing', 0)
                    )

                    st.markdown("---")




                    st.subheader("2. Pembersihan Data Tidak Valid")

                    invalids = rpt.get('invalid_details', {})

                    if isinstance(invalids, dict) and len(invalids) > 0:
                        for col, info in invalids.items():
                            if isinstance(info, dict):
                                cnt = info.get('count', 0)
                                ex = info.get('examples', [])
                            else:
                                cnt = int(info)
                                ex = []

                            st.markdown(f"**Kolom `{col}`** — jumlah data tidak valid: {cnt}")

                            if cnt > 0 and len(ex) > 0:
                                st.write("Contoh data tidak valid:")
                                st.dataframe(pd.DataFrame(ex))
                    else:
                        st.write("Tidak ditemukan data tidak valid.")

                    st.write(
                        "Jumlah data setelah pembersihan data tidak valid:",
                        rpt.get('rows_after_invalid', 0)
                    )

                    st.markdown("---")




                    st.subheader("3. Duplikasi")
                    st.write(f"Duplikasi ditemukan: {rpt.get('duplicates_found',0)}, dibersihkan: {rpt.get('dropped_duplicates',0)}")
                    st.write("Jumlah data setelah pembersihan duplikasi:", rpt.get('after_dup_rows',0))
                    st.markdown("---")

                    st.subheader("4. Transformasi")
                    st.write("Standarisasi numerik menggunakan Z-score.")
                    st.write("Transformasi kategorikal menggunakan Label Encoding.")
                    st.write("Jumlah data setelah transformasi:", rpt.get('after_transform_rows',0))
                    # show preview transform (only selected columns to avoid flooding UI)
                    if st.session_state.get('clean_df') is not None:
                        preview_cols = ['usia','jenis_kelamin','tempat_lahir','status','usia_z' if 'usia_z' in st.session_state['clean_df'].columns else None]
                        # show main transformed columns & label columns
                        show_df = st.session_state['clean_df'].copy()
                        to_show = []
                        for c in ['usia','jenis_kelamin','tempat_lahir','status','jenis_kelamin_label','tempat_lahir_label','status_label']:
                            if c in show_df.columns:
                                to_show.append(c)
                        if len(to_show) == 0:
                            st.write("Tidak ada kolom transformasi yang dapat ditampilkan.")
                        else:
                           st.dataframe(show_df[to_show]) 
                    st.markdown("---")

                    st.subheader("5. Reduksi ")
                    st.write(f"Jumlah baris representatif (unik): {rpt.get('reduced_rows',0)}")
                    st.write("Jumlah data setelah preprocessing:", rpt.get('final_rows',0))
                    if st.session_state.get('reduced_df') is not None:
                        st.dataframe(st.session_state['reduced_df'][['usia','jenis_kelamin','tempat_lahir','status']])
                    st.markdown("---")

    
    # ==============================
    # Cari k terbaik & jalankan klaster
    # ==============================
    
    elif stage == "Klasterisasi":
        st.header("🔢 Proses Klasterisasi")

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
        # PILIH RENTANG K UNTUK DBI
        # ======================================================
        st.subheader("Tentukan rentang k")
        kmin, kmax = st.slider(
            "Pilih rentang jumlah klaster (k)",
            min_value=2,
            max_value=12,
            value=(2, 6),
            step=1
        )

        # ======================================================
        # CARI K TERBAIK (DBI)
        # ======================================================
        if st.button("Cari nilai k terbaik"):
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
                fig, ax = plt.subplots()
                ax.plot(list(dbi_scores.keys()), list(dbi_scores.values()), marker='o')
                ax.set_xlabel("k")
                ax.set_ylabel("DBI")
                ax.set_title("Davies–Bouldin Index vs k")
                st.pyplot(fig)

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
        if st.button("Jalankan K-Prototypes pada data representatif dan tebarkan label ke data asli"):
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
        # TAMPILKAN INTERPRETASI 
        # ======================================================
        if (
            st.session_state.get('klaster_sudah_dijalankan') is True
            and isinstance(st.session_state.get('cluster_results'), dict)
        ):

            st.subheader("📊 Ringkasan Hasil Klasterisasi")

            df_final = st.session_state['cluster_results']['df_final']

            for c in sorted(df_final['cluster'].unique()):
                if c == -1:
                    continue

                sub = df_final[df_final['cluster'] == c]

                usia_min = sub['usia_tahun'].min()
                usia_max = sub['usia_tahun'].max()
                usia_mean = sub['usia_tahun'].mean()

                # Jenis kelamin
                jk_prop = sub['jenis_kelamin'].value_counts(normalize=True) * 100
                jk_dominan = jk_prop.idxmax()
                jk_label = "Laki-laki" if jk_dominan.lower().startswith("l") else "Perempuan"

                # Status pengambilan
                status_prop = sub['status'].value_counts(normalize=True) * 100
                status_dominan = status_prop.idxmax()

                # Top 5 tempat lahir
                top_tempat = sub['tempat_lahir'].value_counts().head(5)
                tempat_teks = ", ".join(top_tempat.index)

                st.markdown(f"### Klaster {c}")
                st.markdown(f"- **Jumlah pemohon** : {len(sub)}")
                st.markdown(f"- **Rentang usia** : {usia_min} – {usia_max} tahun")
                st.markdown(f"- **Rata-rata usia** : {usia_mean:.2f} tahun")
                st.markdown(f"- **Jenis kelamin dominan** : {jk_label} ({jk_prop[jk_dominan]:.2f}%)")
                st.markdown(f"- **Status pengambilan dominan** : {status_dominan} ({status_prop[status_dominan]:.2f}%)")
                st.markdown(f"- **5 tempat lahir terbanyak** : {tempat_teks}")

            
            # ==============================
            # Download Excel
            # ==============================
            if st.session_state.get('merged') is not None and st.session_state.get('labels_rep') is not None and st.session_state.get('reduced_df') is not None:
                filename, buffer = save_excel_with_clusters(
                    original_df=st.session_state.get('clean_df'),
                    df_representative=st.session_state.get('reduced_df'),
                    labels=st.session_state['labels_rep'],
                    now=st.session_state['now'],
                    best_k=st.session_state.get('best_k') or st.session_state.get('sel_k')
                )
                st.download_button(
                    label="📥 Unduh Excel Hasil Klaster",
                    data=buffer,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    # -------------------------
    # VISUALISASI
    # -------------------------
    
    elif stage == "Visualisasi":
        st.header("📊 Visualisasi Hasil Klasterisasi")

        if st.session_state.get('df_final') is None:
            st.warning("Jalankan proses klasterisasi terlebih dahulu.")
            st.stop()

        df_final = st.session_state['df_final']

        # =========================
        # BOXPLOT USIA
        # =========================
        st.subheader("Boxplot Usia per Klaster")
        fig, ax = plt.subplots()
        df_final.boxplot(
            column='usia_tahun',
            by='cluster',
            ax=ax
        )
        ax.set_title("Boxplot Usia per Klaster")
        ax.set_xlabel("Klaster")
        ax.set_ylabel("Usia")
        plt.suptitle("")  # hapus judul default pandas

        st.pyplot(fig)
        plt.close(fig)


        # =========================
        # HISTOGRAM USIA
        # =========================
        st.subheader("Histogram Distribusi Usia per Klaster")
        fig = plt.figure()

        for cluster_id in sorted(df_final['cluster'].unique()):
            subset = df_final[df_final['cluster'] == cluster_id]
            plt.hist(
                subset['usia_tahun'],
                bins=20,
                alpha=0.6,
                label=f'Klaster {cluster_id}'
            )

        plt.xlabel('Usia')
        plt.ylabel('Frekuensi')
        plt.legend()
        st.pyplot(fig)
        plt.close(fig)

        # =========================
        # JENIS KELAMIN
        # =========================
        st.subheader("Distribusi Jenis Kelamin per Klaster")

        jk_cluster = (
            df_final
            .groupby(['cluster', 'jenis_kelamin'])
            .size()
            .unstack(fill_value=0)
        )

        fig, ax = plt.subplots()
        jk_cluster.plot(kind='bar', ax=ax)
        ax.set_xlabel('Klaster')
        ax.set_ylabel('Jumlah')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

        st.pyplot(fig)
        plt.close(fig)


        # =========================
        # STATUS PENGAMBILAN
        # =========================
        st.subheader("Distribusi Status Pengambilan per Klaster")

        status_cluster = (
            df_final
            .groupby(['cluster', 'status'])
            .size()
            .unstack(fill_value=0)
        )

        fig, ax = plt.subplots()
        status_cluster.plot(kind='bar', ax=ax)
        ax.set_xlabel('Klaster')
        ax.set_ylabel('Jumlah')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

        st.pyplot(fig)
        plt.close(fig)


        # =========================
        # TOP TEMPAT LAHIR
        # =========================
        st.subheader("Top 5 Tempat Lahir per Klaster")

        top_n = 5
        for cluster_id in sorted(df_final['cluster'].unique()):
            fig = plt.figure()

            subset = df_final[df_final['cluster'] == cluster_id]
            top_tl = subset['tempat_lahir'].value_counts().head(top_n)

            top_tl.plot(kind='bar')
            plt.title(f'Top {top_n} Tempat Lahir – Klaster {cluster_id}')
            plt.xlabel('Tempat Lahir')
            plt.ylabel('Jumlah')
            plt.xticks(rotation=45, ha='right')

            st.pyplot(fig)
            plt.close(fig)

        # =========================
        # SCATTER USIA (JITTER)
        # =========================
        st.subheader("Scatter Plot Usia Berdasarkan Klaster")

        fig = plt.figure()
        y_jitter = np.random.normal(0, 0.05, size=len(df_final))

        plt.scatter(
            df_final['usia_tahun'],
            y_jitter,
            c=df_final['cluster'],
            alpha=0.6
        )

        plt.xlabel('Usia')
        plt.ylabel('Indeks Dummy (Jitter)')
        plt.title('Scatter Plot Usia Berdasarkan Klaster')

        st.pyplot(fig)
        plt.close(fig)

        # =========================
        # DOWNLOAD PDF
        # =========================
        st.download_button(
            label="📄 Unduh PDF Visualisasi",
            data=save_visuals_pdf(
                df_final,
                st.session_state['now'],
                st.session_state.get('best_k') or 'manual'
            ),
            file_name=f"visualisasi_k{st.session_state.get('best_k','manual')}_{st.session_state['now']}.pdf",
            mime="application/pdf"
        )


# -------------------------
# MUAT ULANG PROSES
# -------------------------
elif main == "Muat Ulang Proses":
    if st.button("Muat Ulang Proses"):
        keys = [
            'raw_df','mapping','clean_df','reduced_df','preproc_report','df_final',
            'kproto_models','dbi_scores','labels_for_k','best_k',
            'merged','labels_rep','cluster_results','klaster_sudah_dijalankan'
        ]
        for k in keys:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state['now'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.success("Sudah direset. Muat ulang halaman atau pilih menu 'Beranda' untuk memulai lagi.")
