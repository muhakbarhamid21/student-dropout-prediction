import pickle

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Prediksi Dropout Mahasiswa",
    layout="wide"
)

@st.cache_resource
def load_model(model_path: str):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model = load_model('model/cat_base_optuna.pkl')

app_mode_options = {
    1: 'Tahap 1 - Kontingen umum',
    2: 'Ordinansi No. 612/93',
    5: 'Tahap 1 - Kontingen khusus (Azores)',
    7: 'Pemegang kursus tinggi lain',
    10: 'Ordinansi No. 854-B/99',
    15: 'Mahasiswa internasional (sarjana)',
    16: 'Tahap 1 - Kontingen khusus (Madeira)',
    17: 'Tahap 2 - Kontingen umum',
    18: 'Tahap 3 - Kontingen umum',
    26: 'Ordinansi No. 533-A/99, b2 (Rencana Berbeda)',
    27: 'Ordinansi No. 533-A/99, b3 (Institusi Lain)',
    39: 'Usia di atas 23 tahun',
    42: 'Transfer',
    43: 'Ganti jurusan',
    44: 'Pemegang diploma spesialisasi teknis',
    51: 'Ganti institusi/jurusan',
    53: 'Pemegang diploma siklus singkat',
    57: 'Ganti institusi/jurusan (Internasional)'
}

prev_qual_options = {
    1: 'Pendidikan menengah',
    2: 'Sarjana (Bachelor)',
    3: 'Gelar tinggi (Degree)',
    4: 'Magister (Master)',
    5: 'Doktor (Doctorate)',
    6: 'Frekuensi pendidikan tinggi',
    9: 'Kelas 12 - belum lulus',
    10: 'Kelas 11 - belum lulus',
    12: 'Lainnya - kelas 11',
    14: 'Kelas 10',
    15: 'Kelas 10 - belum lulus',
    19: 'Pendidikan dasar siklus 3 (9-11)',
    38: 'Pendidikan dasar siklus 2 (6-8)',
    39: 'Kursus spesialisasi teknis',
    40: 'Gelar tinggi siklus 1',
    42: 'Kursus teknis profesional',
    43: 'Magister siklus 2'
}

with st.sidebar:
    st.title("Jaya Jaya Institut")
    st.subheader("Prediksi Dropout")
    st.markdown("---")
    st.write(
        "Aplikasi untuk memprediksi risiko dropout mahasiswa menggunakan model **CatBoost** (Optuna)."
    )
    st.markdown("**Versi:** 1.0.0")
    st.markdown("[Dokumentasi](https://github.com/muhakbarhamid21/student-dropout-prediction/blob/main/README.md)")
    st.markdown("[Source Code](https://github.com/muhakbarhamid21/student-dropout-prediction)")
    st.markdown("---")
    st.write("**Kontak & Dukungan**:")
    st.write("✉️ support@jayajaya.ac.id")
    st.write("© 2025 Jaya Jaya Institut")

st.title("Prediksi Risiko Dropout Mahasiswa")
st.write("Silakan isi data mahasiswa pada form di bawah, lalu klik **PREDIKSI** untuk melihat hasil.")

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
with col2:
    age_at_enrollment = st.number_input("Usia Saat Pendaftaran", min_value=16, max_value=100, value=18)
with col3:
    marital_status = st.selectbox("Status Pernikahan", ["Belum Menikah","Menikah","Duda/Janda","Cerai","Kehidupan Faktual","Pemisahan Hukum"])

col1, col2, col3 = st.columns(3)
with col1:
    previous_qualification = st.selectbox("Kualifikasi Sebelumnya", options=list(prev_qual_options.keys()), format_func=lambda x: f"{x} - {prev_qual_options[x]}")
with col2:
    scholarship_holder = st.selectbox("Penerima Beasiswa", ["Tidak","Ya"])
with col3:
    tuition_fees_up_to_date = st.selectbox("Pembayaran UKT Lancar", ["Tidak","Ya"])

col1, col2, col3 = st.columns(3)
with col1:
    application_mode = st.selectbox("Mode Pendaftaran", options=list(app_mode_options.keys()), format_func=lambda x: f"{x} - {app_mode_options[x]}")
with col2:
    displaced = st.selectbox("Pindah Domisili", ["Tidak","Ya"])
with col3:
    debtor = st.selectbox("Tunggakan Biaya", ["Tidak","Ya"])

col1, col2 = st.columns(2)
with col1:
    curricular_units_1st_sem_approved = st.number_input("SKS Disetujui Semester 1 (0 - 60)", min_value=0, max_value=60, value=0)
with col2:
    curricular_units_2nd_sem_approved = st.number_input("SKS Disetujui Sem 2 (0 - 60)", min_value=0, max_value=60, value=0)

col1, col2 = st.columns(2)
with col1:
    raw_grade1 = st.text_input("Nilai Rata-Rata Semester 1 (0 – 20)", value="0.0")
    try:
        curricular_units_1st_sem_grade = float(raw_grade1)
        if curricular_units_1st_sem_grade < 0 or curricular_units_1st_sem_grade > 20:
            st.error("Nilai semester harus di antara 0 dan 20")
    except:
        curricular_units_1st_sem_grade = 0.0
        if raw_grade1:
            st.error("Masukkan angka desimal yang valid ")
with col2:
    raw_grade2 = st.text_input("Nilai Rata-Rata Semester 2 (0 – 20)", value="0.0")
    try:
        curricular_units_2nd_sem_grade = float(raw_grade2)
        if curricular_units_2nd_sem_grade < 0 or curricular_units_2nd_sem_grade > 20:
            st.error("Nilai semester harus di antara 0 dan 20")
    except:
        curricular_units_2nd_sem_grade = 0.0
        if raw_grade2:
            st.error("Masukkan angka desimal yang valid ")

st.markdown("---")

st.markdown(
    """
    <style>
    .stButton > button {
        width: 100% !important;
        font-size: 1.2rem !important;
        padding: 0.5rem 1rem !important;
        color: white !important;
        background-color: red !important;
        border: none !important;
        border-radius: 5px !important;
        cursor: pointer !important;
        transition: background-color 0.3s ease !important;
        font-weight: bold !important;
    }
    .stButton > button:hover {
        background-color: darkred !important;
    }
    .stButton > button:active {
        transform: scale(0.98) !important;
        background-color: #ff3333 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

predict = st.button("PREDIKSI")

if predict:
    inputs = {
        'Gender': 1 if gender == 'Laki-laki' else 0,
        'Age_at_enrollment': age_at_enrollment,
        'Marital_status': ["Belum Menikah","Menikah","Duda/Janda","Cerai","Kehidupan Faktual","Pemisahan Hukum"].index(marital_status) + 1,
        'Previous_qualification': previous_qualification,
        'Scholarship_holder': 1 if scholarship_holder == 'Ya' else 0,
        'Tuition_fees_up_to_date': 1 if tuition_fees_up_to_date == 'Ya' else 0,
        'Application_mode': application_mode,
        'Displaced': 1 if displaced == 'Ya' else 0,
        'Debtor': 1 if debtor == 'Ya' else 0,
        'Curricular_units_1st_sem_approved': curricular_units_1st_sem_approved,
        'Curricular_units_1st_sem_grade': curricular_units_1st_sem_grade,
        'Curricular_units_2nd_sem_approved': curricular_units_2nd_sem_approved,
        'Curricular_units_2nd_sem_grade': curricular_units_2nd_sem_grade
    }
    df = pd.DataFrame([inputs])

    proba = model.predict_proba(df)[0]
    classes = list(model.classes_)

    drop_idx = classes.index(1)
    grad_idx = classes.index(0)

    p_drop = proba[drop_idx]
    p_grad = proba[grad_idx]
    total = p_drop + p_grad
    p_drop_norm = p_drop / total if total > 0 else 0
    p_grad_norm = p_grad / total if total > 0 else 0

    st.markdown("---")

    st.subheader("Hasil Prediksi")
    if p_drop_norm > p_grad_norm:
        st.error("Mahasiswa diprediksi DROPOUT / KELUAR")
    else:
        st.success("Mahasiswa diprediksi GRADUATE / LULUS")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Probabilitas Dropout", f"{p_drop_norm:.2%}")
    with col2:
        st.metric("Probabilitas Lulus", f"{p_grad_norm:.2%}")

    st.markdown("---")
    with st.expander("Data Masukan"):
        st.dataframe(df)
