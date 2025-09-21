# streamlit_app.py — Dhvani-Shield final demo app
# Put this file in your project root: C:\Projects\Dhvani-Shield

import streamlit as st
import joblib, os, time, hashlib, tempfile, json, sqlite3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import qrcode
import torch, torchaudio
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc
from math import sqrt

# ---------------- Config (edit paths if you put files elsewhere) ----------------
ROOT = Path.cwd()
FEATURES_DIR = ROOT / "ASVspoof_outputs" / "features" / "ASVspoof2019_global"
CSV_BASE = ROOT / "ASVspoof_outputs" / "CSVs"
DB_PATH = ROOT / "biometrics_audit.db"
REPORTS_DIR = ROOT / "reports"
TEMP_FEATS_DIR = ROOT / "temp_feats"
REPORTS_DIR.mkdir(exist_ok=True)
TEMP_FEATS_DIR.mkdir(exist_ok=True)
# -----------------------------------------------------------------------------

# ---------- Helper: DB (biometrics + audit) ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS voices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        note TEXT,
        created_at TEXT,
        sha TEXT,
        emb BLOB,
        emb_dim INTEGER,
        file_path TEXT
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT,
        action TEXT,
        user TEXT,
        filename TEXT,
        sha TEXT,
        details TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_audit(action, user, filename, sha, details):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO audit (ts, action, user, filename, sha, details) VALUES (?,?,?,?,?,?)",
              (time.ctime(), action, user or "unknown", filename or "", sha or "", json.dumps(details)))
    conn.commit()
    conn.close()

def store_embedding_in_db(name, emb: np.ndarray, audio_path: str = None, note: str = ""):
    sha = None
    try:
        with open(audio_path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        sha = hashlib.sha256((name+str(time.time())).encode()).hexdigest()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    emb32 = emb.astype(np.float32)
    blob = sqlite3.Binary(emb32.tobytes())
    c.execute("INSERT INTO voices (name, note, created_at, sha, emb, emb_dim, file_path) VALUES (?,?,?,?,?,?,?)",
              (name, note, time.ctime(), sha, blob, int(emb32.shape[0]), audio_path or ""))
    conn.commit()
    conn.close()
    save_audit("enroll", "ui", audio_path or name, sha, {"name": name})
    return sha

def load_all_db_embeddings():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, sha, emb_dim, emb FROM voices")
    rows = c.fetchall()
    conn.close()
    out = []
    for r in rows:
        bid, name, sha, dim, blob = r
        arr = np.frombuffer(blob, dtype=np.float32).reshape((dim,))
        out.append({"id": bid, "name": name, "sha": sha, "emb": arr})
    return out

# ---------- Audio & HuBERT helpers ----------
@st.cache_resource
def load_hubert(device_str):
    device = torch.device(device_str)
    bundle = torchaudio.pipelines.HUBERT_BASE
    model = bundle.get_model().to(device).eval()
    return bundle, model, device

def extract_embedding_from_audio_bytes(audio_bytes, bundle, model, device, chunk_size=160000):
    """
    Write bytes to a temp file, convert via ffmpeg to 16k mono WAV,
    load waveform, chunk it, run HuBERT and average chunk embeddings.
    """
    import subprocess, librosa
    with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmpf:
        tmpf.write(audio_bytes); tmpf.flush(); tmp_path = tmpf.name
    out_wav = tmp_path + ".wav"
    ffmpeg_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                  "-i", tmp_path, "-ar", str(bundle.sample_rate), "-ac", "1", out_wav]
    converted = False
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        converted = True
    except Exception:
        converted = False
    # load
    try:
        if converted:
            waveform, sr = torchaudio.load(out_wav)
        else:
            y, sr = librosa.load(tmp_path, sr=bundle.sample_rate)
            waveform = torch.tensor(y).unsqueeze(0)
    except Exception as e:
        try:
            # final fallback
            import soundfile as sf
            y, sr = sf.read(tmp_path)
            waveform = torch.tensor(y).unsqueeze(0)
        except Exception as e2:
            for p in (tmp_path, out_wav):
                try: os.remove(p)
                except: pass
            raise RuntimeError("Failed to load audio. Install ffmpeg or try different file.") from e2

    if waveform.ndim > 1 and waveform.shape[0] != 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

    embeddings = []
    with torch.no_grad():
        w = waveform.to(device)
        # split into chunks (samples)
        for i in range(0, w.shape[1], chunk_size):
            chunk = w[:, i:i+chunk_size]
            if chunk.shape[1] < 4000:
                continue
            feat_list, _ = model.extract_features(chunk)
            emb = feat_list[-1].squeeze(0).mean(0).cpu().numpy()
            embeddings.append(emb.astype(np.float32))
    for p in (tmp_path, out_wav):
        try: os.remove(p)
        except: pass
    if len(embeddings) == 0:
        raise RuntimeError("Audio too short for embedding.")
    return np.mean(embeddings, axis=0)

# ---------- PDF report ----------
def make_spectrogram_png(waveform, sr):
    # waveform: 1D numpy
    fig, ax = plt.subplots(figsize=(6,2))
    ax.specgram(waveform, NFFT=1024, Fs=sr, noverlap=512)
    ax.axis('off')
    buf = BytesIO()
    fig.savefig(buf, bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf

def make_pdf_report(audio_name, audio_bytes, label, prob, details: dict, out_pdf_path: Path):
    sha = hashlib.sha256(audio_bytes).hexdigest()
    # attempt to make spectrogram
    try:
        import librosa
        y, sr = librosa.load(BytesIO(audio_bytes), sr=16000)
        spec_png = make_spectrogram_png(y, sr)
    except Exception:
        spec_png = None

    c = canvas.Canvas(str(out_pdf_path), pagesize=A4)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, 800, "Dhvani-Shield — Forensic Detection Report")
    c.setFont("Helvetica", 11)
    c.drawString(40, 780, f"File: {audio_name}")
    c.drawString(40, 765, f"SHA256: {sha}")
    c.drawString(40, 750, f"Result: {'REAL' if label==0 else 'SPOOF'}")
    c.drawString(40, 735, f"Confidence: {prob:.2%}")
    c.drawString(40, 720, f"Generated: {time.ctime()}")

    # details table
    y0 = 690
    for k,v in details.items():
        c.drawString(40, y0, f"{k}: {v}")
        y0 -= 14
        if y0 < 120: break

    # spectrogram
    if spec_png is not None:
        img_path_tmp = out_pdf_path.with_suffix(".spec.png")
        with open(img_path_tmp, "wb") as f:
            f.write(spec_png.read())
        try:
            c.drawImage(str(img_path_tmp), 350, 480, width=200, height=120)
        except Exception:
            pass
        try: os.remove(img_path_tmp)
        except: pass

    # QR code
    qr_img = qrcode.make(f"{audio_name}|{sha}|{label}|{prob:.4f}")
    qr_path = out_pdf_path.with_suffix(".qr.png")
    qr_img.save(qr_path)
    c.drawImage(str(qr_path), 40, 520, width=120, height=120)
    c.showPage()
    c.save()
    try: os.remove(qr_path)
    except: pass

# ---------- Utility: similarity ----------
def cosine_similarity(a, b):
    # a, b: 1-D numpy
    an = a / (np.linalg.norm(a) + 1e-12)
    bn = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(an, bn))

# ---------- UI & logic ----------
init_db()

st.set_page_config(page_title="Dhvani-Shield — Demo", layout="wide", page_icon="🔊")
st.sidebar.title("Settings & Tools")

# Model selection
model_mode = st.sidebar.radio("Model:", ("Baseline", "Augmented"), index=0)
if model_mode == "Augmented":
    SVM_PATH = ROOT / "svm_model_augmented.pkl"
    SCALER_PATH = ROOT / "scaler_augmented.pkl"
    CSV_PATH = CSV_BASE / "asvspoof_augmented_merged.csv"
else:
    SVM_PATH = ROOT / "svm_model.pkl"
    SCALER_PATH = ROOT / "scaler.pkl"
    CSV_PATH = CSV_BASE / "asvspoof_partial_merged.csv"

device_choice = "cuda:0" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"Inference device: `{device_choice}`")
bundle, hubert_model, device = load_hubert(device_choice)

# threshold slider
threshold = st.sidebar.slider("Spoof threshold (probability of SPOOF)", 0.01, 0.99, 0.30, 0.01)

# Evaluate dataset button (danger: heavy). provide warn.
eval_btn = st.sidebar.button("Evaluate model on CSV (quick metrics)")

# Enrollment / DB operations
st.title("🔊 Dhvani-Shield — Deepfake Audio Detector (demo)")
c1, c2 = st.columns([1, 1.2])

with c1:
    uploaded = st.file_uploader("Upload audio", type=["wav","mp3","m4a","flac"])
    save_embedding = st.checkbox("Save embedding to temp_feats/ (debug)", value=False)
    run_detect = st.button("Run detection") if uploaded else None

with c2:
    st.subheader("Database / Forensics")
    enroll_name = st.text_input("Enroll voice: Name")
    enroll_file = st.file_uploader("Upload audio to enroll (for speaker db)", type=["wav","mp3","m4a","flac"], key="enroll")
    enroll_note = st.text_input("Note (optional)")
    if st.button("Enroll voice"):
        if enroll_name.strip() == "" or enroll_file is None:
            st.error("Provide both name and audio file to enroll.")
        else:
            try:
                audio_bytes = enroll_file.read()
                emb = extract_embedding_from_audio_bytes(audio_bytes, bundle, hubert_model, device)
                # store embedding
                tmp_path = TEMP_FEATS_DIR / f"enroll_{int(time.time())}_{enroll_file.name}"
                with open(tmp_path, "wb") as f: f.write(audio_bytes)
                sha = store_embedding_in_db(enroll_name.strip(), emb, str(tmp_path), note=enroll_note)
                st.success(f"Enrolled `{enroll_name}` — sha: {sha[:12]}...")
            except Exception as e:
                st.error(f"Enroll failed: {e}")

    st.markdown("### Verify / Identify")
    db_entries = load_all_db_embeddings()
    names = [f"{r['id']}:{r['name']}" for r in db_entries]
    verify_select = st.selectbox("Select enrolled individual (for verification)", options=["-- none --"] + names)
    verify_file = st.file_uploader("Upload audio to verify (against selected)", type=["wav","mp3","m4a","flac"], key="verify")
    if st.button("Run verification"):
        if verify_select == "-- none --" or verify_file is None:
            st.error("Select a person and upload audio.")
        else:
            sel_id = int(verify_select.split(":")[0])
            target = next((r for r in db_entries if r["id"] == sel_id), None)
            try:
                audio_bytes = verify_file.read()
                emb = extract_embedding_from_audio_bytes(audio_bytes, bundle, hubert_model, device)
                score = cosine_similarity(emb, target["emb"])
                st.write(f"Similarity (cosine): {score:.4f}")
                passed = score >= 0.7
                st.success("VERIFICATION PASSED" if passed else "VERIFICATION FAILED")
                save_audit("verify", "ui", verify_file.name, hashlib.sha256(audio_bytes).hexdigest(),
                           {"target_id": sel_id, "score": score, "passed": passed})
            except Exception as e:
                st.error(f"Verification failed: {e}")

    st.markdown("### Identify")
    identify_file = st.file_uploader("Upload audio to identify (search DB)", type=["wav","mp3","m4a","flac"], key="identify")
    if st.button("Run identification"):
        if identify_file is None:
            st.error("Upload audio to identify.")
        else:
            audio_bytes = identify_file.read()
            try:
                emb = extract_embedding_from_audio_bytes(audio_bytes, bundle, hubert_model, device)
                all_db = load_all_db_embeddings()
                sims = [(r["name"], cosine_similarity(emb, r["emb"])) for r in all_db]
                sims = sorted(sims, key=lambda x: x[1], reverse=True)[:10]
                st.table([{"name": s[0], "score": f"{s[1]:.4f}"} for s in sims])
                save_audit("identify", "ui", identify_file.name, hashlib.sha256(audio_bytes).hexdigest(),
                           {"top_matches": sims[:5]})
            except Exception as e:
                st.error(f"Identify failed: {e}")

# load model & scaler (must exist)
if not SVM_PATH.exists() or not SCALER_PATH.exists():
    st.warning("svm_model.pkl or scaler.pkl not found for the selected model. Train/export first.")
    st.stop()

svm = joblib.load(SVM_PATH)
scaler = joblib.load(SCALER_PATH)

# Detection UI
if uploaded and st.button("▶️ Run detection now"):
    audio_bytes = uploaded.read()
    sha = hashlib.sha256(audio_bytes).hexdigest()
    start = time.time()
    try:
        emb = extract_embedding_from_audio_bytes(audio_bytes, bundle, hubert_model, device)
        if save_embedding:
            np.save(TEMP_FEATS_DIR / (uploaded.name + ".npy"), emb)
        # check scaler dim
        if hasattr(scaler, "mean_") and scaler.mean_.shape[0] != emb.shape[0]:
            # dimension mismatch -> show option
            st.warning(f"Scaler expects {scaler.mean_.shape[0]} dims but embedding is {emb.shape[0]} dims.")
            st.info("You can retrain exporter or use fallback normalization (mean/std of embedding). Using fallback now.")
            emb_scaled = (emb - emb.mean()) / (emb.std() + 1e-12)
            emb_scaled = emb_scaled.reshape(1, -1)
        else:
            emb_scaled = scaler.transform(emb.reshape(1, -1))

        probs = svm.predict_proba(emb_scaled)[0]
        prob_real, prob_spoof = float(probs[0]), float(probs[1])
        if prob_spoof >= threshold:
            pred = 1; label_txt = "⚠️ SPOOF (Deepfake)"; conf = prob_spoof
        else:
            pred = 0; label_txt = "✅ REAL (Bonafide)"; conf = prob_real

        st.markdown(f"### {label_txt} — Confidence: {conf:.2%}")
        st.write("Raw probabilities (real, spoof):", np.round(probs, 4))
        st.progress(float(prob_spoof))

        # save audit & PDF
        details = {"prob_real": prob_real, "prob_spoof": prob_spoof, "model": str(SVM_PATH.name)}
        save_audit("predict", "ui", uploaded.name, sha, details)
        out_pdf = REPORTS_DIR / f"report_{int(time.time())}_{uploaded.name}.pdf"
        make_pdf_report(uploaded.name, audio_bytes, pred, conf, details, out_pdf)
        with open(out_pdf, "rb") as f:
            st.download_button("Download PDF report", data=f.read(), file_name=out_pdf.name)
        st.success(f"Report saved: {out_pdf.name} (also in reports/ folder)")

    except Exception as e:
        st.error(f"Detection failed: {e}")

# dataset evaluation (quick)
if eval_btn:
    st.sidebar.info("Running evaluation — this will load CSV and run predictions (can be heavy).")
    try:
        df = None
        if CSV_PATH.exists():
            import pandas as pd
            df = pd.read_csv(CSV_PATH)
        else:
            st.sidebar.error(f"CSV not found: {CSV_PATH}")
            df = None
        if df is not None:
            st.sidebar.write(f"Rows in CSV: {len(df)}")
            # collect embeddings
            X, y = [], []
            missing = []
            for _, row in df.iterrows():
                stem = row['videoname']
                npy = FEATURES_DIR / f"{stem}.npy"
                if not npy.exists():
                    missing.append(stem); continue
                arr = np.load(npy)
                if arr.ndim > 1: arr = arr.mean(axis=0)
                X.append(arr); y.append(int(row['label']))
            X = np.vstack(X)
            st.sidebar.write(f"Found embeddings: {X.shape[0]}, missing: {len(missing)}")
            # scale check
            if hasattr(scaler, "mean_") and scaler.mean_.shape[0] != X.shape[1]:
                st.sidebar.warning("Scaler dimension mismatch vs dataset embeddings; aborting eval.")
            else:
                Xs = scaler.transform(X)
                y_pred = svm.predict(Xs)
                y_prob = svm.predict_proba(Xs)[:,1]
                st.write("=== Classification report on CSV ===")
                st.text(classification_report(y, y_pred, target_names=["Real (0)","Spoof (1)"]))
                cm = confusion_matrix(y, y_pred)
                st.write("Confusion matrix:\n", cm)
                roc = roc_auc_score(y, y_prob)
                st.write("ROC AUC:", roc)
    except Exception as e:
        st.sidebar.error(f"Eval failed: {e}")

st.sidebar.markdown("---")
st.sidebar.write("Actions:")
if st.sidebar.button("Open reports folder"):
    st.sidebar.write(f"Reports are in: {REPORTS_DIR}")

st.caption("Dhvani-Shield — demo UI. For production, sign reports & deploy behind auth.")
