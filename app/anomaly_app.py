"""
Streamlit – Détection d'anomalies (v2 complète)
==============================================

Cette version finalise le script : tous les sous‑types, colonnes, chargement des
artefacts et affichage des résultats sont opérationnels.

---
Lancer :
```bash
pip install streamlit pandas numpy scikit-learn tensorflow
streamlit run anomaly_app.py
```
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

###############################################################################
# Dossiers et constantes
###############################################################################
BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

###############################################################################
# 1. Sélecteurs hiérarchiques (famille ➜ sous‑types)
###############################################################################
OPTIONS: Dict[str, List[str]] = {
    "UPS": ["UPS_bat", "UPS_I", "UPS_P", "UPS_V"],
    "PDU": ["pdu_fp", "pdu_i"],
    "MET 1": ["met_1_fp", "met_1_i", "met_1_it", "met_1_v"],
    "MET 2": ["met_2_fp", "met_2_i", "met_2_it", "met_2_v"],
}


###############################################################################
# 2. Mapping sous‑type ➜ (modèle | pipeline, scaler or None)
###############################################################################
MODEL_MAP: Dict[str, Tuple[str, Optional[str]]] = {
    # UPS (.keras + scaler)
    "UPS_bat": ("UPS_bat.keras", "scaler_UPS_bat.pkl"),
    "UPS_I":   ("UPS_I.keras",   "scaler_UPS_I.pkl"),
    "UPS_P":   ("UPS_P.keras",   "scaler_UPS_P.pkl"),
    "UPS_V":   ("UPS_V.keras",   "scaler_UPS_V.pkl"),

    # PDU (pipelines)
    "pdu_fp": ("pdu_fp.pkl", None),
    "pdu_i": ("pdu_i.pkl", None),

    # MET 1 (.keras + scaler)
    "met_1_fp": ("meter1_powerfactor.keras", "scaler_meter1_powerfactor.pkl"),
    "met_1_i":  ("meter1_current.keras",     "scaler_meter1_current.pkl"),
    "met_1_it": ("meter1_sequence.keras",    "scaler_meter1_sequence.pkl"),
    "met_1_v":  ("meter1_voltage.keras",     "scaler_meter1_voltage.pkl"),

    # MET 2
    "met_2_fp": ("met_2_fp.pkl", None),
    "met_2_it": ("met_2_it.pkl", None),
    "met_2_i":  ("meter2_current.keras", "scaler_meter2_current.pkl"),
    "met_2_v":  ("meter2_voltage.keras", "scaler_meter2_voltage.pkl"),
}

###############################################################################
# 3. Mapping sous‑type ➜ colonnes
###############################################################################
###############################################################################
# 3-bis.  Taille de fenêtre (time-window) par sous-type
###############################################################################
WINDOW_SIZE_MAP: Dict[str, int] = {
    # MET 1
    "met_1_v": 16,
    "met_1_i": 25,
    "met_1_fp": 12,
    "met_1_it": 16,
    # MET 2
    "met_2_v": 30,
    "met_2_i": 12,
    "met_2_fp": 10,
    "met_2_it": 10,      # ou autre valeur si différente
    # PDU
    "pdu_i": 8,
    "pdu_fp": 24,
    # UPS
    "UPS_V": 25,
    "UPS_I": 16,
    "UPS_P": 16,
    "UPS_bat": 25,
}

COLUMNS_MAP: Dict[str, List[str]] = {
    # MET 1
    "met_1_fp": [
        "m1_power_factor_a",
        "m1_power_factor_b",
        "m1_power_factor_c",
        "m1_total_power_factor",
    ],
    "met_1_i": [
        "m1_ia",
        "m1_ib",
        "m1_ic",
        "m1_i",
        "m1_in",
    ],
    "met_1_it": [
        "m1_ia_seq_p",
        "m1_ia_seq_n",
        "m1_ia_zero_n",
    ],
    "met_1_v": [
        "m1_va",
        "m1_vb",
        "m1_vc",
        "m1_v",
    ],
    # MET 2
    "met_2_fp": [
        "m2_power_factor_a",
        "m2_power_factor_b",
        "m2_power_factor_c",
        "m2_total_power_factor",
    ],
    "met_2_i": [
        "m2_ia",
        "m2_ib",
        "m2_ic",
        "m2_i",
        "m2_in",
    ],
    "met_2_it": [
        "m2_ia_seq_p",
        "m2_ia_seq_n",
        "m2_ia_zero_n",
    ],
    "met_2_v": [
        "m2_va",
        "m2_vb",
        "m2_vc",
        "m2_v",
    ],
    # PDU
    "pdu_fp": [f"pdu_{i}_fp" for i in range(1, 9)],
    "pdu_i": [f"pdu_{i}_i" for i in range(1, 9)],
    
    # UPS
   "UPS_bat": [
        "ups_battery_charge_current",
        "ups_battery_discharge_current",
        "ups_battery_capacity_ah",
        "ups_positive_cell_voltage",
        "ups_negative_cell_voltage",
    ],
    "UPS_I": ["ups_ia_output", "ups_ib_output", "ups_ic_output"],
    "UPS_P": ["ups_pa", "ups_pb", "ups_pc"],
    "UPS_V": [
        "ups_va_input", "ups_vb_input", "ups_vc_input",
        "ups_va_output", "ups_vb_output", "ups_vc_output",
    ],
}
COLUMNS_MAP["pdu_i"] = ["pdu_i"]
COLUMNS_MAP["pdu_fp"] = ["pdu_fp"]
WINDOW_SIZE_MAP["pdu_i"]  = 1     # pas de fenêtrage
WINDOW_SIZE_MAP["pdu_fp"] = 1

###############################################################################
# 4. Cache loaders
###############################################################################

@st.cache_resource(show_spinner=False)
def load_pipeline(pkl_path: Path):
    return joblib.load(pkl_path)


@st.cache_resource(show_spinner=False)
def load_keras_and_scaler(model_path: Path, scaler_path: Path):
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

###############################################################################
# 5. Feature extraction
###############################################################################
def extract_X(df: pd.DataFrame, subtype: str) -> np.ndarray:
    """
    Sélectionne les colonnes nécessaires, en tenant compte :
      • du fenêtrage (base   → base-1-t)
      • des alias met_  → m1_/m2_
      • de la casse (insensible)
    """
    base_cols = COLUMNS_MAP[subtype]
    win       = WINDOW_SIZE_MAP.get(subtype, 1)

    # --- Génère la liste attendue ---
    wanted = []
    for c in base_cols:
        if win > 1:
            wanted.extend([f"{c}-1-{t+1}" for t in range(win)])
        else:
            wanted.append(c)

    # --- Dictionnaire {lower: original} pour matcher sans casse -------------
    col_lookup = {c.lower(): c for c in df.columns}

    found, missing = [], []
    for w in wanted:
        key = w.lower()
        if key in col_lookup:
            found.append(col_lookup[key])
            continue
        # alias met_ ➜ m1_/m2_
        alias = key.replace("met_", "m1_").replace("met_", "m2_")
        if alias in col_lookup:
            found.append(col_lookup[alias])
        else:
            missing.append(w)

    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    return df[found].astype(float).to_numpy()

def window_dataframe(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Découpe df en fenêtres non chevauchantes et aplatit chaque fenêtre
    en une ligne (features × window_size).

    Les noms de colonnes deviennent « base-1-t » où t = 1..window_size.
    """
    n_rows = df.shape[0]
    n_win  = n_rows // window_size
    df_trunc = df.iloc[: n_win * window_size]

    data_3d = df_trunc.values.reshape(n_win, window_size, df.shape[1])          # (N, T, F)
    data_2d = data_3d.transpose(0, 2, 1).reshape(n_win, -1)                     # (N, F*T)

    new_cols = []
    for col in df.columns:
        base = col.replace("met_", "").replace("_1", "").replace("_2", "")
        for t in range(window_size):
            new_cols.append(f"{base}-1-{t+1}")

    return pd.DataFrame(data_2d, columns=new_cols)

###############################################################################
# 6. Analyse
###############################################################################

def analyse(df: pd.DataFrame, subtype: str) -> pd.DataFrame | None:
    if subtype not in MODEL_MAP:
        st.error(f"Sous‑type '{subtype}' non trouvé dans MODEL_MAP")
        return None

    model_name, scaler_name = MODEL_MAP[subtype]
    model_path = ARTIFACTS_DIR / model_name

    try:
        X = extract_X(df, subtype)
    except ValueError as err:
        st.error(str(err))
        return None

    # --- Pipeline (.pkl) ---
    if scaler_name is None:
        if not model_path.exists():
            st.error(f"Pipeline manquant : {model_path}")
            return None
        pipeline = load_pipeline(model_path)
        preds = pipeline.predict(X)
        rec_err = (
            np.mean(np.square(X - preds), axis=1)
            if preds.shape == X.shape
            else np.abs(preds).ravel()
        )
        df["reconstruction_error"] = rec_err
        return df

    # --- Keras + scaler ---
    scaler_path = ARTIFACTS_DIR / scaler_name
    if not model_path.exists() or not scaler_path.exists():
        st.error(f"Artefacts manquants : {model_path} ou {scaler_path}")
        return None

    model, scaler = load_keras_and_scaler(model_path, scaler_path)
    if X.shape[1] != model.input_shape[-1]:
        st.error(
            f"Mismatch features : modèle attend {model.input_shape[-1]} colonnes, CSV en a {X.shape[1]}"
        )
        return None

    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled, verbose=0)
    df["reconstruction_error"] = np.mean(np.square(X_scaled - preds), axis=1)
    return df

###############################################################################
# 7. Interface Streamlit
###############################################################################

st.set_page_config(page_title="Anomaly Detector", layout="wide")

st.title("Détection d'anomalies – Multi-sources")

famille = st.selectbox("Famille", list(OPTIONS.keys()))
subtype = st.selectbox("Sous-type", OPTIONS[famille], key=f"sub_{famille}")

uploaded_file = st.file_uploader("Déposez un CSV", type=["csv"], key=f"upload_{subtype}")

if uploaded_file is not None:
    st.write(f"Fichier : **{uploaded_file.name}** • Sous-type : **{subtype}**")
    try:
        df_raw = pd.read_csv(
            
            
    uploaded_file,
    parse_dates=["time"],
    infer_datetime_format=True,
    
)
        df_new=df_raw.copy()
        # ----------- Compactage df_new pour affichage rapide ----------- 
        N = 300  # taille du bloc pour affichage (réduction de bruit visuel)
        cols_new = df_new.columns.tolist()
        if "time" in cols_new:
            time_col_new = df_new["time"]
            data_cols_new = [c for c in cols_new if c != "time"]
            df_new_compact = (
                df_new[data_cols_new]
                .groupby(np.arange(len(df_new)) // N)
                .mean()
                .reset_index(drop=True)
            )
            times_new = time_col_new.groupby(np.arange(len(time_col_new)) // N).first().reset_index(drop=True)
            df_new_compact.insert(0, "time", times_new)
            df_new = df_new_compact
        else:
            df_new = (
                df_new
                .groupby(np.arange(len(df_new)) // N)
                .mean()
                .reset_index(drop=True)
            )
        # ---------------------------------------------------------

        win = WINDOW_SIZE_MAP.get(subtype, 1)   # 1 ⇒ pas de fenêtrage
        if win > 1:
            # On garde les colonnes utiles à l’entraînement
            cols_needed = COLUMNS_MAP[subtype]
            numeric_df  = df_raw[cols_needed]

            # transformation fenêtre -> ligne
            df_win = window_dataframe(numeric_df, win)

            # on récupère un timestamp par fenêtre (début)
            times = df_raw["time"].iloc[::win].reset_index(drop=True)
            df_win.insert(0, "time", times)

            df_raw = df_win        
            

# ----------- Compactage par blocs de 12 lignes -----------
        N = 300  # taille du bloc
        cols = df_raw.columns.tolist()
        if "time" in cols:
            time_col = df_raw["time"]
            data_cols = [c for c in cols if c != "time"]
            df_compact = (
                df_raw[data_cols]
                .groupby(np.arange(len(df_raw)) // N)
                .mean()
                .reset_index(drop=True)
            )
            # On choisit la valeur du temps du début du bloc (ou .last() si tu préfères la fin)
            times_new = time_col.groupby(np.arange(len(time_col)) // N).first().reset_index(drop=True)
            df_compact.insert(0, "time", times_new)
            df_raw = df_compact
        else:
            df_raw = (
                df_raw
                .groupby(np.arange(len(df_raw)) // N)
                .mean()
                .reset_index(drop=True)
            )
# ---------------------------------------------------------
    
    except Exception as e:
        st.error(f"Erreur lecture CSV : {e}")
        st.stop()

    # ----------- Nouveau : tracé brut des variables -----------
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_new = df_new.select_dtypes(include=[np.number]).columns.tolist()
    if 'time' in df_new.columns and numeric_cols_new:
        st.line_chart(df_new.set_index("time")[numeric_cols_new])
    else:
        st.warning("Pas de colonnes numériques ou colonne 'time' absente.")
   
   
    # ----------------------------------------------------------

    df_result = analyse(df_raw, subtype)
    if df_result is not None:
        st.subheader("Erreur de reconstruction / Anomalies")
        st.line_chart(df_result.set_index("time")["reconstruction_error"], height=300)
        st.dataframe(df_result.head(50), use_container_width=True)

        st.download_button(
            "📥 Télécharger le CSV annoté",
            data=df_result.to_csv(index=False).encode(),
            file_name=f"{Path(uploaded_file.name).stem}_annotated.csv",
            mime="text/csv",
        )
else:
    st.info("Choisissez le type de données puis importez un CSV.")

