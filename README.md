# ⚡  POWER_AI_WISE – Anomaly Detection for Data Center Energy Systems

## Description

PowerAI is a collaborative project between efrei students and efrei research lab  aiming to detect anomalies in the energy consumption and power flow of a data center located in brazil. Using unsupervised deep learning models such as Autoencoders and LSTM, this project focuses on preprocessing high-frequency multi-source electrical data, applying advanced anomaly detection techniques, and delivering a real-time capable solution.

> Final goal: Build a robust anomaly detection system integrated into a Streamlit web interface for industrial monitoring.

---

## Data Description

* **Source**: Internal monitoring system of a real-world data center.
* **Format**: SQLite `.db` files → converted into unified `.csv` via custom extraction script.
* **Size**:

  * `704,248` rows
  * `165` columns
* **Timeframe**: From `2024-10-31` to `2025-02-21`
* **Components Analyzed**:

  * **Meter 1 & 2** (voltage, current, power factor, sequences)
  * **UPS** (battery levels, input/output flow, modes)
  * **PDUs** (current and power factor distribution)

Each component was split into functional groups and analyzed separately.

---

## Methodology

### Data Preprocessing

* Merging tables from `.db` files
* Handling timestamps, null values, duplicate rows
* Splitting into functional subsets: Meter 1, Meter 2, UPS, PDU
* Applying window-based transformation (2–6 minutes)

### Modeling Techniques

| Model            | Description                                 | Used For                     |
| ---------------- | ------------------------------------------- | ---------------------------- |
| Isolation Forest | Tree-based outlier detection                | Simple point anomalies       |
| Autoencoder      | Dense NN for reconstruction error scoring   | Pattern-based anomalies      |
| LSTM Autoencoder | Time series memory for sequential detection | Contextual temporal patterns |

* All models trained on "normal" periods only
* Evaluation based on expert-provided anomaly logs
* Metrics: **Precision**, **Recall**, **F1-score**

---

## Results

* Models compared across multiple time windows (2min, 3min, 5min)
* Voltage-based Autoencoder models achieved best performance on Meter 1
* Final models saved (`.keras` and `.pkl`) and used in Streamlit interface

📌 *See technical report for detailed evaluations, plots, and architecture.*

---

## Web Application

* **Frontend**: Streamlit
  Allows uploading `.csv` subsets and viewing anomalies on plots

* **Backend**: Streamlit
  Dynamically loads the correct model & scaler, applies anomaly detection


## Repository Structure

```
Anomaly_detetction_models/      # main notebooks of the anomaly detection models used on different components 
docs/                      # extra documentation
app/                      # Streamlit code
Technical_Report.pdf      # Detailed report
README.md
```

---

## 👥 Contributors

| Name                     | Role                              |
| ------------------------ | --------------------------------- |
| **Amira Boudaoud**       | Référente de l’équipe — Data & AI |
| **Ghofrane Ben Rhaiem**  | Data & AI                         |
| **Antonin Timbert**      | Data Engineering                  |
| **Mohamed Talhi**        | Business Intelligence & Analysis  |
| **Hind Karim El Alaoui** | Business Intelligence & Analysis  |

---

## License

This project is for academic purposes (WISE 2025 — Industrial AI). Please contact the authors for usage outside this scope.
