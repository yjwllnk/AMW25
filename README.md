# AMW25_readme
# AMW25 — Symmetry Fingerprinting for Thermoelectric Transport ML (KRICT ChemDX Hackathon 2025)

> **Team AMW25 (Team 4)**: Dr. Abhijeet · Yujin/Will Kang · Chaehyeon Moon  
> **Goal**: Explore how **crystal symmetry operations** (space-group “ingredients”) relate to **transport properties** (including **lattice thermal conductivity**) using **machine learning**.

---

## ✨ Motivation

Transport properties in crystalline solids are governed by both chemistry and structure. In particular, **lattice thermal conductivity** is strongly influenced by how phonons propagate and scatter—processes that can be constrained by **crystal symmetry**.

✅ **Working idea**  
There may be meaningful relationships between:
- **Transport coefficients** (e.g., thermal conductivity, electrical conductivity, Seebeck coefficient), and
- **Crystal symmetry elements / symmetry operations** derived from the material’s **space group**.
<p align="center">
  <img src="https://github.com/user-attachments/assets/b74de0dd-9dea-43c3-92c3-ded7c7d61e47" alt="Objective 1" width="49%" />
</p>

We therefore investigate whether symmetry-derived descriptors can provide predictive signal—especially in combination with conventional composition-based descriptors.

---

## 🧭 Approach

We extend LitDX entries with additional descriptors and build a feature table for ML.

### 1) Add symmetry information to LitDX (space group → symmetry operations)
- We enrich each material entry with **space-group information** and corresponding **symmetry-operation descriptors**, then encode them as **symmetry fingerprints**.

### 2) Add composition-based descriptors (Matminer)
- We also generate **elemental / composition fingerprints** using **Matminer**, and combine them with symmetry fingerprints to form a unified feature set.

---

## 🤖 Modeling (brief)

- We train regression models (e.g., **XGBoost**) on the generated fingerprints to predict transport coefficients.
- We compare multiple feature sets (symmetry-only vs. Matminer-only vs. combined) and evaluate model performance using standard regression metrics.
- We also perform quick screening analyses to check whether relationships appear primarily linear or require non-linear modeling.


---

## 📈 Results (summary)

> **Metric note**: We report **MAPE (Mean Absolute Percentage Error)**, a unitless error metric that is comparable across properties with different units. fileciteturn2file0

Below, we summarize the model evaluation trends observed in the project slides (symmetry-only vs. Matminer-only vs. combined). 

### ⚡ Electrical conductivity
- Using **symmetry-only** or **Matminer-only** features gives **poor predictive performance**.
- Using the **combined feature set (symmetry + Matminer)** improves the prediction **substantially** compared to either component alone.   
  → This suggests an **interaction effect**: conductivity appears to benefit from using symmetry information **in the context of** composition descriptors.

<img width="3937" height="1608" alt="Electric conductivity" src="https://github.com/user-attachments/assets/48f918b0-a8bb-4ac2-8c89-9221ae66bac6" />

### 🔥 Thermal conductivity
- **Symmetry-only** features show **weak performance**.
- When **Matminer descriptors** are included, the agreement becomes **much better**.  
  → Practical takeaway: accurate thermal conductivity modeling likely depends strongly on **phonon-related physics**, for which composition/chemistry descriptors provide essential signal.

<img width="3937" height="1608" alt="Thermal conductivity" src="https://github.com/user-attachments/assets/f378050d-73e7-45d2-ac27-ff03f53a8057" />


### 🧊 Seebeck coefficient
- The overall trend is similar to thermal conductivity, but **symmetry-only performance is not bad**. 
  → This indicates the Seebeck coefficient may be relatively **robust** and partially captured by symmetry-derived descriptors, with additional gains from composition features.

<img width="3937" height="1608" alt="Seebeck coefficient" src="https://github.com/user-attachments/assets/07f54b1e-73df-4599-bb3c-d5ed9b652106" />

---
---

## 📦 Installation

### Option A) Install via pip
```bash
git clone git@github.com:boffintocoffin/AMW25.git
cd AMW25
pip install .
```

### Option B) Create a clean environment (recommended)
```bash
micromamba create -n amw python=3.11
micromamba activate amw
pip install .
```

---

## ⚙️ Requirements
```bash
pip install -r requirements.txt
```

---

## 🚀 Quickstart (CLI)

```bash
amw25 --cwd <path_to_working_directory> --config <configuration_file> --mode <feature_mode>
```

---

## 🗂️ Input Data Format

Expected CSV layout under your configured data directory:

```
X_train_<mode>.csv
X_val_<mode>.csv
X_test_<mode>.csv
y_train.csv
y_val.csv
y_test.csv
```

---

## 🧾 Outputs & Logging
✅ Parsed configuration is saved to:
```
<cwd>/parsed_config.yaml
```

✅ Training logs (stdout/stderr) are redirected to:
```
<cwd>/stdout.x
```

---

## 🧑‍🤝‍🧑 Credits
Built for **2025 KRICT ChemDX Hackathon** — Team AMW25 (Team 4).
