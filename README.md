# Avocado Ripening Stage Detection & Classification

Computer vision pipeline to classify avocado ripening stages across 4 categories (`underripe`, `breaking`, `ripe`, `overripe`) using both Classical Machine Learning and Deep Learning (Transfer Learning).

---

## Project Structure

```
Greenpods/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda_and_data_cleaning.ipynb
│   ├── 02_approach_1_classical_ml.ipynb
│   ├── 03_approach_2_deep_learning.ipynb
│   └── 04_model_evaluation_and_demo.ipynb
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── classical_model.py
│   └── dl_model.py
├── models/
├── main_demo.py
├── generate_report.py
└── run_all.py
```

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/KalyanRamGoparaboina/Greenpods.git
cd Greenpods
```

### 2. Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Dataset

Place the extracted dataset into `data/raw/` with the following structure:
```
data/raw/
├── underripe/
├── breaking/
├── ripe/
└── overripe/
```

The dataset can also be downloaded directly using `src.dataset.download_dataset` or via Google Drive:
[Avocado Ripening Dataset Link](https://drive.google.com/file/d/1NMJ0FOxcWH1mwFFeERuY6fgEY-MVgeET/view)

---

## Execution

### Run via Notebooks
Run the notebooks in order:
1. `notebooks/01_eda_and_data_cleaning.ipynb`
2. `notebooks/02_approach_1_classical_ml.ipynb`
3. `notebooks/03_approach_2_deep_learning.ipynb`
4. `notebooks/04_model_evaluation_and_demo.ipynb`

### Run Complete End-to-End Pipeline
```bash
python run_all.py
```

### Run Single-Image Inference
```bash
python main_demo.py --image data/raw/ripe/sample.jpg
```
Or test on a random sample:
```bash
python main_demo.py --demo
```

---

## Results Summary

| Model | Accuracy | Macro F1-Score | Inference Latency (CPU) |
|---|---|---|---|
| SVM (CIELAB + LBP) | ~82% | ~0.80 | ~5 ms |
| Random Forest | ~78% | ~0.76 | ~3 ms |
| MobileNetV2 (Transfer Learning) | ~94% | ~0.93 | ~45 ms |

---

## Methodology

### Approach 1: Classical Computer Vision + ML
- **Features:** 90-dimensional descriptors combining CIELAB color histograms (L*, a*, b*), HSV (H, S), and Local Binary Patterns (LBP) texture features.
- **Classifiers:** Support Vector Machine (RBF kernel, balanced class weights) and Random Forest.
- **Benefits:** Low computational footprint and sub-10ms CPU inference.

### Approach 2: Deep Learning (Transfer Learning)
- **Architecture:** MobileNetV2 initialized with ImageNet pre-trained weights.
- **Strategy:** 2-phase fine-tuning — initial warmup on custom classification head, followed by end-to-end backbone fine-tuning with Adam optimizer.
- **Benefits:** Superior representation learning on subtle ripening transitions.

---

## References

- Howard et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*
- Ojala et al. (2002). *Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns*
- CIE (1976). *CIELAB Color Space*
