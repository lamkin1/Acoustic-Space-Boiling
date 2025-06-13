# Acoustic-Space-Boiling

**Acoustic Data-based Characterization of Incipient Boiling for Space Applications**

This repository contains code, data, and documentation for a project aimed at detecting and characterizing incipient boiling in cryogenic fuel tanks using non-intrusive acoustic sensors. The project was developed in collaboration with NASA Ames Research Center.

---

## Getting Started: Download and Usage

To download and use the Acoustic-Space-Boiling app:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Acoustic-Space-Boiling.git
   cd Acoustic-Space-Boiling
   ```
2. **Install dependencies:**
   - Ensure you have Python 3.10+ installed.
   - Install required packages:
     ```bash
     pip install -r Production/requirements.txt
     ```
3. **Run the pipeline:**
   - To process new data and retrain models, run:
     ```bash
     python Production/retrain.py
     ```
    - You will be prompted to choose whether to run both supervised and unsupervised retraining, or just unsupervised retraining. Type `y` or `n` as instructed.
   - Outputs (features, models, plots) will be saved in `Production/app_resources/output/`.
4. **Launch the web app:**
   - Start the interactive web application:
     ```bash
     python Production/web_app/app.py
     ```
   - Open your browser and go to `http://localhost:5000` to explore results and visualizations.

For more details, see the documentation in the `Production/` folder and the comments in each script.

---

## Adding New Runs and Labels

To add new experimental runs and label them for supervised learning:

1. **Place new data files:**
   - Copy your new acoustic data CSV files into the `Production/new_runs/` directory.
2. **Prepare a label CSV:**
   - Create a CSV file in `Production/new_labels/` with two columns: `filename` and `label`.
   - The `filename` should be just the name of the CSV file (do not include any folder path), and should match the new run's CSV filename (up to the first period is used for matching).
   - The `label` should be an integer corresponding to the current label mapping (see the mapping in `Production/web_app/app.py` for the correspondence between integers and boiling regime categories, or to modify the mapping).
3. **Run the pipeline:**
   - Execute the retraining script:
     ```bash
     python Production/retrain.py
     ```
   - You will be prompted to choose whether to run both supervised and unsupervised retraining, or just unsupervised retraining. Type `y` or `n` as instructed.
   - The pipeline will automatically sort new runs into labeled and unlabeled directories, merge labels, extract features, and update all downstream outputs.

For more details on file formats and troubleshooting, see the documentation in the `Production/` folder.

---

## Project Motivation

Cryogenic fuel tanks store liquefied gases like oxygen and hydrogen for spacecraft. Even minor heat leaks in microgravity can trigger incipient boiling, leading to pressurization and catastrophic failure. Traditional thermal sensors often miss early boiling events due to low resolution and delayed response times.

Our approach uses **accelerometer-based acoustic sensing** to rapidly and reliably detect the onset of boiling—enabling safer cryogenic fuel management in space environments.

---

## Data Overview

- **441 acoustic experiments** conducted at NASA
- Data recorded at **10,000 Hz**
- Each CSV contains:
  - 2 channels (sensor amplitudes)
  - 1–30 seconds of signal data
- Labels correspond to **boiling regimes** (e.g., random, single rhythmic)

---

## Feature Engineering

Two main categories:

### Time-Domain Features
- Peak amplitudes and timing
- Time above threshold
- Peaks per second
- Peak time differences
- Dominant rhythmic patterns

### Frequency-Domain Features
- Power spectral density
- Spectral entropy
- Frequency centroid
- Top peak frequency
- Number of spectral peaks

---

## Rhythmic Pattern Detection Algorithm

A 5-step algorithm detects **dominant rhythmic patterns** in the acoustic data:

1. **Estimate noise level**
2. **Generate candidate rhythms**
3. **Count peak “hits” for each candidate**
4. **Prune weak candidates via statistical tests**
5. **Group similar candidates and extract dominant ones**

The result: number of statistically significant boilings per trial.

---

## ML Unsupervised Pipeline

1. **Feature Extraction** → `features.csv`
2. **Scaling** → for PCA + clustering
3. **PCA** → reduce noise & redundancy (6 PCs > 97% variance)
4. **K-Means Clustering** → unsupervised boiling regime grouping
5. **Classification** → supervised learning with decision trees

---

## ML Supervised Pipeline

1. **Label Training Data** → manually categorize boiling regimes (e.g., Single Rhythmic, Noise)
2. **Feature Extraction** → engineer time-domain and frequency-domain features
3. **Select Model** → evaluate classifiers; use decision trees for interpretability
4. **Fit Model** → train model on PCA-transformed feature set
5. **Test Model** → evaluate using precision, recall, and F1 score per class

---

## Supervised Learning Results

| Class             | Support | Precision | Recall | F1 Score |
|------------------|---------|-----------|--------|----------|
| Single Rhythmic  | 62      | 86.56%    | 85.48% | 85.22%   |
| Random           | 57      | 87.54%    | 80.70% | 83.36%   |
| Noise            | 31      | 90.16%    | 87.09% | 87.38%   |
| Other Classes    | –       | Lower performance due to underrepresentation |

---

## Interactive Web App

A demo web application allows interactive exploration of results:

- **2D/3D PCA visualizations**
- **Detailed time-domain signal plots**
- **Dynamic cluster selection and comparison tools**
- **Supervised learning model visualization**

---

## Authors

| Name               | Role                                              | Email                     |
|--------------------|---------------------------------------------------|---------------------------|
| **Zachary Weinfeld** | Acoustic feature engineering, supervised learning | zweinfeld@calpoly.edu     |
| **James Lamkin**     | Data pipeline, rhythmic detection algorithm      | lamkin@calpoly.edu        |
| **Krish Gupta**      | Web application, system integration              | kgupta15@calpoly.edu      |
| **Andrew Martinez**  | Statistical analysis, clustering                 | amart531@calpoly.edu      |

---

## How to Cite

If you use this work in your research, please cite our project as follows:
```
Krishnanshu Gupta, James Lamkin, Andrew Martinez, and Zachary Weinfeld.
Acoustic-Space-Boiling: Acoustic Feature-Based Characterization of Boiling Regimes in Cryogenic Fuel Tanks.
GitHub repository, 2025. Available at: https://github.com/lamkin1/Acoustic-Space-Boiling.
```
APA Citation: 
```
Gupta, K., Lamkin, J., Martinez, A., & Weinfeld, Z. (2025). Acoustic-Space-Boiling: Acoustic feature-based characterization of boiling regimes in cryogenic fuel tanks. GitHub. https://github.com/lamkin1/Acoustic-Space-Boiling
```
Or, include a link to our GitHub repository:

**[github.com/lamkin1/Acoustic-Space-Boiling](https://github.com/lamkin1/Acoustic-Space-Boiling)**
