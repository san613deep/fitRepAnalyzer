# Barbell Exercise Tracking & Classification

A wearable-sensor machine-learning project that classifies barbell exercises and explores repetition counting from accelerometer and gyroscope time-series data.

The repository follows a research workflow: sensor data preparation, signal processing, feature extraction, model comparison and repetition analysis. It is a learning/portfolio project; model accuracy and generalization should be measured on held-out participants before being reported.

## Problem

Recognize exercises such as squat, bench press, deadlift and overhead press using motion signals from wearable sensors. The project uses time-domain, frequency-domain and dimensionality-reduction features to describe movement patterns.

## Workflow

```text
Raw accelerometer + gyroscope CSV files
          |
          v
Dataset parsing and timestamp alignment
          |
          v
Cleaning and outlier handling
          |
          v
Signal filtering and feature extraction
  ├── Butterworth low-pass filter
  ├── PCA
  ├── rolling-window statistics
  ├── FFT features
  └── KMeans cluster features
          |
          v
Participant-isolated train/test split
          |
          v
Classifier comparison and evaluation
          |
          v
Repetition-counting analysis
```

## Technology

- Python, Pandas and NumPy
- SciPy signal processing
- Scikit-learn (PCA, KMeans and classifiers)
- Matplotlib and Seaborn
- Pytest

## Project structure

```text
data/
  raw/MetaMotion/       wearable sensor CSV files
  interim/              generated intermediate feature datasets
src/
  data/
    make_dataset.py
  features/
    remove_outliers.py
    build_features.py
    TemporalAbstraction.py
    FrequencyAbstraction.py
    DataTransformation.py
    count_repetitions.py
  models/
    LearningAlgorithms.py
    split.py
    train_model.py
tests/
  test_model_split.py
  test_temporal_abstraction.py
requirements.txt
README.md
```

## Setup

Python 3.10+ is recommended.

Windows PowerShell:

```powershell
git clone https://github.com/san613deep/fitRepAnalyzer.git
cd fitRepAnalyzer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS / Linux:

```bash
git clone https://github.com/san613deep/fitRepAnalyzer.git
cd fitRepAnalyzer
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data

The project uses MetaMotion sensor CSV files. The raw data currently tracked in this repository is retained as the sample input. If you replace it, preserve the expected timestamp and sensor columns used by the parsing scripts.

## Run the pipeline

Run the scripts from the repository root in this order, after confirming that the expected input files are present:

1. Dataset preparation: `python src/data/make_dataset.py`
2. Outlier handling: `python src/features/remove_outliers.py`
3. Feature engineering: `python src/features/build_features.py`
4. Model training: `python src/models/train_model.py`
5. Repetition analysis: `python src/features/count_repetitions.py`

The feature-generation and model-training scripts currently contain exploratory plotting and fixed experiment settings. Review their input/output paths before running them in a different environment.

## Participant-level evaluation

Randomly splitting individual sensor rows can place measurements from the same participant in both train and test sets, producing optimistic results. The training workflow now uses `GroupShuffleSplit` with participant IDs as groups, while excluding participant, category and set identifiers from model features.

The selected feature list is retained from the training-time forward-selection result rather than replaced by a sample-specific hard-coded list. Forward selection iterates features in a stable column order for reproducibility.

## Run tests

```bash
python -m pytest tests/ -q
```

The unit tests cover:
- Disjoint participant groups between training and test partitions.
- Exclusion of metadata and target columns from model features.
- Rolling-window mean and standard-deviation calculations.

These tests do not run the full sensor-data pipeline or establish model accuracy. GitHub Actions runs the unit tests on pushes and pull requests.

## Limitations

- The end-to-end scripts are exploratory and include plotting and local data-path assumptions.
- Hyperparameters and signal-filter cutoffs require validation on the dataset and sensor sampling rates being used.
- Participant-isolated holdout is more informative than row-random splitting, but repeated group splits or leave-one-participant-out evaluation would provide stronger evidence.
- Repetition counting needs a separately documented ground-truth protocol and error analysis before accuracy claims are made.
- This project is a research/learning implementation, not a validated fitness or medical measurement product.

## Future improvements

- Refactor exploratory scripts into import-safe functions and a single configurable CLI.
- Add a participant-level cross-validation report with per-participant metrics.
- Track model artifacts and experiment configurations.
- Add repetition-count ground truth and MAE/error distributions.
- Add plots and a concise results table generated from a reproducible run.
