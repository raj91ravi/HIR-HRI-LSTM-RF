# HIR-HRI-10K: Synthetic Multimodal Human Intention Recognition

**Research baseline: Long Short-Term Memory (LSTM) versus Random Forest for human–robot interaction (HRI)**

HIR-HRI-10K is a **synthetic, rule-generated** dataset of 10,000 frame-level observations for studying multimodal human intention recognition. This repository is intended for reproducible, offline machine-learning experiments and for exploring possible future applications in human-aware quadruped robots. **It does not contain recordings of people, real robot sensor measurements, or physical robot validation.**

Repository: https://github.com/raj91ravi/HIR-HRI-10K

## Dataset overview

| Property | Value |
|---|---|
| Observations | 10,000 synthetic frames |
| Independent episodes | 1,000 |
| Frames per episode | 10 |
| Sampling interval | 0.2 seconds |
| Numeric input features | 28 per frame |
| Intention categories | 12 |
| Train / validation / test | 7,000 / 1,500 / 1,500 frames |
| Episode-level split | 700 / 150 / 150 episodes |
| Evaluation unit | Episode (not individual frame) |

The notebook checks that episode and subject identifiers do not overlap across splits, and it fits the feature scaler on training data only. Intention categories are approximately balanced across the full dataset, but individual split proportions need not be identical.

### Intention categories

1. `walking`
2. `approaching_robot`
3. `avoiding_robot`
4. `requesting_assistance`
5. `object_handover`
6. `object_pickup`
7. `following_robot`
8. `stopping`
9. `greeting_robot`
10. `pointing`
11. `sitting_down`
12. `unknown_intention`

The features are structured numeric behavioral/multimodal descriptors, **not raw camera, audio, or wearable sensor recordings**. Consult the dataset documentation and CSV headers for the full feature definitions and generation rules.

## Repository layout

Recommended layout for running the supplied notebook:

```text
HIR-HRI-10K/
├── README.md
├── HIR_Ravi.ipynb
├── DATA_HIR/
│   ├── hir_multimodal_train.csv
│   ├── hir_multimodal_validation.csv
│   ├── hir_multimodal_test.csv
│   └── label_map.csv
└── results/                    # Generated after running notebook
    ├── figures/
    ├── model_comparison_test.csv
    ├── lstm_training_history.csv
    ├── lstm_model.pt
    ├── random_forest_and_scaler.pkl
    └── experiment_metadata.json
```

**Note:** This is a recommended layout based on the uploaded notebook, not a verification of the current remote GitHub file tree. The notebook reads CSVs from `DATA_HIR/`; create that directory or update the four `pd.read_csv` paths. It saves outputs under `results/` and creates `results/figures/` automatically.

## Run in Jupyter Notebook

```bash
git clone https://github.com/raj91ravi/HIR-HRI-10K.git
cd HIR-HRI-10K
python -m pip install numpy pandas scikit-learn torch matplotlib seaborn notebook
jupyter notebook
```

Open `HIR_Ravi.ipynb` and run the cells sequentially after placing the four CSV files in `DATA_HIR/`. The notebook uses random seed `42`; hardware, library versions, and numerical execution can still affect exact outcomes. It can use CUDA if available, otherwise CPU.

## Methods

### Random Forest baseline

Each 10-frame episode is standardized using the training-fitted scaler, then summarized into **84 features**: 28 temporal means, 28 temporal standard deviations, and 28 final-minus-initial differences. The classifier uses 250 trees, `class_weight='balanced_subsample'`, and validation selection across maximum depths `8`/`None` and minimum leaf sizes `1`/`3`.

### LSTM baseline

The LSTM consumes the **10 × 28** sequence directly. It uses one LSTM layer with 64 hidden units, layer normalization, dropout (0.25), and a 32-unit dense layer before the 12-class output. Training uses AdamW (learning rate 0.002), cross-entropy, batch size 64, a maximum of 65 epochs, and early stopping on validation macro-F1 with patience 10.

Both methods use the same episode partitions; the RF and LSTM have different input representations. Models are selected on validation performance and compared on held-out test episodes.

## Results from the supplied notebook

The following are **observed saved notebook outputs**, not independently rerun benchmarks:

| Metric (test, 150 episodes) | Random Forest | LSTM |
|---|---:|---:|
| Accuracy | 0.9733 | 0.9533 |
| Macro precision | 0.9737 | 0.9575 |
| Macro recall | 0.9737 | 0.9554 |
| Macro F1 | 0.9737 | 0.9556 |
| Weighted F1 | 0.9733 | 0.9536 |

Random Forest achieved higher accuracy and macro-F1 on this particular synthetic test split. These results should **not** be interpreted as evidence of superiority on real-world human intention recognition or quadruped robot deployment. The dataset is generated from handcrafted rules and may contain patterns that are easier to learn than natural human behavior.

## Evaluation and research figures

The notebook computes accuracy, precision, recall, macro/weighted F1, per-class classification reports, and the following visualizations:

- Class distribution and example temporal behavioral trends
- LSTM training/validation loss, accuracy, and macro-F1 curves
- Count and normalized confusion matrices for both models
- Comparative model metrics and per-class F1
- Micro-averaged one-vs-rest ROC and precision–recall curves
- Early-intention recognition performance using 1, 3, 5, 7, and 10 observed frames

Figures are saved as **300-dpi PNG** and **PDF** under `results/figures/`. The early-prediction analysis is exploratory: Random Forest is retrained for each prefix length, while the original LSTM is applied to truncated sequences without prefix-specific retraining. Therefore, it is **not a matched optimized early-prediction comparison**.

## Potential quadruped robotics application

A future quadruped robot, such as a Unitree Go2, might use an intention recognition component to support human-aware navigation, assistance requests, or interaction planning. A possible future architecture is:

```text
Human behavior / sensors
          ↓
Multimodal feature extraction
          ↓
Temporal intention recognition (LSTM or RF)
          ↓
Intent-aware decision / interaction planner
          ↓
Quadruped navigation or response
```

**This architecture is a proposed application perspective only.** The supplied experiments do not implement real-time inference, physical robot integration, safety assessment, or real-person validation.

## Reproducibility and responsible interpretation

- The samples and intention labels are synthetically generated by handcrafted rules.
- No real human-subject or real-world quadruped experiments are included.
- Report **episode-level** results, not frame-level results, to avoid inflating the effective test size.
- Keep the train/validation/test partitions fixed and avoid fitting preprocessing on validation/test data.
- Do not treat near-perfect synthetic classification as demonstrated real-world generalization.
- Before deploying a human-facing robot, collect ethically sourced real-world data, assess domain shift, calibrate confidence, and evaluate safety and fairness.

## Data availability

The dataset is associated with this repository: https://github.com/raj91ravi/HIR-HRI-10K. Confirm that the CSV files and documentation have been uploaded before describing the dataset as publicly downloadable in a manuscript.

## Citation

If you use this repository, please cite its URL and the relevant associated paper, **once a formal paper citation is available**. For an archival research release, consider publishing a versioned dataset with a DOI (for example through Zenodo).

## License

No license is asserted here. Add a `LICENSE` file specifying the terms under which others may reuse the data and code.

## Maintainer

GitHub: [raj91ravi](https://github.com/raj91ravi)
