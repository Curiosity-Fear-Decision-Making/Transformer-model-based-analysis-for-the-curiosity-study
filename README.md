# Transformer model of curiosity–threat decision-making

## 1. Purpose

This model predicts whether participants choose to read a trivia fact despite a potential threat or skip it, using regional fMRI signals. It examines regional contributions to task-related representations and how changing attention between brain regions affects predicted choices.

## 2. Model and parameters

BOLD signals from all voxels within 18 ROIs and the supplied small-volume mask are used without PCA. Each trial is resampled to 18 time points. A voxel tokenizer converts regional signals into tokens, spatial embeddings encode ROI coordinates, and a transformer integrates information across regions.

Readout heads predict choice, curiosity, threat, social information, and mouse trajectories.

| Parameter | Value |
| --- | --- |
| Model dimension | 64 |
| Transformer layers | 1 |
| Attention heads | 4 |
| Feed-forward dimension | 128 |
| Dropout | 0.2 |
| BOLD time points per trial | 18 |
| Mouse trajectory points | 50 |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Weight decay | 0.02 |
| Batch size | 16 |
| Maximum epochs | 80 |
| Early-stopping patience | 15 |
| Group penalty / L1 penalty | 0 / 0.001 |
| Random seed | 42 |

Evaluation uses leave-one-subject-out cross-validation across 30 participants.

## 3. Running the analyses

### Generate results using an already-trained model

With trained model weights and the required analysis inputs available, run the Figure 5 scripts directly. Repeating fMRI preprocessing or model training is not required.

From the package folder, install the dependencies using Python 3.12:

```bash
python -m pip install -r requirements.txt
```

Run the analyses:

```bash
python fig5_transformer_beh.py
python fig5_attention.py
python fig5_activitymap.py
```

The scripts read trained models and participant-level results from `results_arch/d64_l1_h4/seed42/`. Keep the supplied `assets/` folder. The attention analysis also requires cached model inputs or prepared BOLD and behavioral data to rebuild them. These data and trained results are not included in this source-code package.

The behavioral and attention scripts display figures. The activity-map script saves NIfTI t maps and masks under `results_2ndlevel_arch/`.

### Train the model from the study data

This step is needed only if you want to train the model yourself rather than use existing trained results.

Download the behavioral and fMRI data from [Science Data Bank](https://doi.org/10.57760/sciencedb.0175h). Prepare these folders inside the package:

- `bold_signals_standard/`: the complete preprocessed 4D BOLD time series for run 3 of each participant, named `standard_bold_par01_s3.nii.gz` through `standard_bold_par30_s3.nii.gz`.
- `beh_fmri/`: matching behavioral records, including task timing, choices, self/risk ratings, and mouse trajectories for all 128 trials. Preserve the original filenames and formats.

If these prepared inputs are already available, use them directly. Otherwise, preprocess the raw fMRI data following the paper. Transform each participant’s FSL `filtered_func_data.nii.gz` time series to MNI152 2 mm space using the corresponding functional-to-standard registration.

The resulting BOLD images must match `assets/aal3_atlas.nii`: 91 × 109 × 91 voxels, 2 mm resolution, and the same voxel-to-world affine. Preserve volume order, TR (1 second), and alignment with behavioral timestamps. The transformer scripts do not perform raw fMRI preprocessing.

Check the prepared inputs:

```bash
CHECK_ONLY=1 bash configure_and_run_me_first.sh local
```

Train the selected model across all 30 cross-validation folds:

```bash
bash configure_and_run_me_first.sh local
```

Missing caches are rebuilt automatically in `cache_voxels/`. Trained models and predictions are saved under `results_arch/d64_l1_h4/seed42/`. After training, run the Figure 5 scripts above.

## 4. Attention and directional interactions

![Transformer architecture](transformer_architecture.gif)

Each regional token produces a query, a key, and a value. A receiving region’s query is compared with the keys of source regions. Softmax converts these scores into attention weights, which determine each source region’s contribution to the receiving region’s updated representation.

These weights are asymmetric: attention from A to B can differ from attention from B to A. The attention-modulation analysis changes a selected source-to-receiver score, recomputes the attention weights, and measures the resulting change in predicted choice.

This tests directed information exchange within the model. It does not establish anatomical directionality or causal influence between brain regions.
