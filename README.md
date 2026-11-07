# Transformer model of curiosity–threat decision-making

## 1. Purpose

This model uses regional fMRI signals to predict whether participants choose to read a trivia fact despite a potential threat or skip it. Auxiliary predictions include curiosity, threat, social information, and mouse trajectories.

The analyses examine regional contributions and how changing attention between brain regions affects predicted choices.

## 2. Model and parameters

The model uses all voxels within 18 ROIs constrained by the supplied small-volume mask, without PCA. Each trial’s BOLD segment is resampled to 18 time points. A voxel tokenizer creates regional tokens, spatial embeddings encode ROI coordinates, and a transformer exchanges information between tokens before readout.

| Parameter | Value |
| --- | --- |
| Model dimension | 64 |
| Transformer layers | 1 |
| Attention heads | 4 |
| Feed-forward dimension | 128 |
| Dropout | 0.2 |
| BOLD time points per trial | 18 |
| Mouse trajectory points per trial | 50 |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Weight decay | 0.02 |
| Batch size | 16 |
| Maximum epochs | 80 |
| Early-stopping patience | 15 |
| Group penalty / L1 penalty | 0 / 0.001 |
| Random seed | 42 |

Evaluation uses leave-one-subject-out cross-validation across 30 participants. Each fold holds out one participant for testing.

## 3. Reproduction

### Download and prepare the data

Download the behavioral and fMRI data from [Science Data Bank](https://doi.org/10.57760/sciencedb.0175h). Use the fMRI participants and their matching behavioral records, preserving participant correspondence from `par01` to `par30`.

Create the following input folders inside the package:

```text
analysis_transformer_model_curiosity_project/
├── assets/
├── bold_signals_standard/
│   ├── standard_bold_par01_s3.nii.gz
│   ├── standard_bold_par02_s3.nii.gz
│   └── ... standard_bold_par30_s3.nii.gz
├── beh_fmri/
│   ├── output_formal_fix_par01.txt
│   ├── ... other behavioral files for par01–par30
│   └── output_par01_mousePos/
│       ├── trial_00.txt
│       └── ... trial_127.txt
├── configure_and_run_me_first.sh
├── transformer_architecture.gif
└── ... Python scripts and requirements.txt
```

**BOLD inputs**

Each `standard_bold_parXX_s3.nii.gz` contains the complete preprocessed 4D BOLD time series for the participant’s third run, corresponding to the decision task. These files are not raw scans, beta maps, t maps, or ROI-averaged signals.

1. Preprocess the downloaded fMRI data following the procedure described in the paper.
2. In the original FSL workflow, use the preprocessing FEAT output `filtered_func_data.nii.gz`.
3. Apply the corresponding functional-to-standard registration, `reg/example_func2standard.mat`, to every volume using the MNI152 2 mm reference grid.
4. Save the registered 4D time series using the filenames shown above.

The registration must correspond to the same participant and run. The resulting images must have a spatial shape of **91 × 109 × 91**, 2 mm voxels, and the same voxel-to-world affine as `assets/aal3_atlas.nii`.

Preserve the original volume order and TR in the NIfTI header; the original run has a TR of 1 second. Behavioral timestamps must remain aligned with the BOLD volumes. The loader calculates trial onsets from column 3 minus column 2 of `output_formal_fix_parXX.txt`, in milliseconds. It extracts each trial from its onset to the next trial’s onset, with the final trial ending at the last volume, and resamples each segment to 18 time points.

The transformer scripts begin with these prepared inputs. They do not perform raw fMRI preprocessing.

**Behavioral inputs**

Place the matching behavioral files in `beh_fmri/`, retaining their original names and tab-separated, headerless format. Each participant has 128 task trials.

| File pattern | Required contents |
| --- | --- |
| `output_formal_fix_parXX.txt` | Run reference and trial onset timestamps |
| `output_formal_FLevel_parXX.txt` | Objective threat level, column 2 |
| `output_formal_CLevel_parXX.txt` | Social-information level, column 2 |
| `output_formal_clickbaitLabel_parXX.txt` | Trivia identifier, column 2 |
| `output_formal_clickbaitRes_parXX.txt` | Choice, column 2; 1 indicates story |
| `output_formal_ori1_parXX.txt` | Task orientation variable, column 2 |
| `output_formal_ori2_parXX.txt` | Task orientation variable, column 2 |
| `output_formal_oriSpace_parXX.txt` | Task orientation variable, column 2 |
| Files beginning `output_self_rating_parXX` | Trivia identifier and self-rating, columns 2 and 3 |
| Files beginning `output_risk_riskLevel_parXX` | Risk-rating level, column 2 |
| Files beginning `output_risk_ratingResponse_parXX` | Risk-rating response, column 2 |
| `output_parXX_mousePos/trial_00.txt` through `trial_127.txt` | Mouse x/y coordinates in the first two columns |

Column numbers start at 1. Mouse-trial filenames start at 0. Keep one matching self/risk-rating file per participant and file type, and retain all supplied files in `assets/`.

### Install and run

Run the following commands from the package folder using Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Check the prepared inputs.
CHECK_ONLY=1 bash configure_and_run_me_first.sh local

# Train the selected model across all 30 cross-validation folds.
bash configure_and_run_me_first.sh local

# Generate the Figure 5 analyses.
python fig5_transformer_beh.py
python fig5_attention.py
python fig5_activitymap.py
```

Training uses CPU by default. For CUDA:

```bash
DEVICE=cuda bash configure_and_run_me_first.sh local
```

Missing caches are rebuilt automatically in `cache_voxels/`. Trained models and predictions are saved to `results_arch/d64_l1_h4/seed42/`.

The behavioral and attention scripts display figures. The activity-map script saves NIfTI t maps and masks to `results_2ndlevel_arch/`. Raw data, caches, and trained results are not included in this repository.

## 4. Attention and directional interactions

![Transformer architecture](transformer_architecture.gif)

Each regional token produces a query, a key, and a value. A receiving region’s query is compared with the keys of source regions. Softmax converts these scores into attention weights, determining how strongly each source region contributes to the receiving region’s updated representation.

Attention is asymmetric: the weight from region A to region B need not equal the weight from B to A. This allows the model to represent directed information exchange between regional representations.

The attention-modulation analysis changes a selected source-to-receiver attention score, recomputes the softmax weights, and measures the resulting change in predicted choice. It tests the behavioral influence of a directed connection within the model.

These directions describe model computations; they do not establish anatomical directionality or causal influence between brain regions.
