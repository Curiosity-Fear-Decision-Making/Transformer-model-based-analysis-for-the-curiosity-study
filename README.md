# Transformer model of curiosity–threat decision-making

## 1. Purpose

This model predicts participants’ choices from regional fMRI signals during a curiosity–threat decision-making task. Participants choose whether to read a trivia fact despite a potential threat or skip it.

The model learns representations related to choice, curiosity, threat, social information, and mouse trajectories. Analyses examine regional contributions to these representations and how changing attention between brain regions affects predicted choices.

## 2. Model and parameters

Preprocessed BOLD signals are extracted from all voxels within 18 regions of interest, constrained by the supplied small-volume mask. No PCA is applied. Each trial is resampled to 18 time points.

A voxel tokenizer converts each region’s signals into a token. Spatial embeddings derived from ROI coordinates are added, and a transformer encoder exchanges information between regional tokens. Readout heads predict choice and the auxiliary targets.

| Parameter | Value |
|---|---|
| Brain regions | 18 |
| BOLD time points per trial | 18 |
| Model dimension | 64 |
| Transformer layers | 1 |
| Attention heads | 4 |
| Feed-forward dimension | 128 |
| Dropout | 0.2 |
| Mouse trajectory length | 50 points |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Weight decay | 0.02 |
| Batch size | 16 |
| Maximum epochs | 80 |
| Early-stopping patience | 15 |
| Group penalty | 0 |
| L1 penalty | 0.001 |
| Random seed | 42 |

Evaluation uses leave-one-subject-out cross-validation across 30 participants. Each fold holds out one participant for testing.

## 3. Reproduction

Install the dependencies using Python 3.12:

```bash
pip install -r requirements.txt
```

Provide the preprocessed BOLD data and behavioral files, including mouse trajectories. Keep the supplied atlas and ROI assets in `assets/`. Raw data, trained weights, and large caches are not included in this repository.

Set the input paths and Python executable:

```bash
export PYTHON_BIN="$(command -v python)"
export BOLD_TS_ROOT="/path/to/bold_signals_standard"
export BEH_DIR="/path/to/beh_fmri"
```

Check the inputs:

```bash
CHECK_ONLY=1 bash configure_and_run_me_first.sh local
```

Train the model:

```bash
bash configure_and_run_me_first.sh local
```

This runs the selected model configuration across all 30 cross-validation folds. Missing voxel caches are rebuilt automatically from the input data. Training results are saved under `results_arch/d64_l1_h4/seed42/`.

After training, run:

```bash
python fig5_transformer_beh.py
python fig5_attention.py
python fig5_activitymap.py
```

The behavioral and attention scripts display figures with `plt.show()`. The activity-map script generates NIfTI t maps and masks in `results_2ndlevel_arch/`.

## 4. Attention and directional interactions

![Transformer architecture](transformer_architecture.gif)

Each brain-region token produces a query, a key, and a value. A receiving region’s query is compared with the keys of source regions. Softmax converts these comparisons into attention weights, which determine how strongly each source region’s value contributes to the receiving region’s updated representation.

Attention is asymmetric: the weight from region A to region B need not equal the weight from B to A. This allows the model to represent directed information exchange between regional representations.

The attention-modulation analysis changes a selected source-to-receiver attention score, recomputes the softmax weights, and measures the resulting change in predicted choice. It therefore tests the behavioral influence of a directed connection within the model.

These directions describe model computations. They do not establish anatomical directionality or causal influence between brain regions.
