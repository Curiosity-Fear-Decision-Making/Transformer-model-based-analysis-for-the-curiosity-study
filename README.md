## 3. Running the analyses

### Generate results using an already-trained model

With the trained model weights and required analysis inputs available, run the Figure 5 scripts directly. Repeating fMRI preprocessing or model training is not required.

Install the dependencies with Python 3.12:

```bash
python -m pip install -r requirements.txt
```

Run the analyses from the package folder:

```bash
python fig5_transformer_beh.py
python fig5_attention.py
python fig5_activitymap.py
```

The scripts use model results under `results_arch/d64_l1_h4/seed42/`. The behavioral and attention scripts display figures; the activity-map script saves NIfTI t maps and masks under `results_2ndlevel_arch/`.

Keep the supplied `assets/` folder. The attention analysis also needs cached model inputs or the prepared BOLD and behavioral data to rebuild them. Source code and atlas assets alone are insufficient to generate the results.

### Train the model from the study data

This step is needed only if you want to train the model yourself rather than use existing trained results.

Download the behavioral and fMRI data from [Science Data Bank](https://doi.org/10.57760/sciencedb.0175h). Prepare these input folders:

- `bold_signals_standard/`: preprocessed 4D BOLD time series for run 3, named `standard_bold_par01_s3.nii.gz` through `standard_bold_par30_s3.nii.gz`.
- `beh_fmri/`: matching behavioral records, including task timing, choices, self/risk ratings, and mouse trajectories for all 128 trials, with their original filenames and formats.

If these prepared inputs are already available, use them directly. Otherwise, preprocess the raw fMRI data following the paper and transform each participant’s `filtered_func_data.nii.gz` to MNI152 2 mm space using the corresponding functional-to-standard registration.

The BOLD images must match `assets/aal3_atlas.nii`: 91 × 109 × 91 voxels, 2 mm resolution, and the same voxel-to-world affine. Preserve volume order, TR (1 second), and alignment with behavioral timestamps.

Check the inputs and train:

```bash
CHECK_ONLY=1 bash configure_and_run_me_first.sh local
bash configure_and_run_me_first.sh local
```

This trains the selected model across 30 leave-one-subject-out folds. Missing caches are rebuilt automatically, and model results are saved under `results_arch/d64_l1_h4/seed42/`. Then run the Figure 5 scripts above.
