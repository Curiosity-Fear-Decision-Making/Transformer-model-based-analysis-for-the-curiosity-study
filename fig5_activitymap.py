# Foldername: analysis_transformer_model_curiosity_project
# Filename: fig5_activitymap.py
from pathlib import Path
from time import perf_counter
import glob
import hashlib
import importlib.util
import os
import re as _re

from nibabel.processing import resample_from_to
from nilearn.image import resample_to_img
from scipy import stats
from scipy.ndimage import generate_binary_structure, label
import nibabel as nib
import numpy as np
import pandas as pd
import torch

from functools import partial

import sys

UTILITY_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
if not (UTILITY_DIR / 'utility_transformer.py').is_file():
    UTILITY_DIR = UTILITY_DIR / 'analysis_transformer_model_curiosity_project'
if not (UTILITY_DIR / 'utility_transformer.py').is_file():
    raise FileNotFoundError(f'utility_transformer.py not found in {UTILITY_DIR}. Set the Console working directory to the package or Dean_Code_c.')
sys.path.insert(0, str(UTILITY_DIR))

from utility_transformer import (
        activity_build_original_roi_label_lookup as build_original_roi_label_lookup,
        activity_build_roi_table as build_roi_table,
        activity_cache_id as cache_id,
        activity_calculate_held_out_linear_roi_contributions as calculate_held_out_linear_roi_contributions,
        activity_calculate_linear_roi_contributions as calculate_linear_roi_contributions,
        activity_contribution_roi_indices as contribution_roi_indices,
        activity_correct_roi_map_family as correct_roi_map_family,
        activity_find_matching_behavior_prediction_path as find_matching_behavior_prediction_path,
        activity_load_subject_voxels as load_subject_voxels,
        activity_load_trial_onsets_seconds as load_trial_onsets_seconds,
        activity_map_roi_to_merged_name as map_roi_to_merged_name,
        activity_resample_time_series as resample_time_series,
        activity_save_array_cache_atomically as save_array_cache_atomically,
        activity_shared_calculate_roi_one_sample_statistics as shared_calculate_roi_one_sample_statistics,
    )

if '__file__' in globals():
    SCRIPT_DIR = Path(__file__).resolve().parent
else:
    SCRIPT_DIR = Path.cwd()
    if not (SCRIPT_DIR / 'analysis_transformer_train.py').is_file():
        SCRIPT_DIR = SCRIPT_DIR / 'analysis_transformer_model_curiosity_project'
if not (SCRIPT_DIR / 'analysis_transformer_train.py').is_file():
    raise FileNotFoundError(
        'Cannot locate the transformer package. In PyCharm Console, set the working '
        'directory to analysis_transformer_model_curiosity_project or its parent directory.'
    )
MANUSCRIPT_DIR = Path(os.environ.get('MANUSCRIPT_DIR', str(SCRIPT_DIR)))
_HERE = str(SCRIPT_DIR)
DATA_DIR = Path(os.environ.get('TRANSFORMER_DATA_DIR', str(SCRIPT_DIR))).expanduser()
if 'TRANSFORMER_DATA_DIR' not in os.environ and not (DATA_DIR / 'results_arch').is_dir():
    existing = SCRIPT_DIR.parent / 'analysis_transformer_model_and_assets'
    if (existing / 'results_arch').is_dir():
        DATA_DIR = existing
# Reuse the exact architecture that trained this folder's checkpoints.
_model_spec = importlib.util.spec_from_file_location(
    'activity_training_model', Path(_HERE) / 'analysis_transformer_train.py'
)
training_model = importlib.util.module_from_spec(_model_spec)
_model_spec.loader.exec_module(training_model)
BOLD_TS_ROOT = os.environ.get('BOLD_TS_ROOT', str(SCRIPT_DIR / 'bold_signals_standard'))
BOLD_TS_PATTERN = 'standard_bold_par{sub_num:02d}_s3.nii.gz'
BEH_DIR = os.environ.get('BEH_DIR', str(SCRIPT_DIR / 'beh_fmri'))
NUM_TRIALS = 128
T_BOLD = int(os.environ.get('T_BOLD', '18'))
K_PCA = 0
TR_FALLBACK = 1.0


AAL3_PATH = os.path.join(_HERE, 'assets', 'aal3_atlas.nii')
SVC_MASK_PATH = os.path.join(_HERE, 'assets', 'svc_mask.nii')
LABEL_PATH = os.path.join(_HERE, 'assets', 'aal3_labels.tsv')
CACHE_DIR = os.environ.get('SMALLVOLUME_CACHE_DIR', os.path.join(DATA_DIR, 'cache_voxels'))
CACHE_PREFIX = 'smallvolume_exactsvc_allvoxels_v1'
MERGE_ROIS = bool(int(os.environ.get('MERGE_ROIS', '1')))
_MERGE_TAG = '_bilateral' if MERGE_ROIS else '_hemisphere'


SEED = 42
ARCH = 'd64_l1_h4'
SUBJ_DIR = os.path.join(DATA_DIR, 'results_arch', ARCH, f'seed{SEED}', 'per_subject')
OUT_DIR = os.path.join(_HERE, 'results_2ndlevel_arch')
N_PERM = 10000
PERM_SEED = 0
ALPHA = 0.05
SVC_MASK = os.path.join(_HERE, 'assets', 'svc_mask.nii')
EXCLUDE_ROI_PATTERNS = ('^Temporal_Pole',)
EXCLUDE_ROI = None
SVC_VOX = None
INFERENCE_BATCH_SIZE = 32
SELECTED_ARCH = ARCH


if SELECTED_ARCH is not None:
    ARCH = SELECTED_ARCH
SUBJ_DIR = os.path.join(DATA_DIR, 'results_arch', ARCH, f'seed{SEED}', 'per_subject')
OUT_DIR = os.path.join(_HERE, 'results_2ndlevel_arch')
files = sorted(glob.glob(os.path.join(SUBJ_DIR, 'contrib_*.npz')))
missing_inputs = []
for number in range(1, 31):
    subject = f'par{number:02d}'
    required = [
        Path(SUBJ_DIR) / f'contrib_{subject}.npz',
        Path(SUBJ_DIR) / f'behav_{subject}.npz',
        DATA_DIR / 'results_arch' / ARCH / f'seed{SEED}' / 'models' / f'loso_{subject}.pt',
    ]
    missing_inputs.extend(str(path) for path in required if not path.is_file())
if missing_inputs:
    raise FileNotFoundError(
        'Activity maps require trained results. Set TRANSFORMER_DATA_DIR to the folder '
        'containing results_arch, or run training first. Missing files:\n'
        + '\n'.join(missing_inputs)
    )
rt = build_roi_table(atlas_path=AAL3_PATH, context=globals())
labels = rt['labels']
n_roi = rt['n_roi']
expected = [f'par{i:02d}' for i in range(1, 31)]
found = [os.path.basename(f).removeprefix('contrib_').removesuffix('.npz') for f in files]
if found != expected:
    raise ValueError(f'Expected contribution files for {expected}; found {found}')
alignment = {}
for sub, f in zip(expected, files):
    checkpoint = os.path.join(
        DATA_DIR, 'results_arch', ARCH, f'seed{SEED}', 'models', f'loso_{sub}.pt'
    )
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    indices, trained_n = contribution_roi_indices(ck, rt, context=globals())
    alignment[sub] = indices
table = build_roi_table(context=globals())
nm = {
    int(roi_labels_lab): str(name) for roi_labels_lab, name in zip(table['labels'], table['names'])
}
names = [nm.get(int(labels[i]), f'L{int(labels[i])}') for i in range(n_roi)]
vol = np.zeros(tuple(rt['atlas_shape']), np.int16)
for lab, vox in zip(labels, rt['roi_voxels']):
    vol.ravel()[vox] = int(lab)
atlas = nib.Nifti1Image(vol, rt['affine'])
EXCLUDE_ROI = np.array([any((_re.search(p, nm_) for p in EXCLUDE_ROI_PATTERNS)) for nm_ in names])
atlas_img = atlas
svc = nib.load(SVC_MASK)
if svc.shape[:3] != atlas_img.shape[:3] or not np.allclose(
    svc.affine, atlas_img.affine, atol=0.001
):
    svc = resample_to_img(svc, atlas_img, interpolation='nearest')
SVC_VOX = np.squeeze(np.asarray(svc.dataobj)) > 0
interaction_trials = {}
for subject_index, sub in enumerate(expected, 1):
    started = perf_counter()
    interaction_trials[sub] = calculate_held_out_linear_roi_contributions(sub, rt, context=globals())
trial_counts = []
maps, subjects_by_map = ({}, {})
for choice_value, choice_tag in [(1, 'storyonly'), (0, 'skiponly'), (None, 'alltrial')]:
    for target, key in [
        ('curiosity', 'cur_contrib_trial'),
        ('threat', 'risk_contrib_trial'),
        ('interaction', None),
    ]:
        rows, subjects, counts = ([], [], [])
        for sub, f in zip(expected, files):
            if target == 'interaction':
                values = interaction_trials[sub]
            else:
                with np.load(f) as d:
                    values = d[key][:, alignment[sub]].astype(float)
            with np.load(find_matching_behavior_prediction_path(f, context=globals())) as d:
                choice = d['chose_story']
            selected = (
                np.ones(len(values), dtype=bool) if choice_value is None else choice == choice_value
            )
            n_selected = int(selected.sum())
            if target == 'curiosity':
                trial_counts.append(
                    dict(
                        subject=sub, choice=choice_tag, n_trials=n_selected, included=n_selected > 0
                    )
                )
            if n_selected == 0:
                continue
            counts.append(n_selected)
            subjects.append(sub)
            rows.append(values[selected].mean(axis=0))
        maps[target, choice_tag] = np.asarray(rows)
        subjects_by_map[target, choice_tag] = subjects
# Exploratory P95(|t|) across all six maps and included ROIs; not max-T FWER.
split_maps = {key: value for key, value in maps.items() if key[1] != 'alltrial'}
alltrial_maps = {key: value for key, value in maps.items() if key[1] == 'alltrial'}
correction = correct_roi_map_family(split_maps, subjects_by_map, expected, EXCLUDE_ROI, context=globals())
# All-trial effects are a separate exploratory family: three maps x 18 ROIs.
correction.update(correct_roi_map_family(alltrial_maps, subjects_by_map, expected, EXCLUDE_ROI, context=globals()))
tvalues = {}
for (target, choice_tag), matrix in maps.items():
    tag = f'{target}_{choice_tag}'
    M = matrix
    roi_statistics_labels = labels
    roi_statistics_names = names
    roi_statistics_atlas = atlas
    roi_statistics_subjects = subjects_by_map[target, choice_tag]
    roi_statistics_correction = correction[target, choice_tag]
    M = np.asarray(M, float)
    mean, tval, pval = shared_calculate_roi_one_sample_statistics(M, context=globals())
    p_perm95, t_crit = roi_statistics_correction
    significant = (
        np.isfinite(tval) & (tval > 0) & np.isfinite(p_perm95) & (p_perm95 < ALPHA) & ~EXCLUDE_ROI
    )
    atlas_values = np.squeeze(np.asarray(roi_statistics_atlas.dataobj)).astype(np.int32)
    tdata = np.zeros(atlas_values.shape, dtype=np.float32)
    for roi_statistics_j, roi_statistics_lab in enumerate(roi_statistics_labels):
        if np.isfinite(tval[roi_statistics_j]) and (not EXCLUDE_ROI[roi_statistics_j]):
            tdata[atlas_values == roi_statistics_lab] = tval[roi_statistics_j]
    if SVC_VOX is not None:
        tdata[~SVC_VOX] = 0
    selected_voxels = (
        np.isfinite(tdata) & (tdata > 0) & np.isin(atlas_values, roi_statistics_labels[significant])
    )
    components, n_clusters = label(selected_voxels, structure=generate_binary_structure(3, 1))
    mask = (components > 0).astype(np.uint8)
    output = Path(OUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'tmap_{tag}_{ARCH}_seed{SEED}_raw_nroi{n_roi}.nii.gz'
    image = nib.Nifti1Image(tdata, roi_statistics_atlas.affine)
    image.header.set_intent('t test', (len(roi_statistics_subjects) - 1,))
    nib.save(image, path)

    if choice_tag == 'alltrial':
        path = output / f'mask_permP95_05_{tag}_{ARCH}_seed{SEED}_permP95_nroi{n_roi}.nii.gz'
        nib.save(nib.Nifti1Image(mask, roi_statistics_atlas.affine), path)

    tvalues[target, choice_tag] = tval
combined_mask_correction = correction
combined_mask_labels = labels
combined_mask_atlas = atlas
combined_mask_n_roi = n_roi
combined_mask_tvalues = tvalues
output = Path(OUT_DIR)
output.mkdir(parents=True, exist_ok=True)
atlas_values = np.asarray(combined_mask_atlas.dataobj).astype(np.int32)
for combined_mask_target in ('curiosity', 'threat', 'interaction'):
    flags = {}
    for combined_mask_choice in ('storyonly', 'skiponly'):
        adjusted = np.asarray(
            combined_mask_correction[combined_mask_target, combined_mask_choice][0]
        )
        tval = np.asarray(combined_mask_tvalues[combined_mask_target, combined_mask_choice])
        flags[combined_mask_choice] = (
            np.isfinite(tval)
            & (tval > 0)
            & np.isfinite(adjusted)
            & (adjusted < ALPHA)
            & ~EXCLUDE_ROI
        )
    codes = flags['storyonly'].astype(np.uint8) + 2 * flags['skiponly'].astype(np.uint8)
    volume = np.zeros(atlas_values.shape, dtype=np.uint8)
    for combined_mask_lab, code in zip(combined_mask_labels, codes):
        volume[atlas_values == combined_mask_lab] = code
    if SVC_VOX is not None:
        volume[~SVC_VOX] = 0
    path = (
        output
        / f'mask_permP95_05_{combined_mask_target}_story_skip_{ARCH}_seed{SEED}_permP95_nroi{combined_mask_n_roi}.nii.gz'
    )
    nib.save(nib.Nifti1Image(volume, combined_mask_atlas.affine), path)
