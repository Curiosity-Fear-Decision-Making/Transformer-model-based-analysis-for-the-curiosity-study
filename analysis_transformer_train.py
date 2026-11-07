# Foldername: analysis_transformer_model_curiosity_project
# Filename: analysis_transformer_train.py
from pathlib import Path
import argparse
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import sys

UTILITY_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
if not (UTILITY_DIR / 'utility_transformer.py').is_file():
    UTILITY_DIR = UTILITY_DIR / 'analysis_transformer_model_curiosity_project'
if not (UTILITY_DIR / 'utility_transformer.py').is_file():
    raise FileNotFoundError(f'utility_transformer.py not found in {UTILITY_DIR}. Set the Console working directory to the package or Dean_Code_c.')
sys.path.insert(0, str(UTILITY_DIR))

from utility_transformer import BoldToChoice, roi18_load_roi18_table

from functools import partial

from utility_transformer import (
        training__atomic_npz as _atomic_npz,
        training__beh_arr as _beh_arr,
        training__ensure_dirs as _ensure_dirs,
        training__ensure_training_dirs as _ensure_training_dirs,
        training__find_dir as _find_dir,
        training__fit_one_fold as _fit_one_fold,
        training__fix_onsets_sec as _fix_onsets_sec,
        training__group_index as _group_index,
        training__load_pooled as _load_pooled,
        training__minmax01 as _minmax01,
        training__read_beh_col as _read_beh_col,
        training__resample_trajectory as _resample_trajectory,
        training__resample_ts as _resample_ts,
        training__resolve_device as _resolve_device,
        training__roi_name_map as _roi_name_map,
        training__subject_cache_path as _subject_cache_path,
        training__to_tensor as _to_tensor,
        training__zscore as _zscore,
        training_build_roi_table as build_roi_table,
        training_build_subject as build_subject,
        training_cache_id as cache_id,
        training_compute_losses as compute_losses,
        training_find_subject_cache as find_subject_cache,
        training_load_subject_behavior as load_subject_behavior,
        training_load_subject_mouse as load_subject_mouse,
        training_load_subject_voxels as load_subject_voxels,
        training_make_batch as make_batch,
        training_partial_r as partial_r,
        training_roi_network_map as roi_network_map,
        training_run_epoch as run_epoch,
        training_sparse_group_penalty as sparse_group_penalty,
        training_train_loocv as train_loocv,
        training_verify_inputs as verify_inputs,
    )

if '__file__' in globals():
    _HERE = Path(__file__).resolve().parent
else:
    _HERE = Path.cwd()
    if not (_HERE / 'analysis_transformer_train.py').is_file():
        _HERE = _HERE / 'analysis_transformer_model_curiosity_project'
if not (_HERE / 'analysis_transformer_train.py').is_file():
    raise FileNotFoundError(
        'Cannot locate the transformer package. In PyCharm Console, set the working '
        'directory to analysis_transformer_model_curiosity_project or its parent directory.'
    )


roi18_ROOT = Path(_HERE).resolve()
roi18_ATLAS_PATH = roi18_ROOT / 'assets/aal3_atlas.nii'
roi18_SVC_MASK_PATH = roi18_ROOT / 'assets/svc_mask.nii'
roi18_LABEL_PATH = roi18_ROOT / 'assets/aal3_labels.tsv'
roi18_MANIFEST_PATH = roi18_ROOT / 'assets/roi_manifest.json'
roi18_ROI_NAMES = (
    'ACC_pre',
    'ACC_sub',
    'ACC_sup',
    'Amygdala',
    'Caudate',
    'Frontal_Med_Orb',
    'Frontal_Sup_Medial',
    'Hippocampus',
    'Insula',
    'OFCant',
    'OFCmed',
    'OFCpost',
    'Olfactory',
    'Putamen',
    'Rectus',
    'SN_pc',
    'VTA',
    'Vent_Str',
)
roi18_ROI_LABELS = dict(
    zip(
        roi18_ROI_NAMES,
        (
            'pgACC',
            'sgACC',
            'dACC',
            'AMY',
            'CAU',
            'ORBsupmed',
            'dmPFC',
            'HPC',
            'INS',
            'aOFC',
            'mOFC',
            'pOFC',
            'OLF',
            'PUT',
            'vmPFC',
            'SNc',
            'VTA',
            'NAc',
        ),
    )
)
roi18_ROI_NETWORK_LABELS = roi18_ROI_LABELS
roi18_ROI_FULL_NAMES = dict(
    zip(
        roi18_ROI_NAMES,
        (
            'Pregenual anterior cingulate cortex',
            'Subgenual anterior cingulate cortex',
            'Supracallosal anterior cingulate cortex',
            'Amygdala',
            'Caudate nucleus',
            'Superior frontal gyrus, medial orbital part',
            'Superior frontal gyrus, medial part',
            'Hippocampus',
            'Insula',
            'Anterior orbitofrontal cortex',
            'Medial orbitofrontal cortex',
            'Posterior orbitofrontal cortex',
            'Olfactory cortex',
            'Putamen',
            'Gyrus rectus',
            'Substantia nigra pars compacta',
            'Ventral tegmental area',
            'Nucleus accumbens',
        ),
    )
)
roi18_SCHEMA_POSITIONS = {
    'ACC_pre': (47, 17),
    'ACC_sub': (28, 4),
    'ACC_sup': (35, 31),
    'Amygdala': (-5, -38),
    'Caudate': (2, 36),
    'Frontal_Med_Orb': (57, 1),
    'Frontal_Sup_Medial': (54, 48),
    'Hippocampus': (-55, -9),
    'Insula': (-28, 20),
    'OFCant': (55, -33),
    'OFCmed': (35, -28),
    'OFCpost': (17, -33),
    'Olfactory': (39, -9),
    'Putamen': (-12, 2),
    'Rectus': (70, -17),
    'SN_pc': (-34, -31),
    'VTA': (-47, -21),
    'Vent_Str': (12, -12),
}

_PARENT = os.path.dirname(_HERE)


BOLD_TS_ROOT = os.environ.get('BOLD_TS_ROOT', _find_dir('bold_signals_standard', context=globals()))
BOLD_TS_PATTERN = 'standard_bold_par{sub_num:02d}_s3.nii.gz'
BEH_DIR = os.environ.get('BEH_DIR', _find_dir('beh_fmri', context=globals()))
AAL3_PATH = os.path.join(_HERE, 'assets', 'aal3_atlas.nii')
SVC_MASK_PATH = os.path.join(_HERE, 'assets', 'svc_mask.nii')
LABEL_PATH = os.path.join(_HERE, 'assets', 'aal3_labels.tsv')
CACHE_DIR = os.environ.get('SMALLVOLUME_CACHE_DIR', os.path.join(_HERE, 'cache_voxels'))
SUBJECTS = [f'par{i:02d}' for i in range(1, 31)]
NUM_TRIALS = 128
T_BOLD = int(os.environ.get('T_BOLD', '18'))
K_PCA = 0
T_TRAJ = 50
TR_FALLBACK = 1.0
SCREEN_W, SCREEN_H = (1920, 1080)
SCREEN_CX, SCREEN_CY = (SCREEN_W / 2.0, SCREEN_H / 2.0)
LABEL_MODE = 'subj'
CACHE_PREFIX = 'smallvolume_exactsvc_allvoxels_v1'
MERGE_ROIS = bool(int(os.environ.get('MERGE_ROIS', '1')))
_MERGE_TAG = '_bilateral' if MERGE_ROIS else '_hemisphere'
ATLAS_PATH = AAL3_PATH
EXCLUDE_ROIS = set()


NETWORKS = [
    'frontal',
    'parietal',
    'temporal',
    'occipital',
    'cingulo_limbic',
    'subcortical',
    'cerebellum',
    'brainstem',
]
_NET_RULES = [
    ('cerebellum', ['Cerebellum', 'Vermis']),
    ('brainstem', ['Red_N', 'SN_', 'SN', 'VTA', 'Raphe', 'LC_']),
    ('subcortical', ['Caudate', 'Putamen', 'Pallidum', 'Thal', 'Thalamus', 'N_Acc']),
    (
        'cingulo_limbic',
        [
            'Cingulate',
            'ACC',
            'Insula',
            'Hippocampus',
            'ParaHippocampal',
            'Amygdala',
            'OFC',
            'Rectus',
            'Olfactory',
        ],
    ),
    ('occipital', ['Occipital', 'Calcarine', 'Cuneus', 'Lingual', 'Fusiform']),
    ('temporal', ['Temporal', 'Heschl']),
    (
        'parietal',
        ['Parietal', 'Postcentral', 'SupraMarginal', 'Angular', 'Precuneus', 'Paracentral'],
    ),
    ('frontal', ['Frontal', 'Precentral', 'Rolandic_Oper', 'Supp_Motor']),
]


SEED = int(os.environ.get('TRAIN_SEED', '42'))
LR = 0.0003
WEIGHT_DECAY = 0.02
MAX_EPOCHS = 80
PATIENCE = 15
BATCH_SIZE = 16
D_MODEL = 64
NUM_LAYERS = 1
NHEAD = 4
DIM_FF = 128
DROPOUT = 0.2
Z_DIM = 64
COORD_N_FREQ = 16
BALANCE = True
SCALE_BOLD = True
SAVE_MODELS = True
W_CHOICE, W_TRAJ = (1.0, 1.0)
W_CUR, W_RISK = (1.0, 1.0)
LAMBDA_GROUP = 0.0
LAMBDA_L1 = 0.001
GAM = False
INTER_RANK = 0
VAL_FRAC = 0.15
RESULTS_DIR = os.environ.get('RESULTS_DIR') or os.path.join(
    _HERE, 'results_arch', os.environ.get('MODEL_ARCH', 'd64_l1_h4'), f'seed{SEED}'
)
MODEL_DIR = os.path.join(RESULTS_DIR, 'models')
SUBJ_DIR = os.path.join(RESULTS_DIR, 'per_subject')
KEYS_INPUT = ['ori_opt1', 'ori_opt2', 'ori_space']
KEYS_LABEL = ['chose_story', 'curiosity_label', 'risk_label', 'social_label']
_GIDX_CACHE = {}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--cache-subject', choices=SUBJECTS, help='Prepare one participant cache without training'
    )
    ap.add_argument('--verify-inputs', action='store_true', help='Check inputs without training')
    ap.add_argument('--held-out', '--subject', dest='held_out', default=None)
    ap.add_argument('--epochs', type=int, default=MAX_EPOCHS)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--rebuild-cache', action='store_true')
    ap.add_argument('--all', action='store_true', help='run all 30 LOSO folds in one process')
    ap.add_argument('--d-model', type=int, default=D_MODEL)
    ap.add_argument('--num-layers', type=int, default=NUM_LAYERS)
    ap.add_argument('--nhead', type=int, default=NHEAD)
    ap.add_argument('--dim-ff', type=int, default=DIM_FF)
    ap.add_argument('--lambda-group', type=float, default=LAMBDA_GROUP)
    ap.add_argument('--lambda-l1', type=float, default=LAMBDA_L1)
    ap.add_argument('--gam', action='store_true', default=GAM)
    ap.add_argument('--inter-rank', type=int, default=INTER_RANK)
    a, _ = ap.parse_known_args()

    if a.cache_subject is not None:
        build_subject(
            a.cache_subject, rebuild=a.rebuild_cache or find_subject_cache(a.cache_subject, context=globals()) is None
        , context=globals())
        raise SystemExit(0)

    if a.verify_inputs:
        verify_inputs(context=globals())
        raise SystemExit(0)

    D_MODEL, NUM_LAYERS, NHEAD, DIM_FF = (a.d_model, a.num_layers, a.nhead, a.dim_ff)
    LAMBDA_GROUP, LAMBDA_L1, GAM = (a.lambda_group, a.lambda_l1, a.gam)
    INTER_RANK = a.inter_rank

    if a.all:
        pooled, subj, info = _load_pooled(rebuild=a.rebuild_cache, context=globals())

        for s in SUBJECTS:
            train_loocv(s, epochs=a.epochs, device=a.device, pooled=pooled, subj=subj, info=info, context=globals())
    else:
        held = a.held_out or SUBJECTS[int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))]
        train_loocv(held, epochs=a.epochs, device=a.device, rebuild=a.rebuild_cache, context=globals())
