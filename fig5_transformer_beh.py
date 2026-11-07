# Foldername: analysis_transformer_model_curiosity_project
# Filename: fig5_transformer_beh.py
from pathlib import Path
import glob
import json
import os

from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import Patch
from matplotlib.text import Text
from scipy import stats
from scipy.stats import gaussian_kde, ttest_1samp
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


shared_FONT_SIZE = 8
shared_TICK_FONT_SIZE = 8
shared_AXIS_LINE_WIDTH = 0.6
shared_TITLE_FONT_SIZE = 8
shared_ANNOTATION_FONT_SIZE = 8
shared_MEAN_LINE_WIDTH = 1.2
shared_LARGE_MEAN_POINT_SIZE = 16
shared_ERROR_BAR_LINE_WIDTH = 0.8
shared_ERROR_BAR_CAP_SIZE = 1.6
shared_POINT_ALPHA = 0.22
shared_POINT_OFFSET = 0.13
shared_CURIOSITY_COLOR = '#6a51a3'
shared_LIGHT_CURIOSITY_COLOR = '#bcbddc'


from functools import partial

import sys

UTILITY_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
if not (UTILITY_DIR / 'utility_transformer.py').is_file():
    UTILITY_DIR = UTILITY_DIR / 'analysis_transformer_model_curiosity_project'
if not (UTILITY_DIR / 'utility_transformer.py').is_file():
    raise FileNotFoundError(f'utility_transformer.py not found in {UTILITY_DIR}. Set the Console working directory to the package or Dean_Code_c.')
sys.path.insert(0, str(UTILITY_DIR))

from utility_transformer import (
        behavior_calculate_curiosity_by_threat_cell_means as calculate_curiosity_by_threat_cell_means,
        behavior_calculate_marginal_label_chance_accuracy as calculate_marginal_label_chance_accuracy,
        behavior_calculate_one_sample_t_test as calculate_one_sample_t_test,
        behavior_draw_observed_predicted_choice_cells as draw_observed_predicted_choice_cells,
        behavior_load_unique_participant_behavior_table as load_unique_participant_behavior_table,
        behavior_performance_combined_row as performance_combined_row,
        behavior_performance_violin_panel as performance_violin_panel,
        behavior_read_behavior_table_as_array as read_behavior_table_as_array,
        behavior_shared_configure_plot_style as shared_configure_plot_style,
        behavior_shared_darken_plot_color as shared_darken_plot_color,
        behavior_shared_format_significance_label as shared_format_significance_label,
        behavior_shared_performance_check_font as shared_performance_check_font,
        behavior_shared_scale_values_to_unit_interval_with_bounds as shared_scale_values_to_unit_interval_with_bounds,
        behavior_shared_style_plot_axes as shared_style_plot_axes,
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
DATA_DIR = Path(os.environ.get('TRANSFORMER_DATA_DIR', str(SCRIPT_DIR))).expanduser()
if 'TRANSFORMER_DATA_DIR' not in os.environ and not (DATA_DIR / 'results_arch').is_dir():
    existing = SCRIPT_DIR.parent / 'analysis_transformer_model_and_assets'
    if (existing / 'results_arch').is_dir():
        DATA_DIR = existing
_HERE = str(DATA_DIR)
HERE = Path(_HERE)
SEED = 42
ARCH = 'd64_l1_h4'
TITLE_PAD = 11.5
ROOT = os.path.join(_HERE, 'results_arch', ARCH, f'seed{SEED}')
SUBJ_DIR = os.path.join(ROOT, 'per_subject')
KEYS = ['choice_acc', 'choice_auc', 'cur_r', 'risk_r', 'social_r', 'cur_partial_social_r', 'traj']
CHANCE = {
    'choice_acc': 0.5,
    'choice_auc': 0.5,
    'cur_r': 0.0,
    'risk_r': 0.0,
    'traj': None,
    'social_r': 0.0,
    'cur_partial_social_r': 0.0,
}
FORMATS = ('png',)
PNG_DPI = 600
EXTRA_WIDTH_IN = 5.0 / 25.4
HEIGHT_SCALE = 1.1


_hvm_draw = draw_observed_predicted_choice_cells


_violin_panel = performance_violin_panel


_combined_row = performance_combined_row


files = sorted(glob.glob(os.path.join(SUBJ_DIR, '*.json')))
files = [f for f in files if not os.path.basename(f).startswith('contrib')]
rows = []
for f in files:
    with open(f) as handle:
        row = json.load(handle)['agg']
    subject = Path(f).stem
    root = SUBJ_DIR
    sub = subject
    project = HERE
    root = Path(root)
    project = Path(project)
    beh = Path(os.environ.get('BEH_DIR', str(project / 'beh_fmri')))
    if 'BEH_DIR' not in os.environ and not beh.is_dir() and (project.parent / 'beh_fmri').is_dir():
        beh = project.parent / 'beh_fmri'
    with np.load(root / f'contrib_{sub}.npz') as d:
        prob = np.asarray(d['prob'], float)
        c, prediction_t, cp, tp = [
            np.asarray(d[prediction_k], float)
            for prediction_k in ('curiosity', 'risk', 'curiosity_pred', 'risk_pred')
        ]
    names = read_behavior_table_as_array(beh / f'output_formal_clickbaitLabel_{sub}.txt', context=globals())[:, 1]
    sr = load_unique_participant_behavior_table(beh, 'output_self_rating_', sub, context=globals())
    lookup = {
        str(prediction_n): float(prediction_r)
        for prediction_n, prediction_r in zip(sr[:, 1], sr[:, 2])
    }
    rawc = np.asarray([lookup.get(str(prediction_n), np.nan) for prediction_n in names], float)[
        : len(c)
    ]
    observed = np.isfinite(rawc)
    filled = np.where(observed, rawc, np.nanmean(rawc))
    scaled, clo, chi = shared_scale_values_to_unit_interval_with_bounds(filled, context=globals())
    rawt = read_behavior_table_as_array(beh / f'output_formal_FLevel_{sub}.txt', context=globals())[:, 1].astype(
        float
    )[: len(prediction_t)]
    scaled, tlo, thi = shared_scale_values_to_unit_interval_with_bounds(rawt, context=globals())
    c_rawpred = cp * (chi - clo) + clo
    t_rawpred = tp * (thi - tlo) + tlo
    categorical = observed & np.isclose(rawc, np.round(rawc), atol=1e-06)
    c_pred = np.floor(c_rawpred[categorical] + 0.5).astype(int)
    c_true = np.round(rawc[categorical]).astype(int)
    cacc = float(np.mean(c_pred == c_true)) if chi > clo else np.nan
    cchance = calculate_marginal_label_chance_accuracy(c_pred, c_true, context=globals()) if chi > clo else np.nan
    unique = np.unique(rawt)
    if len(unique) > 1:
        predicted = unique[np.argmin(abs(t_rawpred[:, None] - unique[None, :]), axis=1)]
        t_pred = np.argmin(abs(t_rawpred[:, None] - unique[None, :]), axis=1)
        t_true = np.searchsorted(unique, rawt)
        tacc = float(np.mean(t_pred == t_true))
        tchance = calculate_marginal_label_chance_accuracy(t_pred, t_true, context=globals())
    else:
        tacc = tchance = np.nan
    with np.load(root / f'behav_{sub}.npz') as b:
        choice = np.asarray(b['chose_story'], int)
    choice_pred = (prob > 0.5).astype(int)
    choice_acc = float(np.mean(choice_pred == choice))
    choice_chance = calculate_marginal_label_chance_accuracy(choice_pred, choice, context=globals())
    metrics = dict(
        cur_mae=float(np.mean(abs(cp[observed] - c[observed]))),
        risk_mae=float(np.mean(abs(tp - prediction_t))),
        cur_accuracy=cacc,
        risk_accuracy=tacc,
        cur_accuracy_chance=cchance,
        risk_accuracy_chance=tchance,
        choice_acc=choice_acc,
        choice_acc_chance=choice_chance,
        cur_mae_raw=float(np.mean(abs(c_rawpred[observed] - rawc[observed]))),
        risk_mae_raw=float(np.mean(abs(t_rawpred - rawt))),
        cur_n=int(observed.sum()),
        risk_n=len(prediction_t),
        risk_n_levels=len(unique),
    )
    row['choice_acc_chance'] = metrics['choice_acc_chance']
    rows.append(row)
for k in KEYS:
    vals = [r.get(k) for r in rows]
    clean = [v for v in vals if v is not None and np.isfinite(v)]
    if not clean:
        continue
    if CHANCE[k] is None:
        pass
    else:
        if k == 'choice_acc':
            vals = [r['choice_acc'] - r['choice_acc_chance'] for r in rows]
            t, p, n = calculate_one_sample_t_test(vals, 0.0, context=globals())
        else:
            t, p, n = calculate_one_sample_t_test(vals, CHANCE[k], context=globals())
shared_configure_plot_style(context=globals())
shared_performance_check_font(context=globals())
_combined_row(rows, context=globals())
