# Foldername: analysis_transformer_model_curiosity_project
# Filename: fig5_attention.py
from pathlib import Path
import hashlib
import json
import os
import pickle
import sys

from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import TwoSlopeNorm, to_rgb
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.text import Text
from matplotlib.textpath import TextPath
from matplotlib.ticker import LogLocator, MaxNLocator
from matplotlib.transforms import Bbox
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.special import expit
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


UTILITY_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
if not (UTILITY_DIR / 'utility_transformer.py').is_file():
    UTILITY_DIR = UTILITY_DIR / 'analysis_transformer_model_curiosity_project'
if not (UTILITY_DIR / 'utility_transformer.py').is_file():
    raise FileNotFoundError(f'utility_transformer.py not found in {UTILITY_DIR}. Set the Console working directory to the package or Dean_Code_c.')
sys.path.insert(0, str(UTILITY_DIR))

from utility_transformer import BoldToChoice, roi18_load_roi18_table

from functools import partial

from utility_transformer import (
        attention_apply_fixed_schema_layout as apply_fixed_schema_layout,
        attention_attention_glm_plot_attention_network as attention_glm_plot_attention_network,
        attention_attention_intervention as attention_intervention,
        attention_attention_place_perimeter_labels as attention_place_perimeter_labels,
        attention_attention_plot_attention_distribution as attention_plot_attention_distribution,
        attention_attention_plot_intervention_results as attention_plot_intervention_results,
        attention_attention_plot_matrix_matched_networks as attention_plot_matrix_matched_networks,
        attention_attention_plot_schema_color_legend as attention_plot_schema_color_legend,
        attention_attention_plot_value_connection_table as attention_plot_value_connection_table,
        attention_attention_result_table_exists as attention_result_table_exists,
        attention_attention_scale_from_means as attention_scale_from_means,
        attention_brain_content_bbox as brain_content_bbox,
        attention_build_attention_glm_display_reference as build_attention_glm_display_reference,
        attention_build_roi_table as build_roi_table,
        attention_cache_id as cache_id,
        attention_calculate_transmitted_value_signs as calculate_transmitted_value_signs,
        attention_calibration_id as calibration_id,
        attention_condition_effects_from_cache as condition_effects_from_cache,
        attention_display_completed_figure as display_completed_figure,
        attention_empirical_attention_calibration as empirical_attention_calibration,
        attention_empirical_subject_predictions as empirical_subject_predictions,
        attention_filter_retained_analysis_rois as filter_retained_analysis_rois,
        attention_format_attention_figure as format_attention_figure,
        attention_get_attention_table_cache_path as get_attention_table_cache_path,
        attention_get_training_table_cache_path as get_training_table_cache_path,
        attention_glm_compute_matrix_statistics as glm_compute_matrix_statistics,
        attention_glm_design_matrix as glm_design_matrix,
        attention_glm_fit_fractional_logit as glm_fit_fractional_logit,
        attention_glm_group_statistics as glm_group_statistics,
        attention_glm_subject_fit as glm_subject_fit,
        attention_grouped_map_roi_to_analysis_group as grouped_map_roi_to_analysis_group,
        attention_load_subject_voxels as load_subject_voxels,
        attention_load_trial_onsets_seconds as load_trial_onsets_seconds,
        attention_matrix_effect_statistics as matrix_effect_statistics,
        attention_model_input_id as model_input_id,
        attention_network_publication_filename as network_publication_filename,
        attention_normalize_roi_table_labels as normalize_roi_table_labels,
        attention_original_attention_scores as original_attention_scores,
        attention_read_attention_result_table as read_attention_result_table,
        attention_resample_time_series as resample_time_series,
        attention_restore_replot_statistics as restore_replot_statistics,
        attention_route_connections as route_connections,
        attention_run_attention_glm as run_attention_glm,
        attention_save_array_cache_atomically as save_array_cache_atomically,
        attention_save_attention_result_table_cache as save_attention_result_table_cache,
        attention_shared_calculate_point_to_polyline_distances as shared_calculate_point_to_polyline_distances,
        attention_shared_trim_curve as shared_trim_curve,
        attention_tight_schema_export_bbox as tight_schema_export_bbox,
        attention_value_cached_model_tokens as value_cached_model_tokens,
        attention_value_message_logits as value_message_logits,
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
roi18_ATLAS_PATH = Path(_HERE) / 'assets/aal3_atlas.nii'
roi18_SVC_MASK_PATH = Path(_HERE) / 'assets/svc_mask.nii'
roi18_LABEL_PATH = Path(_HERE) / 'assets/aal3_labels.tsv'
roi18_MANIFEST_PATH = Path(_HERE) / 'assets/roi_manifest.json'
roi18_ROI_NAMES = ('ACC_pre', 'ACC_sub', 'ACC_sup', 'Amygdala', 'Caudate', 'Frontal_Med_Orb', 'Frontal_Sup_Medial', 'Hippocampus', 'Insula', 'OFCant', 'OFCmed', 'OFCpost', 'Olfactory', 'Putamen', 'Rectus', 'SN_pc', 'VTA', 'Vent_Str')
roi18_ROI_LABELS = dict(zip(roi18_ROI_NAMES, ('pgACC', 'sgACC', 'dACC', 'AMY', 'CAU', 'ORBsupmed', 'dmPFC', 'HPC', 'INS', 'aOFC', 'mOFC', 'pOFC', 'OLF', 'PUT', 'vmPFC', 'SNc', 'VTA', 'NAc')))
roi18_ROI_FULL_NAMES = dict(zip(roi18_ROI_NAMES, ('Pregenual anterior cingulate cortex', 'Subgenual anterior cingulate cortex', 'Supracallosal anterior cingulate cortex', 'Amygdala', 'Caudate nucleus', 'Superior frontal gyrus, medial orbital part', 'Superior frontal gyrus, medial part', 'Hippocampus', 'Insula', 'Anterior orbitofrontal cortex', 'Medial orbitofrontal cortex', 'Posterior orbitofrontal cortex', 'Olfactory cortex', 'Putamen', 'Gyrus rectus', 'Substantia nigra pars compacta', 'Ventral tegmental area', 'Nucleus accumbens')))


DISPLAY_FIGURES = True
WIDTH_MM = 92.0
HEIGHT_MM = 82.8
AXES_RECT = (0.13, 0.035, 0.85, 0.93)
PREVIEW_DPI = 600
POSITIONS = {
    'Frontal_Sup_Medial': (54, 50),
    'ACC_sup': (30, 33),
    'vACC': (43, 11),
    'Rectus': (72, -12),
    'amOFC': (49, -28),
    'pmOFC': (22, -35),
    'Olfactory': (37, -5),
    'Vent_Str': (7, -12),
    'Caudate': (3, 36),
    'Putamen': (-18, 4),
    'Insula': (-29, 21),
    'Amygdala': (-8, -38),
    'Hippocampus': (-55, -8),
    'SN_pc': (-33, -29),
    'VTA': (-47, -20),
}
_RESULT_TABLES = {}
EXCLUDED_ANALYSIS_ROIS = frozenset({'Hippocampus', 'Frontal_Med_Orb', 'OFCant'})
ARCH = 'd64_l1_h4'
ANALYSIS_MODE = 'replot'
SEED = 42
RES_DIR = Path(_HERE) / 'results_arch' / ARCH / f'seed{SEED}'
SUBJ_DIR = RES_DIR / 'per_subject'
MODEL_DIR = RES_DIR / 'models'
OUTPUT_DIR = Path(_HERE) / 'results_2ndlevel_arch' / ARCH / 'attention_intervention' / f'seed{SEED}'
PREFIX = f'{ARCH}_seed{SEED}'
NUM_TRIALS = 128
SUBJECTS = [f'par{i:02d}' for i in range(1, 31)]
torch.set_num_threads(int(os.environ.get('ATTENTION_THREADS', '4')))
BOLD_TS_ROOT = os.environ.get('BOLD_TS_ROOT', str(DATA_DIR.parent / 'bold_signals_standard') if DATA_DIR != SCRIPT_DIR else str(SCRIPT_DIR / 'bold_signals_standard'))
BOLD_TS_PATTERN = 'standard_bold_par{sub_num:02d}_s3.nii.gz'
BEH_DIR = os.environ.get('BEH_DIR', str(DATA_DIR.parent / 'beh_fmri') if DATA_DIR != SCRIPT_DIR else str(SCRIPT_DIR / 'beh_fmri'))
AAL3_PATH = os.path.join(_HERE, 'assets', 'aal3_atlas.nii')
SVC_MASK_PATH = os.path.join(_HERE, 'assets', 'svc_mask.nii')
LABEL_PATH = os.path.join(_HERE, 'assets', 'aal3_labels.tsv')
CACHE_DIR = os.environ.get('SMALLVOLUME_CACHE_DIR', os.path.join(_HERE, 'cache_voxels'))
NUM_TRIALS = 128
T_BOLD = 18
K_PCA = 0
TR_FALLBACK = 1.0
CACHE_PREFIX = 'smallvolume_exactsvc_allvoxels_v1'
MERGE_ROIS = True
_MERGE_TAG = '_bilateral' if MERGE_ROIS else '_hemisphere'


BEH_DIR = os.environ.get('BEH_DIR', str(DATA_DIR.parent / 'beh_fmri') if DATA_DIR != SCRIPT_DIR else str(SCRIPT_DIR / 'beh_fmri'))


ATT_LAYER = 1
ROI_NAMES = None
GAIN_LEVELS = tuple(range(1, 7))
_EMPIRICAL_CALIBRATION = None
BATCH_SIZE = 128
SCENARIOS_PER_BATCH = 16
CONDITIONS = [
    ('highC_highT', True, True, 'High C / High T'),
    ('highC_lowT', True, False, 'High C / Low T'),
    ('lowC_highT', False, True, 'Low C / High T'),
    ('lowC_lowT', False, False, 'Low C / Low T'),
]


plot_intervention_results = attention_plot_intervention_results
GLM_TERMS = ['Intercept', 'C', 'T', 'C:T', 'D']
GLM_LABELS = {
    'Intercept': 'Intercept at extrapolated attention level 0',
    'C': 'Self-interest rating',
    'T': 'Objective threat level',
    'C:T': 'Self-interest × objective threat',
    'D': 'Pre-softmax score level (1–6; zero to filtered common maximum mean)',
}
GLM_VERSION = 'fractional_logit_CT_presoftmax_zero_to_mean_3sd_max6_v12_nohpc_noamofc'
GLM_OUT = (
    Path(_HERE) / 'results_2ndlevel_arch' / ARCH / 'attention_glm' / f'seed{SEED}' / GLM_VERSION
)
GLM_PER_SUB = SUBJ_DIR
GLM_PREFIX = f'{PREFIX}_layer{ATT_LAYER}'
GLM_NETWORK_EDGES_PER_SIGN = None
SCHEMA_HIDDEN_NODES = {'Hippocampus', 'vACC'}
SCHEMA_ADDED_NODES = {'VTA', 'Caudate'}
CONNECTION_TABLE_GROUPS = [
    ('Threat', ('Frontal_Sup_Medial', 'ACC_sup', 'Putamen', 'Insula', 'Amygdala')),
    ('Curiosity', ('Rectus',)),
    ('Choice modulation', ('pmOFC',)),
    ('Dopamine', ('SN_pc', 'Vent_Str', 'Olfactory')),
]
SCHEMA_HIDE_WITHIN_ROIS = {
    'Caudate',
    'Insula',
    'Olfactory',
    'Vent_Str',
    'SN_pc',
    'Amygdala',
    'Putamen',
}
SKIP_SCHEMA_CONNECTIONS = {
    ('pmOFC', target) for target in ('vmPFC', 'OLF', 'NAc', 'SNc', 'INS', 'CAU')
} | {
    ('SNc', 'vmPFC'),
    ('dACC', 'SNc'),
    ('dmPFC', 'PUT'),
    ('vmPFC', 'dmPFC'),
    ('PUT', 'dmPFC'),
    ('PUT', 'dACC'),
    ('AMY', 'pmOFC'),
    ('OLF', 'pmOFC'),
    ('OLF', 'vmPFC'),
}


# Analysis ROI groups
ANALYSIS_GROUPS = {'vACC': ('ACC_pre', 'ACC_sub'), 'pmOFC': ('OFCmed', 'OFCpost')}
ANALYSIS_SOURCE_TO_GROUP = {
    source: group for group, sources in ANALYSIS_GROUPS.items() for source in sources
}


ANALYSIS_SELECTED_ROIS = filter_retained_analysis_rois(roi18_ROI_NAMES, context=globals())
ANALYSIS_ROI_NAMES = tuple(
    dict.fromkeys((grouped_map_roi_to_analysis_group(n, context=globals()) for n in ANALYSIS_SELECTED_ROIS))
)
ANALYSIS_ROI_LABELS = {
    grouped_map_roi_to_analysis_group(n, context=globals()): roi18_ROI_LABELS[n] for n in ANALYSIS_SELECTED_ROIS
}
ANALYSIS_ROI_LABELS.update(vACC='vACC', pmOFC='pmOFC')
ANALYSIS_ROI_FULL_NAMES = {
    grouped_map_roi_to_analysis_group(n, context=globals()): roi18_ROI_FULL_NAMES[n] for n in ANALYSIS_SELECTED_ROIS
}
ANALYSIS_ROI_FULL_NAMES.update(
    vACC='Ventral anterior cingulate cortex', pmOFC='Medial–posterior orbitofrontal cortex'
)
ANALYSIS_SCHEMA_POSITIONS = {
    name: point for name, point in POSITIONS.items() if name in ANALYSIS_ROI_NAMES
}
SKIP_SCHEMA_CONNECTIONS |= {
    (a, b)
    for a in ANALYSIS_ROI_LABELS.values()
    for b in ANALYSIS_ROI_LABELS.values()
    if a != b and 'vACC' in (a, b)
}
GLM_FIGURE_WIDTH_MM = 92
GLM_SCHEMA_WIDTH_MM = 87.5
GLM_BRAIN_SIZE_SCALE = 1.141076475
GLM_FONT_SIZE_PT = 8
GLM_EXPORT_DPI = 600
MANUSCRIPT_FIGURE_DIR = Path(
    os.environ.get(
        'MANUSCRIPT_DIR',
        str(MANUSCRIPT_DIR),
    )
)


format_figure = format_attention_figure


plot_attention_distribution = attention_plot_attention_distribution


place_perimeter_labels = attention_place_perimeter_labels


glm_plot_attention_network = attention_glm_plot_attention_network


plot_schema_color_legend = attention_plot_schema_color_legend


plot_matrix_matched_networks = attention_plot_matrix_matched_networks


plot_value_connection_table = attention_plot_value_connection_table


if ANALYSIS_MODE == 'distribution':
    with plt.ioff():
        figures = plot_attention_distribution(show=DISPLAY_FIGURES, context=globals())
else:
    path = get_attention_table_cache_path(context=globals())
    if not path.is_file():
        path = get_training_table_cache_path(context=globals())
    if path.is_file():
        with path.open('rb') as handle:
            tables = pickle.load(handle)
        _RESULT_TABLES.update(
            {
                key.replace(
                    'analysis_transformer_smallvolume_social_nopca_objthreat', Path(_HERE).name
                ): normalize_roi_table_labels(value, context=globals())
                for key, value in tables.items()
            }
        )
    with plt.ioff():
        if ANALYSIS_MODE == 'replot':
            show = DISPLAY_FIGURES
            table = build_roi_table(context=globals())
            names = table['names'].tolist()
            out = (
                Path(_HERE)
                / 'results_2ndlevel_arch'
                / ARCH
                / 'value_message_choice'
                / f'seed{SEED}'
            )
            strength_path = GLM_OUT / f'{GLM_PREFIX}_group_betas.csv'
            participant_path = GLM_OUT / f'{GLM_PREFIX}_participant_betas.csv'
            missing = [
                str(p)
                for p in (strength_path, participant_path)
                if not attention_result_table_exists(p, context=globals())
            ]
            if missing:
                restore_replot_statistics(context=globals())
                save_attention_result_table_cache(context=globals())
            strength = read_attention_result_table(strength_path, context=globals())
            for frame, term in ((strength, 'D'),):
                selected = frame[frame.term == term]
            specification = out / 'analysis_specification.json'
            metadata = json.loads(specification.read_text()) if specification.is_file() else {}
            save_attention_result_table_cache(context=globals())
            replot_figures = glm_compute_matrix_statistics(strength, names, context=globals())
            replot_figures += plot_matrix_matched_networks(names, show=show, context=globals())
            save_attention_result_table_cache(context=globals())
            figures = replot_figures
        elif ANALYSIS_MODE == 'value_choice':
            show = DISPLAY_FIGURES
            table = build_roi_table(context=globals())
            names = table['names'].tolist()
            strength_table = GLM_OUT / f'{GLM_PREFIX}_group_betas.csv'
            if attention_result_table_exists(strength_table, context=globals()):
                glm_compute_matrix_statistics(read_attention_result_table(strength_table, context=globals()), names, context=globals())
            else:
                run_attention_glm(show=show, draw_network=False, context=globals())
            selected, _, _ = build_attention_glm_display_reference(names, context=globals())
            edges = [(s, t) for s in selected for t in selected if s != t]
            indices = [(names.index(s), names.index(t)) for s, t in edges]
            out = (
                Path(_HERE)
                / 'results_2ndlevel_arch'
                / ARCH
                / 'value_message_choice'
                / f'seed{SEED}'
            )
            effects = []
            participants = []
            for sub in SUBJECTS:
                model, tokens = value_cached_model_tokens(sub, table, context=globals())
                with np.load(SUBJ_DIR / f'contrib_{sub}.npz') as d:
                    c = d['curiosity'].astype(float)
                    t = d['risk'].astype(float)
                cz = (c - c.mean()) / c.std(ddof=1)
                tz = (t - t.mean()) / t.std(ddof=1)
                X = np.column_stack([np.ones(128), cz, tz, cz * tz])
                fingerprint = hashlib.sha256(
                    tokens.numpy().tobytes()
                    + repr([ARCH, SEED, ATT_LAYER, edges, 'exact_value_removal_v1']).encode()
                    + b''.join(
                        (
                            v.detach().numpy().tobytes()
                            for k, v in model.state_dict().items()
                            if not k.startswith(('tokenizer.', 'decoder.'))
                        )
                    )
                ).hexdigest()
                cache = Path(CACHE_DIR) / f'{GLM_PREFIX}_{sub}_value_choice_nohpc_noamofc.npz'
                delta = None
                if cache.is_file():
                    with np.load(cache) as d:
                        if str(d['fingerprint']) == fingerprint:
                            delta = d['delta_story_logit'].copy()
                with torch.no_grad():
                    baseline, _ = model.choice_head(model.enc_norm(model.encoder(tokens)))
                    if delta is None:
                        chunks = []
                        for start in range(0, len(indices), SCENARIOS_PER_BATCH):
                            removed = value_message_logits(
                                model, tokens, indices[start : start + SCENARIOS_PER_BATCH]
                            , context=globals())
                            chunks.append((baseline[None] - removed).numpy())
                        delta = np.concatenate(chunks)
                        save_array_cache_atomically(
                            cache,
                            delta_story_logit=delta,
                            fingerprint=fingerprint,
                            source=np.array([e[0] for e in edges]),
                            target=np.array([e[1] for e in edges]),
                        context=globals())
                beta = np.linalg.lstsq(X, delta.T, rcond=None)[0][0]
                effects.append(beta)
                for (value_choice_source, value_choice_target), v in zip(edges, beta):
                    participants.append(
                        dict(
                            subject=sub,
                            source=value_choice_source,
                            target=value_choice_target,
                            adjusted_story_logit_contribution=v,
                        )
                    )
            effects = np.asarray(effects)
            rows = []
            family = len(edges)
            for value_choice_i, (value_choice_source, value_choice_target) in enumerate(edges):
                v = effects[:, value_choice_i]
                test = stats.ttest_1samp(v, 0.0)
                rows.append(
                    dict(
                        source=value_choice_source,
                        target=value_choice_target,
                        term='value_choice',
                        n=len(v),
                        mean_beta=v.mean(),
                        sem_beta=stats.sem(v),
                        t=test.statistic,
                        p_raw=test.pvalue,
                        p_bonf=min(1.0, test.pvalue * family),
                        family_size=family,
                    )
                )
            summary = pd.DataFrame(rows)
            _RESULT_TABLES[str(out / f'{GLM_PREFIX}_participant_value_choice.csv')] = pd.DataFrame(
                participants
            )
            _RESULT_TABLES[str(out / f'{GLM_PREFIX}_group_value_choice.csv')] = summary
            selected_frame = summary[summary.source.isin(names) & summary.target.isin(names)]
            calculate_transmitted_value_signs(selected_frame, out, context=globals())
            plot_matrix_matched_networks(names, show=show, context=globals())
            figures = (effects, summary)
        elif ANALYSIS_MODE == 'glm':
            figures = run_attention_glm(show=DISPLAY_FIGURES, context=globals())
        elif ANALYSIS_MODE == 'intervention':
            show = DISPLAY_FIGURES
            calibration = empirical_attention_calibration(context=globals())
            names = calibration['names']
            chosen = names if ROI_NAMES is None else filter_retained_analysis_rois(ROI_NAMES, context=globals())
            rows = []
            prefix = f'{PREFIX}_layer{ATT_LAYER}'
            for sub in SUBJECTS:
                result = empirical_subject_predictions(sub, calibration, persist=True, context=globals())
                baseline = result['baseline']
                base_skip = baseline <= 0.5
                base_story = baseline > 0.5
                for intervention_i, (intervention_source, intervention_target, level) in enumerate(
                    zip(result['source'], result['target'], result['level'])
                ):
                    if intervention_source not in chosen or intervention_target not in chosen:
                        continue
                    probability = result['probability'][intervention_i]
                    rows.append(
                        dict(
                            subject=sub,
                            source=intervention_source,
                            target=intervention_target,
                            level=level,
                            target_score=result['target_score'][intervention_i],
                            mean_pstory=probability.mean(),
                            baseline_pstory=baseline.mean(),
                            delta_pp=100 * (probability - baseline).mean(),
                            baseline_skip_n=int(base_skip.sum()),
                            mean_pstory_baseline_skip=(
                                float(probability[base_skip].mean()) if base_skip.any() else np.nan
                            ),
                            skip_to_story_fraction=(
                                float((probability[base_skip] > 0.5).mean())
                                if base_skip.any()
                                else np.nan
                            ),
                            baseline_story_n=int(base_story.sum()),
                            story_to_skip_fraction=(
                                float((probability[base_story] <= 0.5).mean())
                                if base_story.any()
                                else np.nan
                            ),
                            score_before_min=result['score_before_min'][intervention_i],
                            score_before_max=result['score_before_max'][intervention_i],
                            score_after_min=result['score_after_min'][intervention_i],
                            score_after_max=result['score_after_max'][intervention_i],
                        )
                    )
            frame = pd.DataFrame(rows)
            family = len(chosen) * (len(chosen) - 1) * len(GAIN_LEVELS)
            summary = matrix_effect_statistics(frame, 'delta_pp', family, context=globals())
            _RESULT_TABLES[str(OUTPUT_DIR / f'{prefix}_participant_effects.csv')] = frame
            _RESULT_TABLES[str(OUTPUT_DIR / f'{prefix}_statistics.csv')] = summary
            plot_intervention_results(frame, summary, show=show, context=globals())
            figures = (frame, summary)
        else:
            raise ValueError(
                "ANALYSIS_MODE must be 'distribution', 'replot', 'value_choice', 'glm' or 'intervention'"
            )
