"""NumPy full-group quantity-flow corrections to the unchanged S anchor."""

import json
from pathlib import Path

import numpy as np

from quantity_flow_features import DIMENSIONS, extract_table, group_features, variant_values
from submission_ensemble import choose_scores
from submission_question_operations import candidate_scores as surface_scores


def candidate_scores(bank, arrays, model):
    result = surface_scores(bank, arrays, model['anchor_raw'], 'numeric_surface')
    values = variant_values(extract_table(bank)[0], model['variant'])
    for q, groups in enumerate(group_features(bank, arrays, values)):
        for feature, indices in groups:
            result[q, indices] += feature @ model['raw_v']
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite quantity-flow scores')
    return result


def load_checkpoint(checkpoint):
    directory = Path(checkpoint)
    config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
    if config.get('schema_version')!=1 or config.get('model')!='surface_quantity_flow' or config.get('variant') not in DIMENSIONS:
        raise ValueError('Unsupported quantity-flow checkpoint')
    with np.load(directory/'weights.npz', allow_pickle=False) as saved:
        if set(saved.files)!={'anchor_raw','raw_v'}:
            raise ValueError('Unexpected inference arrays')
        model={name:saved[name].copy() for name in saved.files}
    if model['anchor_raw'].shape!=(1034,) or model['raw_v'].shape!=(2*DIMENSIONS[config['variant']],) \
            or not all(np.isfinite(value).all() for value in model.values()):
        raise ValueError('Invalid quantity-flow coefficients')
    return config,{**model,'variant':config['variant']}


def predict(bank, arrays, checkpoint):
    _,model=load_checkpoint(checkpoint)
    return choose_scores(candidate_scores(bank,arrays,model),bank,arrays)
