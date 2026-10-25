"""Bounded syntactic quantity sources and checked-result reuse summaries."""

import ast
from fractions import Fraction
import hashlib
import json
import math

import numpy as np

from arithmetic_features import candidate_features as old_candidate_features
from common import parse_answer
from question_operation_features import question_numbers
from submission_answer_groups import representations


VARIANTS = ('format_only', 'local_arithmetic', 'quantity_bag', 'canonical_order', 'temporal_order')
DIMENSIONS = {'format_only': 3, 'local_arithmetic': 11, 'quantity_bag': 19,
              'canonical_order': 27, 'temporal_order': 27}
TRIVIAL = frozenset(map(Fraction, (-1, 0, 1)))


def numeric_atoms(left):
    """Collect source-text numeric AST atoms after the old checker accepts left."""
    source = left.strip().replace('^', '**')
    tree = ast.parse(source, mode='eval')
    atoms = set()
    def visit(node):
        if isinstance(node, ast.Expression):
            visit(node.body)
        elif isinstance(node, ast.Constant) and type(node.value) in (int, float):
            atoms.add(Fraction(ast.get_source_segment(source, node)))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            if isinstance(node.operand, ast.Constant) and type(node.operand.value) in (int, float):
                value = Fraction(ast.get_source_segment(source, node.operand))
                atoms.add(-value if isinstance(node.op, ast.USub) else value)
            else:
                visit(node.operand)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            visit(node.left); visit(node.right)
        else:
            raise ValueError('Atoms require the accepted numeric AST whitelist')
    visit(tree)
    return frozenset(atoms - TRIVIAL)


def checked_records(checks):
    return [{**row, 'atoms': numeric_atoms(row['left']), 'result': Fraction(row['right_value'])}
            for row in checks]


def canonical_key(record):
    return tuple(record[name] for name in ('left', 'right', 'left_value', 'right_value', 'mismatch'))


def directed_matches(records):
    return [(i, j) for i, row in enumerate(records) if row['result'] not in TRIVIAL
            for j, other in enumerate(records) if i != j and row['result'] in other['atoms']]


def bag_features(records, sources, final):
    operands = set().union(*(row['atoms'] for row in records)) if records else set()
    results = {row['result'] for row in records}
    nontrivial = [row for row in records if row['result'] not in TRIVIAL]
    matches = directed_matches(records)
    nonempty = [row for row in records if row['atoms']]
    return np.asarray([float(bool(nonempty)), len(sources & operands)/len(sources) if sources else 0.,
        len(sources & (results-TRIVIAL))/len(sources) if sources else 0.,
        len(operands-(sources|results))/len(operands) if operands else 0.,
        len({i for i, _ in matches})/len(nontrivial) if nontrivial else 0., math.log1p(len(matches)),
        sum(row['atoms'] <= sources for row in nonempty)/len(nonempty) if nonempty else 0.,
        float(final is not None and final in results)], dtype=np.float64)


def ordered_features(records, sources, final):
    K = len(records)
    matches = directed_matches(records)
    forward = [(i, j) for i, j in matches if i < j]
    occurrences = sum(len(row['atoms']) for row in records)
    earlier, late_needed, known_operands, first_ungiven = set(), 0, 0, 0.
    for i, row in enumerate(records):
        later = {r['result'] for r in records[i+1:]}
        late_needed += sum(x not in sources and x not in earlier and x in later for x in row['atoms'])
        known_operands += sum(x in sources or x in earlier for x in row['atoms'])
        if not first_ungiven and not row['atoms'] <= (sources | earlier):
            first_ungiven = (i+1)/(K+1)
        earlier.add(row['result'])
    adjacent = sum(records[i-1]['result'] not in TRIVIAL and records[i-1]['result'] in records[i]['atoms']
                   for i in range(1, K))
    known = set(sources)
    grounded = []
    for row in records:
        good = not row['mismatch'] and row['atoms'] <= known
        grounded.append(good)
        if good and row['result'] not in TRIVIAL:
            known.add(row['result'])
    nonempty = sum(bool(row['atoms']) for row in records)
    return np.asarray([len(forward)/len(matches) if matches else 0.,
        late_needed/occurrences if occurrences else 0., known_operands/occurrences if occurrences else 0.,
        adjacent/max(K-1, 1),
        sum(bool(row['atoms']) and good for row, good in zip(records, grounded))/nonempty if nonempty else 0.,
        float(final is not None and any(good and row['result']==final for row, good in zip(records, grounded))),
        first_ungiven, sum((j-i)/max(K-1,1) for i,j in forward)/len(forward) if forward else 0.], dtype=np.float64)


def extract_table(bank):
    """Return [old11, bag8, canonical8, temporal8], with input-only metadata."""
    table = np.empty((len(bank), 4, 35), dtype=np.float64)
    counts = np.empty((len(bank), 4), dtype=np.int64)
    for q, row in enumerate(bank):
        sources = set(question_numbers(row['question'])) - TRIVIAL
        for i, candidate in enumerate(row['candidates']):
            text = candidate['text']
            ff, aa, checks = old_candidate_features(text)
            records = checked_records(checks)
            parsed = parse_answer(text)
            final = Fraction(parsed) if parsed is not None else None
            table[q, i] = np.r_[ff, aa, bag_features(records, sources, final),
                ordered_features(sorted(records, key=canonical_key), sources, final),
                ordered_features(records, sources, final)]
            counts[q, i] = len(checks)
    if not np.isfinite(table).all():
        raise ValueError('Nonfinite quantity-flow table')
    return table, {'checked_equation_counts': counts}


def input_fingerprint(bank, arrays, source_indices):
    digest = hashlib.sha256(json.dumps(bank, sort_keys=True, ensure_ascii=False,
                                      separators=(',', ':')).encode('utf-8'))
    for value in (np.asarray(arrays['embeddings']), np.asarray(source_indices)):
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def build_cache(bank, arrays, source_indices):
    table, metadata = extract_table(bank)
    return {'table': table, **metadata, 'fingerprint': input_fingerprint(bank, arrays, source_indices),
            'table_sha256': hashlib.sha256(table.tobytes()).hexdigest(),
            'row_fingerprints': [input_fingerprint([row], {'embeddings':arrays['embeddings'][i:i+1]},
                                np.asarray(source_indices)[i:i+1]) for i,row in enumerate(bank)]}


def cached_values(bank, arrays, source_indices, cache=None):
    if cache is None:
        return extract_table(bank)[0]
    table = np.asarray(cache['table'])
    if table.shape != (len(bank), 4, 35) or not np.isfinite(table).all() or \
            cache['fingerprint'] != input_fingerprint(bank, arrays, source_indices) or \
            cache['table_sha256'] != hashlib.sha256(table.tobytes()).hexdigest():
        raise ValueError('Quantity-flow caller cache differs')
    return table


def slice_cache(cache, indices, bank, arrays, source_indices):
    expected = [input_fingerprint([row], {'embeddings':arrays['embeddings'][i:i+1]},
                np.asarray(source_indices)[i:i+1]) for i,row in enumerate(bank)]
    if [cache['row_fingerprints'][i] for i in indices] != expected or \
            hashlib.sha256(cache['table'].tobytes()).hexdigest()!=cache['table_sha256']:
        raise ValueError('Cache slice row provenance differs')
    table = cache['table'][indices]
    return {'table': table, 'fingerprint': input_fingerprint(bank, arrays, source_indices),
            'table_sha256': hashlib.sha256(table.tobytes()).hexdigest()}


def variant_values(table, variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown frozen quantity-flow variant')
    if variant == 'temporal_order':
        return np.concatenate([table[..., :19], table[..., 27:35]], axis=-1)
    return table[..., :DIMENSIONS[variant]]


def group_features(bank, arrays, values):
    return [[(np.r_[values[q, indices].mean(axis=0), values[q, indices].max(axis=0)], indices)
             for _, indices in groups] for q, groups in enumerate(representations(bank, arrays, 'mean'))]
