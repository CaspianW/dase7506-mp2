"""Bounded numeric equality checks; unsupported expressions are missing data."""

import ast
from fractions import Fraction
import re
import numpy as np

from common import parse_answer


FORMAT_NAMES = ['has_equals', 'log_equals', 'log_equation_lines']
ARITHMETIC_NAMES = ['has_checked_equations', 'log_checked_equations', 'log_mismatches',
                    'mismatch_fraction', 'log_max_relative_residual',
                    'last_result_final_known', 'last_result_final_agrees',
                    'last_result_final_disagrees']
ABS_TOLERANCE = Fraction(5, 1000)
REL_TOLERANCE = Fraction(1, 1000000)


def close(a, b):
    return abs(a - b) <= ABS_TOLERANCE + REL_TOLERANCE * max(abs(a), abs(b))


def numeric_value(expression):
    """Interpret at most64 whitelisted AST nodes; never invoke Python eval."""
    expression = expression.strip()
    if not expression or len(expression) > 160 or re.search(r'[^\d\s.()+*/^\-]', expression):
        raise ValueError('Unsupported numeric expression')
    expression = expression.replace('^', '**')
    tree = ast.parse(expression, mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError('Too many expression nodes')

    def bounded(value):
        if abs(value) > 10**12 or abs(value.numerator) > 10**18 or value.denominator > 10**18:
            raise ValueError('Numeric magnitude bound exceeded')
        return value

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            token = ast.get_source_segment(expression, node)
            if not re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)', token):
                raise ValueError('Unsupported constant')
            return bounded(Fraction(token))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return bounded(left + right)
            if isinstance(node.op, ast.Sub):
                return bounded(left - right)
            if isinstance(node.op, ast.Mult):
                return bounded(left * right)
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ValueError('Division by zero')
                return bounded(left / right)
            if right.denominator != 1 or abs(right) > 6 or (left == 0 and right < 0):
                raise ValueError('Unsupported exponent')
            return bounded(left ** int(right))
        raise ValueError('AST node is outside whitelist')

    return visit(tree)


def normalize(text):
    text = text[:20000]
    text = re.sub(r'(?<!\w)\*\*([^*\n]+)\*\*(?!\w)', r'\1', text)
    text = text.replace('×', '*').replace('÷', '/').replace('−', '-')
    for source, target in [('\\times', '*'), ('\\cdot', '*'), ('\\div', '/'),
                           ('\\left', ''), ('\\right', ''), ('\\,', ' '), ('\\!', '')]:
        text = text.replace(source, target)
    for _ in range(4):
        updated = re.sub(r'\\(?:d?frac)\s*\{([^{}]{1,80})\}\s*\{([^{}]{1,80})\}',
                         r'((\1)/(\2))', text)
        if updated == text:
            break
        text = updated
    text = re.sub(r'\^\{(-?\d{1,2})\}', r'^(\1)', text)
    for delimiter in ['\\(', '\\)', '\\[', '\\]', '$']:
        text = text.replace(delimiter, ' ')
    # Remove grouping commas only when the whole token follows thousands grouping.
    text = re.sub(r'(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w,])',
                  lambda m: m.group().replace(',', ''), text)
    return text


def side_expression(part, left):
    pattern = r'[\d\s.()+*/^\-]+$' if left else r'^[\d\s.()+*/^\-]+'
    match = re.search(pattern, part)
    if not match or not re.search(r'\d', match.group()):
        raise ValueError('No numeric side')
    before = part[:match.start()].rstrip()
    raw_after = part[match.end():]
    after = raw_after.lstrip()
    # Prevent harvesting the tail of variables such as 2N+100, or percentages.
    if before and (before[-1].isalnum() or before[-1] in '_%}\\'):
        raise ValueError('Ambiguous prefix')
    if after and (after[0] in '%_\\{' or after[0].isdigit()):
        raise ValueError('Ambiguous suffix')
    # A prefix such as "2" in "2N+100" is not a complete numeric result.
    # The greedy numeric span can include separating spaces, so inspect the
    # original boundary as well as symbolic continuations after whitespace.
    if after and after[0].isalpha():
        if match.group()[-1:].strip() or re.match(r'^[A-Za-z](?:\W|$)', after):
            raise ValueError('Numeric expression continues into a variable')
        if re.search(r'[\d=+*/^]', after):
            raise ValueError('Text suffix contains a further expression')
    expression = match.group().strip()
    if not left:
        expression = re.sub(r'(?<=\d)\.$', '', expression)
    return expression


def checked_equalities(text):
    checked = []
    opportunities = 0
    for line in normalize(text).splitlines():
        # A spaced markdown list marker is not a unary arithmetic sign.
        line = re.sub(r'^\s*[-+*]\s+(?=\d|\()', '', line)
        parts = line.split('=')
        for left, right in zip(parts, parts[1:]):
            opportunities += 1
            if opportunities > 64:
                return checked
            try:
                lhs, rhs = side_expression(left, True), side_expression(right, False)
                a, b = numeric_value(lhs), numeric_value(rhs)
            except (ValueError, SyntaxError, OverflowError, ZeroDivisionError, RecursionError):
                continue
            checked.append({'left': lhs, 'right': rhs, 'left_value': str(a), 'right_value': str(b),
                            'mismatch': not close(a, b),
                            'relative_residual': float(abs(a - b) / max(Fraction(1), abs(a), abs(b)))})
    return checked


def candidate_features(text):
    count = text.count('=')
    format_features = [float(count > 0), np.log1p(count), np.log1p(sum('=' in line for line in text.splitlines()))]
    checks = checked_equalities(text)
    mismatch = sum(row['mismatch'] for row in checks)
    final = parse_answer(text)
    known = bool(checks and final is not None)
    agrees = known and close(Fraction(checks[-1]['right_value']), Fraction(final))
    arithmetic = [float(bool(checks)), np.log1p(len(checks)), np.log1p(mismatch),
                  mismatch / len(checks) if checks else 0.,
                  np.log1p(max([row['relative_residual'] for row in checks], default=0.)),
                  float(known), float(agrees), float(known and not agrees)]
    return format_features, arithmetic, checks


def feature_names(variant):
    if variant not in ('embedding', 'format', 'arithmetic'):
        raise ValueError('Unknown arithmetic experiment variant')
    names = [f'embedding_{i}' for i in range(1024)]
    if variant != 'embedding':
        names += FORMAT_NAMES
    if variant == 'arithmetic':
        names += ARITHMETIC_NAMES
    return names


def build_features(bank, arrays, variant):
    names = feature_names(variant)
    x = np.asarray(arrays['embeddings'], dtype=np.float64)
    if x.shape != (len(bank), 4, 1024) or not np.isfinite(x).all():
        raise ValueError('Expected finite [N,4,1024] embeddings')
    if not bank:
        return np.empty((0, 4, len(names)))
    if variant == 'embedding':
        return x.copy()
    extra = []
    for row in bank:
        group = []
        for candidate in row['candidates']:
            ff, aa, _ = candidate_features(candidate['text'])
            group.append(ff + aa if variant == 'arithmetic' else ff)
        extra.append(group)
    return np.concatenate([x, np.asarray(extra)], axis=-1)
