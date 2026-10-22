"""Resolve exact score ties using input content."""

import numpy as np
from selection_utils import candidate_answers

def choose_scores(scores, bank, arrays):
    answers = candidate_answers(bank)
    embeddings = np.asarray(arrays['embeddings'], dtype=np.float64)
    choices = []
    for row_index, (row, parsed) in enumerate(zip(bank, answers)):
        usable = [i for i, value in enumerate(parsed) if value is not None] or list(range(4))
        maximum = max(scores[row_index, i] for i in usable)
        tied = [i for i in usable if scores[row_index, i] == maximum]
        # Content resolves exact score ties without depending on IDs or positions.
        choice = min(tied, key=lambda i: (parsed[i] or '', row['candidates'][i]['text'],
                                        embeddings[row_index, i].tobytes()))
        choices.append(int(choice))
    return choices
