"""Small, model-specific FP32 reference galleries without storage or tracking."""
from numbers import Integral, Real

import numpy as np


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(name + ' must be a positive integer')
    return int(value)


def _person_id(value):
    if isinstance(value, str) and value.strip():
        return value
    if not isinstance(value, (bool, np.bool_)) and isinstance(value, Integral) and value > 0:
        return int(value)
    raise ValueError('person_id must be a nonempty string or a positive integer')


class ReferenceMatrix:
    """Keep normalized references in one expandable matrix, with an optional cap.

    Only ``matrix[:count]`` participates in matching. Similarity is the maximum
    over each person's samples; margin compares two different people. With one
    person the margin is similarity + 1, not a confidence or correctness rate.
    The caller owns synchronization and the decision to trust a new sample.
    """

    def __init__(self, dimension, model_id, *, max_samples=None, initial_capacity=256,
                 duplicate_threshold=.98):
        self.dimension = _positive_integer(dimension, 'dimension')
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError('model_id must be a nonempty string')
        self.model_id = model_id
        self.max_samples = None if max_samples is None else _positive_integer(max_samples, 'max_samples')
        initial_capacity = _positive_integer(initial_capacity, 'initial_capacity')
        if (isinstance(duplicate_threshold, (bool, np.bool_))
                or not isinstance(duplicate_threshold, Real)
                or not np.isfinite(duplicate_threshold)
                or not -1 <= duplicate_threshold <= 1):
            raise ValueError('duplicate_threshold must be finite and in [-1, 1]')
        self.duplicate_threshold = float(duplicate_threshold)
        self.matrix = np.empty((initial_capacity, self.dimension), dtype=np.float32)
        self._person_ids = []
        self._manual = []
        self._added_order = []
        self._person_rows = {}
        self._next_order = 0

    @property
    def count(self):
        return len(self._person_ids)

    @property
    def capacity(self):
        return self.matrix.shape[0]

    def __len__(self):
        return self.count

    @property
    def sample_counts(self):
        return {person: len(rows) for person, rows in self._person_rows.items()}

    @property
    def stats(self):
        manual = sum(self._manual)
        return dict(model_id=self.model_id, dimension=self.dimension, dtype='float32',
                    count=self.count, capacity=self.capacity, people=len(self._person_rows),
                    max_samples=self.max_samples, manual_samples=manual,
                    automatic_samples=self.count - manual, matrix_bytes=self.matrix.nbytes)

    def _normalize(self, feature):
        value = np.asarray(feature)
        if value.ndim != 1 or value.shape[0] != self.dimension or value.dtype.kind not in 'iuf':
            raise ValueError('feature must be a numeric vector matching the model dimension')
        value = value.astype(np.float64, copy=True)
        if not np.isfinite(value).all():
            raise ValueError('feature must contain only finite values')
        scale = np.max(np.abs(value))
        if scale == 0:
            raise ValueError('feature must have a nonzero norm')
        # Scaling first also handles finite inputs whose squared norm overflows.
        value /= scale
        value /= np.linalg.norm(value)
        return value.astype(np.float32)

    def _reserve(self):
        if self.count < self.capacity:
            return
        matrix = np.empty((self.capacity * 2, self.dimension), dtype=np.float32)
        matrix[:self.count] = self.matrix[:self.count]
        self.matrix = matrix

    def add(self, person_id, feature, *, manual=False):
        """Return (added/replaced/skipped, reason), protecting manual samples.

        A manual near-duplicate promotes an automatic row. A full person's
        gallery replaces only its oldest automatic row, regardless of whether
        the new sample is manual or automatic. Other people's rows are untouched.
        """
        person_id = _person_id(person_id)
        if not isinstance(manual, bool):
            raise ValueError('manual must be a boolean')
        vector = self._normalize(feature)
        rows = self._person_rows.get(person_id, set())
        duplicates = [row for row in rows
                      if float(self.matrix[row] @ vector) >= self.duplicate_threshold]
        reason = 'new_sample'
        row = None
        if duplicates:
            if not manual or any(self._manual[index] for index in duplicates):
                return 'skipped', 'duplicate'
            row = min(duplicates, key=lambda index: self._added_order[index])
            reason = 'promoted_to_manual'
        elif self.max_samples is not None and len(rows) >= self.max_samples:
            automatic = [index for index in rows if not self._manual[index]]
            if not automatic:
                return 'skipped', 'manual_limit'
            row = min(automatic, key=lambda index: self._added_order[index])
            reason = 'oldest_automatic'
        if row is None:
            self._reserve()
            row = self.count
            self._person_ids.append(person_id)
            self._manual.append(manual)
            self._added_order.append(self._next_order)
            self._person_rows.setdefault(person_id, set()).add(row)
            action = 'added'
        else:
            self._manual[row] = manual
            self._added_order[row] = self._next_order
            action = 'replaced'
        self.matrix[row] = vector
        self._next_order += 1
        return action, reason

    def match(self, feature):
        """Return the best person's raw score and distinct-person margin, or None."""
        vector = self._normalize(feature)
        if not self.count:
            return None
        scores = np.clip(self.matrix[:self.count] @ vector, -1., 1.)
        ranked = sorted(((person, float(max(scores[row] for row in rows)))
                         for person, rows in self._person_rows.items()),
                        key=lambda item: item[1], reverse=True)
        person, score = ranked[0]
        return dict(person_id=person, similarity=score,
                    margin=score - ranked[1][1] if len(ranked) > 1 else score + 1.)

    def contains(self, person_id):
        return _person_id(person_id) in self._person_rows

    def remove(self, person_id):
        """Delete a person's rows, filling each hole with the last live row."""
        person_id = _person_id(person_id)
        removed = len(self._person_rows.get(person_id, ()))
        while person_id in self._person_rows:
            row = next(iter(self._person_rows[person_id]))
            last = self.count - 1
            self._person_rows[person_id].remove(row)
            if not self._person_rows[person_id]:
                del self._person_rows[person_id]
            if row != last:
                moved = self._person_ids[last]
                self.matrix[row] = self.matrix[last]
                self._person_ids[row] = moved
                self._manual[row] = self._manual[last]
                self._added_order[row] = self._added_order[last]
                self._person_rows[moved].remove(last)
                self._person_rows[moved].add(row)
            self._person_ids.pop()
            self._manual.pop()
            self._added_order.pop()
        return removed

    def clear(self):
        """Forget all references while retaining the allocated matrix for reuse."""
        self._person_ids.clear()
        self._manual.clear()
        self._added_order.clear()
        self._person_rows.clear()
        self._next_order = 0
