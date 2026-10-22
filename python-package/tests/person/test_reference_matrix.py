import numpy as np
import pytest

from insightface.app.person.matrix import ReferenceMatrix


def gallery(max_samples=4, **kwargs):
    return ReferenceMatrix(3, 'test-model', max_samples=max_samples, **kwargs)


def assert_rows_consistent(matrix):
    assert len(matrix._person_ids) == len(matrix._manual) == len(matrix._added_order) == matrix.count
    indexed = {row for rows in matrix._person_rows.values() for row in rows}
    assert indexed == set(range(matrix.count))
    for person, rows in matrix._person_rows.items():
        assert all(matrix._person_ids[row] == person for row in rows)
        if matrix.max_samples is not None:
            assert len(rows) <= matrix.max_samples
    assert matrix.matrix.dtype == np.float32
    if matrix.count:
        np.testing.assert_allclose(np.linalg.norm(matrix.matrix[:matrix.count], axis=1), 1, atol=1e-6)


def test_default_capacity_and_only_live_rows_are_searched():
    matrix = gallery()
    assert matrix.capacity == 256 and matrix.count == 0
    assert matrix.match([1, 0, 0]) is None
    matrix.add('a', [3, 0, 0], manual=True)
    matrix.matrix[1:] = np.nan
    assert matrix.match([2, 0, 0]) == {'person_id': 'a', 'similarity': 1., 'margin': 2.}
    assert matrix.stats['matrix_bytes'] == 256 * 3 * 4


def test_growth_doubles_capacity_and_preserves_fp32_references():
    matrix = gallery(initial_capacity=2)
    values = ([1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0])
    capacities = []
    for index, value in enumerate(values, 1):
        assert matrix.add(index, value, manual=True)[0] == 'added'
        capacities.append(matrix.capacity)
    assert capacities == [2, 2, 4, 4, 8]
    matrix.matrix[matrix.count:] = 1000
    for index, value in enumerate(values, 1):
        assert matrix.match(value)['person_id'] == index
    assert matrix.sample_counts == {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
    assert_rows_consistent(matrix)


def test_duplicate_is_per_person_and_manual_promotes_automatic():
    matrix = gallery(max_samples=2)
    assert matrix.add('a', [1, 0, 0]) == ('added', 'new_sample')
    assert matrix.add('a', [100, 0, 0]) == ('skipped', 'duplicate')
    assert matrix.add('b', [1, 0, 0]) == ('added', 'new_sample')
    assert matrix.add('a', [1, .01, 0], manual=True) == ('replaced', 'promoted_to_manual')
    assert matrix.count == 2 and matrix.stats['manual_samples'] == 1
    assert matrix.add('a', [1, 0, 0]) == ('skipped', 'duplicate')
    assert matrix.add('a', [1, 0, 0], manual=True) == ('skipped', 'duplicate')
    assert_rows_consistent(matrix)


def test_automatic_replaces_oldest_automatic_and_protects_manual():
    matrix = gallery(max_samples=3)
    matrix.add('a', [1, 0, 0], manual=True)
    matrix.add('a', [0, 1, 0])
    matrix.add('a', [0, 0, 1])
    assert matrix.add('a', [-1, 0, 0]) == ('replaced', 'oldest_automatic')
    assert matrix.match([0, 1, 0])['similarity'] == 0.
    assert matrix.match([1, 0, 0])['similarity'] == 1.
    assert matrix.add('a', [0, -1, 0]) == ('replaced', 'oldest_automatic')
    assert matrix.match([0, 0, 1])['similarity'] == 0.
    assert matrix.stats['manual_samples'] == 1
    assert_rows_consistent(matrix)


def test_manual_replaces_auto_and_full_manual_gallery_is_unchanged():
    matrix = gallery(max_samples=2)
    matrix.add(1, [1, 0, 0], manual=True)
    matrix.add(1, [0, 1, 0])
    assert matrix.add(1, [0, 0, 1], manual=True) == ('replaced', 'oldest_automatic')
    before = matrix.matrix[:matrix.count].copy()
    for manual in (False, True):
        assert matrix.add(1, [-1, 0, 0], manual=manual) == ('skipped', 'manual_limit')
        np.testing.assert_array_equal(matrix.matrix[:matrix.count], before)
    assert matrix.stats['manual_samples'] == 2
    assert matrix.stats['automatic_samples'] == 0


def test_removal_swaps_keep_labels_indices_and_replacement_age_correct():
    matrix = gallery(max_samples=2, initial_capacity=2)
    matrix.add('a', [1, 0, 0], manual=True)
    matrix.add(2, [0, 1, 0])
    matrix.add('a', [0, 0, 1])
    matrix.add(2, [-1, 0, 0])
    matrix.add('c', [0, 0, -1], manual=True)
    assert matrix.remove('a') == 2
    assert not matrix.contains('a') and matrix.contains(2)
    assert matrix.match([0, 0, -1])['person_id'] == 'c'
    assert matrix.add(2, [0, -1, 0]) == ('replaced', 'oldest_automatic')
    assert matrix.match([-1, 0, 0])['person_id'] == 2
    assert_rows_consistent(matrix)
    assert matrix.remove('missing') == 0
    assert matrix.remove(2) == 2
    assert matrix.remove('c') == 1
    assert matrix.count == 0 and matrix.match([1, 0, 0]) is None
    assert_rows_consistent(matrix)


def test_remove_when_last_row_belongs_to_same_person():
    matrix = gallery()
    matrix.add('a', [1, 0, 0])
    matrix.add('b', [0, 1, 0])
    matrix.add('a', [0, 0, 1])
    assert matrix.remove('a') == 2
    assert matrix.sample_counts == {'b': 1}
    assert matrix.match([0, 1, 0])['person_id'] == 'b'
    assert_rows_consistent(matrix)


def test_margin_compares_people_not_two_samples_of_one_person():
    matrix = gallery()
    matrix.add('a', [.9, np.sqrt(1 - .9**2), 0])
    matrix.add('a', [.85, 0, np.sqrt(1 - .85**2)])
    matrix.add(2, [.7, 0, np.sqrt(1 - .7**2)])
    result = matrix.match([1, 0, 0])
    assert result['person_id'] == 'a'
    assert result['similarity'] == pytest.approx(.9)
    assert result['margin'] == pytest.approx(.2)
    matrix.add('tie', [.9, np.sqrt(1 - .9**2), 0])
    assert matrix.match([1, 0, 0])['margin'] == pytest.approx(0.)


def test_clear_reuses_capacity_without_searching_old_rows():
    matrix = gallery(initial_capacity=1)
    matrix.add('a', [1, 0, 0])
    matrix.add('b', [0, 1, 0])
    capacity = matrix.capacity
    matrix.clear()
    assert matrix.capacity == capacity and matrix.count == 0
    assert matrix.stats['people'] == 0 and matrix.sample_counts == {}
    assert matrix.match([1, 0, 0]) is None
    matrix.add('c', [0, 0, 1], manual=True)
    assert matrix.match([0, 1, 0])['person_id'] == 'c'
    assert_rows_consistent(matrix)


def test_normalization_is_stable_and_does_not_mutate_input():
    matrix = gallery()
    value = np.array([1e308, 1e308, 0.], dtype=np.float64)
    before = value.copy()
    matrix.add(np.int64(7), value)
    np.testing.assert_array_equal(value, before)
    assert matrix.match([1., 1., 0.])['similarity'] == pytest.approx(1.)
    assert matrix.contains(7)
    assert_rows_consistent(matrix)


@pytest.mark.parametrize('feature', [[], [1, 0], [1, 0, 0, 0], [[1, 0, 0]],
    [0, 0, 0], [np.nan, 1, 0], [np.inf, 1, 0], [-np.inf, 1, 0],
    ['1', '0', '0'], [True, False, False], [1j, 0, 0]])
def test_invalid_vectors_are_rejected_without_changing_gallery(feature):
    matrix = gallery()
    matrix.add('a', [1, 0, 0], manual=True)
    before = matrix.matrix[:matrix.count].copy()
    with pytest.raises(ValueError):
        matrix.add('b', feature)
    with pytest.raises(ValueError):
        matrix.match(feature)
    assert matrix.sample_counts == {'a': 1}
    np.testing.assert_array_equal(matrix.matrix[:matrix.count], before)


@pytest.mark.parametrize('person', [None, '', '  ', 0, -1, 1.5, True, np.bool_(True)])
def test_invalid_person_ids_are_rejected(person):
    matrix = gallery()
    with pytest.raises(ValueError):
        matrix.add(person, [1, 0, 0])
    with pytest.raises(ValueError):
        matrix.remove(person)
    with pytest.raises(ValueError):
        matrix.contains(person)


@pytest.mark.parametrize('kwargs', [dict(dimension=0), dict(dimension=True),
    dict(dimension=3.5), dict(model_id=''), dict(model_id=None), dict(max_samples=0),
    dict(max_samples=True), dict(initial_capacity=0), dict(initial_capacity=1.5),
    dict(duplicate_threshold=np.nan), dict(duplicate_threshold=1.1), dict(duplicate_threshold=True)])
def test_invalid_configuration(kwargs):
    options = dict(dimension=3, model_id='test', max_samples=2)
    options.update(kwargs)
    with pytest.raises(ValueError):
        ReferenceMatrix(**options)


def test_manual_flag_requires_bool():
    matrix = gallery()
    with pytest.raises(ValueError):
        matrix.add('a', [1, 0, 0], manual=1)
    assert matrix.count == 0
