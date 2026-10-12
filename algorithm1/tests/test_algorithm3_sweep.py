"""Guard against changing stochastic samples when batching long sweep jobs."""
import numpy as np
from numpy.testing import assert_array_equal
from algorithm3 import build_active_petz, mps_local_magnetization
from local_petz import image_graph
from sweep_algorithm3 import chunked_estimate


def test_chunked_estimator_preserves_all_samples_and_uncertainty():
    graph = image_graph([[0, 0], [255, 255]])
    model = build_active_petz(graph, [-.5, np.nan, np.nan, np.nan], b=2.)
    expected, original = mps_local_magnetization(model, 2, 9, 8, 1e-12, 77)
    progress = []
    actual, batched = chunked_estimate(model, 2, 9, 8, 1e-12, 77,
                                      chunk=4, progress=progress.append)
    assert progress == [4, 9]
    assert_array_equal(actual, expected)
    assert_array_equal(batched['samples'], original['samples'])
    assert_array_equal(batched['se'], original['se'])
    for key in ['max_discarded_weight_sum', 'max_probability_sum_error', 'svd_fallbacks']:
        assert batched[key] == original[key]
