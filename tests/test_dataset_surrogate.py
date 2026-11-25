import numpy as np
import pytest

from rbc import dataset as ds
from rbc.evaluate import r2, rel_l2
from rbc.surrogate import Transform, fit_gp, train_mlp


def test_latin_hypercube_one_point_per_bin():
    u = ds.latin_hypercube(20, 3, np.random.default_rng(0))
    for j in range(3):
        assert sorted(np.floor(u[:, j] * 20).astype(int)) == list(range(20))


def test_split_disjoint_complete_reproducible():
    a, b = ds.split_indices(100, seed=1), ds.split_indices(100, seed=1)
    all_idx = np.concatenate(list(a.values()))
    assert sorted(all_idx) == list(range(100))
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_run_roundtrip(tmp_path):
    paths = ds.generate_channel_runs(tmp_path, n=3, seed=0)
    run = ds.load_run(paths[0])
    assert set(run) >= {"y", "u", "tau", "gdot", "G", "params"}
    X, Y, ids = ds.build_table(tmp_path)
    assert X.shape == (3, len(ds.FEATURES)) and Y.shape == (3, len(ds.TARGETS)) and len(ids) == 3


def test_transform_inverse():
    A = np.array([[1.0, -2.0], [10.0, 3.0], [100.0, 0.5]])
    t = Transform(np.array([True, False])).fit(A)
    assert np.allclose(t.inverse(t.forward(A)), A)


def test_metrics_perfect_prediction():
    Y = np.random.default_rng(0).random((10, 2)) + 1.0
    assert np.allclose(rel_l2(Y, Y), 0.0) and np.allclose(r2(Y, Y), 1.0)


def test_mlp_and_gp_fit_smooth_function():
    rng = np.random.default_rng(0)
    X = rng.uniform(1.0, 10.0, (40, 2))
    Y = np.column_stack([X[:, 0] * X[:, 1], np.sin(X[:, 0])])
    log_x, log_y = np.array([True, True]), np.array([True, False])
    mlp = train_mlp(X[:30], Y[:30], X[30:], Y[30:], log_x, log_y, epochs=1500, seed=0)
    gp = fit_gp(X[:30], Y[:30], log_x, log_y)
    for model in (mlp, gp):
        assert rel_l2(model.predict(X[30:]), Y[30:])[0] < 0.1
