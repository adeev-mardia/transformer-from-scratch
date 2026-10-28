import numpy as np

from transformer_scratch.autograd import Tensor, cross_entropy_loss


def _numerical_grad(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f1 = f(x)
        x[idx] = orig - eps
        f2 = f(x)
        x[idx] = orig
        grad[idx] = (f1 - f2) / (2 * eps)
    return grad


def test_matmul_gradient():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 3))
    B = rng.standard_normal((3, 5))

    def forward(a):
        return float((Tensor(a) @ Tensor(B)).sum().data)

    def analytic(a):
        t = Tensor(a, requires_grad=True)
        out = (t @ Tensor(B, requires_grad=False)).sum()
        out.backward()
        return t.grad

    num = _numerical_grad(forward, A.copy())
    ana = analytic(A.copy())
    assert np.allclose(num, ana, atol=1e-5)


def test_gelu_gradient():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((5, 5)) * 2

    def forward(v):
        return float(Tensor(v).gelu().sum().data)

    def analytic(v):
        t = Tensor(v, requires_grad=True)
        out = t.gelu().sum()
        out.backward()
        return t.grad

    num = _numerical_grad(forward, x.copy())
    ana = analytic(x.copy())
    assert np.allclose(num, ana, atol=1e-4)


def test_softmax_gradient():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 6))
    weights = rng.standard_normal((4, 6))

    def forward(v):
        s = Tensor(v).softmax(axis=-1)
        return float((s * Tensor(weights)).sum().data)

    def analytic(v):
        t = Tensor(v, requires_grad=True)
        s = t.softmax(axis=-1)
        out = (s * Tensor(weights, requires_grad=False)).sum()
        out.backward()
        return t.grad

    num = _numerical_grad(forward, x.copy())
    ana = analytic(x.copy())
    assert np.allclose(num, ana, atol=1e-5)


def test_layer_norm_gradient():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((3, 8))
    gamma_val = rng.standard_normal(8)
    beta_val = rng.standard_normal(8)

    def forward(v):
        g = Tensor(gamma_val, requires_grad=False)
        b = Tensor(beta_val, requires_grad=False)
        return float(Tensor(v).layer_norm(g, b).sum().data)

    def analytic(v):
        t = Tensor(v, requires_grad=True)
        g = Tensor(gamma_val, requires_grad=False)
        b = Tensor(beta_val, requires_grad=False)
        out = t.layer_norm(g, b).sum()
        out.backward()
        return t.grad

    num = _numerical_grad(forward, x.copy())
    ana = analytic(x.copy())
    assert np.allclose(num, ana, atol=1e-4)


def test_cross_entropy_gradient():
    rng = np.random.default_rng(4)
    logits = rng.standard_normal((6, 7)) * 0.5
    targets = rng.integers(0, 7, size=6)

    def forward(v):
        return float(cross_entropy_loss(Tensor(v), targets).data)

    def analytic(v):
        t = Tensor(v, requires_grad=True)
        loss = cross_entropy_loss(t, targets)
        loss.backward()
        return t.grad

    num = _numerical_grad(forward, logits.copy())
    ana = analytic(logits.copy())
    assert np.allclose(num, ana, atol=1e-4)


def test_embedding_lookup_gradient_accumulates_duplicates():
    rng = np.random.default_rng(5)
    table = rng.standard_normal((5, 3))
    idx = np.array([0, 1, 0, 2, 0])  # index 0 repeated three times

    t = Tensor(table, requires_grad=True)
    out = t.embedding_lookup(idx)
    out.backward()  # default upstream grad of ones

    expected = np.zeros_like(table)
    for i in idx:
        expected[i] += 1.0
    assert np.allclose(t.grad, expected)
