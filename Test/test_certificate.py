"""Exactness check for Alg. 4: on a small graph, enumerate every ring the sampler (Alg. 2) can output for
every signer, compute the true Bayes posterior, and compare it with certify(). Run: python -m pytest -q tests"""
import itertools
import math

import numpy as np
from scipy.sparse import coo_matrix

from primering.core import Domain, Near, certify, synthesize


def small_domain(seed=3, n=14, p=0.18):
    rng = np.random.default_rng(seed)
    iu, ju = np.triu_indices(n, 1)
    keep = rng.random(len(iu)) < p
    s, d = iu[keep], ju[keep]
    G = coo_matrix((np.ones(2 * len(s)), (np.r_[s, d], np.r_[d, s])), shape=(n, n)).tocsr()
    bucket = np.array([i % 2 for i in range(n)])
    return Domain("tiny", G, bucket, ["a", "b"])


def test_posterior_is_exact():
    dom = small_domain()
    slots = np.array([2, 2])                     # n = 4, two members per bucket
    near = Near(dom.G, 2)
    active = np.ones(dom.N, dtype=bool)
    prior = np.ones(dom.N) / dom.N
    mass = {}
    for u in range(dom.N):
        need = slots.copy(); need[dom.bucket[u]] -= 1
        excl = set(near(u).tolist()) | {u}
        pools = [[v for v in range(dom.N) if dom.bucket[v] == a and v not in excl] for a in range(2)]
        if any(len(pools[a]) < need[a] for a in range(2)):
            continue
        p_ring = prior[u] / math.prod(math.comb(len(pools[a]), need[a]) for a in range(2))
        for pick in itertools.product(*[itertools.combinations(pools[a], need[a]) for a in range(2)]):
            R = frozenset((u,) + sum(pick, ()))
            mass.setdefault(R, {}).setdefault(u, 0.0)
            mass[R][u] += p_ring
    assert mass, "no feasible ring"
    checked = 0
    for R, m in mass.items():
        Rarr = np.array(sorted(R))
        post = certify(dom, Rarr, slots, near, active, prior)
        tot = sum(m.values())
        truth = np.array([m.get(int(v), 0.0) / tot for v in Rarr])
        assert np.allclose(post, truth, atol=1e-9), (Rarr, post, truth)
        checked += 1
    assert checked > 50


def test_synthesis_hits_target_counts():
    dom = small_domain()
    slots = np.array([2, 2])
    near = Near(dom.G, 2)
    rng = np.random.default_rng(0)
    for u in range(dom.N):
        ring, _ = synthesize(dom, u, slots, near, np.ones(dom.N, dtype=bool), rng, 4)
        if ring is not None and ring.covers == 0:
            assert np.array_equal(np.bincount(dom.bucket[ring.real], minlength=2), slots)
