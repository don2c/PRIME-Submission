"""Governed opening: the opening handle oh, Open, JudgeOpen, authorisation records and the audit log.

oh encrypts a 32-byte key identifier of the signer's epoch key under the auditor key apk.
The PKE is a matrix-LWE scheme (dimension 256, q = 3329, centred binomial noise, eta = 2)
made deterministic with the Fujisaki-Okamoto transform: oh = (Enc(apk, seed; G(seed)), H(seed) xor id).
Open recovers seed and id. The opening proof pi_open is seed itself. JudgeOpen re-encrypts
with G(seed) and checks both parts, so anyone can check that oh opens to the claimed id.
Parameters are for testing correctness, not a security claim."""
import hashlib
import hmac

import numpy as np

N_LWE, Q, ETA, MSG_BITS = 256, 3329, 2, 256


def _cbd(rng, shape):
    b = rng.integers(0, 2, size=(2 * ETA,) + tuple(shape))
    return (b[:ETA].sum(0) - b[ETA:].sum(0)).astype(np.float64)


class AuditorPKE:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.A = rng.integers(0, Q, size=(N_LWE, N_LWE)).astype(np.float64)
        self.S = _cbd(rng, (N_LWE, MSG_BITS))
        self.B = np.mod(self.A @ self.S + _cbd(rng, (N_LWE, MSG_BITS)), Q)

    @staticmethod
    def _coins(seed_bytes):
        r = np.random.default_rng(int.from_bytes(hashlib.sha256(b"G" + seed_bytes).digest()[:8], "little"))
        return _cbd(r, (N_LWE,)), _cbd(r, (N_LWE,)), _cbd(r, (MSG_BITS,))

    def enc_batch(self, seeds):
        """seeds: list of 32-byte values. Returns (U, V) with one row per seed."""
        k = len(seeds)
        R = np.empty((k, N_LWE)); E1 = np.empty((k, N_LWE)); E2 = np.empty((k, MSG_BITS))
        M = np.empty((k, MSG_BITS))
        for i, s in enumerate(seeds):
            R[i], E1[i], E2[i] = self._coins(s)
            M[i] = np.unpackbits(np.frombuffer(s, dtype=np.uint8))
        U = np.mod(R @ self.A + E1, Q)
        V = np.mod(R @ self.B + E2 + np.floor(Q / 2) * M, Q)
        return U.astype(np.int32), V.astype(np.int32)

    def dec(self, U, V):
        W = np.mod(V.astype(np.float64) - U.astype(np.float64) @ self.S, Q)
        bits = ((W > Q / 4) & (W < 3 * Q / 4)).astype(np.uint8)
        return [np.packbits(b).tobytes() for b in np.atleast_2d(bits)]


def mask(seed):
    return hashlib.sha256(b"H" + seed).digest()


def xor(a, b):
    return bytes(x ^ y for x, y in zip(a, b))


def make_handles(pke, key_ids, rng):
    seeds = [rng.bytes(32) for _ in key_ids]
    U, V = pke.enc_batch(seeds)
    c2 = [xor(mask(s), kid) for s, kid in zip(seeds, key_ids)]
    return U, V, c2


def judge_open(pke, U, V, c2, claimed_id, pi_open):
    U2, V2 = pke.enc_batch([pi_open])
    return bool(np.array_equal(U2[0], U) and np.array_equal(V2[0], V) and xor(mask(pi_open), c2) == claimed_id)


class GovernedOpener:
    """Holds okey. Decrypts only after a valid authorisation record is appended to a hash-chained log."""

    def __init__(self, pke, gov_key):
        self.pke, self.gov_key = pke, gov_key
        self.log, self.head = [], b"\x00" * 32

    def authz_valid(self, tid, req, tag):
        return tag is not None and hmac.compare_digest(
            tag, hmac.new(self.gov_key, f"{tid}|{req}".encode(), hashlib.sha256).digest())

    def open(self, tid, req, tag, U, V, c2):
        if not self.authz_valid(tid, req, tag):
            return None
        seed = self.pke.dec(U[None, :], V[None, :])[0]
        U2, V2 = self.pke.enc_batch([seed])
        if not (np.array_equal(U2[0], U) and np.array_equal(V2[0], V)):
            return None                    # tampered or malformed handle: refuse
        kid = xor(mask(seed), c2)
        rec = f"{tid}|{req}|{kid.hex()}".encode()
        self.head = hashlib.sha256(self.head + rec).digest()
        self.log.append((tid, req, kid, self.head))
        return kid, seed

    def log_consistent(self):
        h = b"\x00" * 32
        for tid, req, kid, head in self.log:
            h = hashlib.sha256(h + f"{tid}|{req}|{kid.hex()}".encode()).digest()
            if h != head:
                return False
        return True


def issue_authz(gov_key, tid, req):
    return hmac.new(gov_key, f"{tid}|{req}".encode(), hashlib.sha256).digest()
