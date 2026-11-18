# Numbers quoted in the text

## Q1 calibration
- mean tau_hat per domain: E-health 0.181, OSN-Facebook 0.240, OSN-Twitter 0.208
- mean tau_hat range: 0.18--0.24
- mean |R|: E-health 16.0, OSN-Facebook 16.0, OSN-Twitter 16.0
- max over runs of p(u0) - tau_hat: E-health 0.000, OSN-Facebook 0.000, OSN-Twitter 0.000
- released (%): E-health 0.0, OSN-Facebook 0.0, OSN-Twitter 0.0
- mean tau_hat inside Bayes-success CI: E-health True, OSN-Facebook True, OSN-Twitter True

## Q2 builders
- Neighbour Bayes success range: 0.840--1.000
- Uniform mean max posterior range: 0.062
- PRIME mean max posterior range: 0.18--0.24
- pass at tau (%): E-health: Uniform 100.0, Neighbour 0.0, Same-bucket 100.0, PRIME 0.0; OSN-Facebook: Uniform 100.0, Neighbour 0.0, Same-bucket 100.0, PRIME 0.0; OSN-Twitter: Uniform 100.0, Neighbour 0.0, Same-bucket 100.0, PRIME 0.0

## Sparsity and churn
- H2 values: [3.0]
- max cover fraction: 0.000
- pool depletion (share of runs) by cell: 
- max change of mean tau_hat across churn within a setting and domain: 0.054
- released (%) max over all cells: 0.0
- hard setting mean tau_hat range: 0.19--0.40

## Constant shape (figure)
- mean timing deviation (ms) range over accepted runs: 0.000--0.004
- rejection causes (%): pool max 0.0, length max 12.0, time max 0.0
- per cell: E-health/0.0: reject 4.0 (pool 0.0, len 4.0, time 0.0), pad 79.7 B; E-health/0.25: reject 0.0 (pool 0.0, len 0.0, time 0.0), pad 83.2 B; E-health/0.5: reject 2.0 (pool 0.0, len 2.0, time 0.0), pad 91.0 B; E-health/0.7: reject 0.0 (pool 0.0, len 0.0, time 0.0), pad 86.4 B; OSN-Twitter/0.0: reject 4.0 (pool 0.0, len 4.0, time 0.0), pad 72.3 B; OSN-Twitter/0.25: reject 0.0 (pool 0.0, len 0.0, time 0.0), pad 73.0 B; OSN-Twitter/0.5: reject 2.0 (pool 0.0, len 2.0, time 0.0), pad 71.3 B; OSN-Twitter/0.7: reject 0.0 (pool 0.0, len 0.0, time 0.0), pad 73.6 B; OSN-Facebook/0.0: reject 0.0 (pool 0.0, len 0.0, time 0.0), pad 53.6 B; OSN-Facebook/0.25: reject 8.0 (pool 0.0, len 8.0, time 0.0), pad 62.3 B; OSN-Facebook/0.5: reject 12.0 (pool 0.0, len 12.0, time 0.0), pad 46.5 B; OSN-Facebook/0.7: reject 8.0 (pool 0.0, len 8.0, time 0.0), pad 54.9 B
- plans: E-health: n=16, L=273449 B, Theta=27.04 ms; OSN-Twitter: n=16, L=273434 B, Theta=34.23 ms; OSN-Facebook: n=16, L=273409 B, Theta=20.73 ms

## Overhead and scaling
- synthesis 0.11 ms, certification 0.89 ms, attack search 20.2 ms, sign 30.6 ms, verify incl. padding 27.0 ms
- tau_AS >= tau_hat in 100% of searches
- scaling synthesis (ms): n=8: 0.08, n=16: 0.09, n=32: 0.10, n=64: 0.11
- certification at largest n: 3.0 ms

## Governed opening
- E-health: 45000 transcripts, 4582 requests, opened 4582, correct 4582, JudgeOpen accepted 4582, unauthorised opened 0, non-signer accepted 0, wrong proof accepted 0, tampered 245 (opened 0), logged once 4582, duplicates 0, hash chain ok True
- OSN-Twitter: 50000 transcripts, 4925 requests, opened 4925, correct 4925, JudgeOpen accepted 4925, unauthorised opened 0, non-signer accepted 0, wrong proof accepted 0, tampered 260 (opened 0), logged once 4925, duplicates 0, hash chain ok True
- OSN-Facebook: 5000 transcripts, 468 requests, opened 468, correct 468, JudgeOpen accepted 468, unauthorised opened 0, non-signer accepted 0, wrong proof accepted 0, tampered 32 (opened 0), logged once 468, duplicates 0, hash chain ok True

## Freshness and forward security
- E-health: fresh 1.000, same-epoch replay 0.000, older-epoch replay 0.000, stale status 0.000, past-epoch forgery 0.000, crypto linker 0.513, ring-overlap linker 0.648 (192 transcripts, 1000 pairs)
- OSN-Twitter: fresh 1.000, same-epoch replay 0.000, older-epoch replay 0.000, stale status 0.000, past-epoch forgery 0.000, crypto linker 0.473, ring-overlap linker 0.977 (192 transcripts, 1000 pairs)
- OSN-Facebook: fresh 1.000, same-epoch replay 0.000, older-epoch replay 0.000, stale status 0.000, past-epoch forgery 0.000, crypto linker 0.519, ring-overlap linker 0.946 (192 transcripts, 1000 pairs)

## AOV hiding
- E-health/Policy body: best attacker logreg acc 0.446, chance 0.455, with ring membership 0.328, positive control 1.000
- E-health/Attribute value: best attacker logreg acc 0.018, chance 0.021, with ring membership 0.071, positive control 0.815
- E-health/Policy mask: best attacker logreg acc 0.439, chance 0.455, with ring membership 0.336, positive control 1.000
- OSN-Twitter/Policy body: best attacker logreg acc 0.444, chance 0.453, with ring membership 0.416, positive control 1.000
- OSN-Twitter/Attribute value: best attacker logreg acc 0.022, chance 0.027, with ring membership 0.025, positive control 0.780
- OSN-Twitter/Policy mask: best attacker logreg acc 0.701, chance 0.701, with ring membership 0.692, positive control 1.000
- OSN-Facebook/Policy body: best attacker logreg acc 0.408, chance 0.407, with ring membership 0.349, positive control 1.000
- OSN-Facebook/Attribute value: best attacker gbt acc 0.018, chance 0.017, with ring membership 0.023, positive control 0.778
- OSN-Facebook/Policy mask: best attacker logreg acc 0.575, chance 0.575, with ring membership 0.535, positive control 1.000
- transcripts: E-health: 1500 sampled, 9 length-rejected; OSN-Twitter: 1500 sampled, 28 length-rejected; OSN-Facebook: 1500 sampled, 131 length-rejected

## Ablations
- No shape policy 0.832, attr 0.046 (policy-size model: PROXY: distinct granted attribute tuples per policy handle)
- Signer not counted: Bayes success 0.436, bucket 1.000
- No support control: Bayes success 0.104, release 42.8%

