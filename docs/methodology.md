# Experimental Methodology

| | |
|---|---|
| Status | Phase 1 protocol, 2026-10-26. Written **before** any evaluation data exists. Later changes are listed in §12 with a reason and a date. |
| Related | Hypotheses H1–H6 and sub-questions SQ1–SQ6 are defined in [proposal.md §3](proposal.md#3-research-question-sub-questions-and-hypotheses). |

## 1. Principles

1. **Pre-declaration.** Detector thresholds, the operating false-alarm budget, trial counts, metrics,
   and decision rules are fixed in this document before evaluation runs.
2. **No tuning on fault data.** Thresholds are functions of the false-alarm budget α and the claimed
   min-entropy H of the nominal source model, and of nothing else.
3. **Disjoint randomness.** Development and debugging use seeds derived from a *development* master
   seed. Reported results use a separate *evaluation* master seed, recorded in the configuration
   before the runs start.
4. **Formal versus exploratory.** Detectors that can raise a *critical* alarm must have an exact null
   distribution or a rigorous bound at the configured α. Approximate statistics (normal or χ²
   approximations in extreme tails) may only raise *warnings*. They are labelled exploratory
   everywhere they appear.
5. **Report everything.** Every configured cell is reported, including failures and inconclusive
   results.

## 2. Simulated entropy-source models

The research path monitors **simulated raw noise samples**. It never touches the host operating
system's RNG.

**Primary sample model: binary noise samples** (one bit per sample, stored as `uint8` 0/1). This is
the case for which SP 800-90B specifies the APT window W = 1024. A byte-sample mode (W = 512) is
supported for RCT, APT, repeated-block, and availability detectors. The binary-specific supplementary
detectors are disabled in byte mode.

The simulation randomness comes from NumPy's PCG64 generator, seeded through `SeedSequence`. PCG64 is
**not** a cryptographic generator. It is used only to model the statistical behaviour of a noise
source, a role for which a statistically strong PRNG is appropriate.

| Model | Definition | Claimed min-entropy H per sample | Role |
|---|---|---|---|
| S0 Ideal | IID Bernoulli(0.5) | 1 | Healthy reference |
| S1 Imperfect-IID | IID Bernoulli(p₀) with p₀ = 0.55 | −log₂(0.55) ≈ 0.8625 | Healthy but biased source: tests that detectors are calibrated to the nominal model rather than to ideal uniformity |
| S2 Imperfect-Markov | Two-state Markov chain with P(xᵢ₊₁ = xᵢ) = 0.55 | −log₂(0.55) ≈ 0.8625 (conservative per-sample bound) | Healthy but correlated source |

In a real system, H would come from an SP 800-90B entropy assessment of the physical source. Here it
is known by construction. **This is a major simplification**, discussed in §11.

## 3. Fault taxonomy

Each fault is applied to a healthy stream from onset sample t₀ onward, either permanently or in
bursts. Every sample carries a ground-truth label (`healthy` or `faulty`, plus the fault ID).

| ID | Fault family | Parameters (sweep range) | Reference intensity (used for H1) | Gross? |
|---|---|---|---|---|
| F1 | Constant output (stuck-at) | value ∈ {0, 1} | permanent constant | yes |
| F2 | Stuck bit (byte mode only) | bit position, value | permanent | yes (byte mode) |
| F3 | Bias | P(1) = p ∈ [0.50, 0.75] | p = 0.65 | yes |
| F4 | Short repeating pattern | period P ∈ [2, 256] samples, pattern drawn once from the healthy model | P = 16 | yes |
| F5 | Repeated blocks (buffer replay) | block length B ∈ [64, 4096] samples, replay distance | B = 512, re-emitted from recent history | yes |
| F6 | Lag correlation | P(xᵢ₊₁ = xᵢ) = q ∈ [0.55, 0.80] | q = 0.70 | no (sweep only) |
| F7 | Abrupt distribution change | change-point t₀ to one of F3/F6 at given intensity | p = 0.60 | no (sweep only) |
| F8a | Reseeded replay (VM-snapshot analogue) | the post-onset stream replays an earlier segment generated from the same seed | replay of a 2^14-sample segment | no (sweep only) |
| F8b | Low-entropy deterministic substitution | the output is replaced by an LCG seeded from a 16-bit seed space, with no replay | n/a | no: **expected undetectable** |
| F9 | Source outage | exception, short read, or timeout at t₀ (one-off, intermittent, or permanent) | permanent exception | yes |
| F10 | Intermittent or bursty fault | any of F1–F7 active in on/off bursts; geometric durations with means (μ_on, μ_off) | F1 bursts, μ_on = 64, μ_off = 4096 | no (sweep only) |
| N1 | **Negative control:** zero-secrecy deterministic stream | ChaCha20 keystream under a fixed, published key | n/a | expected undetectable (H6) |

Onset t₀ is drawn uniformly from [W_up, W_up + 2^16], where W_up = 2^16 samples of warm-up. The
post-onset observation budget is N_max = 2^20 samples. Gross faults form the H1 set.

## 4. Detectors and threshold derivations

Notation: α is the per-evaluation false-alarm budget of one detector; H is the claimed min-entropy
per sample of the nominal model; p₀ is the nominal P(1).

### 4.1 Formal health tests (SP 800-90B §4.4 style), critical level

These follow the structure of SP 800-90B §4.4. The implementation will be checked line by line
against the specification text in Phase 3. This is **not** a claim of SP 800-90B conformance.

| Detector | Statistic | Cutoff | Evidence reported |
|---|---|---|---|
| RCT: Repetition Count Test | Length of the current run of identical samples | C = 1 + ⌈−log₂(α) / H⌉ | Run length, sample index, value, C, α, H |
| APT: Adaptive Proportion Test | Within each non-overlapping window of W samples (1024 binary, 512 byte), the count of occurrences of the window's first sample value | C = 1 + CRITBINOM(W, 2^−H, 1 − α), the smallest c such that P(Binomial(W, 2^−H) ≥ c) ≤ α, computed exactly with SciPy | Count, window index, value, C, α, H, W |

Worked RCT examples at the operating α = 2^−30 (§4.4 below explains this choice):
S0 (H = 1): C = 1 + ⌈30⌉ = 31. S1/S2 (H ≈ 0.8625): C = 1 + ⌈34.78⌉ = 36.

**Verification against the standard (Phase 3).** The implementation was checked against the text of
SP 800-90B §4.4. Unit tests reproduce all ten APT cutoffs in its Table 2 (α = 2^−20; binary
W = 1024: 941, 840, 748, 664, 589; non-binary W = 512: 410, 311, 177, 62, 13) and both RCT worked
examples (C = 11 for H = 2 at 2^−20; C = 6 for H = 8 at 2^−40).

**Finding: the APT's exact false-alarm probability can exceed α.** The standard's cutoff
C = 1 + CRITBINOM(W, 2^−H, 1 − α) bounds P(Binomial(W, 2^−H) ≥ C). The test as specified, however,
always counts the window's first sample (B starts at 1), so its statistic is
1 + Binomial(W − 1, P(A)). The exact worst-case probability P(1 + Binomial(W − 1, 2^−H) ≥ C) is:
- up to about 1.13 α for binary sources;
- about **3.5 α** for uniform bytes (H = 8, W = 512) at α = 2^−30, and **5.2 α** at α = 2^−40.

The implementation keeps the standard's cutoff and reports the exact probability next to it
(`DetectorSpec.false_alarm_probability`). A unit test confirms empirically, on 20,000 windows at a
relaxed α, that observed alarm counts match the exact probability and reject the formula's.
Consequence for H2: the pre-declared rule compares observed rates with nominal α, and APT rates are
expected to exceed it slightly by construction. The report will show both comparisons. This
finding was not checked against the literature; it may already be known.

### 4.2 Supplementary detectors with exact nulls, critical level

| Detector | Null model and statistic | Exact threshold | Targets |
|---|---|---|---|
| Repeated block | Each aligned block of k samples is compared exactly with **every** k-sample window, at any offset, that ends before the block and starts within the last L = 2^16 samples. For independent samples, two non-overlapping windows match with probability ≤ 2^(−kH). | Choose the smallest k with L·2^(−kH) ≤ α (union bound per block). At 2^−30: k = 46 for S0, 54 for S1/S2. | F4, F5, F8a |
| Windowed frequency | Count of ones in a window of n samples is Binomial(n, p₀) under the IID nominal model | Two-sided exact binomial quantiles at α/2 per tail | F3, F7 |
| Runs (transition count) | Number of value changes among n consecutive IID Bernoulli(p₀) samples. Its exact distribution is computed once per (n, p₀) by dynamic programming over (last value, transitions so far). | Two-sided exact tail at α/2 per tail | F4, F6 |
| Lag-k agreement, k ∈ {1, …, 8} | Disjoint pairs (x_j, x_{j+k}) taken in blocks of 2k so that pairs are independent. The agreement count is Binomial(n/2, p₀² + (1 − p₀)²). | Exact binomial at α/(2·8) per tail (Bonferroni over k) | F4 (period ≤ 8), F6 |
| Source availability | Exceptions, short reads, timeouts. No statistics involved. | Any event is immediately critical | F9 |

Windowed detectors (frequency, runs, lag-k, autocorrelation, byte χ²) use non-overlapping windows of
**4096 samples**. This value was not fixed before Phase 3. It was chosen for sensitivity to the
smallest bias in the F3 sweep, never tuned on evaluation data.

The windowed frequency, runs, and lag-k detectors assume IID samples. For the Markov model S2 they are
re-parameterised with the chain's exact distributions, or disabled where no exact null is
implemented. A disabled detector is recorded in the run metadata; it never silently passes.

### 4.3 Exploratory detectors, warning level only

| Detector | Statistic | Why warning-only |
|---|---|---|
| Lag-k autocorrelation (k ≤ 64) on overlapping samples | Sample autocorrelation, z = r_k·√n | Normal approximation is unreliable at extreme α |
| Byte χ² (byte mode) | Pearson χ² against the nominal distribution | χ² tail approximation at extreme α |

The sliding-window entropy estimate (plug-in Shannon entropy of 8-sample patterns) has no exact null
distribution, so it is a descriptive metric in `analysis/entropy_metrics.py` (Phase 4), not a
detector.

Warnings use α_warn = 10^−3 per evaluation. The dashboard shows the expected number of warnings per
evaluation count next to the observed count, so that warnings are read as the expected background
they are.

### 4.4 Choosing the operating α

SP 800-90B recommends per-test false-positive probabilities in the range 2^−40 ≤ α ≤ 2^−20. The RCT is
evaluated at every sample, so its false-alarm rate is on the order of α per sample. For S0 there are
about n/2 runs in n samples, each reaching length C with probability 2^−(C−1) ≤ α, so the expected
number of false alarms is about n·α/2. Analytically, a binary source at α = 2^−20 would raise on the
order of 2^33·2^−21 = 4096 false RCT alarms per GiB of raw samples. **This analytic figure is checked
empirically in E2.**

**Pre-declared rule:** the operating α is the largest value in {2^−20, 2^−30, 2^−40} for which the
analytic expected number of false critical alarms per trial (all critical detectors, trial length
2^21 samples) is ≤ 0.01.

- At 2^−20 the RCT alone contributes about 2^21·2^−21 = 1 per trial. Rejected.
- At 2^−30 the RCT contributes about 2^−10 ≈ 0.001. Window-based detectors contribute about
  2^11 windows × 2^−30 each, which is negligible. Accepted.

**Operating α = 2^−30 per detector evaluation.** The aggregate per-evaluation budget is bounded by the
sum over critical detectors (union bound). E5 also reports results at 2^−20 and 2^−40 to show the
latency/false-alarm trade-off.

### 4.5 Alert levels and fail-closed aggregation

| Detector result | Meaning |
|---|---|
| `PASS` | Evaluated with sufficient data; statistic within threshold |
| `WARNING` | An exploratory detector exceeded α_warn |
| `CRITICAL` | A formal or exact supplementary detector exceeded its threshold, or a source-availability event occurred |
| `ERROR` | The detector raised an exception or received invalid input |
| `INSUFFICIENT_DATA` | Not enough samples to evaluate |

| Source status | Condition | Output gate |
|---|---|---|
| `HEALTHY` | Every enabled critical detector returned `PASS` on sufficient data | Open |
| `DEGRADED` | As `HEALTHY`, but with ≥ 1 `WARNING` | Open; warning logged |
| `FAILED` | ≥ 1 `CRITICAL`. **Latched** until an explicit reset, mirroring SP 800-90B's error state. | Closed |
| `UNKNOWN` | Any `ERROR`, any `INSUFFICIENT_DATA` on a critical detector, or an invalid configuration | Closed |

No code path maps an exception to `HEALTHY` (acceptance criterion AC3).

## 5. Pipelines under comparison

All pipelines consume **the same raw trace** for a given trial seed.

| Pipeline | Description |
|---|---|
| P0 No monitoring | Raw → SHA-256 conditioning (fixed input block size) → output. Never alarms; establishes exposure. |
| P1 Output tests only | P0 plus the SP 800-22 subset applied to consecutive output blocks of 2^16 bits (each covering 2^17 raw samples). The subset is frequency, block frequency (M chosen by the standard's recommendation), runs, longest run in a block (exact class probabilities), and cumulative sums forward and reverse. Holm correction across the battery at family level 0.01; inapplicable tests are excluded. A rejection is attributed to the last raw sample of the block. |
| P2 Raw-source monitor (proposed) | Raw → health monitor → (gate) → SHA-256 → output |
| P3 Conditioning + DRBG | Conditioned output feeds a stateful DRBG stand-in: for each 32-byte block dᵢ, kᵢ = SHA-256(kᵢ₋₁ ‖ dᵢ), and 32 bytes of ChaCha20 keystream under kᵢ are emitted. Output tests as in P1 show what is visible after a DRBG. |
| Ablation variants of P2 | Single detectors; RCT + APT only; leave-one-out |

## 6. Metrics

A **trial** is one stream: warm-up, then onset, then N_max post-onset samples. Faulty trials contain
one fault schedule. Healthy trials contain none.

| Metric | Definition |
|---|---|
| Detection (trial level) | ≥ 1 `CRITICAL` at a sample index ≥ t₀ in a faulty trial |
| False alarm (trial level) | ≥ 1 `CRITICAL` in a healthy trial, or before t₀ in a faulty trial |
| False-alarm rate (per evaluation) | Alarms ÷ evaluations on healthy data, per detector. Evaluation units: one run (RCT), one window (APT and windowed detectors), one aligned block (repeated block). Each unit raises at most one alarm, so no reset is needed. It is compared with both nominal α and the exact null probability. |
| Detection rate (recall) | Detected faulty trials ÷ faulty trials |
| Excess detection | Detection rate minus the trial-level alarm rate of healthy controls over the same post-onset span. This guards against crediting background false alarms as detections. |
| Precision | TP / (TP + FP) at a **stated prevalence** (the design mix). Precision depends on prevalence, so it is reported with that caveat. |
| Detection latency | Samples from t₀ to the first critical alarm, for detected trials only. Reported as median and 95th percentile, alongside detection rate (undetected trials are right-censored at N_max). |
| Exposure | Faulty raw samples (and the corresponding conditioned bytes, 32 per 512 samples) released before the gate closed. Gating is per buffer of 2^16 samples: the buffer containing the first alarm, and every later one, is withheld. P0 exposes everything. |
| Throughput | MiB/s of raw input processed (P0–P3), or of output produced (generator benchmarks) |
| Latency per call | Wall time per `generate(n)` or `monitor(buffer)` call: mean, median, p95 |
| CPU time | `time.process_time_ns()` per measured block |
| Peak memory | `tracemalloc` peak, in separate runs (tracemalloc distorts timing) |

## 7. Statistical analysis

- **Proportions:** Wilson score 95% intervals; Clopper–Pearson exact intervals when the count is 0
  or n. For example, 200 detections in 200 trials gives a two-sided 95% Clopper–Pearson lower bound
  of 0.025^(1/200) ≈ 0.982.
- **Zero-event false alarms:** one-sided 95% upper bound 1 − 0.05^(1/n) (≈ 3/n).
- **Medians and ratios** (latency, throughput ratio): percentile bootstrap, 10,000 resamples, fixed
  bootstrap seed recorded in the result file.
- **Calibration (H2):** one-sided exact binomial test of observed false alarms against nominal α, per
  calibration level. Holm correction across levels and detectors at family level 0.05.
- **Comparisons between pipelines** (H4, H5): difference in detection proportions with Newcombe's
  hybrid-score interval. Pipelines run on identical traces, so paired analysis (McNemar's exact test)
  is also reported.
- **Multiple cells:** each cell is reported with its own interval. H1's "every gross class" rule is an
  intersection test, which is conservative and needs no multiplicity correction.
- p-values are reported to three significant figures, never as "p = 0". A p-value is never described
  as the probability that a hypothesis is true.

## 8. Experiments and decision rules

| Exp. | Design | Decision rule | Proves | Does not prove |
|---|---|---|---|---|
| **E1 Detection** | Each gross fault at reference intensity on S0 and S1, pipeline P2, 200 trials per cell | **H1 supported** if every gross cell has a Clopper–Pearson/Wilson 95% lower bound on detection rate ≥ 0.95 **and** excess detection > 0 | That the implemented monitor detects these simulated faults within N_max | Detection of real hardware or OS failures |
| **E2 False alarms** | (a) 1,000 healthy trials each for S0, S1, S2 at operating α; (b) calibration at α ∈ {2^−6, 2^−8, 2^−10} with ≥ 10^5 evaluations per detector per level | **H2 supported** if no calibration level shows a significant excess (Holm, 0.05), **and** the operating-α count lies within the central 95% Poisson range of its analytic expectation | That implementation matches theory under the nominal models | Behaviour on non-stationary real sources |
| **E3 Placement** | F1, F3, F4, F6 traces through P0–P3, 200 trials per cell | **H4 supported** if P1 detection on F3 and F6 traces is not significantly above P1's healthy-control alarm rate (McNemar/Newcombe), P2 detects per H1, and P3 output detection ≈ alarm rate for all faults | That conditioning masks source faults from output tests | That the conditioned output is secure, or insecure |
| **E4 Ablation** | P2 variants across all fault families at reference or mid-sweep intensity | **H5 supported** if ≥ 1 family shows a full-vs-(RCT+APT) detection difference with a 95% interval excluding 0. The false-alarm cost of each supplementary detector is reported from E2. | Marginal contribution of each detector on this fault model | Optimality of the detector set |
| **E5 Sensitivity** | Intensity sweeps (F3 p, F4 P, F5 B, F6 q, F10 μ_on) × α ∈ {2^−20, 2^−30, 2^−40}, 200 trials per point | Descriptive: detection-rate and latency curves, with the 50% and 95% detection boundaries interpolated | Where detection stops working | Universal thresholds |
| **E6 Overhead** | P0 vs P2 at buffer sizes {1 KiB, 16 KiB, 64 KiB, 256 KiB, 1 MiB}; generator benchmarks (§9) | **H3 supported** if the median throughput ratio P2/P0 is ≥ 0.5 at every buffer size ≥ 64 KiB and its bootstrap 95% interval excludes < 0.5 | Cost on this machine and stack | Cost of optimised native implementations |
| **E7 Negative control** | N1 and F8b streams on the S0 model (whose statistics they match) through P2 and P1, 200 trials each. On S1, the same streams *are* flagged by the frequency detector, as a shift from p₀ = 0.55 to 0.5; this is reported as an illustration that detectors measure deviation from the nominal model, not predictability. | **H6 confirmed** (as a limitation) if detection is not significantly above the healthy-control alarm rate | That statistical monitoring cannot see deterministic, zero-secrecy streams | Anything about the security of a monitored source |

## 9. Benchmark methodology

**Generators:** `secrets.token_bytes` (OS CSPRNG); LCG, both a pure-Python loop and a NumPy-vectorised
version, labelled separately; ChaCha20 keystream via `cryptography` (encrypting zero bytes).

**Buffer sizes:** 64 B, 1 KiB, 16 KiB, 64 KiB, 1 MiB.

**Procedure:**

1. Run 5 warm-up calls per cell, then ≥ 50 measured repetitions.
2. Randomise cell order with a recorded seed to spread thermal and power drift across cells.
3. Use `time.perf_counter_ns` for wall time and `time.process_time_ns` for CPU time.
4. Leave the garbage collector at its default and record that in metadata.
5. Measure memory in separate runs.

No administrator-only tuning (CPU pinning, frequency locking) is used, and this is recorded as a
limitation.

**Metadata recorded:** CPU model and logical core count, OS and version, Python version and build,
versions of NumPy, SciPy, `cryptography` and OpenSSL, package version, git commit (if available),
power source when detectable, and timestamp.

**Interpretation rules:**

- Throughput is **not** a security metric. The fastest generator is not the most secure.
- APIs differ. `secrets.token_bytes` performs a system call per request. ChaCha20 runs in OpenSSL's C
  code. The LCG runs in Python or NumPy. Comparisons describe cost on this stack, not intrinsic
  algorithm speed.

### 9.1 Output-level tests: verification of the implementation

SP 800-22r1a's worked examples contain inconsistencies, so its printed values are not used
uncritically. Each test is checked three ways (`tests/test_statistics.py`): against an independent
transcription of the formula, against exact distributions where available, and against the
published value where those agree. Discrepancies found in the publication:

- **Longest run (2.4):** two different answers for the same example (χ² = 4.882605, p = 0.180598
  versus 4.882457, p = 0.180609). The first comes from the rounded four-digit table; exact class
  probabilities give the second.
- **Probability tables (3.4):** as listed in the change log (M = 512, 1000, 10^4).
- **Cumulative sums (2.13):** "z = 1.6 / 1.9" is printed where z = max|Sₖ| = 16 / 19. The p-value
  is printed as 0.4116588 and as 0.411658 (truncated).
- **Runs (2.3):** the 10-bit example says both that the test is not run and that it is
  applicable. The 100-bit example lists τ = 0.02, but 2/√n = 0.2.

SP 800-90B's most-common-value example also has a typo: −log₂(0.6822) is printed where the stated
result 0.5363 = −log₂(0.6895). These checks were made against the documents' own text; the
IACR ePrint audit of SP 800-22 (2025/856) could not be retrieved (HTTP 403) for comparison.

## 10. Reproducibility protocol

- One JSON configuration per experiment in `experiments/configs/`, validated by Pydantic models, with
  no executable content.
- Master seed → `SeedSequence.spawn` per cell → per trial. Each trial's spawn key is stored, so any
  single trial can be re-run in isolation.
- Each result file records: schema version, configuration (verbatim), master seed and spawn keys,
  package version and git commit, dependency versions, platform metadata, start and end timestamps,
  and per-trial outcomes (labels, alarm indices, detector evidence).
- Result files **never** contain secure-source output, ChaCha20 keys other than the published
  negative-control key, or raw secret material.
- A SHA-256 digest of each result file's canonical JSON is stored in the experiment history. This
  detects accidental modification only; it is not a signature against deliberate tampering.

## 11. Threats to validity

| Type | Threat | Mitigation or acknowledgement |
|---|---|---|
| External | Synthetic models do not represent real hardware failures (analogue drift, temperature, EM injection). | Stated in every report. Results apply to the model only. |
| External | H is known by construction. In practice, entropy estimation is hard and error-prone [Kelsey-2015]. | Discussed. A sensitivity check with a mis-specified H is planned as future work. |
| Internal | Circularity: the author designs both faults and detectors. | Thresholds derived from (α, H) only; boundary sweeps; negative controls; all cells reported. |
| Internal | Artefacts of the simulation PRNG. | PCG64 is statistically strong for this purpose; healthy controls use the same generator, so artefacts would show up as E2 miscalibration. |
| Construct | Latency is measured in samples, not seconds. | Seconds depend on a source's sample rate. A conversion table for example rates is given in the report. |
| Statistical conclusion | Many cells and tests. | Per-cell intervals, Holm correction where families are defined, conservative intersection rule for H1. |

## 12. Change log

| Date | Change | Reason |
|---|---|---|
| 2026-10-26 | Initial protocol | Phase 1 |
| 2026-10-26 | Repeated-block detector compares each aligned block with windows at **every** offset in the history; k from L·2^(−kH) ≤ α instead of (m choose 2)·2^(−kH) over aligned blocks | Aligned-only comparison misses replays whose distance is not a multiple of k. Found while implementing, before any evaluation data. |
| 2026-10-26 | Per-evaluation false alarms counted per natural unit (run, window, block), with no monitor reset | Each unit raises at most one alarm, so a reset is unnecessary and the counts map directly onto the null probabilities |
| 2026-10-26 | Window length 4096 fixed for windowed detectors; exposure gating buffer 2^16 fixed | Not specified in the Phase 1 protocol |
| 2026-10-26 | RCT/APT alarms are counted once per run/window, where the SP 800-90B pseudo-code signals at every further repetition | Keeps counts per evaluation unit. The time of the first alarm is unchanged. |
| 2026-10-26 | APT exact false-alarm probability reported next to nominal α; H2 reported against both | See the finding in §4.1. The pre-declared decision rule itself is unchanged. |
| 2026-10-26 | E7 specified on S0 | N1 and F8b match S0's statistics, not S1's (see the E7 row) |
| 2026-10-26 | Sliding-window entropy estimate moved from detector to descriptive metric | It has no exact null distribution |
| 2026-10-26 | Development smoke runs used master seed 12345 and test seeds 0–19 and 20261026 | Recorded so that the evaluation master seed can be chosen disjoint from them |
| 2026-10-27 | P1 block length 2^16 output bits; battery of six tests (cusum in both modes); block-frequency M from the SP 800-22 recommendation | Not fixed in Phase 1. 2^16 bits meets every test's input-size recommendation while keeping about 16 blocks per trial. |
| 2026-10-27 | P3 is a stateful hash-chain DRBG stand-in emitting 32 bytes per conditioned block, rather than an independent ChaCha20 seeding per block with 1024-byte expansion | Independent reseeding re-emits identical output for identical inputs, unlike a real DRBG. The 32x expansion would also multiply output-test false alarms per trial. Decided before any evaluation data. |
| 2026-10-27 | H3 "end-to-end" includes simulating the raw samples; a processing-only variant (sample generation excluded) is reported alongside | Clarifies "end-to-end throughput of the simulated pipeline". The decision rule uses end-to-end. |
| 2026-10-27 | Longest-run class probabilities computed exactly instead of using the printed tables | Two independent exact methods agree to 4e-16. The printed SP 800-22 tables differ from them by up to 0.013 (M = 512), 0.008 (M = 1000) and 0.0016 (M = 10^4); the M = 512 table sums to 1.0059. |
| 2026-10-27 | Development seed 99 and 5–7 (tests) and 12345 (CLI smoke configs) | Recorded so that the evaluation master seed differs |
