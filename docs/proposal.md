# EntropyLens: Project Proposal

**An adaptive, auditable cryptographic randomness generator and entropy-health research platform**

| | |
|---|---|
| Document status | Phase 1 proposal, 2026-10-26. **No experimental results exist yet.** Hypotheses are stated before any evaluation data is collected. |
| Companion documents | [architecture.md](architecture.md) · [threat_model.md](threat_model.md) · [methodology.md](methodology.md) · [references.md](references.md) |

---

## 1. Summary

Cryptographic systems fail quietly when their randomness fails. Keys generated from a broken entropy
source look like any other keys. Output from a deterministic generator with a guessable seed passes
statistical tests. Documented failures include Debian's 2006–2008 OpenSSL seeding bug [Debian-2008],
widespread shared RSA factors in network devices [Heninger-2012], repeated randomness after virtual
machine snapshot resets [Ristenpart-2010, Everspaugh-2014], power-supply frequency injection that
disables a TRNG's noise [Markettos-2009], and a certified hardware TRNG with measurable bias
[HurleySmith-2018].

Standards respond to this by requiring **continuous health tests on the raw noise source**, before
any conditioning hides its defects (NIST SP 800-90B §4 [NIST-90B]; BSI AIS 20/31 [BSI-AIS2031]).
Health tests are cheap and conservative. How well a given configuration of tests detects a given
failure, how quickly it does so, and how many false alarms it costs are usually reported only inside
proprietary validation documents, if at all.

EntropyLens is a Python research platform that makes these questions measurable and visible. It
combines:

1. A **secure production path**, which takes bytes only from the operating-system CSPRNG via
   Python's `secrets`.
2. **Educational baselines**: an insecure Linear Congruential Generator (LCG) and a deterministic
   ChaCha20 keystream generator from the `cryptography` library.
3. A **simulated entropy-source laboratory** with labelled, parameterised fault injection.
4. A **source-health monitor**: SP 800-90B-style Repetition Count and Adaptive Proportion tests,
   plus supplementary detectors. Thresholds are derived from stated false-alarm budgets, and every
   alert carries its evidence.
5. A **reproducible experiment harness** that reports detection rate, false-alarm rate, detection
   latency, exposure, and overhead with uncertainty intervals, including ablation and baseline
   comparisons.
6. A **Streamlit dashboard** and an HTML report generator that present the results, and the limits of
   what they show.

The centrepiece of both the evaluation and the demonstration is the **Fault Injection Lab**. A
simulated source is made to fail in a controlled way. The audience sees which detector fires, why,
after how many samples, and what output would have escaped without the monitor. Repeated trials then
turn that single demonstration into measured detection and false-alarm rates.

## 2. Problem statement

Developers and students commonly judge randomness by testing **output**: run a statistical battery,
see p-values above 0.01, conclude "random". This is unreliable in two directions:

- **Statistical tests cannot establish unpredictability.** A deterministic generator with a known
  seed, such as ChaCha20 under a published key, produces output that passes every practical
  statistical test while having zero secrecy. NIST's own 2022 decision to revise SP 800-22 includes
  the goal of rejecting its use for assessing cryptographic RNGs [NIST-22].
- **Conditioning masks source defects.** Once raw noise has passed through a hash function or DRBG,
  a source that has lost most of its entropy still yields output that looks uniform. That is why
  SP 800-90B places health tests on the raw noise samples [NIST-90B].

A monitoring layer is useful only if three things are known about it: what it detects, how quickly,
and at what false-alarm cost. Its limits must also be stated: what it cannot detect, and why passing
it is not a security guarantee. This project builds the apparatus to measure those properties for a
transparent set of detectors under a documented fault model.

## 3. Research question, sub-questions, and hypotheses

### 3.1 Primary research question

> Can a lightweight, explainable entropy-source monitoring layer detect injected entropy-source
> failures with an acceptable false-alarm rate and performance overhead, while clearly distinguishing
> observable statistical anomalies from cryptographic security guarantees?

"Acceptable" is not left vague. Section 3.3 turns it into pre-declared numeric criteria. These are
**operating requirements chosen by the project, not scientific constants**, and they are justified in
[methodology.md](methodology.md).

### 3.2 Sub-questions

| ID | Sub-question | Main experiment |
|---|---|---|
| SQ1 | For each fault class and intensity, what detection rate and detection latency (raw samples from fault onset to the first critical alarm) does the monitor achieve? | E1, E5 |
| SQ2 | On healthy sources, including healthy but imperfect ones, do empirical false-alarm rates match the configured false-alarm budget α? What is the expected false-alarm count per GiB of raw data? | E2 |
| SQ3 | How do raw-source monitoring and output-level statistical testing after conditioning compare on identical fault traces? | E3 |
| SQ4 | Which detectors add detection capability that the others lack, and at what false-alarm cost (ablation)? | E4 |
| SQ5 | What throughput, latency, CPU, and memory overhead does monitoring add? | E6 |
| SQ6 | Which failures are undetectable by any statistical monitor by construction (negative controls)? | E7 |

### 3.3 Hypotheses, pre-declared before data collection

Each hypothesis has an explicit decision rule. A result is reported as **supported**, **not supported**,
or **inconclusive**. A hypothesis that fails is reported as a finding, not removed.

| ID | Hypothesis | Decision rule (summary; full rule in methodology §8) |
|---|---|---|
| H1 Detection | For each *gross* fault class at its reference intensity (stuck-at/constant, short repeating pattern, repeated blocks, strong bias, source outage), the full monitor detects the fault within N_max = 2^20 raw samples after onset. | Supported if, for every gross class, the lower bound of the 95% two-sided Clopper–Pearson interval on trial-level detection rate is ≥ 0.95 (n = 200 trials per cell). |
| H2 False alarms | On healthy sources, including healthy but imperfect models, false alarms do not exceed the configured budget. | Supported if, at relaxed calibration levels α ∈ {2^-6, 2^-8, 2^-10}, a one-sided exact binomial test shows no significant excess over nominal α (Holm-corrected, family level 0.05), and at the operating α the observed false-alarm count is consistent with its expected count. |
| H3 Overhead | Monitoring costs at most a 50% reduction in end-to-end throughput of the simulated pipeline at buffer sizes ≥ 64 KiB. | Supported if the median throughput ratio (monitored / unmonitored) is ≥ 0.5 and its 95% bootstrap interval excludes values < 0.5. This is an engineering target; missing it is reported. |
| H4 Placement | On identical fault traces, output-level tests after SHA-256 conditioning detect faults only when the conditioner input repeats exactly. They do not detect faults that reduce but do not eliminate entropy, and they detect nothing when a DRBG follows conditioning. | Supported if output-level detection for entropy-reducing faults (bias, correlation) is statistically indistinguishable from the false-alarm rate, while raw-source monitoring detects the same traces per H1. |
| H5 Ablation | The SP 800-90B pair (RCT + APT) alone misses at least one fault class at reference intensity that a supplementary detector catches; supplementary detectors add measurable false-alarm cost. | Supported if at least one class shows a detection-rate difference whose 95% interval excludes 0 between the full monitor and RCT+APT only. |
| H6 Negative control | No detector distinguishes a zero-secrecy deterministic stream (ChaCha20 under a fixed, published key) from a healthy source. | Expected outcome, stated as a limitation: detection rate is not significantly above the false-alarm rate. This shows that **passing the monitor is not evidence of security**. |

H4 and H6 are deliberately *negative* hypotheses. They exist so the project shows, with data, what
statistical monitoring cannot do.

## 4. Proposed contribution and novelty assessment

### 4.1 What the contribution is

The contribution is offered as a **hypothesis to be tested**, not a claim of novelty:

> A reproducible, open, software-only evaluation harness that quantifies the detection-rate,
> detection-latency, false-alarm, exposure, and overhead trade-offs of SP 800-90B-style health tests
> and supplementary detectors under a parameterised fault taxonomy. It includes ablation, a side-by-side
> comparison of raw-source monitoring versus post-conditioning output testing on identical traces, and
> human-readable evidence for every alert.

Concretely, the project delivers:

1. **A labelled fault laboratory.** Ten fault families with documented parameters, onset schedules,
   and ground-truth labels at sample resolution.
2. **A transparent monitor.** Each detector documents its null model, its parameters, its derivation
   from α and the claimed min-entropy H, and the evidence it reports. The aggregate status is
   fail-closed.
3. **An evaluation methodology.** Thresholds are fixed *before* evaluation, with separate seed streams
   for development and evaluation, uncertainty intervals, multiple-testing control, and negative
   controls.
4. **Measured results** for H1–H6 on the stated machine, in a structured, reproducible format.
5. **A teaching instrument.** A dashboard that shows *why* an alarm fired, and what it does *not*
   prove.

### 4.2 What the contribution is not

- **Not a new health test.** RCT and APT come from SP 800-90B. Supplementary detectors use textbook
  statistics.
- **Not an entropy-source validation.** No physical noise source is assessed. Simulated sources are
  statistical models, not hardware.
- **Not a security proof.** Detection results describe the simulated fault model only.
- **Not NIST-compliant or NIST-certified.** No CMVP/ESV process applies.

### 4.3 Preliminary literature scan

A systematic review is scheduled for Phase 6. The table records what the Phase 1 scan found.
References are in [references.md](references.md).

| Area | Established work | Overlap with EntropyLens |
|---|---|---|
| Continuous health tests | SP 800-90B §4.4 defines RCT and APT with cutoffs derived from α and H [NIST-90B]. AIS 20/31 defines online and total-failure tests [BSI-AIS2031]. NIST staff have said future revisions should base health tests on a source's stochastic model and failure modes [Kelsey-2024]. | **High.** EntropyLens implements these tests rather than inventing new ones. |
| On-the-fly hardware testing | Compact FPGA implementations of SP 800-22 tests for online monitoring [Veljkovic-2012, Yang-2015]. Tests designed from attack effects [Yang-2016]. "Canary numbers" for lightweight online testability [Rozic-2016]. | **Medium.** These target hardware cost and specific TRNG designs. EntropyLens targets software-level, design-agnostic measurement of detection/false-alarm trade-offs. |
| Synthetic failure modelling | Lin (2026) models observable raw-output artefacts of CMOS TRNGs and QRNGs with synthetic sources. It studies their effect on SP 800-90B IID screening and min-entropy bounds, and reports RCT/APT alarm rates by artefact family [Lin-2026]. | **High, and the closest work found.** Its focus is the validation pipeline (IID decision, entropy bounds) using approximate test proxies. As read in Phase 1, it does not report detection latency, false-alarm calibration against the budget with intervals, throughput overhead, ablation of supplementary detectors, or raw-versus-conditioned comparison on identical traces. Phase 6 will confirm this by a full reading. |
| Statistical test suites and their limits | SP 800-22 [NIST-22], TestU01 [LEcuyer-2007]. Documented errors in SP 800-22r1a [Riesinger-2025]. Certified TRNGs whose bias the certification tests missed [HurleySmith-2018]. | **Low** (used as context). EntropyLens uses output tests as a baseline and explains their limits. |
| Generator robustness theory | Robustness of PRNGs with input [Dodis-2013]. Linux RNG analysis [Gutterman-2006]. | **Low** (background for the threat model). |

**Assessment.** The individual components are established. The proposed contribution is their
combination into a reproducible evaluation harness with the specific metrics, ablation, placement
comparison, and explanation layer described above. This framing is defensible for an academic
project. Whether the combination is new in the published literature is **unverified** until the
Phase 6 review. The project will not describe itself as novel unless that review supports it. If
closer prior work turns up, the contribution will be reframed as a replication and teaching
platform.

## 5. Objectives

| # | Objective (from the brief) | Where addressed |
|---|---|---|
| O1 | Generate cryptographically secure bytes via OS facilities | Layer A, `security/secure_source.py` |
| O2 | Demonstrate PRNG vs CSPRNG | Layers B and C, Randomness Lab page, demo steps 2–4 |
| O3 | Monitor observable characteristics and simulated source health | Layer D, `monitoring/` |
| O4 | Inject controlled faults; evaluate detection | `experiments/`, Fault Injection Lab, E1, E5 |
| O5 | Reproducible statistical and performance comparison | `analysis/`, E6, benchmark page |
| O6 | Professional dashboard | `dashboard/` (Phase 5) |
| O7 | Downloadable results and reports | `storage/`, `reporting/` (Phase 6) |
| O8 | Explain limitations of statistical tests | Every analysis page, [threat_model.md](threat_model.md), H4, H6 |
| O9 | Show effects of deterministic seeds, bias, repeats, failures | Fault taxonomy F1–F10, seed-recovery demo |
| O10 | Evaluate whether monitoring helps without excessive false alarms or overhead | H1–H5 |

## 6. Scope

**In scope:** simulated raw noise sources; SP 800-90B-style health tests; supplementary detectors;
SHA-256 conditioning in the simulated pipeline; output-level statistical tests (an SP 800-22 subset,
implemented and documented); benchmarking of the OS CSPRNG, LCG, and ChaCha20; experiment history;
HTML reports; a dashboard on localhost.

**Out of scope:** physical entropy sources; entropy assessment of real hardware; modifying or
degrading the host OS RNG (never attempted); FIPS 140-3 or CMVP/ESV validation; side-channel attacks;
key management for real systems; networked deployment.

## 7. Evaluation plan (summary)

Full protocol: [methodology.md](methodology.md).

- **Trials.** 200 independent trials per (fault class × intensity) cell. Fault onset is drawn uniformly
  within a window. Healthy controls: 1,000 trials for trial-level false-alarm rate, plus ≥ 10^5
  test evaluations per relaxed-α calibration level.
- **Metrics.** Detection rate (recall), false-alarm rate (per trial and per test evaluation), precision
  at a stated prevalence, detection latency in samples (median and 95th percentile), exposure (output
  bytes derived from faulty raw data before the alarm), throughput (MiB/s), latency percentiles, CPU
  time, and peak memory.
- **Uncertainty.** Wilson score intervals for proportions, Clopper–Pearson when counts are 0 or n,
  percentile bootstrap (10,000 resamples, fixed seed) for medians and throughput ratios.
- **Controls.** No-monitor baseline, output-tests-only baseline, ablation configurations, a
  healthy-but-imperfect source, and a zero-secrecy negative control.
- **Integrity.** Thresholds are derived from (α, H) only and are never tuned on fault data.
  Development and evaluation use disjoint seed streams. Every run records its configuration, seeds,
  software versions, and platform metadata.

| Experiment | Purpose | What it can show | What it cannot show |
|---|---|---|---|
| E1 Detection | H1 | Detection rate and latency per fault class on the simulated model | Detection of real hardware failures |
| E2 False alarms | H2 | Calibration of implemented tests against theory | False-alarm behaviour of real, non-stationary sources |
| E3 Placement | H4 | Masking of source faults by conditioning | That conditioned output is secure |
| E4 Ablation | H5 | Marginal detection contribution of each detector | Optimality of the detector set |
| E5 Sensitivity | SQ1 | Detection boundary versus intensity and α | Universal thresholds |
| E6 Overhead | H3 | Cost on this machine and software stack | Performance of optimised native implementations |
| E7 Negative control | H6 | Blindness of statistical monitoring to deterministic, zero-secrecy streams | Anything about the security of the monitored source |

## 8. Demonstration plan (8–12 minutes, fault injection as centrepiece)

| Time | Step | What the professor sees |
|---|---|---|
| 0:00–1:00 | Why randomness matters | Debian 2008 and shared RSA factors (one slide-like panel on the Overview page). |
| 1:00–2:30 | Insecure baseline | LCG output in the Randomness Lab, then the next outputs predicted from three observed values. |
| 2:30–3:30 | Predictable seeds | A time-seeded LCG: its seed is recovered by brute force over a one-hour window of timestamps. |
| 3:30–4:00 | Secure source | Switch to `secrets` (OS CSPRNG). It exposes no seed, and the preview is labelled non-secret. |
| 4:00–7:00 | **Fault Injection Lab (centrepiece)** | A simulated source runs healthy, then a stuck-at or repeating-pattern fault is injected at a labelled sample. The RCT or repeated-block detector fires. The alert panel shows the evidence (for example, "run of identical samples reached the cutoff C"), the threshold derivation from α and H, the detection latency, and what the alarm does **not** establish. |
| 7:00–8:30 | Monitor on vs off | The same trace without monitoring: faulty bytes flow through SHA-256, and the output passes output-level tests. Exposure counter: N bytes released. |
| 8:30–10:00 | Measured evidence | E1/E2/E6 results: detection rate with intervals, false-alarm rate against budget, overhead. Pre-computed runs, with one cell re-run live. |
| 10:00–11:00 | Limits | Negative control (E7): a ChaCha20 stream with a published key passes every detector. Statistical testing ≠ security. |
| 11:00–12:00 | Conclusion | Research question, status of each hypothesis (supported, not supported, inconclusive), limitations, future work. |

The demo runs offline from prepared configuration files. All live steps use synthetic sources only.

## 9. Timeline

Estimates assume one student working part-time with tool assistance. Each phase ends only when its
files exist, its tests pass, and the next phase's interfaces have been checked.

| Phase | Content | Estimate | Exit criterion |
|---|---|---|---|
| 1 | Proposal, threat model, architecture, methodology, references; scaffold; first tested module | Week 1 | Docs reviewed; environment builds; `secure_source` tests pass |
| 2 | Generator protocol, LCG, ChaCha20 reference, configuration models, tests | Week 1–2 | Unit tests incl. RFC 8439 keystream vector |
| 3 | Synthetic sources, fault injection, thresholds, detectors, monitor, evaluation metrics | Weeks 2–3 (largest phase) | Detector tests on known faults; false-alarm calibration smoke run |
| 4 | Statistical tests, entropy metrics, intervals, benchmark harness | Week 4 | Independently recomputed test vectors; benchmark schema tests |
| 5 | Streamlit dashboard (8 pages), experiment history (SQLite) | Weeks 5–6 | AppTest smoke tests for each page |
| 6 | Report builder, exports, limitations doc, literature review, packaging | Weeks 6–7 | HTML report from real runs; export round-trip tests |
| 7 | Full experiment runs, defect fixing, final report, demo rehearsal | Weeks 7–8 | All acceptance criteria in §10 met or explicitly reported as unmet |

## 10. Acceptance criteria

| ID | Criterion | Verified by |
|---|---|---|
| AC1 | The full test suite passes in a fresh virtual environment built from the pinned requirements; tests need no network access. | `pytest` run log |
| AC2 | Security-sensitive output comes only from `secrets`/`os.urandom`; the security package never imports `random`. | Automated static test |
| AC3 | Health status is fail-closed: an exception in a detector, an invalid configuration, or insufficient data never yields `HEALTHY`. | Unit tests with injected detector failures |
| AC4 | Every detector documents its null model, α, threshold derivation, and evidence fields, and has tests on a known fault and on healthy fixtures. | Code review checklist plus tests |
| AC5 | Re-running an experiment configuration with the same master seed reproduces identical detection results (timings excluded). | Reproducibility test |
| AC6 | Result files record configuration, seeds, versions, and platform metadata, and contain no generated output from the secure source. | Schema tests plus a content scan test |
| AC7 | Experiments E1–E7 run with the stated trial counts, and the report states every hypothesis as supported, not supported, or inconclusive by its pre-declared rule. | Generated report |
| AC8 | All eight dashboard pages load without exceptions. Every alert shows its evidence and a limitation statement. | Streamlit AppTest smoke tests plus manual check |
| AC9 | `ruff check` and `ruff format --check` are clean; `mypy` is clean on the core package. | CI-style local run |
| AC10 | The demonstration runs offline in ≤ 12 minutes from prepared configurations. | Rehearsal timing |
| AC11 | No document or UI text claims the system is secure, novel, NIST-compliant, or certified. | Wording review before submission |

## 11. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Circular evaluation (detectors tuned to the faults the author designed) | Inflated detection rates | Thresholds derived only from (α, H); disjoint seed streams; boundary-intensity sweeps (E5); negative controls; all results reported |
| Health tests false-alarm frequently at high sample rates (RCT fires about once per α⁻¹ samples) | Misleading "monitor is noisy" or "monitor is quiet" conclusions | Report expected false alarms per GiB alongside α; sweep α ∈ {2^-20, 2^-30, 2^-40} |
| Healthy-but-imperfect sources trip uniformity-based detectors | High false-alarm rate | Detectors are parameterised by the nominal source model, not ideal uniformity; E2 includes such sources |
| Python performance makes overhead look worse than native code | H3 fails | Vectorise with NumPy; report honestly; note that native implementations would differ |
| SP 800-22 worked examples contain errors [Riesinger-2025] | Wrong test vectors | Recompute expected values independently (SciPy) and cross-check implementations against each other |
| Literature review finds closer prior work | Contribution overstated | Contribution is framed as a hypothesis; reframe as replication or teaching platform if needed |

## 12. Phase 1 deliverables and immediate next steps

Phase 1 deliverables in this repository:

- [x] This proposal (research question, hypotheses, contribution, novelty assessment, evaluation
  summary, demo plan, timeline, acceptance criteria).
- [x] [architecture.md](architecture.md): layered architecture diagram, final directory structure,
  dependency plan, interfaces.
- [x] [threat_model.md](threat_model.md): assets, adversaries, threats, mitigations, non-goals.
- [x] [methodology.md](methodology.md): full experimental protocol and decision rules.
- [x] [references.md](references.md): verified bibliography with a verification log.
- [x] Project scaffold (`pyproject.toml`, pinned `requirements.txt`, lint and test configuration).
- [x] Smallest coherent, tested component: `entropylens/security/secure_source.py` with
  `tests/test_secure_source.py`.

**Phase 2 (next):** `security/base.py` (generator protocol), `security/educational_lcg.py`
(including seed-recovery and next-output prediction for the demo), `security/crypto_reference.py`
(ChaCha20 keystream via `cryptography`, checked against the RFC 8439 test vector), `config.py`
(validated experiment configuration models), and their tests.
