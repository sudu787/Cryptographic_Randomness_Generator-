# Threat Model

| | |
|---|---|
| Status | Phase 1, 2026-10-26 |
| System | EntropyLens research prototype: a Python library, CLI and local Streamlit dashboard |
| Bottom line | EntropyLens is a **research and teaching prototype, not a security product**. Its secure path delegates entirely to the operating system's CSPRNG. Its monitoring results describe simulated sources under stated models and are never evidence that a randomness source is secure. |

## 1. Assets

| ID | Asset | Why it matters |
|---|---|---|
| A1 | Integrity of the production randomness path: security-relevant bytes come **only** from the OS CSPRNG | A single substitution of a deterministic generator would silently destroy unpredictability |
| A2 | Confidentiality of production output | Random bytes used as keys or tokens are secrets |
| A3 | Correctness of the health-status signal | A false `HEALTHY` is worse than no monitor, because it creates unjustified confidence |
| A4 | Integrity and honesty of experimental results and audit records | The project's scientific claims rest on them |
| A5 | Integrity of the host system | The project must not degrade the OS RNG or need elevated privileges |

## 2. Trust boundaries and assumptions

1. **The OS CSPRNG is trusted.** `secrets.token_bytes` → `os.urandom` → BCryptGenRandom (Windows,
   Python ≥ 3.11) or `getrandom()` (Linux) [CPY-URANDOM]. EntropyLens does not and cannot monitor
   the kernel's internal noise sources. Any statistics computed on OS CSPRNG output are output-level
   sanity checks, **not health tests**.
2. **The Python runtime and the pinned dependencies are trusted.** Supply-chain compromise is out of
   scope; the pinned lock file reduces accidental drift.
3. **The local user is trusted.** The dashboard has no authentication and is bound to `localhost`.
   Exposing it on a network is unsupported.
4. **Simulated sources are models.** A detector's behaviour on a model says nothing definitive about
   real hardware.

## 3. Adversary models

| ID | Adversary | Knowledge and capability | Outcome against EntropyLens components |
|---|---|---|---|
| ADV1 | Passive output observer | Sees generator output | **LCG:** predicts all future outputs from a few observations (demonstrated). **ChaCha20 with a secret random key:** no practical prediction is known. **OS CSPRNG:** out of scope (trusted). |
| ADV2 | Seed guesser | Knows the seeding procedure, such as a timestamp seed | Recovers small or predictable seeds by enumeration (demonstrated on the LCG over a one-hour window of timestamps). This is the failure class behind [Debian-2008] and [Heninger-2012]. |
| ADV3 | State compromise | Learns a deterministic generator's internal state or key | **LCG:** all past and future outputs, since the recurrence is invertible. **ChaCha20 reference:** the entire keystream for that key and nonce, past and future. It has no backtracking or prediction resistance, unlike SP 800-90A DRBGs with reseeding [NIST-90A]. **SecureSource:** holds no state of its own. |
| ADV4 | Non-malicious source failure | The noise source degrades: stuck, biased, correlated, replaying, intermittent, unavailable | **This is the primary research target.** Detection is measured (E1, E5), never assumed. |
| ADV5 | Adaptive malicious source | Controls the raw source and emits data designed to pass health tests, such as a PRNG with an attacker-known seed | **Not detectable by statistical monitoring.** The negative control N1 (E7) demonstrates this. No claim of detection is made. |
| ADV6 | Local dashboard user supplying input | Provides parameters, configurations and file names through the UI | Mitigated by input validation, size caps, path confinement, JSON-only parsing, and no code execution |

## 4. Threats and mitigations

| ID | Threat | Asset | Mitigation | Residual risk |
|---|---|---|---|---|
| T1 | A non-cryptographic generator (`random`, the LCG, NumPy PCG64) is used for security-relevant output | A1 | One production entry point (`SecureSource`). A static test forbids importing `random` in the security package. Every generator carries `security_status`. Educational generators are labelled insecure in code, docs and UI. | A downstream developer could still call the LCG directly. This is documented. |
| T2 | Manual seeding of the OS CSPRNG | A1 | `SecureSource` exposes no seed or state API. Documentation explains why. | None identified |
| T3 | Predictable or reused seeds leak from experiments into production use | A1 | Seeds exist only in Layers B and C and in simulation. Deterministic generators report `deterministic=True`. The UI shows a banner whenever deterministic output is displayed. | A user may copy demo output; the UI warns against it. |
| T4 | Random output is logged, displayed or persisted | A2 | Result files store configuration, metadata and aggregate statistics only. The serialiser refuses `bytes` outright, and a content-scan test plants a marker in the secure source and checks it never reaches a result file. Demonstration buffers live in memory and are labelled "demonstration only — never use as a key". EntropyLens never generates keys for real use. | Process memory may retain buffers; Python offers no reliable zeroisation (see §6). |
| T5 | Fail-open health status: an exception, invalid configuration or too little data reported as healthy | A3 | Status semantics with `UNKNOWN` and `FAILED`, both of which close the output gate. Latching on failure. Dedicated unit tests (AC3). | Bugs in detector logic itself; mitigated by tests against exact distributions |
| T6 | Statistical results are misread as security evidence | A3, A4 | Every result view carries a limitation statement. Wording rules forbid "secure", "certified" and "NIST-compliant". Multiple-testing control. Negative controls (H4, H6). | Readers may still over-interpret. The report repeats the limits. |
| T7 | Circular evaluation: thresholds tuned on the faults being evaluated | A4 | Thresholds derived only from (α, H). Pre-declared protocol. Disjoint development and evaluation seeds. All cells reported. | The author designed the fault model. Acknowledged as a threat to validity. |
| T8 | ChaCha20 misuse: (key, nonce) reuse, counter overflow, key disclosure | A1, A2 | Random mode draws a fresh key and nonce from `SecureSource` per instance and rekeys at the counter limit. Seeded mode derives the key from a recorded experiment seed and is labelled zero-secrecy. Raw keys are never accepted in configuration files. Fixed-key modes raise before the counter wraps. The key is not retained as an attribute, is excluded from `repr`, and instances refuse pickling and copying. | Keys reside in the cipher context in process memory |
| T9 | Unsafe deserialisation or code injection through configuration files or uploads | A5 | JSON only, Pydantic-validated. No `eval`, `exec`, `pickle` or `yaml.load`. Uploaded files are never executed. | Vulnerabilities in parsers of trusted libraries |
| T10 | Path traversal when reading or writing results | A4, A5 | Paths resolved and confined to `experiments/configs` and `experiments/results`. Names validated against an allow-list pattern. | None identified |
| T11 | Resource exhaustion through very large requests | A5 | Per-request byte caps; bounded UI inputs; trial-count limits in the configuration schema | A determined local user can still exhaust memory |
| T12 | Tampering with stored results | A4 | SHA-256 digest of each result file's canonical JSON detects **accidental** modification | No protection against a deliberate editor; there is no signing |
| T13 | Dashboard exposed on a network | A2, A5 | `.streamlit/config.toml` binds to localhost, keeps XSRF protection on, disables usage statistics, and allows no uploads. Exposure is documented as unsupported. The dashboard never displays or offers for download the output of the OS CSPRNG, or of ChaCha20 under a random key; only labelled deterministic demonstration output can be previewed. No raw HTML is rendered from data. | A user can override the settings |
| T15 | Spreadsheet formula injection through exported CSV | A4, A5 | Text cells starting with `= + - @`, tab or carriage return get a leading apostrophe on export (tested) | Other consumers of the CSV must still treat it as data |
| T16 | SQL injection through history filters | A4 | Every value is a bound parameter. Column names come from an allow-list. Free-text search escapes LIKE wildcards (tested with an injection string). | None identified |
| T14 | Degradation of the host OS RNG | A5 | Never attempted. Fault injection exists only in simulated sources. No writes to OS entropy interfaces. | None identified |

## 5. Three different questions

The project keeps these apart in code, documentation and UI.

| Question | Technique | What a "pass" means | What it cannot establish |
|---|---|---|---|
| **Is the raw noise source behaving as its model predicts?** | Source health testing (Layer D: RCT, APT, supplementary detectors) | No evidence of the specific failure modes the tests target, at the configured α | That the source has the claimed entropy; that it is not adversarially controlled |
| **Does the output look statistically random?** | Output-level testing (SP 800-22 subset, entropy metrics) | No detectable deviation from uniformity for the tests run, at their sample size | Unpredictability. Deterministic generators with known seeds pass. Conditioning hides source defects. |
| **Is the generator cryptographically secure?** | Design analysis and proofs (DRBG security, cipher cryptanalysis), validated entropy assessment, implementation review | Out of scope for EntropyLens | — |

**High empirical Shannon entropy is not unpredictability.** The plug-in Shannon entropy of
ChaCha20 output under a published key is near 8 bits per byte, yet an attacker who knows the key
predicts every byte. Unpredictability is a property of the generating process and the attacker's
knowledge, not of any finite sample.

## 6. Known limitations of the security posture

- **No memory hygiene guarantees.** Python `bytes` objects are immutable and may be copied by the
  runtime. EntropyLens cannot zeroise buffers. This is acceptable for a research prototype and
  unacceptable for a key-management product.
- **No validation or certification.** No CMVP/ESV entropy validation, no FIPS 140-3 module
  boundary, no independent review.
- **Simulated sources only.** No physical entropy source is modelled at the physics level.
- **Detection is model-relative.** A fault outside the taxonomy, or one that preserves the tested
  statistics, can go undetected (ADV5, F8b, N1).
- **The entropy claim H is given, not estimated.** A wrong H makes the thresholds wrong. Real entropy
  estimation is difficult [Kelsey-2015].

## 7. Why this is a research prototype and not a security product

A security product for randomness would need, at minimum: a validated entropy source with an
SP 800-90B assessment; an approved DRBG construction per SP 800-90A/C [NIST-90A, NIST-90C]; a defined
module boundary with self-tests, zeroisation and error states; independent validation; and an
operational security process. EntropyLens implements none of these. It is built to **measure and
explain**, using the OS CSPRNG wherever real security-relevant randomness is needed.
