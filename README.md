# EntropyLens

**An auditable cryptographic randomness generator and entropy-health research platform.**

EntropyLens studies how well a lightweight, explainable monitoring layer detects failures of an
entropy source, how quickly it does so, and at what false-alarm and performance cost. It also shows
clearly what such monitoring **cannot** establish about cryptographic security.

> **Research prototype.** EntropyLens is not a certified or NIST-compliant security product.
> Security-relevant randomness always comes from the operating system's CSPRNG through Python's
> `secrets` module. Fault injection only ever affects *simulated* sources; the host RNG is never
> modified. See [docs/threat_model.md](docs/threat_model.md).

## Project status

| Phase | Content | Status |
|---|---|---|
| 1 | Proposal, threat model, architecture, methodology, references; scaffold; secure source | **Complete** |
| 2 | Generator protocol, educational LCG, ChaCha20 reference, configuration | **Complete** |
| 3 | Synthetic sources, fault injection, health detectors, evaluation | **Complete** |
| 4 | Statistical tests, entropy metrics, benchmarks, CLI | **Complete** |
| 5 | Streamlit dashboard, experiment history | **Complete** |
| 6 | Reports, exports, documentation, packaging | Planned |
| 7 | Full experiments, final report, demonstration | Planned |

No evaluation results exist yet; only development smoke runs have been made. The hypotheses in
[docs/proposal.md](docs/proposal.md) were written before any data was collected.

## Running experiments

Run these from the repository root. Configurations live in `experiments/configs/`, and result files
are written to `experiments/results/`.

```powershell
python -m entropylens list
python -m entropylens run smoke
python -m entropylens benchmark --quick
python -m entropylens metadata
```

## Dashboard

```powershell
streamlit run entropylens/dashboard/app.py
```

Run this from the repository root, then open http://localhost:8501. The server binds to localhost
only (`.streamlit/config.toml`); it has no authentication and must not be exposed on a network.

The dashboard has eight pages:

- **Overview:** the project, its generators, and the latest runs.
- **Fault Injection Lab:** the demonstration centrepiece. Inject a fault, see which detector fires
  and why, and compare the exposure with monitoring on and off.
- **Entropy & Health Monitor:** the detectors, their thresholds, and a monitored stream.
- **Randomness Lab:** generate bytes, and break the LCG.
- **Statistical Analysis:** the SP 800-22 subset and entropy metrics, with their limits.
- **Benchmarking:** generator throughput and monitoring overhead.
- **Experiment History:** a SQLite index of the result files, with filters, comparison, integrity
  checks and CSV/JSON export.
- **Research Report:** a Markdown report of selected runs.

Interactive runs use development seeds and are not evaluation results.

`run` executes a detection experiment (or, with `"analysis": "placement"`, the E3 comparison of
output tests, raw monitoring and DRBG output). It prints a summary per cell and writes a JSON result
file with its SHA-256 digest. Configuration files are strictly validated JSON; nothing in them is
executed.

## Documentation

- [Proposal](docs/proposal.md): research question, hypotheses, contribution and novelty assessment,
  demonstration plan, timeline, acceptance criteria
- [Methodology](docs/methodology.md): source models, fault taxonomy, detector thresholds, metrics,
  decision rules
- [Architecture](docs/architecture.md): layers, directory structure, dependencies, interfaces
- [Threat model](docs/threat_model.md): assets, adversaries, threats, limits
- [References](docs/references.md): verified bibliography

## Setup

Requires Python 3.12 or later. Commands for Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

On Linux or macOS, activate with `source .venv/bin/activate` instead.

`requirements.txt` pins the exact versions used to verify the project. Supported version ranges
are declared in `pyproject.toml`; `pip install -e ".[dashboard,dev]"` installs from those ranges
instead.

## Running the checks

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy
```

## Licence

MIT; see [LICENSE](LICENSE).
