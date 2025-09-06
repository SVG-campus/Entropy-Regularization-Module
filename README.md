# Entropy Regularization Module (Paper 5)

This repository contains a clean, production‑ready implementation of the **Entropy Regularization Module** described in the accompanying paper. Entropy regularization penalizes overly concentrated allocations to encourage diversification and robustness by adding an entropy term \(H(w) = -\sum_i w_i \log w_i\) to the objective with strength \(\gamma\).

We provide:
- A reference implementation with **exponentiated‑gradient** updates using the entropy gradient \(\nabla H(w) = -(1 + \log w)\).
- A projected‑simplex utility to maintain non‑negativity and \(\ell_1\)-normalization.
- Unit tests, CI, and full Zenodo/ORCID metadata.

---

## Quick start

```bash
# (optional) create venv
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate

pip install -r requirements.txt
pytest -q
python examples.py
```

---

## Usage

```python
import numpy as np
from entropy_module import (
    entropy, entropy_gradient, entropy_regularized_update,
    projected_simplex, portfolio_variance_gradient
)

# Example: one step using a variance objective gradient
w = np.array([0.25, 0.25, 0.25, 0.25])
cov = np.array([[0.02,0.01,0.00,0.00],
                [0.01,0.03,0.00,0.00],
                [0.00,0.00,0.01,0.00],
                [0.00,0.00,0.00,0.04]])
grad_L = portfolio_variance_gradient(w, cov)

w_next = entropy_regularized_update(
    weights=w, grad_L=grad_L, eta=0.05, gamma=0.05, project=True
)
print(w_next, w_next.sum())
```

Key properties:
- Always returns non‑negative weights that **sum to 1**.
- The **\(\gamma\)** parameter controls spread (higher \(\gamma\) → more uniform weights).
- Compatible with any differentiable base objective via `grad_L` (pass a gradient vector or a callable `f(weights)->grad`).

---

## Files included

- `entropy_module.py` — reference implementation (entropy, gradient, update, simplex projection).
- `tests/test_entropy_module.py` — unit tests covering invariants and \(\gamma\) behavior.
- `tests/test_pdfs_exist.py` — validates that the paper and test PDF notebooks exist (skips gracefully if missing).
- `requirements.txt` — minimal Python dependencies.
- `.github/workflows/ci.yml` — CI for tests on push/PR.
- `.github/workflows/release.yml` — GitHub Release flow on tags (works with Zenodo’s GitHub integration).
- `CITATION.cff` — citation metadata (with your ORCID).
- `.zenodo.json` — **Zenodo** deposition metadata.
- `examples.py` — runnable example and sanity checks.
- `CHANGELOG.md` — version history.
- `LICENSE-CODE` (MIT) and `LICENSE-DOCS` (CC BY 4.0).

> The repository should also include these research artifacts at the root (already present in your repo):
>
> - `Entropy_Regularization_Module.pdf`
> - `Entropy_Regularization_Module(Test).pdf`
> - `Entropy_Regularization_Module_in_Hilbert_Spaces(Test).ipynb`

---

## ORCID & Zenodo integration (what’s already set up and what this adds)

- **ORCID**: Your iD is **https://orcid.org/0009-0004-9601-5617**. Once a **DOI** is minted (via Zenodo), add it to your ORCID **Works**. Zenodo also pushes ORCID metadata if `.zenodo.json` includes your iD (it does).
- **Zenodo ↔ GitHub**: With GitHub connected to Zenodo, **creating a GitHub release** will trigger a Zenodo deposition and mint a DOI. This repo includes:
  - `.zenodo.json` — authorship (with ORCID), title, description, keywords, license, etc.
  - A `release.yml` workflow that creates a GitHub release whenever you push a tag like `v0.1.0`.

**Checklist to publish a citable version**  
1) Push all code, paper, and tests.  
2) Update version in `CHANGELOG.md` and `CITATION.cff`.  
3) Push a tag, e.g. `git tag v0.1.0 && git push --tags`.  
4) Zenodo will create a record with a **DOI**.  
5) Copy the DOI back into `README.md` (badge), `CITATION.cff` (`identifiers:` block), and (optionally) the repo description.  
6) Add the DOI to your **ORCID Works** if it doesn’t auto‑appear.

---

## Citing

See `CITATION.cff`. After the first Zenodo release, replace the placeholder DOI in the README badge and the BibTeX below.

```bibtex
@misc{entropyreg2025,
  title        = {Entropy Regularization Module},
  author       = {Villalobos-Gonzalez, Santiago de Jesus},
  year         = {2025},
  note         = {Code and preprint. DOI to be added after first Zenodo release.},
  howpublished = {GitHub + Zenodo}
}
```

---

## Licensing

- **Code**: MIT (see `LICENSE-CODE`).
- **Text/figures/PDFs**: CC BY 4.0 (see `LICENSE-DOCS`).

---

## Acknowledgements

This code implements the entropy‑regularized optimization and exponentiated‑gradient update discussed in the paper and test materials. Unit tests include checks on invariants and on the effect of \(\gamma\) on diversification.
