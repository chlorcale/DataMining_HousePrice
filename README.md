# Ames Housing Static Site

This is a standalone, static GitHub Pages site. It contains no Python backend and
no raw dataset. The fitted five-feature Ridge model is exported as
`model/ridge_model.json`; the four-model comparison table is
`model/research_results.json`.

The form predicts historic Ames, Iowa transaction prices in USD. It is an
academic model, not a current market quote or a Vietnamese valuation.

## Publish from a new GitHub repository

1. Create a new empty repository on your GitHub account. For a project URL such
   as `https://USERNAME.github.io/ames-housing-static/`, use a repository named
   `ames-housing-static`. To use the root URL `https://USERNAME.github.io/`, the
   repository name must be exactly `USERNAME.github.io`.
2. Open PowerShell in this folder and connect this already initialized Git
   repository to the new remote:

   ```powershell
   git remote add origin https://github.com/USERNAME/REPOSITORY.git
   git push -u origin main
   ```

3. In the repository, open **Settings → Pages**. Under **Build and deployment**,
   set **Source** to **GitHub Actions**. The workflow in
   `.github/workflows/pages.yml` deploys the site on pushes to `main` and can
   also be started manually from the **Actions** tab.
4. Wait for the **Publish static site** workflow to finish. GitHub shows the live
   URL under **Settings → Pages**.

Replace `USERNAME` and `REPOSITORY` with your GitHub account and new repository.
Do not run `git init` from the parent coursework folder; this folder is already
separate so it can have its own Git history and remote.

## Local preview

Run from this folder:

```powershell
python -m http.server 8000
```

Then open `http://localhost:8000`. The site fetches model JSON, so use an HTTP
server rather than opening `index.html` as a `file://` URL.

## Updating trained results

Training scripts and datasets are intentionally kept in the coursework
repository. If you retrain, export the browser Ridge artifact and the comparison
metrics JSON from that coursework repository, then copy the resulting
`ridge_model.json` and `research_results.json` into this site's `model/` folder.
Keep the current schemas: `ridge_model.json` has five ordered feature objects,
an intercept, and matching ordered coefficients; `research_results.json` has a
`models` array and its holdout scope. Commit the two exported JSON files here.
The browser form evaluates `expm1(intercept + Σ(coefficient ×
((input − mean) / scale)))`; if the artifact or feature order changes, update the
browser implementation to match and verify parity against the coursework
export. Do not copy the dataset or training scripts into this website repo.

## Site structure

- `index.html`: home page and results overview.
- `pages/data.html`: dataset scope, real-row walkthrough, and preprocessing steps.
- `pages/algorithms.html`: switchable algorithm comparison.
- `pages/models/`: theory and experiment notes for the four models.
- `pages/evaluation.html`: split protocol and metrics.
- `pages/demo.html`: five-feature Ridge inference in the browser.
- `pages/ames_samples.json`: eight example transactions extracted from the Ames CSV.
- `model/`: browser model coefficients and experiment metrics.

Use the buttons on the data page to move through a real record, preprocessing, model selection, and holdout evaluation. The record's actual sale price is shown for teaching; it is not presented as a model prediction.