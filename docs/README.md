# GitHub Pages site

This directory is the static site published by `.github/workflows/pages.yml`.
On GitHub, open **Settings → Pages** and set **Build and deployment → Source**
to **GitHub Actions**. A push to `main` that changes `docs/**` will deploy it.

With the repository's current remote (`nguyenduc13475/DataMining_HousePrice`),
the project-site URL is expected to be
`https://nguyenduc13475.github.io/DataMining_HousePrice/` after Pages is enabled.
A root user site such as `https://username.github.io/` requires a repository
named exactly `username.github.io`. A separate custom domain requires a domain
you own, DNS records, and configuration in repository Pages settings.

The valuation form runs a small fitted Ridge model directly in JavaScript. The
model uses five numeric pre-sale features and a JSON export of coefficients,
imputation medians, scaler values, target transform, and evaluation metadata.
The browser formula is checked against the fitted scikit-learn pipeline during
export. GitHub Pages does not run Python; model training remains a local step.

Train and refresh the browser model and comparison table from the repository
root with:

```powershell
.\.venv\Scripts\python.exe members\member_a\train_static_ridge.py
.\.venv\Scripts\python.exe scripts\export_web_results.py
```

The full 86-feature Ridge and Random Forest pipelines remain separate research
models; the static form uses a restricted five-feature Ridge because those
inputs can be represented directly in a small browser form.

The site's current research scope uses the Ames Housing dataset included in this
repository: historical transactions in Ames, Iowa. It does not use the Vietnam
Real Estates Hugging Face dataset, and must not be described as a Vietnamese
market valuation product.

For modeling theory, evaluation protocol, leakage, and browser inference details, see
[`LEARNING_GUIDE_MEMBER_A.md`](LEARNING_GUIDE_MEMBER_A.md).

The interactive Python application is a separate Streamlit deployment (GitHub Pages
cannot run Python). Run `python -m streamlit run app/streamlit_app.py` from the
repository root; see [`STREAMLIT_GUIDE.md`](STREAMLIT_GUIDE.md) for the workflow,
inference behavior, and limitations.

The app's Markdown lessons with embedded Python examples are maintained in
[`tutorials/`](tutorials/README.md) and rendered in the corresponding learning
pages.
