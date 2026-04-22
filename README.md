# Does Job Training Cause Higher Earnings?

**A Causal Analysis of the NSW Program Using Double Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B.svg)](https://econ5200-final-project-3mvbvspgbtb9vn7gnj63a3.streamlit.app)
[![Method](https://img.shields.io/badge/Method-Double_ML-1F3864.svg)](https://arxiv.org/abs/1608.00060)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Course:** ECON 5200 — Causal Machine Learning & Applied Analytics
> **Institution:** Northeastern University, Spring 2026
> **Author:** Zehan Qin
> **Client (hypothetical):** U.S. Department of Labor

**🔗 Live Dashboard:** https://econ5200-final-project-3mvbvspgbtb9vn7gnj63a3.streamlit.app

---

## 📌 Headline Result

| Method | Estimate | 95% CI | Interpretation |
|---|---|---|---|
| Naive OLS (no controls) | −$8,498 | [−$9,893, −$7,102] | ❌ Wrong direction |
| OLS + 8 controls | $699 | [−$374, $1,773] | ⚠️ Right direction, too small |
| DML (GBR nuisance) | $365 | [−$814, $1,543] | ⚠️ Robustness check; CI crosses 0 |
| **DML (RF) — primary** | **$1,541** | **[$229, $2,853]** | ✅ **CI excludes 0** |
| RCT benchmark | $1,794 | (experimental) | 🎯 Validation target |

The causal DML estimate is **statistically significant** and **closely aligned** with the experimental benchmark — validating the observational identification strategy.

---

## 🎯 Research Question

> **Does participation in the NSW job training program cause higher post-program earnings?**

Answering this is harder than it looks. A naive comparison of treated vs. untreated workers yields **−$8,498** — suggesting training *reduces* earnings. This is wrong, and the direction is wrong, because NSW participants were selected precisely because they had the weakest labor market histories (pre-treatment earnings of $2,096 vs. $14,017 for CPS controls).

This project uses **Double Machine Learning (DML)** (Chernozhukov et al., 2018) to recover the causal effect while correcting for severe selection bias, then validates the result against the original NSW randomized experiment.

---

## 🧪 Methodology

**Identification strategy:** Double Machine Learning with cross-fitting

- **Outcome model** `g(X) = E[Y | X]`: RandomForestRegressor / GradientBoostingRegressor
- **Treatment model** `m(X) = P(T = 1 | X)`: RandomForestClassifier / GradientBoostingClassifier
- Second stage: regress residuals `Y − g(X)` on `T − m(X)` to recover ATE
- `discrete_treatment=True` to ensure valid propensity scores in [0, 1]

**Key identifying assumption:** Conditional independence (unconfoundedness)

```
{Y(0), Y(1)} ⊥ T | X
```

Given covariates X (age, education, race, marital status, pre-treatment earnings), treatment assignment is independent of potential outcomes.

**Why not other methods?** IV (no plausible instrument), DiD (thin panel), PSM (unstable across comparison groups per Smith & Todd 2005), RCT (not available in the constructed observational sample).

---

## 💡 Why This Project Matters: Prediction vs. Causation

A predictive Random Forest trained on the same data achieves **R² = 0.42** — respectable predictive performance. But the feature importance of `treat` is only **0.0025**, nearly the lowest of all features. A policy analyst reading a predictive model output could wrongly conclude the training program is irrelevant to earnings.

**The causal DML estimate tells the opposite story.** For policy questions — where the goal is to decide whether to change something, not merely forecast it — causal methods are indispensable.

---

## 📁 Repository Structure

```
ECON5200-FINAL-PROJECT/
├── README.md                                  ← You are here
├── requirements.txt                           ← Streamlit dashboard dependencies
├── app.py                                     ← Streamlit dashboard (deployed)
├── checkpoint/                                ← Apr 19 checkpoint submission
│   ├── Research_Proposal.pdf
│   └── checkpoint_ECON5200_Final_Project_v2.ipynb
├── final_complete_notebooks/                  ← Final analysis notebook
│   └── Zehan_Qin_ECON5200_Final_Project_COMPLETE_v6.ipynb
└── deliverables/                              ← Final submission PDFs
    ├── Executive_Summary.pdf
    ├── Technical_Report.pdf
    ├── Threats_to_Identification.pdf
    └── AI_Methodology_Appendix.pdf
```

---

## 🚀 How to Reproduce

### 1. Clone the repo

```bash
git clone https://github.com/ZehanQin/ECON5200-FINAL-PROJECT.git
cd ECON5200-FINAL-PROJECT
```

### 2. Set up the environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

**For the Streamlit dashboard** (lightweight — only needs `streamlit`, `pandas`, `numpy`, `plotly`):

```bash
pip install -r requirements.txt
```

**To additionally run the analysis notebook** (needs `econml`, `causaldata`, `scikit-learn`, `matplotlib`):

```bash
pip install econml causaldata scikit-learn matplotlib jupyter
```

### 3. Run the analysis notebook

```bash
jupyter notebook final_complete_notebooks/Zehan_Qin_ECON5200_Final_Project_COMPLETE_v6.ipynb
```

Executing all cells reproduces the full analysis: data loading, balance checks, naive benchmarks, primary DML estimate, robustness checks, and all figures.

### 4. Launch the Streamlit dashboard locally

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501` with live what-if controls for treatment intensity, confidence level, nuisance model, program cost, and effect persistence.

---

## 📊 Dataset

**Source:** Lalonde (1986) / Dehejia-Wahba (1999), loaded via the `causaldata` Python package.

| Source | N | Description |
|---|---|---|
| NSW treated | 185 | Randomized into the job training program |
| CPS-1 controls | 15,992 | General population workers (Current Population Survey) |
| **Total** | **16,177** | Combined observational sample |

The NSW experimental control group is deliberately discarded to simulate a realistic policy setting where only observational controls are available. The experimental ATE (~$1,794) is reserved as a *validation benchmark*, not a training input.

---

## 🎛️ Interactive Dashboard

The deployed [Streamlit dashboard](https://econ5200-final-project-3mvbvspgbtb9vn7gnj63a3.streamlit.app) supports:

- **Parameter sliders** — treatment intensity multiplier, confidence level, nuisance model, program cost, effect persistence
- **Dynamic confidence intervals** — uncertainty bounds update in real time as parameters change
- **Counterfactual scenarios** — "If treatment intensity changes by X%, estimated effect becomes Y (CI: [a, b])"
- **Method comparison chart** — naive vs. causal estimates side-by-side
- **ROI calculator** — payback period, net benefit per participant, break-even analysis

---

## 📑 Deliverables

| Deliverable | File | Purpose |
|---|---|---|
| Executive Summary | [`deliverables/Executive_Summary.pdf`](deliverables/Executive_Summary.pdf) | SCR-structured 1-page decision memo |
| Technical Report | [`deliverables/Technical_Report.pdf`](deliverables/Technical_Report.pdf) | 8-page methodology, results, robustness |
| Threats to Identification | [`deliverables/Threats_to_Identification.pdf`](deliverables/Threats_to_Identification.pdf) | Honest adversarial assessment |
| AI Methodology Appendix | [`deliverables/AI_Methodology_Appendix.pdf`](deliverables/AI_Methodology_Appendix.pdf) | P.R.I.M.E. documentation of AI-assisted workflow |
| Streamlit Dashboard | [Live deployment](https://econ5200-final-project-3mvbvspgbtb9vn7gnj63a3.streamlit.app) | Interactive what-if scenarios |

---

## ⚠️ Limitations

1. **Unobserved confounding.** NSW referrals screened on motivation, mental health, and criminal history — none observed in our data.
2. **Overlap violation.** |SMD| up to 2.43; DML may extrapolate into regions with no treated units.
3. **External validity.** Data is 1975–1978; modern labor markets and training technologies differ substantially.
4. **Specification sensitivity.** RF and GBR nuisance models yield a factor-of-four difference in point estimates; directional conclusion is robust, magnitude is not.
5. **ROI is conditional.** At the dashboard's default 5-year persistence assumption, net benefit is negative (−$4,796). The program delivers positive social returns only under sustained-effect assumptions (8+ years).

See [`deliverables/Threats_to_Identification.pdf`](deliverables/Threats_to_Identification.pdf) for the full adversarial analysis.

---

## 📚 References

- Chernozhukov, V. et al. (2018). *Double/debiased machine learning for treatment and structural parameters.* The Econometrics Journal, 21(1), C1–C68.
- Dehejia, R. H., & Wahba, S. (1999). *Causal effects in nonexperimental studies: Reevaluating the evaluation of training programs.* JASA, 94(448), 1053–1062.
- LaLonde, R. J. (1986). *Evaluating the econometric evaluations of training programs with experimental data.* American Economic Review, 76(4), 604–620.
- Smith, J. A., & Todd, P. E. (2005). *Does matching overcome LaLonde's critique of nonexperimental estimators?* Journal of Econometrics, 125(1–2), 305–353.
- Battocchi, K. et al. (2019). *EconML: A Python package for ML-based heterogeneous treatment effects estimation.* Microsoft Research.

---

## 🛠️ Tech Stack

`Python 3.10` · `pandas` · `numpy` · `scikit-learn` · `econml` · `causaldata` · `matplotlib` · `plotly` · `streamlit`

---

## 📄 License

MIT License — feel free to reuse this code with attribution.

---

*This project was completed for ECON 5200 at Northeastern University, Spring 2026. AI tools (Claude) were used for code generation, drafting, and methodology review; all causal interpretation and final decisions are the author's own. See the [AI Methodology Appendix](deliverables/AI_Methodology_Appendix.pdf) for full P.R.I.M.E. documentation.*
