# AxiomAlpha: IJCNN 2027 Research Plan

**Status:** Working research plan; no proposed research result has been established.  
**Prepared:** 21 November 2026; conference information checked on this date.  
**Target:** International Joint Conference on Neural Networks (IJCNN) 2027, regular paper.  
**Intended author:** Kartik Mehra, sole author. Confirm affiliation, contact information and ORCID before submission.  
**Primary project question:** When does the structure of a stock network add reliable information for predicting future market stress beyond conventional indicators and non-graph neural models?

> This document is a working plan, not a manuscript, preregistration, proof of novelty, evidence of predictive power, or a submission. Its design choices are provisional until the author reviews and freezes the experiment specification. Existing AxiomAlpha figures, reports and backtests do not answer the proposed question.

## 1. Conference requirements and submission decisions

| Item | Current information and action |
|---|---|
| Official call | [IJCNN 2027 call for papers](https://ijcnn.org/2027/authors/call-for-papers). Target the **regular paper** track; confirm the selected track and current dates on the official site. |
| Dates | The call lists **1 December 2026** as submissions opening, **31 January 2027** as the regular-paper deadline, 15 March 2027 for notification, 12 April 2027 for camera ready, and 14–18 June 2027 for the conference. Confirm the actual submission cutoff and time zone in CMT; do not assume AoE for the initial deadline. Aim to finish by 27 January. |
| Format | The [initial author instructions](https://ijcnn.org/2027/authors/initial-author-instructions) specify IEEE format, double-blind review and six standard pages including references, figures, tables and appendices. They permit up to four extra pages at a stated USD 100 per page, with initial-submission charges potentially retained even if pages are later removed. Plan for six pages and check the latest rules. |
| Originality | The instructions say the paper must be original and not under review elsewhere. Check venue policy before publishing or presenting substantially similar work. Preprints are discussed in the instructions; do not assume every previous presentation is permitted. |
| Single author | The author list should contain Kartik only, provided the intellectual contribution and authorship are accurate. Outside feedback or tools do not automatically constitute authorship. Disclose assistance and funding where applicable and follow IEEE and institutional policies. Decide authorship before submission; the conference says authors cannot be added after acceptance. |
| AI disclosure | The author instructions currently require disclosure of AI-generated text in acknowledgements and citations to the AI system in affected sections. Verify the latest IEEE/IJCNN language and how to satisfy it while preserving anonymization; contact the organizers if unclear. The author must verify all prose, methods, figures, citations and code and remains responsible for the work. |
| Registration | The [registration page](https://ijcnn.org/2027/registration-information) says an accepted paper requires an eligible full non-student author registration; student registration alone does not cover publication. The fee amounts were placeholders when checked. Confirm the actual amount, applicable deadline and whether an author registration can be covered by institutional funds. |
| Presentation | [Initial author instructions](https://ijcnn.org/2027/authors/initial-author-instructions) describe remote presentation by prerecorded video under conditions, with a registration-format deadline of 12 April 2027. Confirm eligibility, availability, author-registration coverage and current rules before relying on remote presentation. |
| Indexing | The author instructions state that accepted IJCNN papers will be published in IEEE Xplore; confirm final presentation and proceedings conditions. Acceptance and publication are not guaranteed. |

## 2. Executive research position

### 2.1 Proposed working title

**When Do Dynamic Stock Networks Improve Market-Stress Forecasts? A Reliability-Aware Neural Study**

The title is provisional. It must change if the results or final contribution require it. Do not use “reliable,” “causal,” “systemic,” “early warning,” or “improves” as established claims without evidence.

### 2.2 Primary question

At a clearly specified prediction cutoff, does a financial graph provide incremental value for forecasting future high volatility beyond past volatility, market-wide correlation and a non-graph temporal neural model?

### 2.3 Secondary question

Can a model use a past-only measure of graph-estimation stability to decide how much to rely on graph-based forecasts, and does it outperform a generic learned gate or simple ensemble?

### 2.4 Research logic

1. A graph captures a proposed pattern of relationships between assets.
2. Estimated relationships depend on noisy observations, graph rules and market conditions.
3. A graph model can benefit from node features, network summaries, additional parameters or model averaging even if the exact topology is not informative.
4. Matched non-graph models and carefully constructed graph controls can help determine which information contributes to forecasts.
5. A stability-aware gating mechanism is a testable candidate, not a guarantee of better forecasts.

**Potential contribution:** A tightly controlled evaluation plus a modest neural mechanism whose graph contribution, stability signal and calibration can be separately ablated. A sufficiently novel neural contribution for IJCNN is not established; complete the related-work review before making that claim.

### 2.5 Non-claims

- A correlation edge does not establish direct causation, bilateral exposure or contagion.
- Daily close-to-close volatility proxies are not equivalent to intraday realized volatility.
- High market volatility is a market-stress proxy, not direct proof of institutional systemic failure.
- A randomized-graph experiment does not establish causal effects; null models preserve some graph properties while changing others.
- Improved forecast ranking is not proof of calibrated probabilities, portfolio gains, investment profitability or future deployment performance.
- No current AxiomAlpha result proves the proposed method works.

## 3. Literature: what exists and what to verify

The first four papers were reviewed in our project discussion. Two previously identified connectedness papers were available only as abstracts or publisher summaries in that review. Do not characterize unread sections as omissions. For every reference, record the version, exact data, target, forecast origin, split, preprocessing, graphs, controls, metrics and code availability. Recheck literature before claiming novelty.

| Reference | Evidence and relevance | Implication for our work |
|---|---|---|
| [R1] Zhang, Pu, Cucuringu and Dong, “Forecasting realized volatility with spillover effects: Perspectives from graph neural networks,” *International Journal of Forecasting*, 2025, [DOI](https://doi.org/10.1016/j.ijforecast.2024.09.002); [2023 preprint](https://arxiv.org/pdf/2308.01419). | Graph-enhanced HAR and GNNs for stock realized volatility; tests graph hops and MSE/QLIKE. The accessible preprint reports no clear general benefit from multi-hop information. | Financial volatility GNNs and depth comparisons already exist. Confirm final published methods against the preprint; compare against simple HAR-style and graph-based methods. |
| [R2] Gorduza, Zohren and Dong, “Understanding stock market instability via graph auto-encoders,” *EPJ Data Science*, 2025, [DOI](https://doi.org/10.1140/epjds/s13688-025-00523-3). | A graph-reconstruction indicator is used in downstream market-volatility forecasts; node ablations and rewired edges are studied. | Graph-based market instability and basic rewiring already exist. Reproduction needs explicit alignment of snapshot, indicator and hourly outcome; do not infer leakage merely from ambiguous notation. |
| [R3] Kumar, Umeorah and Alochukwu, “Regime-Dependent Graph Neural Networks for Enhanced Volatility Prediction in Financial Markets,” *Mathematics*, 2026, [DOI](https://doi.org/10.3390/math14020289). | Temporal neural/graph volatility model; the paper reports little systematic fixed-versus-dynamic benefit in its expanding-window check and strongly negative R² in that comparison. | Fixed versus dynamic alone is not novel. Recheck main partition-specific graph estimation against prediction-time information; a potential timing concern inferred from prose is not a verified finding about the authors’ implementation. |
| [R4] Lee and Cho, “Graph-Based Stock Volatility Forecasting with Effective Transfer Entropy and Hurst-Based Regime Adaptation,” *Fractal and Fractional*, 2025, [DOI](https://doi.org/10.3390/fractalfract9060339). | Compares relationship definitions and Hurst-triggered updates/retraining. Some aggregate improvements are reported, with variation across years and metrics. | Triggered adaptation and directional financial graphs already exist. Verify target overlap, scaling, tuning periods and graph availability before reproducing. |
| [R5] Bracht et al., “Learning from Structure, Not Sequences: Forecasting Equity Markets Using GNNs on Financial Correlation Networks,” [SSRN preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7154273), 2026, listed as under review when found. | Reports graph and label placebo controls for a short-horizon equity-direction task. | Randomized graph controls are not new. Read the complete accessible version and check its status and exact null designs before specifying our distinction. |
| [R6] Corsi, “A Simple Approximate Long-Memory Model of Realized Volatility,” *Journal of Financial Econometrics*, 2009, [DOI](https://doi.org/10.1093/jjfinec/nbp001). | HAR-RV uses volatility information over multiple time scales. | Use a clearly identified HAR-*inspired* daily-data baseline; do not call it exact HAR-RV if inputs or task differ. |
| [R7] Feng et al., “Temporal Relational Ranking for Stock Prediction,” [arXiv](https://arxiv.org/abs/1809.09441), 2019. | Uses temporal stock information and relational learning for stock ranking. | Foundational financial temporal graph reference; ranking and market-stress classification are different tasks. |
| [R8] Wang et al., “Systemic risk prediction using machine learning: Does network connectedness help prediction?,” 2024, [DOI](https://doi.org/10.1016/j.irfa.2024.103147). | Accessible abstract and publisher sections indicate a study of connectedness and systemic-risk prediction. | Full text was not available during prior review. Obtain legally through library or author before attributing detailed limitations or missing baselines. |
| [R9] Song, Zhang and So, “On the predictive power of network statistics for financial risk indicators,” 2021, [author-institution record](https://researchportal.hkust.edu.hk/en/publications/on-the-predictive-power-of-network-statistics-for-financial-risk-/). | Abstract reports network statistics from a large international stock universe for financial risk indicators. | Full text was not available during prior review. Do not claim the broad idea of network-based risk forecasting is new. |
| [R10] “An Entropy-Regularised AI Framework for Multi-Asset Volatility Spillover Forecasting and CVaR-Constrained Portfolio Allocation,” *Entropy*, 2026, [article](https://www.mdpi.com/1099-4300/28/7/756). | The abstract describes dynamic financial graphs and calibrated uncertainty. | Uncertainty or calibration alone is not a new claim. Inspect the full method and data before asserting a gap. |

### 3.1 Related-work completion checklist

- [ ] Read R5 and R10 in full where available; obtain R8/R9 through legitimate access if feasible.
- [ ] Search financial GNN graph uncertainty, edge-stability estimation, gating, regime adaptation, calibration, market-stress classification and randomized controls.
- [ ] Review bibliographies and papers citing R1–R4; separate peer-reviewed publications from preprints.
- [ ] Create a comparison table with task, data, information cutoff, model, graph control, uncertainty method and evaluation.
- [ ] Write a specific distinction from the closest prior papers; label any unresolved novelty risk.
- [ ] Freeze the research question only after this review and pilot audit.

## 4. Data inventory and acquisition

### 4.1 Current AxiomAlpha pilot: available, exploratory

The checkout inspected for this plan contains `data/processed/log_returns.csv` with 1,508 dates, 30 stocks, and dates from 2019-01-03 through 2024-12-30. `dynamic_graph_metrics.csv` has 73 dated rows; `centrality_scores_daily.csv` has 43,470 rows. These observations are an inventory, not proof of clean point-in-time construction.

The existing universe is a fixed selected set of stocks with selection/survivorship limitations. Existing `RESULTS.md` reports a prior direction-model AUC and exploratory backtest metrics; these are not results for the proposed market-stress task. News data is synthetic and must not be used as real independent news or as evidence of news alpha.

**Actions:**

- Audit `data/raw/prices.csv`, `data/processed/log_returns.csv`, date coverage, symbols, adjustment conventions, missing values, duplicates and corporate actions.
- Recompute experimental features directly from a frozen input snapshot. Do not rely on cached artifacts without proving their provenance and time availability.
- Inspect existing notebooks for full-sample graphs, normalization, same-day data and join timing. Different artifact frequencies require explicit as-of joins; forward fill only after an observation is available.
- The 2019–2024 data has already been explored. Label initial analyses **pilot/development**, not untouched final evidence.
- Retain existing project artifacts; experimental runs write to an isolated research output location and never silently overwrite production artifacts.

### 4.2 Additional public data: French industry portfolios

The [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) and [30 industry portfolio details](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library/det_30_ind_port.html) offer a long daily historical series and documented industry assignment. Record the downloaded snapshot, access date, units, missing-value codes, license/terms and checksums; the library notes historical series may be reconstructed on updates.

Industry portfolios are not individual stocks. Treat them as an **additional industry-node setting** and do not claim stock-level validation from it. Document any overlap between inputs and the market target, definition of market return, and how portfolio membership and published data versions are handled.

### 4.3 Longer stock data and market target

Investigate university access to WRDS/CRSP for historical returns, identifier histories, delistings and point-in-time universe formation; confirm actual subscription, use terms and constituent availability. Public alternatives can be used with explicit universe and survivorship limitations. No purchase or external access is assumed.

Select one documented market-return source with appropriate history, stable units, timestamp semantics and usage rights. FRED notes a 10-year daily-history limit for S&P and Dow series: [FRED S&P 500](https://fred.stlouisfed.org/series/SP500). An equal-weight proxy from the same stock universe is acceptable only if clearly defined, its overlap with node inputs is disclosed, and it does not masquerade as the S&P 500.

### 4.4 Data manifest

For each data snapshot save: source URL and access date; retrieval method and license; date range, timezone, ticker/identifier rules and universe; price adjustment and return units; missing-data policy; split/calendar rules; checksum; scripts and configuration version. Never publish restricted raw data or keys.

## 5. Pilot task and point-in-time protocol

### 5.1 Forecast cutoff

Forecast **after the US market close on trading day t**, once each input is actually available. Features and graphs may use observations through t; labels must start after t. Record data-provider publication timing and calendar alignment; end-of-day historical availability is not automatically proven by a CSV date. For different markets, explicitly align closes and availability rather than assuming same-date information is synchronous.

### 5.2 Primary target

Provisional pilot target: future 20-trading-day root mean squared market return,

```text
V[t,20] = sqrt((1/20) * sum_{i=1..20} r_market[t+i]^2)
y[t] = 1 if V[t,20] > q else 0
```

`q` is the 80th percentile of fully matured training-period target values, estimated within each training fold and frozen for the associated evaluation. This defines relative high-volatility events, not a fixed economic danger threshold. Record units, annualization choice and handling of missing trading days. Consider log returns versus simple returns explicitly; do not mix them silently.

Before running the final test, freeze the target, percentile, forecast horizon, market proxy and threshold policy. If event frequency is inadequate, revise based on development data only and document the revision. Secondary horizons or thresholds are exploratory unless preregistered.

### 5.3 Temporal split and purge

Use expanding or rolling training folds, validation folds for model/hyperparameter decisions, and a final untouched chronological test if an actually untouched period is available. Do not call a repeatedly inspected period a holdout. Predeclare dates only after verifying data availability; record why they were chosen.

For a label with horizon H=20, an observation can enter a training/refit set only when **all outcomes t+1 through t+20 are available before the next evaluation starts**. Purge overlapping labels at train/validation and validation/test boundaries and at every walk-forward refit. Preserve any needed historical lookback for computing features without using future outcomes. Verify date-level disjointness, matured labels, cutoff semantics and purge behavior programmatically.

- Fit imputation, scaling, feature selection, threshold selection, graph hyperparameters, models, decision thresholds and calibration only on permitted training/validation data.
- Fit the calibrator on a chronologically later validation/calibration subset or genuinely out-of-fold predictions; never use the final test.
- Prevent the same target window or graph future observation appearing in training and test.
- Record every inspected evaluation and every experiment, including failures. Do not tune on the final test.

### 5.4 Required early tests

- Future-return mutation cannot alter features, graph, scaling or prediction at an earlier time.
- Target uses t+1…t+20 exactly; final incomplete horizons are excluded.
- Purging removes every training label crossing a split/refit cutoff.
- Thresholds, imputation, feature selection and scalers use only eligible history.
- Graph construction and stability resampling use only data available by each forecast cutoff.
- Calendar alignment, duplicates, missing assets, newly listed/delisted nodes and adjusted prices have documented behavior.
- Test-only results cannot update model selection, calibration or threshold settings.

## 6. Models and controlled comparisons

All models share target, forecast dates, features allowed by the comparison, folds, training windows, risk threshold, evaluation and test observations. Start small; choose model families and tuning budgets before the final holdout.

| ID | Model | Purpose |
|---|---|---|
| B0 | Training-period stress prevalence | Constant-probability baseline; report prevalence because PR-AUC depends on it. |
| B1 | Past-volatility / persistence and logistic regression | Interpretable risk baseline. |
| B2 | HAR-inspired daily/weekly/monthly volatility features + logistic model | Strong simple multiscale baseline; not an exact intraday HAR-RV reproduction. |
| B3 | B2 plus average pairwise stock correlation and optionally graph density | Tests whether simple network summaries explain gains. |
| B4 | Gradient-boosted tree classifier | Strong nonlinear non-graph baseline. |
| N0 | Small temporal MLP, GRU or LSTM | Same time information without edges; select one primary architecture in validation. |
| G1 | Same temporal backbone plus fixed graph | Measures graph contribution. |
| G2 | Same backbone plus trailing-window periodic graph | Tests changing graph estimates under matched training. |
| G3 | G2 with randomized-graph control | Tests dependence on a particular graph realization. Use repeated, documented null samples. |
| E0 | Fixed-weight graph/non-graph ensemble | Tests if simple averaging explains improvement. |
| E1 | Generic learned gate without stability input | Tests whether ordinary mixing, extra capacity or optimization explains improvement. |
| E2 | Candidate graph-stability-aware gate | Tests the specific candidate contribution; report even if it loses. |

Where feasible include a directly relevant published graph baseline with implementation and deviations documented. Match information availability, parameter counts/compute budgets as reasonably possible, and tune all model families fairly. Compare equivalent forecasts on the same dates, not disjoint samples. Use multiple training seeds for neural variability; seeds are not independent market histories.

### 6.1 Graph controls

Consider edge-count-preserving rewiring, degree-preserving swaps where feasible, sector-based graph, density/average-correlation-only summaries and node-feature ablations. State exactly which graph properties each null preserves or changes. Use repeated random seeds and record graph diagnostics. Controls are diagnostic, not causal proof. Do not claim rewiring is new: R2 and R5 already use related controls.

## 7. Candidate neural mechanism: stability-aware fusion

A compact candidate combines a temporal probability `p_temporal` with a graph probability `p_graph`:

```text
p = (1 - g) * p_temporal + g * p_graph,  0 <= g <= 1
```

`g` is a learned gate receiving graph-stability summaries and selected past-only state variables. A simple way to estimate stability is repeated block-resampling of the trailing return window, constructing a graph per resample and summarizing edge consistency. Tune block length and number of resamples on development data only; record computational costs. Compare a generic gate with the same inputs except the proposed stability signal, and a gate given simple correlation/density summaries to detect confounding.

**Key limitations:** bootstrap samples are dependent on the observed window; edge stability is not a calibrated probability of true connections; changes in graph density, volatility, missingness and window length may affect the score. A gate does not establish causality, guarantee improvement or guarantee calibration.

### 7.1 Architecture and implementation

Use PyTorch already present in the checked environment. PyTorch Geometric was not found there; assess compatibility and installation only if needed, or implement a small auditable graph layer in PyTorch. Do not change the project dependency environment silently. Record Python, package versions, CPU/GPU, random seeds, model size and runtime. Keep the architecture compact enough for an independent solo-author project and fair comparison.

### 7.2 Controlled adaptation study

Secondary, only after baseline validity: compare fixed/updated graph crossed with fixed/periodic or triggered model refits. Use matched information cutoffs and budgets; changing the graph while holding weights fixed can be unfair if the model expects a different graph distribution. Separate graph update effects from weight retraining effects; include non-graph retraining controls. Regime-triggered retraining is already studied in R4.

## 8. Evaluation, inference and decision thresholds

### Primary metric

**Average precision (area under the precision-recall curve)** for the predefined binary event. State the estimator used. Report stress prevalence in every fold and dataset. Interpret changes against the prevalence baseline; do not assume PR-AUC is comparable across populations with different event rates.

### Secondary metrics

ROC-AUC, Brier score, log loss, calibration curve, precision/recall and false-alarm rate at a decision threshold selected on validation data. Report absolute performance, paired differences and uncertainty, not just improvements or significance. Calibration-bin choices and prevalence must be disclosed.

### Uncertainty and robustness

- Report sample count, event prevalence and the number of distinct/approximately separated stress episodes, not only positive days.
- Use paired time-block bootstrap or another justified dependent-data method on forecast differences; document block length and sensitivity. Overlapping 20-day outcomes create serial dependence. Report uncertainty on calibration and ranking where feasible.
- Report seed-to-seed variability separately from market-period uncertainty; seeds are not independent crises.
- Predefine main comparisons; log all trials and discuss multiple-testing/selection risk.
- Report per-period results, graph-window/threshold sensitivity and available universe checks without selecting only favorable periods.
- Retrospective stress-period analysis must not be presented as a regime label known ahead of time.
- If positive events are too few, intervals are too wide or the final dataset cannot be obtained, reduce claims or defer submission.

### Go/no-go criteria

Before final test, define a smallest meaningful effect and success criteria on validation data. A positive claim requires improvement against matched strong baselines and generic gate, with uncertainty compatible with the claim and stability on a separate dataset or period. If models tie, fail, or perform differently across settings, document that outcome and reassess the contribution. Do not promise a target accuracy or publication.

## 9. Optional portfolio analysis — only after core study

Do not make a trading claim from a stress classifier alone. If time permits and forecasts are valid, run a secondary, predeclared decision experiment comparing equal risk budgets and a volatility-only risk-control baseline over identical dates. Include transaction costs, cash yield, initial allocation costs, turnover, exposure, decision timing and uncertainty. Existing AxiomAlpha backtest metrics are exploratory and cannot be presented as evidence for the proposed model. Omit this analysis if it compromises the primary forecast study or cannot be tested rigorously.

## 10. Reproducible workspace and workflow

Proposed implementation layout; create incrementally, not all at once:

```text
IJCNN/
  RESEARCH_PLAN.md             # this document
  protocol/                    # frozen decisions and experiment log
  literature/                  # citation matrix and reading notes; no unlicensed PDFs
  configs/                     # dataset, split and model configurations
  src/                         # isolated data, graph, model and evaluation code
  tests/                       # timing, target and graph tests
  results/                     # generated summaries, manifests and figures
  paper/                       # anonymous manuscript when evidence is ready
```

Data flow:

1. Source snapshot -> provenance manifest and validated returns.
2. Training-fold target threshold and past-only features -> timestamped dataset.
3. Point-in-time graph snapshots and matched null graphs.
4. Purged chronological training/validation -> fixed models and calibration.
5. Locked final evaluation -> predictions, metrics and block-bootstrap intervals.
6. Reproducible tables and figures -> manuscript claims tied to result IDs.

Store outputs with: input hashes, data/split dates, configuration, model and software version, seed, graph-rule and threshold, code revision, run time, predictions and metrics. Keep large/generated results and credentials out of Git unless deliberately reviewed. Re-run an experiment from saved configuration and verify output consistency before reporting.

**Do not publish or push this research plan or results automatically.** Git status and any prior pending changes must be checked before any future commit/push; follow the user’s current instructions.

## 11. Schedule and decision gates

| Stage | Working dates | Deliverable | Gate |
|---|---|---|---|
| 1. Literature and data audit | 21–27 Nov 2026 | Source matrix, audit, preliminary protocol | Verify data and confirm the question is not already resolved. |
| 2. Pilot dataset and baselines | 28 Nov–10 Dec | Point-in-time pilot, tests, B0–B3 | Enough mature labels and usable baselines; pilot only. |
| 3. Neural and graph baselines | 11–24 Dec | B4, N0, G1/G2 and initial controls | Determine whether there is evidence graph structure can add value. |
| 4. Candidate fusion | 25 Dec–8 Jan | E0/E1/E2 and matched ablations | Is stability input better than generic mixing? |
| 5. Robustness | 9–18 Jan | Additional dataset, controls, uncertainty | Are claims stable and supported, with enough distinct episodes? |
| 6. Manuscript | 19–27 Jan | Six-page anonymous IEEE paper and traceable figures | Honest, specific contribution and complete reproducibility. |
| 7. Final review | 28–31 Jan | Policy, authorship, AI disclosure and submission check | Submit only if ready before verified deadline. |

These dates are planning targets, not guaranteed implementation or acceptance. Replan if data access, compute or results invalidate a milestone.

## 12. Manuscript and review checklist

- [ ] One focused research question and accurate task definition.
- [ ] Literature matrix demonstrates a specific distinction from R1–R5 and other close work.
- [ ] All paper claims link to saved run IDs and complete data/split specifications.
- [ ] No invented results, references, baselines or unsupported novelty claims.
- [ ] Report negative and mixed results; no claims of causal contagion from correlation.
- [ ] Single-author details and attribution correct; blind submission anonymized, including code links and metadata where required.
- [ ] Follow the latest IJCNN and IEEE AI-disclosure instructions; resolve any ambiguity with conference organizers without compromising anonymity.
- [ ] Follow original-work and concurrent-submission policies; review prior presentation/preprint rules.
- [ ] Figure labels include units, horizons, sample, uncertainty and any synthetic data label; check color accessibility.
- [ ] Record data access and redistribution rights and ensure anonymous reviewers can assess permitted artifacts.
- [ ] Validate exact deadline, time zone, page rules, submission portal, author registration, presentation and indexing requirements on official pages.

## 13. Immediate next actions

1. Audit source data and document whether graph artifacts are point-in-time; do not modify existing results.
2. Obtain/read missing literature and create a side-by-side comparison matrix, distinguishing verified evidence from inference.
3. Freeze a pilot protocol: target, cutoff, matured-label threshold, purged splits, primary metric and event count.
4. Implement and test baseline dataset construction in an isolated research module.
5. Run B0–B3 on development data only, save manifests and report results as pilot observations with limitations.
6. Decide from the pilot whether to proceed to non-graph neural and graph baselines. Do not claim new results or submit until the evidence exists.

## 14. Work started: 21 November 2026

- Initial pilot source audit and exploratory 2022 baseline: [`protocol/PILOT_AUDIT.md`](protocol/PILOT_AUDIT.md).
- Initial reading and comparison matrix: [`protocol/LITERATURE_MATRIX.md`](protocol/LITERATURE_MATRIX.md); restricted papers remain unread in full.
- Reproducible development runner: [`src/pilot.py`](src/pilot.py); forecasts and metrics: [`results/pilot/`](results/pilot/). This pilot is not a frozen preregistration or an independent final evaluation.
- Timing and split checks: [`tests/test_pilot.py`](tests/test_pilot.py). The pilot tests and existing static pipeline tests passed in the observed environment.
- The 2022 validation period has been inspected. Existing 2023–2024 results were previously explored in the project; no current period is claimed as a clean final holdout.
- Remaining Stage 1: source/adjustment provenance, full restricted-paper review where legitimately available, longer/independent data, and a frozen walk-forward and final evaluation protocol.

## 15. Walk-forward development check: 21 November 2026

Added [`src/walk_forward.py`](src/walk_forward.py), [`tests/test_walk_forward.py`](tests/test_walk_forward.py) and [`protocol/WALK_FORWARD_AUDIT.md`](protocol/WALK_FORWARD_AUDIT.md). Annual 2022–2024 folds are exploratory, with matured-label purging and training-fold thresholds. The 2023 and 2024 folds contain no positive stress outcomes under the selected rule, so event-ranking comparisons are not informative. Before neural/graph claims, obtain longer data, review the target definition, and reserve a genuinely independent final evaluation. Existing years are not untouched holdouts.
