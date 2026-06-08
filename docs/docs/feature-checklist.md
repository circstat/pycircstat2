

### 1. Descriptive Statistics

| Feature                             | PyCircStat2           | PyCircStat                | CircStat (MATLAB)  | CircStats (R) | circular (R)                          |
|-------------------------------------|-----------------------|---------------------------|--------------------|---------------|---------------------------------------|
| **Measures of Central Tendency**    |                       |                           |                    |               |                                       |
| Circular Mean                       | `circ_mean`           | `mean(alpha)`             | `circ_mean(alpha)` | `circ.mean`   | `mean.circular`                       |
| Circular Mean CI                    | `circ_mean_ci`        | `mean(alpha, ci=95)`      | `circ_confmean`    | -             | `mle.vonmises.bootstrap.ci`           |
| Circular Median (Fisher/Mardia)     | `circ_median` (method=`"deviation"`/`"count"`) [^median-grouped] | `median`                  | `circ_median`      | -             | `median.circular`                     |
| Hodges-Lehmann Median               | `circ_median` (method=`"HL1"`/`"HL2"`/`"HL3"`) | -                         | -                  | -             | `medianHL.circular`[^medianHL-broken] |
| Circular Median CI                  | `circ_median_ci`      | -                         | -                  | -             | -                                     |
| Circular Quantile                   | `circ_quantile`       | -                         | -                  | -             | `quantile.circular`                   |
| **Measures of Spread & Dispersion** |                       |                           |                    |               |                                       |
| Resultant Vector Length             | `circ_r`              | `resultant_vector_length` | `circ_r`           | `est.rho`     | `rho.circular`                        |
| Angular Variance                    | `angular_var`         | `avar`                    | `circ_var`         | -             | `angular.variance`                    |
| Angular Standard Deviation          | `angular_std`         | `astd`                    | `circ_std`         | -             | `angular.deviation`                   |
| Circular Variance                   | `circ_var`            | `var`                     | `circ_var`         | `circ.disp`   | `var.circular`                        |
| Circular Standard Deviation         | `circ_std`            | `std`                     | `circ_std`         | -             | `sd.circular`                         |
| Circular Dispersion                 | `circ_dispersion`     | -                         | -                  | -             | -                                     |
| Circular Range                      | `circ_range`          | -                         | -                  | `circ.range`  | `range.circular`                      |
| Concentration Parameter (κ)         | `circ_kappa`          | `kappa`                   | `circ_kappa`       | `est.kappa`   | `A1inv`                               |
| **Higher-Order Statistics**         |                       |                           |                    |               |                                       |
| Circular Moment                     | `circ_moment`         | `moment`                  | `circ_moment`      | `tri.moment`  | `trigonometric.moment`                |
| Circular Skewness                   | `circ_skewness`       | `skewness`                | `circ_skewness`    | -             | -                                     |
| Circular Kurtosis                   | `circ_kurtosis`       | `kurtoisis`               | `circ_kurtosis`    | -             | -                                     |
| **Distance & Pairwise Comparisons** |                       |                           |                    |               |                                       |
| Mean deviation [^mean-dev]          | `circ_mean_deviation` | -                         | -                  | -             | `meandeviation`                       |
| Circular Distance                   | `circ_dist`           | `cdist`                   | `circ_dist`        | -             | -                                     |
| Pairwise Circular Distance          | `circ_pairdist`       | `pairwise_cdiff`          | `circ_dist2`       | -             | `dist.circular`                       |

### 2. Hypothesis Testing

#### One-Sample Tests for Significance

| Feature                     | H0                                  | PyCircStat2         | PyCircStat | CircStat (MATLAB) | CircStats (R) | circular (R)    |
|-----------------------------|-------------------------------------|---------------------|------------|-------------------|---------------|-----------------|
| **Mean Direction**          |                                     |                     |            |                   |               |                 |
| Rayleigh Test               | $\rho=0$ [^uniform]                 | `rayleigh_test`     | `rayleigh` | `circ_rtest`      | `r.test`      | `rayleigh.test` |
| V-Test                      | $\rho=0$                            | `V_test`            | `vtest`    | `circ_vtest`      | `v0.test`     | `rayleigh.test(., mu=)`[^vtest-circular] |
| One-sample Test             | $\tilde\mu=μ_0$                     | `one_sample_test`   | `mtest`    | `circ_mtest`      | -             | -               |
| Change Point Test           | no change point                     | `change_point_test` | -          | -                 | `change.pt`   | `change.point`  |
| **Median Direction**        |                                     |                     |            |                   |               |                 |
| Hodges-Ajne (omnibus) Test  | $\rho=0$                            | `omnibus_test`      | `omnibus`  | `circ_otest`      | -             | -               |
| Batschelet Test             | $\rho=0$                            | `batschelet_test`   | -          | -                 | -             | -               |
| Binomial Test               | $\tilde\theta = \theta_0$ [^median] | `binomial_test`     | `medtest`  | `circ_medtest`    | -             | -               |
| Symmetry Test around median | $\text{symmetry}$                   | `symmetry_test`     | `symtest`  | `circ_symtest`    | -             | -               |

#### Multi-Sample Tests for Significance

| Feature                         | H0                                            | PyCircStat2                  | PyCircStat        | CircStat (MATLAB) | CircStats (R)     | circular (R)           |
|---------------------------------|-----------------------------------------------|------------------------------|-------------------|-------------------|-------------------|------------------------|
| **Mean Direction**              |                                               |                              |                   |                   |                   |                        |
| Circular Analysis of Variance   | $\mu_1 = \dots = \mu_n$                       | `circ_anova`                 | -                 | -                 | -                 | `aov.circular`         |
| Watson-Williams Test [^one-way] | $\mu_1 = \dots = \mu_n$                       | `watson_williams_test`       | `watson_williams` | `circ_wwtest`     | -                 | `watson.williams.test` |
| Harrison-Kanji Test[^two-way]   | $\mu_1 = \dots = \mu_n$                       | `harrison_kanji_test`        | `hktest`          | `circ_hktest`     | -                 | -                      |
| **Median Direction**            |                                               |                              |                   |                   |                   |                        |
| Common Median Test              | $\tilde{\theta}_1 = \dots = \tilde{\theta}_n$ | `common_median_test`         | `cmtest`          | `circ_cmtest`     | -                 | -                      |
| **Concentration**               |                                               |                              |                   |                   |                   |                        |
| Concentration Test (F-test)     | $\kappa_1 = \dots = \kappa_n$                 | `concentration_test`         | -                 | `circ_ktest`      | -                 | -                      |
| Equal Kappa Test                | $\kappa_1 = \dots = \kappa_n$                 | `equal_kappa_test`           | -                 | -                 | -                 | `equal.kappa.test`     |
| **Distribution Homogeneity**    |                                               |                              |                   |                   |                   |                        |
| Watson's U2 Test                | $F_1 = F_2$ [^F]                              | `watson_u2_test`             | -                 | -                 | `watson.two`      | `watson.two.test`      |
| Two-sample Kuiper Test[^kuiper-2samp] | $F_1 = F_2$                             | `kuiper_two_test`            | `kuiper`          | `circ_kuipertest` | -                 | -                      |
| Wallraff Test                   | $F_1 = F_2$                                   | `wallraff_test`              | -                 | -                 | -                 | `wallraff.test`        |
| Wheeler-Watson Test             | $F_1 = F_2$                                   | `wheeler_watson_test`        | -                 | -                 | -                 | `watson.wheeler.test`  |
| Angular Randomization Test      | $F_1 = F_2$                                   | `angular_randomisation_test` | -                 | -                 | -                 | -                      |
| Rao's Tests for Homogeneity     | $F_1 = F_2$                                   | `rao_homogeneity_test`       | -                 | -                 | `rao.homogeneity` | `rao.test`             |

#### Goodness-of-fit Tests

| Feature             | H0         | PyCircStat2        | PyCircStat   | CircStat (MATLAB) | CircStats (R) | circular (R)       |
|---------------------|------------|--------------------|--------------|-------------------|---------------|--------------------|
| Kuiper’s Test (one-sample)[^kuiper-1samp] | $\rho = 0$ | `kuiper_test`      | -            | -                 | `kuiper`      | `kuiper.test`      |
| Rao’s Spacing Test  | $\rho = 0$ | `rao_spacing_test` | `raospacing` | `circ_raotest`    | `rao.spacing` | `rao.spacing.test` |
| Watson's Test       | $\rho = 0$ | `watson_test`      | -            | -                 | `watson`      | `watson.test`      |
| Watson's Test (von Mises GoF)[^vm-gof] | von Mises | `watson_test(dist="vonmises")` | -            | -                 | `watson(dist="vm")` | `watson.test(dist="vonmises")` |
| Circular Range Test | $\rho = 0$ | `circ_range_test`  | -            | -                 | `circ_range`  | `range.circular`   |


### 3. Correlation & Regression
| Feature                                       | PyCircStat2    | PyCircStat | CircStat (MATLAB) | CircStats (R) | circular (R)              |
|-----------------------------------------------|----------------|------------|-------------------|---------------|---------------------------|
| Circular-Circular Correlation                 | `circ_corrcc`  | `corrcc`   | `circ_corrcc`     | `circ.cor`    | `cor.circular`            |
| Circular-Linear Correlation                   | `circ_corrcl`  | `corrcl`   | `circ_corrcl`     | -             | -                         |
| Circular-Circular Regression                  | `CCRegression` | -          | -                 | `circ.reg`    | `lm.circular(type="c-c")` |
| Circular-Linear Regression [^cl-resp]         | `CLRegression` | -          | -                 | -             | `lm.circular(type="c-l")` |
| Linear-Circular Regression (harmonic) [^lc-resp] | `LCRegression` | -          | -                 | -             | -                         |



### 4. Circular Distributions

All circular distributions assume angles are on ``[0, 2π)``. Inputs are automatically wrapped to that support as a convenience. We remove SciPy's ``loc``/``scale`` convention—parameters like ``mu``, ``rho``, etc. are the only inputs. 

#### Symmetric Circular Distributions

| Feature              | Method | PyCircStat2               | PyCircStat       | CircStat (MATLAB) | CircStats (R) | circular (R)        |
|----------------------|--------|---------------------------|------------------|-------------------|---------------|---------------------|
| Circular Uniform     | PDF    | `circularuniform.pdf`     | -                | -                 | -             | `dcircularuniform`  |
|                      | CDF    | `circularuniform.cdf`     | -                | -                 | -             | -                   |
|                      | PPF    | `circularuniform.ppf`     | -                | -                 | -             | -                   |
|                      | RVS    | `circularuniform.rvs`     | -                | -                 | -             | `rcircularuniform`  |
|                      | Fit    | `circularuniform.fit`     | -                | -                 | -             | -                   |
| Triangular           | PDF    | `triangular.pdf`          | `triangular.pdf` | -                 | `dtri`        | -                   |
|                      | CDF    | `triangular.cdf`          | `triangular.cdf` | -                 | -             | -                   |
|                      | PPF    | `triangular.ppf`          | `triangular.ppf` | -                 | -             | -                   |
|                      | RVS    | `triangular.rvs`          | `triangular.rvs` | -                 | `rtri`        | -                   |
|                      | Fit    | `triangular.fit`          | `triangular.fit` | -                 | -             | -                   |
| Cardioid             | PDF    | `cardioid.pdf`            | `cardioid.pdf`   | -                 | `dcard`       | `dcardioid`         |
|                      | CDF    | `cardioid.cdf`            | `cardioid.cdf`   | -                 | -             | -                   |
|                      | PPF    | `cardioid.ppf`            | `cardioid.ppf`   |                   | -             | -                   |
|                      | RVS    | `cardioid.rvs`            | `cardioid.rvs`   | -                 | `rcard`       | `rcardioid`         |
|                      | Fit    | `cardioid.fit`            | `cardioid.fit`   | -                 | -             |                     |
| Cartwright           | PDF    | `cartwright.pdf`          | -                | -                 | -             | `dcarthwrite`       |
|                      | CDF    | `cartwright.cdf`          | -                | -                 | -             | -                   |
|                      | PPF    | `cartwright.ppf`          | -                | -                 | -             | -                   |
|                      | RVS    | `cartwright.rvs`          | -                | -                 | -             | -                   |
|                      | Fit    | `cartwright.fit`          | -                | -                 | -             | -                   |
| Wrapped Normal       | PDF    | `wrapnorm.pdf`            | -                | -                 | `dwrpnorm`    | `dwrappednormal`    |
|                      | CDF    | `wrapnorm.cdf`            | -                | -                 | -             | `pwrappednormal`    |
|                      | PPF    | `wrapnorm.ppf`            | -                | -                 | -             | `qwrappednormal`    |
|                      | RVS    | `wrapnorm.rvs`            | -                | -                 | `rwrpnorm`    | `rwrappednormal`    |
|                      | Fit    | `wrapnorm.fit`            | -                | -                 | -             | `mle.wrappednormal` |
| Wrapped Cauchy       | PDF    | `wrapcauchy.pdf`          | -                | -                 | `dwrpcauchy`  | `dwrappedcauchy`    |
|                      | CDF    | `wrapcauchy.cdf`          | -                | -                 | -             | -                   |
|                      | PPF    | `wrapcauchy.ppf`          | -                | -                 | -             | -                   |
|                      | RVS    | `wrapcauchy.rvs`          | -                | -                 | `rwrpcauchy`  | `rwrappedcauchy`    |
|                      | Fit    | `wrapcauchy.fit`          | -                | -                 | -             | `mle.wrappedcauchy` |
| Von Mises            | PDF    | `vonmises.pdf`            | -                | `circ_vmpdf`      | `dvm`         | `dvonmises`         |
|                      | CDF    | `vonmises.cdf`            | -                | -                 | `pvm`         | `pvonmises`         |
|                      | PPF    | `vonmises.ppf`            | -                | -                 | -             | `qvonmises`         |
|                      | RVS    | `vonmises.rvs`            | -                | `circ_vmrnd`      | `rvm`         | `rvonmises`         |
|                      | Fit    | `vonmises.fit`            | -                | `circ_vmpar`      | `vm.ml`       | `mle.vonmises`      |
| Flattopped Von Mises | PDF    | `vonmises_flattopped.pdf` | -                | -                 | -             | -                   |
|                      | CDF    | `vonmises_flattopped.cdf` | -                | -                 | -             | -                   |
|                      | PPF    | `vonmises_flattopped.ppf` | -                | -                 | -             | -                   |
|                      | RVS    | `vonmises_flattopped.rvs` | -                | -                 | -             | -                   |
|                      | Fit    | `vonmises_flattopped.fit` | -                | -                 | -             | -                   |
| Jones-Pewsey         | PDF    | `jonespewsey.pdf`         | -                | -                 | -             | `djonespewsey`      |
|                      | CDF    | `jonespewsey.cdf`         | -                | -                 | -             | -                   |
|                      | PPF    | `jonespewsey.ppf`         | -                | -                 | -             | -                   |
|                      | RVS    | `jonespewsey.rvs`         | -                | -                 | -             | -                   |
|                      | Fit    | `jonespewsey.fit`         | -                | -                 | -             | -                   |

#### Asymmetric Circular Distributions
| Feature                  | Method | PyCircStat2                  | PyCircStat | CircStat (MATLAB) | CircStats (R) | circular (R)     |
|--------------------------|--------|------------------------------|------------|-------------------|---------------|------------------|
| Jones-Pewsey Sine-Skewed | PDF    | `jonespewsey_sineskewed.pdf` | -          | -                 | -             | -                |
|                          | CDF    | `jonespewsey_sineskewed.cdf` | -          | -                 | -             | -                |
|                          | PPF    | `jonespewsey_sineskewed.ppf` | -          | -                 | -             | -                |
|                          | RVS    | `jonespewsey_sineskewed.rvs` | -          | -                 | -             | -                |
|                          | Fit    | `jonespewsey_sineskewed.fit` | -          | -                 | -             | -                |
| Jones-Pewsey Asymmetric  | PDF    | `jonespewsey_asym.pdf`       | -          | -                 | -             | -                |
|                          | CDF    | `jonespewsey_asym.cdf`       | -          | -                 | -             | -                |
|                          | PPF    | `jonespewsey_asym.ppf`       | -          | -                 | -             | -                |
|                          | RVS    | `jonespewsey_asym.rvs`       | -          | -                 | -             | -                |
|                          | Fit    | `jonespewsey_asym.fit`       | -          | -                 | -             | -                |
| Inverse Batschelet       | PDF    | `inverse_batschelet.pdf`     | -          | -                 | -             | -                |
|                          | CDF    | `inverse_batschelet.cdf`     | -          | -                 | -             | -                |
|                          | PPF    | `inverse_batschelet.ppf`     | -          | -                 | -             | -                |
|                          | RVS    | `inverse_batschelet.rvs`     | -          | -                 | -             | -                |
|                          | Fit    | `inverse_batschelet.fit`     | -          | -                 | -             | -                |
| Kato-Jones               | PDF    | `katojones.pdf`              | -          | -                 | -             | `dkatojones`     |
|                          | CDF    | `katojones.cdf`              | -          | -                 | -             | -                |
|                          | PPF    | `katojones.ppf`              | -          | -                 | -             | -                |
|                          | RVS    | `katojones.rvs`              | -          | -                 | -             | `rkatojones`     |
|                          | Fit    | `katojones.fit`              | -          | -                 | -             | -                |
| Wrapped Stable           | PDF    | `wrapstable.pdf`             | -          | -                 | -             | -                |
|                          | CDF    | `wrapstable.cdf`             | -          | -                 | -             | -                |
|                          | PPF    | `wrapstable.ppf`             | -          | -                 | -             | -                |
|                          | RVS    | `wrapstable.rvs`             | -          | -                 | `rwrpstab`    | -                |
|                          | Fit    | `wrapstable.fit`             | -          | -                 | -             | -                |
| Asymmetric Triangular    | PDF    | -                            | -          | -                 | -             | `dasytriangular` |
| Projected Normal         | PDF    | -                            | -          | -                 | -             | `dpnorm`         |
|                          | RVS    | -                            | -          | -                 | -             | `rpnorm`         |

[^uniform]: $\rho=0$ stands for uniform distributed.
[^median]: $\theta$ stands for median.
[^F]: $F$ stands for distributions.
[^one-way]: Yet another one-way ANOVA.
[^two-way]: Two-way ANOVA.
[^median-grouped]: For grouped data (non-uniform `w`), `circ_median` uses the
  Mardia (1972) interpolation; the `method` argument is ignored on that path.
[^medianHL-broken]: As of `circular` 0.5-2 (CRAN, 2025-09-24), `medianHL.circular`
  builds the pair-mean array but its C primitive calls the deviation median on
  the original `x` instead of on the pair-means, so HL1/HL2/HL3 all return the
  regular `median.circular` value. See Otieno (2002) §3.4 for the intended
  algorithm.
[^mean-dev]: Different signatures: pycircstat2's `circ_mean_deviation(α, β)`
  evaluates Fisher (1993) eq. 2.32 — `d(β) = π − (1/n)Σ|π−|αᵢ−β||` — at every
  reference angle `β` (vector output). R's `meandeviation(x)` is the scalar
  `d(median(x))`. Same formula, different evaluation points.
[^cl-resp]: Circular response, linear predictor.
[^lc-resp]: Linear response, circular predictor (harmonic regression à la
  Pewsey et al. 2014, §8.4).
[^vtest-circular]: `circular`'s `rayleigh.test(x, mu = θ)` *is* the V-test: with
  `mu` supplied it computes the modified Rayleigh statistic
  `z₀ = √(2n)·mean(cos(x − θ))` and `p = 1 − Φ(z₀)`. Only the no-`mu` call is the
  ordinary Rayleigh test. (Verified in `circular`'s `rayleigh.test.R`.)
[^kuiper-1samp]: This row is the **one-sample** Kuiper test of uniformity (GoF vs the
  circular uniform). PyCircStat's `kuiper` and MATLAB CircStat's `circ_kuipertest`
  are **two-sample** tests (`H0: F₁ = F₂`) and belong in *Distribution Homogeneity*
  below — they are not one-sample GoF tests, despite sharing the Kuiper name.
[^kuiper-2samp]: The **two-sample** Kuiper test, the circular analogue of the
  two-sample Kolmogorov–Smirnov test (`H0: the two samples share a distribution`;
  sensitive to differences in location *or* dispersion). Distinct from the
  one-sample Kuiper GoF in the goodness-of-fit section.
[^vm-gof]: Goodness-of-fit against a **von Mises** null (not uniformity): estimate κ
  by ML, probability-integral-transform the data through the fitted von Mises CDF,
  then apply Watson's U². `circular`'s `watson.test(dist = "vonmises")` and CircStats'
  `watson(dist = "vm")` both do this; the other packages test uniformity only.
