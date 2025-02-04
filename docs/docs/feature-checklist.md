

### 1. Descriptive Statistics

| Feature                             | PyCircStat2           | PyCircStat                | CircStat (MATLAB)  | CircStats (R) | circular (R)                          |
| ----------------------------------- | --------------------- | ------------------------- | ------------------ | ------------- | ------------------------------------- |
| **Measures of Central Tendency**    |                       |                           |                    |               |                                       |
| Circular Mean                       | `circ_mean`           | `mean(alpha)`             | `circ_mean(alpha)` | `circ.mean`   | `mean.circular`                       |
| Circular Mean CI                    | `circ_mean_ci`        | `mean(alpha, ci=95)`      | `circ_confmean`    | -             | `mle.vonmises.bootstrap.ci`           |
| Circular Median                     | `circ_median`         | `median`                  | `circ_median`      | -             | `median.circular`/`medianHL.circular` |
| Circular Median CI                  | `circ_median_ci`      | -                         | -                  | -             | -                                     |
| Circular Quantile                   | `circ_quantile`       | -                         | -                  | -             | `quantile.circular`                   |
| **Measures of Spread & Dispersion** |                       |                           |                    |               |                                       |
| Resultant Vector Length             | `circ_r`              | `resultant_vector_length` | `circ_r`           | `est.rho`     | `rho.circular`                        |
| Angular Variance                    | `angular_var`         | `avar`                    | `circ_var`         | -             | `angular.variance`                    |
| Angular Standard Deviation          | `angular_std`         | `astd`                    | `circ_std`         | -             | `angular.deviation`                   |
| Circular Variance                   | `circ_var`            | `var`                     | `circ_var`         | `circ.disp`   | `var.circular`                        |
| Circular Standard Deviation         | `circ_std`            | `std`                     | `circ_std`         | -             | `sd.circular`                         |
| Circular Dispersion                 | `circ_dispersion`     | -                         | -                  | -             | -                                     |
| **Higher-Order Statistics**         |                       |                           |                    |               |                                       |
| Circular Moment                     | `circ_moment`         | `moment`                  | `circ_moment`      | `tri.moment`  | `trigonometric.moment`                |
| Circular Skewness                   | `circ_skewness`       | `skewness`                | `circ_skewness`    | -             | -                                     |
| Circular Kurtosis                   | `circ_kurtosis`       | `kurtoisis`               | `circ_kurtosis`    | -             | -                                     |
| **Distance & Pairwise Comparisons** |                       |                           |                    |               |                                       |
| Mean deviation                      | `circ_mean_deviation` | -                         | -                  | -             | `meandeviation`                       |
| Circular Distance                   | `circ_dist`           | `cdist`                   | `circ_dist`        | -             | -                                     |
| Pairwise Circular Distance          | `circ_pairdist`       | `pairwise_cdiff`          | `circ_dist2`       | -             | `dist.circular`                       |

### 2. Hypothesis Testing

#### One-Sample Tests for Significance

| Feature                     | H0                        | PyCircStat2         | PyCircStat | CircStat (MATLAB) | CircStats (R) | circular (R)    |
| --------------------------- | ------------------------- | ------------------- | ---------- | ----------------- | ------------- | --------------- |
| **Mean Direction**          |                           |                     |            |                   |               |                 |
| Rayleigh Test               | $\rho=0$                  | `rayleigh_test`     | `rayleigh` | `circ_rtest`      | `r.test`      | `rayleigh.test` |
| V-Test                      | $\rho=0$                  | `V_test`            | `vtest`    | `circ_vtest`      | `v0.test`     | -               |
| One-sample Test             | $\tilde\mu=μ_0$           | `one_sample_test`   | `mtest`    | `circ_mtest`      | -             | -               |
| Change Point Test           | no change point           | `change_point_test` | -          | -                 | `change.pt`   | `change.point`  |
| **Median Direction**        |                           |                     |            |                   |               |                 |
| Hodges-Ajne (omnibus) Test  | $\rho=0$                  | `omnibus_test`      | `omnibus`  | `circ_otest`      | -             | -               |
| Batschelet Test             | $\rho=0$                  | `batschelet_test`   | -          | -                 | -             | -               |
| Binomial Test               | $\tilde\theta = \theta_0$ | `binomial_test`     | `medtest`  | `circ_medtest`    | -             | -               |
| Symmetry Test around median | $\text{symmetry}$         | `symmetry_test`     | `symtest`  | `circ_symtest`    | -             | -               |

#### Multi-Sample Tests for Significance

| Feature                         | H0                                            | PyCircStat2            | PyCircStat        | CircStat (MATLAB) | CircStats (R)     | circular (R)           |
| ------------------------------- | --------------------------------------------- | ---------------------- | ----------------- | ----------------- | ----------------- | ---------------------- |
| **Mean Direction**              |                                               |                        |                   |                   |                   |                        |
| Circular Analysis of Variance   | $\mu_1 = \dots = \mu_n$                       | `circ_anova`           | -                 | -                 | -                 | `aov.circular`         |
| Watson-Williams Test [^one-way] | $\mu_1 = \dots = \mu_n$                       | `watson_williams_test` | `watson_williams` | `circ_wwtest`     | -                 | `watson.williams.test` |
| Harrison-Kanji Test[^two-way]   | $\mu_1 = \dots = \mu_n$                       | `harrison_kanji_test`  | `hktest`          | `circ_hktest`     | -                 | -                      |
| **Median Direction**            |                                               |                        |                   |                   |                   |                        |
| Common Median Test              | $\tilde{\theta}_1 = \dots = \tilde{\theta}_n$ | `common_median_test`   | `cmtest`          | `circ_cmtest`     | -                 | -                      |
| **Concentration**               |                                               |                        |                   |                   |                   |                        |
| Concentration Test (F-test)     | $\kappa_1 = \dots = \kappa_n$                 | `concentration_test`   | -                 | `circ_ktest`      | -                 | -                      |
| Equal Kappa Test                | $\kappa_1 = \dots = \kappa_n$                 | `equal_kappa_test`     | -                 | -                 | -                 | `equal.kappa.test`     |
| **Distribution Homogeneity**    |                                               |                        |                   |                   |                   |                        |
| Watson's U2 Test                | $F_1 = F_2$                                   | `watson_u2_test`       | -                 | -                 | `watson.two`      | `watson.two.test`      |
| Wallraff Test                   | $F_1 = F_2$                                   | `wallraff_test`        | -                 | -                 | -                 | `wallraff.test`        |
| Wheeler-Watson Test             | $F_1 = F_2$                                   | `wheeler_watson_test`  | -                 | -                 | -                 | `watson.wheeler.test`  |
| Rao's Tests for Homogeneity     | $F_1 = F_2$                                   | `rao_homogeneity_test` | -                 | -                 | `rao.homogeneity` | `rao.test`             |

#### Goodness-of-fit Tests

| Feature             | H0         | PyCircStat2        | PyCircStat   | CircStat (MATLAB) | CircStats (R) | circular (R)       |
| ------------------- | ---------- | ------------------ | ------------ | ----------------- | ------------- | ------------------ |
| Kuiper’s Test       | $\rho = 0$ | `circ_kuiper_test` | `kupier`     | `circ_kuipertest` | `kuiper`      | `kuiper.test`      |
| Rao’s Spacing Test  | $\rho = 0$ | `rao_spacing_test` | `raospacing` | `circ_raotest`    | `rao.spacing` | `rao.spacing.test` |
| Watson's Test       | $\rho = 0$ | `watson_test`      | -            | -                 | `watson`      | `watson.test`      |
| Circular Range Test | $\rho = 0$ | `circ_range_test`  | -            | -                 | `circ_range`  | `range.circular`   |


### 3. Correlation & Regression
| Feature                       | PyCircStat2    | PyCircStat | CircStat (MATLAB) | CircStats (R) | circular (R)              |
| ----------------------------- | -------------- | ---------- | ----------------- | ------------- | ------------------------- |
| Circular-Circular Correlation | `aacorr`       | `corrcc`   | `circ_corrcc`     | `circ.cor`    | `cor.circular`            |
| Circular-Linear Correlation   | `alcorr`       | `corrcl`   | `circ_corrcl`     | -             | -                         |
| Circular-Circular Regression  | `CCRegression` | -          | -                 | `circ.reg`    | `lm.circular(type="c-c")` |
| Circular-Linear Regression    | `CLRegression` | -          | -                 | -             | `lm.circular(type="c-l")` |



### 4. Circular Distributions

#### Symmetric Circular Distributions

| Feature              | Method | PyCircStat2               | PyCircStat       | CircStat (MATLAB) | CircStats (R) | circular (R)        |
| -------------------- | ------ | ------------------------- | ---------------- | ----------------- | ------------- | ------------------- |
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
| Kato-Jones           | PDF    | -                         | -                | -                 | -             | `dkatojones`        |
|                      | CDF    | -                         | -                | -                 | -             | -                   |
|                      | PPF    | -                         | -                | -                 | -             | -                   |
|                      | RVS    | -                         | -                | -                 | -             | `rkatojones`        |
|                      | Fit    | -                         | -                | -                 | -             | -                   |

#### Asymmetric Circular Distributions
| Feature                  | Method | PyCircStat2                  | PyCircStat | CircStat (MATLAB) | CircStats (R) | circular (R)     |
| ------------------------ | ------ | ---------------------------- | ---------- | ----------------- | ------------- | ---------------- |
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
| Wrapped Stable           | PDF    | `wrapstable.pdf`             | -          | -                 | -             | -                |
|                          | CDF    | `wrapstable.cdf`             | -          | -                 | -             | -                |
|                          | PPF    | `wrapstable.ppf`             | -          | -                 | -             | -                |
|                          | RVS    | `wrapstable.rvs`             | -          | -                 | `rwrpstab`    | -                |
|                          | Fit    | `wrapstable.fit`             | -          | -                 | -             | -                |
| Asymmetric Trangular     | PDF    | -                            | -          | -                 | -             | `dasytriangular` |
| Projected Normal         | PDF    | -                            | -          | -                 | -             | `dpnorm`         |
|                          | RVS    | -                            | -          | -                 | -             | `rpnorm`         |

[^uniform]: $\rho=0$ stands for uniform distributed.
[^median]: $\theta$ stands for median.
[^one-way]: Yet anothr one-way ANOVA.
[^two-way]: Two-way ANOVA.
[^F]: $F$ stands for distributions.

