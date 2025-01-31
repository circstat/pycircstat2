

### 1. Descriptive Statistics

| Feature                     | PyCircStat2 (Python) | PyCircStat (Python)       | CircStat (MATLAB)  | CircStats (R) | circular (R)        |
| --------------------------- | -------------------- | ------------------------- | ------------------ | ------------- | ------------------- |
| Circular Mean               | `circ_mean`          | `mean(alpha)`             | `circ_mean(alpha)` | `circ.mean`   | `mean.circular`     |
| Circular Mean CI            | `circ_mean_ci`       | `mean(alpha, ci=95)`      | `circ_confmean`    | -             |                     |
| Circular Moment             | `circ_moment`        | `moment`                  | `circ_moment`      | `tri.moment`  |                     |
| Cencentration Parameter     | `circ_kappa`         | `kappa`                   | `circ_kappa`       | `est.kappa`   |                     |
| Resultant Vector Length     | `circ_r`             | `resultant_vector_length` | `circ_r`           | `est.rho`     | `rho.circular`      |
| Circular Variance           | `circ_var`           | `var`                     | `circ_var`         | -             | `var.circular`      |
| Circular Standard Deviation | `circ_std`           | `std`                     | `circ_std`         | -             | `sd.circular`       |
| Angular Variance            | `angular_var`        | `avar`                    | `circ_var`         | -             | `angular.variance`  |
| Angular Standard Deviation  | `angular_std`        | `astd`                    | `circ_std`         | -             | `angular.deviation` |
| Circular Median             | `circ_median`        | `median`                  | `circ_median`      | -             | `median.circular`   |
| Circular Median CI          | `circ_median_ci`     | -                         | -                  | -             |                     |
| Circular Skewness           | `circ_skewness`      | `skewness`                | `circ_skewness`    | -             |                     |
| Circular Kurtosis           | `circ_kurtosis`      | `kurtoisis`               | `circ_kurtosis`    | -             |                     |
| Circular Dispersion         | `circ_dispersion`    | -                         | -                  | `circ.disp`   | -                   |
| Circular Distance           | `circ_dist`          | `cdist`                   | `circ_dist`        | -             |                     |
| Pairwaise Circular Distance | `circ_pairdist`      | `pairwise_cdiff`          | `circ_dist2`       | -             |                     |


### 2. Hypothesis Testing

#### Testing Significance of the Mean Direction / Cencentration

| Feature           | PyCircStat2 (Python) | PyCircStat (Python) | CircStat (MATLAB) | CircStats (R) | circular (R)    |
| ----------------- | -------------------- | ------------------- | ----------------- | ------------- | --------------- |
| Rayleigh Test     | `rayleigh_test`      | `rayleigh`          | `circ_rtest`      | `r.test`      | `rayleigh.test` |
| V-Test            | `V_test`             | `vtest`             | `circ_vtest`      | `v0.test`     | -               |
| One-sample Test   | `one_sample_test`    | `mtest`             | `circ_mtest`      | -             | -               |
| Change Point Test | `change_point_test`  | -                   | -                 | `change.pt`   | `change.point`  |

#### Testing Significance of the Median Direction

| Feature                     | PyCircStat2 (Python) | PyCircStat (Python) | CircStat (MATLAB) | CircStats (R) | circular (R) |
| --------------------------- | -------------------- | ------------------- | ----------------- | ------------- | ------------ |
| Hodges-Ajne (omnibus) Test  | `omnibus_test`       | `omnibus`           | `circ_otest`      | -             | -            |
| Batschelet Test             | `batschelet_test`    | -                   | -                 | -             |              |
| Binomial Test               | `binomial_test`      | `medtest`           | `circ_medtest`    | -             |              |
| Symmetry Test around median | `symmetry_test`      | `symtest`           | `circ_symtest`    | -             |              |

#### Multi-sample testing of Mean Directions / Concentration

| Feature                       | PyCircStat2 (Python)   | PyCircStat (Python) | CircStat (MATLAB) | CircStats (R)     | circular (R)           |
| ----------------------------- | ---------------------- | ------------------- | ----------------- | ----------------- | ---------------------- |
| Circular Analysis of Variance | `circ_anova_test`      | -                   | -                 | -                 | `aov.circular`         |
| Watson-Williams Test [^1]     | `watson_williams_test` | `watson_williams`   | `circ_wwtest`     | -                 | `watson.williams.test` |
| Harrison-Kanji Test[^2]       | `harrison_kanji_test`  | `hktest`            | `circ_hktest`     | -                 | -                      |
| Watson's U2 Test              | `watson_u2_test`       | -                   | -                 | `watson.two`      | `watson.two.test`      |
| Wallraff Test                 | `wallraff_test`        | -                   | -                 | -                 | `wallraff.test`        |
| Concentration Test (F-test)   | `concentration_test`   | -                   | `circ_ktest`      | -                 | -                      |
| Rao's Tests for Homogeneity   | `rao_homogeneity_test` | -                   | -                 | `rao.homogeneity` | `rao.test`             |

#### Goodness-of-fit Tests

| Feature             | PyCircStat2 (Python) | PyCircStat (Python) | CircStat (MATLAB) | CircStats (R) | circular (R)       |
| ------------------- | -------------------- | ------------------- | ----------------- | ------------- | ------------------ |
| Kuiper’s Test       | `circ_kuiper_test`   | `kupier`            | `circ_kuipertest` | `kuiper`      | `kuiper.test`      |
| Rao’s Spacing Test  | `rao_spacing_test`   | `raospacing`        | `circ_raotest`    | `rao.spacing` | `rao.spacing.test` |
| Watson's Test       | `watson_test`        | -                   | -                 | `watson`      |                    |
| Circular Range Test | `circ_range_test`    | -                   | -                 | `circ_range`  |                    |


### 3. Correlation & Regression
| Feature                       | PyCircStat2 (Python) | PyCircStat (Python) | CircStat (MATLAB) | CircStats (R) | circular (R)              |
| ----------------------------- | -------------------- | ------------------- | ----------------- | ------------- | ------------------------- |
| Circular-Circular Correlation | `aacorr`             | `corrcc`            | `circ_corrcc`     | `circ.cor`    | `cor.circular`            |
| Circular-Linear Correlation   | `alcorr`             | `corrcl`            | `circ_corrcl`     | -             | -                         |
| Circular-Circular Regression  | `CCRegression`       | -                   | -                 | `circ.reg`    | `lm.circular(type="c-c")` |
| Circular-Linear Regression    | `CLRegression`       | -                   | -                 | -             | `lm.circular(type="c-l")` |



### 4. Circular Distributions

#### Symmetric Circular Distributions

| Feature              | Method | PyCircStat2 (Python)      | PyCircStat (Python) | CircStat (MATLAB) | CircStats (R) | circular (R)     |
| -------------------- | ------ | ------------------------- | ------------------- | ----------------- | ------------- | ---------------- |
| Circular Uniform     | PDF    | `circularuniform.pdf`     | -                   | -                 | -             | -                |
|                      | CDF    | `circularuniform.cdf`     | -                   | -                 | -             | -                |
|                      | PPF    | `circularuniform.ppf`     | -                   | -                 | -             | -                |
|                      | RVS    | `circularuniform.rvs`     | -                   | -                 | -             | -                |
|                      | Fit    | `circularuniform.fit`     | -                   | -                 | -             | -                |
| Triangular           | PDF    | `triangular.pdf`          | `triangular.pdf`    | -                 | `dtri`        | -                |
|                      | CDF    | `triangular.cdf`          | `triangular.cdf`    | -                 | -             | -                |
|                      | PPF    | `triangular.ppf`          | `triangular.ppf`    | -                 | -             | -                |
|                      | RVS    | `triangular.rvs`          | `triangular.rvs`    | -                 | `rtri`        | -                |
|                      | Fit    | `triangular.fit`          | `triangular.fit`    | -                 | -             | -                |
| Cardioid             | PDF    | `cardioid.pdf`            | `cardioid.pdf`      | -                 | `dcard`       | `dcardioid`      |
|                      | CDF    | `cardioid.cdf`            | `cardioid.cdf`      | -                 | -             | -                |
|                      | PPF    | `cardioid.ppf`            | `cardioid.ppf`      |                   | -             | -                |
|                      | RVS    | `cardioid.rvs`            | `cardioid.rvs`      | -                 | `rcard`       | `rcardioid`      |
|                      | Fit    | `cardioid.fit`            | `cardioid.fit`      | -                 | -             |                  |
| Cartwright           | PDF    | `cartwright.pdf`          | -                   | -                 | -             | `dcarthwrite`    |
|                      | CDF    | `cartwright.cdf`          | -                   | -                 | -             | -                |
|                      | PPF    | `cartwright.ppf`          | -                   | -                 | -             | -                |
|                      | RVS    | `cartwright.rvs`          | -                   | -                 | -             | -                |
|                      | Fit    | `cartwright.fit`          | -                   | -                 | -             | -                |
| Wrapped Normal       | PDF    | `wrapnorm.pdf`            | -                   | -                 | `dwrpnorm`    | `dwrappednormal` |
|                      | CDF    | `wrapnorm.cdf`            | -                   | -                 | -             | `pwrappednormal` |
|                      | PPF    | `wrapnorm.ppf`            | -                   | -                 | -             | `qwrappednormal` |
|                      | RVS    | `wrapnorm.rvs`            | -                   | -                 | `rwrpnorm`    | `rwrappednormal` |
|                      | Fit    | `wrapnorm.fit`            | -                   | -                 | -             | -                |
| Wrapped Cauchy       | PDF    | `wrapcauchy.pdf`          | -                   | -                 | `dwrpcauchy`  | `dwrappedcauchy` |
|                      | CDF    | `wrapcauchy.cdf`          | -                   | -                 | -             | -                |
|                      | PPF    | `wrapcauchy.ppf`          | -                   | -                 | -             | -                |
|                      | RVS    | `wrapcauchy.rvs`          | -                   | -                 | `rwrpcauchy`  | `rwrappedcauchy` |
|                      | Fit    | `wrapcauchy.fit`          | -                   | -                 | -             | -                |
| Von Mises            | PDF    | `vonmises.pdf`            | -                   | `circ_vmpdf`      | `dvm`         | `dvonmises`      |
|                      | CDF    | `vonmises.cdf`            | -                   | -                 | `pvm`         | `pvonmises`      |
|                      | PPF    | `vonmises.ppf`            | -                   | -                 | -             | `qvonmises`      |
|                      | RVS    | `vonmises.rvs`            | -                   | `circ_vmrnd`      | `rvm`         | `rvonmises`      |
|                      | Fit    | `vonmises.fit`            | -                   | `circ_vmpar`      | `vm.ml`       | -                |
| Flattopped Von Mises | PDF    | `vonmises_flattopped.pdf` | -                   | -                 | -             | -                |
|                      | CDF    | `vonmises_flattopped.cdf` | -                   | -                 | -             | -                |
|                      | PPF    | `vonmises_flattopped.ppf` | -                   | -                 | -             | -                |
|                      | RVS    | `vonmises_flattopped.rvs` | -                   | -                 | -             | -                |
|                      | Fit    | `vonmises_flattopped.fit` | -                   | -                 | -             | -                |
| Jones-Pewsey         | PDF    | `jonespewsey.pdf`         | -                   | -                 | -             | `djonespewsey`   |
|                      | CDF    | `jonespewsey.cdf`         | -                   | -                 | -             | -                |
|                      | PPF    | `jonespewsey.ppf`         | -                   | -                 | -             | -                |
|                      | RVS    | `jonespewsey.rvs`         | -                   | -                 | -             | -                |
|                      | Fit    | `jonespewsey.fit`         | -                   | -                 | -             | -                |
| Kato-Jones           | PDF    | -                         | -                   | -                 | -             | `dkatojones`     |
|                      | CDF    | -                         | -                   | -                 | -             | -                |
|                      | PPF    | -                         | -                   | -                 | -             | -                |
|                      | RVS    | -                         | -                   | -                 | -             | `rkatojones`     |
|                      | Fit    | -                         | -                   | -                 | -             | -                |

#### Asymmetric Circular Distributions
| Feature                  | Method | PyCircStat2 (Python)         | PyCircStat (Python) | CircStat (MATLAB) | CircStats (R) | circular (R) |
| ------------------------ | ------ | ---------------------------- | ------------------- | ----------------- | ------------- | ------------ |
| Jones-Pewsey Sine-Skewed | PDF    | `jonespewsey_sineskewed.pdf` | -                   | -                 | -             | -            |
|                          | CDF    | `jonespewsey_sineskewed.cdf` | -                   | -                 | -             | -            |
|                          | PPF    | `jonespewsey_sineskewed.ppf` | -                   | -                 | -             | -            |
|                          | RVS    | `jonespewsey_sineskewed.rvs` | -                   | -                 | -             | -            |
|                          | Fit    | `jonespewsey_sineskewed.fit` | -                   | -                 | -             | -            |
| Jones-Pewsey Asymmetric  | PDF    | `jonespewsey_asym.pdf`       | -                   | -                 | -             | -            |
|                          | CDF    | `jonespewsey_asym.cdf`       | -                   | -                 | -             | -            |
|                          | PPF    | `jonespewsey_asym.ppf`       | -                   | -                 | -             | -            |
|                          | RVS    | `jonespewsey_asym.rvs`       | -                   | -                 | -             | -            |
|                          | Fit    | `jonespewsey_asym.fit`       | -                   | -                 | -             | -            |
| Inverse Batschelet       | PDF    | `inverse_batschelet.pdf`     | -                   | -                 | -             | -            |
|                          | CDF    | `inverse_batschelet.cdf`     | -                   | -                 | -             | -            |
|                          | PPF    | `inverse_batschelet.ppf`     | -                   | -                 | -             | -            |
|                          | RVS    | `inverse_batschelet.rvs`     | -                   | -                 | -             | -            |
|                          | Fit    | `inverse_batschelet.fit`     | -                   | -                 | -             | -            |
| Wrapped Stable           | PDF    | `wrapstable.pdf`             | -                   | -                 | -             | -            |
|                          | CDF    | `wrapstable.cdf`             | -                   | -                 | -             | -            |
|                          | PPF    | `wrapstable.ppf`             | -                   | -                 | -             | -            |
|                          | RVS    | `wrapstable.rvs`             | -                   | -                 | `rwrpstab`    | -            |
|                          | Fit    | `wrapstable.fit`             | -                   | -                 | -             | -            |


[^1]: Yet anothr one-way ANOVA.
[^2]: Two-way ANOVA.