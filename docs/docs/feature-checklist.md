

### 1. Descriptive Statistics

| Feature                     | PyCircStat2 (Python) | PyCircStat (Python)       | CircStat (MATLAB)  | CircStats (R) | circular (R)      |
| --------------------------- | -------------------- | ------------------------- | ------------------ | ------------- | ----------------- |
| Circular Mean               | `circ_mean`          | `mean(alpha)`             | `circ_mean(alpha)` | `circ.mean`   | `mean.circular`   |
| Circular Mean CI            | `circ_mean_ci`       | `mean(alpha, ci=95)`      | `circ_confmean`    | -             |                   |
| Circular Moment             | `circ_moment`        | `moment`                  | `circ_moment`      | `tri.moment`  |                   |
| Cencentration Parameter     | `circ_kappa`         | `kappa`                   | `circ_kappa`       | `est.kappa`   |                   |
| Resultant Vector Length     | `circ_r`             | `resultant_vector_length` | `circ_r`           | `est.rho`     | `rho.circular`    |
| Circular Variance           | `circ_var`           | `var`                     | `circ_var`         | -             | `var.circular`    |
| Circular Standard Deviation | `circ_std`           | `std`                     | `circ_std`         | -             | `sd.circular`     |
| Angular Variance            | `angular_var`        | `avar`                    | `circ_var`         | -             |                   |
| Angular Standard Deviation  | `angular_std`        | `astd`                    | `circ_std`         | -             |                   |
| Circular Median             | `circ_median`        | `median`                  | `circ_median`      | -             | `median.circular` |
| Circular Median CI          | `circ_median_ci`     | -                         | -                  | -             |                   |
| Circular Skewness           | `circ_skewness`      | `skewness`                | `circ_skewness`    | -             |                   |
| Circular Kurtosis           | `circ_kurtosis`      | `kurtoisis`               | `circ_kurtosis`    | -             |                   |
| Circular Dispersion         | `circ_dispersion`    | -                         | -                  | `circ.disp`   | -                 |
| Circular Distance           | `circ_dist`          | `cdist`                   | `circ_dist`        | -             |                   |
| Pairwaise Circular Distance | `circ_pairdist`      | `pairwise_cdiff`          | `circ_dist2`       | -             |                   |


### 2. Hypothesis Testing

#### Testing Significance of the Mean Direction / Cencentration

| Feature           | PyCircStat2 (Python) | PyCircStat (Python) | CircStat (MATLAB) | CircStats (R) | circular (R)    |
| ----------------- | -------------------- | ------------------- | ----------------- | ------------- | --------------- |
| Rayleigh Test     | `rayleigh_test`      | `rayleigh`          | `circ_rtest`      | `r.test`      | `rayleigh.test` |
| V-Test            | `V_test`             | `vtest`             | `circ_vtest`      | `v0.test`     | `v.test`        |
| One-sample Test   | `one_sample_test`    | `mtest`             | `circ_mtest`      | -             |                 |
| Change Point Test | `change_point_test`  | -                   | -                 | `change.pt`   |                 |

#### Testing Significance of the Median Direction

| Feature                     | PyCircStat2 (Python) | PyCircStat (Python) | CircStat (MATLAB) | CircStats (R) | circular (R) |
| --------------------------- | -------------------- | ------------------- | ----------------- | ------------- | ------------ |
| Hodges-Ajne (omnibus) Test  | `omnibus_test`       | `omnibus`           | `circ_otest`      | -             |              |
| Batschelet Test             | `batschelet_test`    | -                   | -                 | -             |              |
| Binomial Test               | `binomial_test`      | `medtest`           | `circ_medtest`    | -             |              |
| Symmetry Test around median | `symmetry_test`      | `symtest`           | `circ_symtest`    | -             |              |

#### Multi-sample testing of Mean Directions / Concentration

| Feature                     | PyCircStat2 (Python)   | PyCircStat (Python) | CircStat (MATLAB) | CircStats (R)     | circular (R)           |
| --------------------------- | ---------------------- | ------------------- | ----------------- | ----------------- | ---------------------- |
| Watson-Williams Test [^1]   | `watson_williams_test` | `watson_williams`   | `circ_wwtest`     | -                 | `watson.williams.test` |
| Harrison-Kanji Test[^2]     | `harrison_kanji_test`                      | `hktest`            | `circ_hktest`     | -                 |                        |
| Watson's U2 Test            | `watson_u2_test`       | -                   | -                 | `watson.two`      |                        |
| Concentration Test (F-test) | `concentration_test`   | -                   | `circ_ktest`      | -                 |                        |
| Rao's Tests for Homogeneity | `rao_homogeneity_test` | -                   | -                 | `rao.homogeneity` |                        |

#### Goodness-of-fit Tests

| Feature             | PyCircStat2 (Python) | PyCircStat (Python) | CircStat (MATLAB) | CircStats (R) | circular (R)       |
| ------------------- | -------------------- | ------------------- | ----------------- | ------------- | ------------------ |
| Kuiper’s Test       | `circ_kuiper_test`   | `kupier`            | `circ_kuipertest` | `kuiper`      | `kuiper.test`      |
| Rao’s Spacing Test  | `rao_spacing_test`   | `raospacing`        | `circ_raotest`    | `rao.spacing` | `rao.spacing.test` |
| Watson's Test       | `watson_test`        | -                   | -                 | `watson`      |                    |
| Circular Range Test | `circ_range_test`    | -                   | -                 | `circ_range`  |                    |


### 3. Correlation & Regression
| Feature                       | PyCircStat2 (Python) | PyCircStat (Python) | CircStat (MATLAB) | CircStats (R) | circular (R)   |
| ----------------------------- | -------------------- | ------------------- | ----------------- | ------------- | -------------- |
| Circular-Circular Correlation | `aacorr`             | `corrcc`            | `circ_corrcc`     | `circ.cor`    | `cor.circular` |
| Circular-Linear Correlation   | `alcorr`             | `corrcl`            | `circ_corrcl`     | -             | `cor.circular` |
| Circular-Circular Regression  | `CCRegression`       | -                   | -                 | `circ.reg`    | `lm.circular`  |
| Circular-Linear Regression    | `CLRegression`       | -                   | -                 | -             | `lm.circular`  |



### 4. Circular Distributions
| Feature                    | Method | PyCircStat2 (Python) | PyCircStat (Python) | CircStat (MATLAB) | CircStats (R) | circular (R) |
| -------------------------- | ------ | -------------------- | ------------------- | ----------------- | ------------- | ------------ |
| **Von Mises Distribution** | PDF    | `vonmises.pdf`       | `vonmises`          | `circ_vmpdf`      | `dvm`         | `dvonmises`  |
|                            | CDF    | `vonmises.cdf`       | -                   | -                 | `pvm`         | -            |
|                            | PPF    | `vonmises.ppf`       | -                   | -                 | -             | -            |
|                            | RVS    | `vonmises.rvs`       | -                   | `circ_vmrnd`      | `rvm`         | -            |
|                            | Fit    | `vonmises.fit`       | -                   | `circ_vmpar`      | `vm.ml`       | -            |


[^1]: One-way ANOVA.
[^2]: Two-way ANOVA.