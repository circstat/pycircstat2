

### 1. Descriptive Statistics

| Feature                     | CircStat (MATLAB)  | PyCircStat (Python)       | CircStats (R) | circular (R)      | PyCircStat2 (Python) |
| --------------------------- | ------------------ | ------------------------- | ------------- | ----------------- | -------------------- |
| Circular Mean               | `circ_mean(alpha)` | `mean(alpha)`             | `circ.mean`   | `mean.circular`   | `circ_mean`          |
| Circular Mean CI            | `circ_confmean`    | `mean(alpha, ci=95)`      |               |                   | `circ_mean_ci`       |
| Circular Moment             | `circ_moment`      | `moment`                  |               |                   | `circ_moment`        |
| Resultant Vector Length     | `circ_r`           | `resultant_vector_length` | -             | `rho.circular`    | `circ_r`             |
| Circular Variance           | `circ_var`         | `var`                     | `circ.var`    | `var.circular`    | `circ_var`           |
| Circular Standard Deviation | `circ_std`         | `std`                     | `circ.sd`     | `sd.circular`     | `circ_std`           |
| Angular Variance            | `circ_var`         | `avar`                    |               |                   | `angular_var`        |
| Angular Standard Deviation  | `circ_std`         | `astd`                    |               |                   | `angular_std`        |
| Circular Median             | `circ_median`      | `median`                  | `circ.median` | `median.circular` | `circ_median`        |
| Circular Median CI          | -                  | -                         |               |                   | `circ_median_ci`     |
| Circular Skewness           | `circ_skewness`    | `skewness`                |               |                   | `circ_skewness`      |
| Circular Kurtosis           | `circ_kurtosis`    | `kurtoisis`               |               |                   | `circ_kurtosis`      |
| Circular Dispersion         | -                  | -                         | -             | -                 | `circ_dispersion`    |
| Circular Distance           | `circ_dist`        | `cdist`                   |               |                   | `circ_dist`          |
| Pairwaise Circular Distance | `circ_dist2`       | `pairwise_cdiff`          |               |                   | `circ_pairdist`      |


### 2. Hypothesis Testing

#### Testing Significance of the Mean Direction

| Feature         | CircStat (MATLAB) | PyCircStat (Python) | CircStats (R) | circular (R)    | PyCircStat2 (Python) |
| --------------- | ----------------- | ------------------- | ------------- | --------------- | -------------------- |
| Rayleigh Test   | `circ_rtest`      | `rayleigh`          | `r.test`      | `rayleigh.test` | `rayleigh_test`      |
| V-Test          | `circ_vtest`      | `vtest`             | `v.test`      | `v.test`        | `V_test`             |
| One-sample test | `circ_mtest`      | `mtest`             |               |                 | `one_sample_test`    |

#### Testing Significance of the Median Direction

| Feature                     | CircStat (MATLAB) | PyCircStat (Python) | CircStats (R) | circular (R) | PyCircStat2 (Python) |
| --------------------------- | ----------------- | ------------------- | ------------- | ------------ | -------------------- |
| Hodges-Ajne (omnibus) Test  | `circ_otest`      | `omnibus`           |               |              | `omnibus_test`       |
| Batschelet Test             | -                 | -                   |               |              | `batschelet_test`    |
| Binomial Test               | `circ_medtest`    | `medtest`           |               |              | `binomial_test`      |
| Symmetry Test around median | `circ_symtest`    | `symtest`           |               |              | `symmetry_test`      |

#### Multi-sample testing of Mean Directions and Concentration

| Feature                              | CircStat (MATLAB) | PyCircStat (Python) | CircStats (R)     | circular (R)           | PyCircStat2 (Python)   |
| ------------------------------------ | ----------------- | ------------------- | ----------------- | ---------------------- | ---------------------- |
| Watson-Williams Test (one-way ANOVA) | `circ_wwtest`     | `watson_williams`   | -                 | `watson.williams.test` | `watson_williams_test` |
| Harrison-Kanji Test (two-way ANOVA)  | `circ_hktest`     | `hktest`            | -                 |                        | -                      |
| Watson's U2 Test                     | -                 | -                   | `watson.two`      |                        | `watson_u2_test`       |
| Concentration Test (F-test)          | `circ_ktest`      | -                   | -                 |                        | `concentration_test`   |
| Rao's Tests for Homogeneity          | -                 | -                   | `rao.homogeneity` |                        | `rao_homogeneity_test` |

#### Goodness-of-fit Tests

| Feature             | CircStat (MATLAB) | PyCircStat (Python) | CircStats (R) | circular (R)       | PyCircStat2 (Python) |
| ------------------- | ----------------- | ------------------- | ------------- | ------------------ | -------------------- |
| Kuiper’s Test       | `circ_kuipertest` | `kupier`            | `kuiper`      | `kuiper.test`      | `circ_kuiper_test`   |
| Rao’s Spacing Test  | `circ_raotest`    | `raospacing`        | `rao.spacing` | `rao.spacing.test` | `rao_spacing_test`   |
| Watson's Test       | -                 | -                   | `watson`      |                    | `watson_test`        |
| Circular Range Test | -                 | -                   | `circ_range`  |                    | `circ_range_test`    |


### 3. Correlation & Regression
| Feature                       | CircStat (MATLAB) | PyCircStat (Python) | CircStats (R) | circular (R)   | PyCircStat2 (Python) |
| ----------------------------- | ----------------- | ------------------- | ------------- | -------------- | -------------------- |
| Circular-Circular Correlation | `circ_corrcc`     | `corrcc`            | `circ.cor`    | `cor.circular` | `aacorr`             |
| Circular-Linear Correlation   | `circ_corrcl`     | `corrcl`            | -             | `cor.circular` | `alcorr`             |
| Circular-Circular Regression  | -                 | -                   | `circ.reg`    | `lm.circular`  | `CCRegression`       |
| Circular-Linear Regression    | -                 | -                    | -             | `lm.circular`  | `CLRegression`       |



### 4. Circular Distributions
| Feature                     | CircStat (MATLAB)                        | PyCircStat (Python) | CircStats (R)         | circular (R)     | PyCircStat2 (Python) |
| --------------------------- | ---------------------------------------- | ------------------- | --------------------- | ---------------- | -------------------- |
| Von Mises Distribution      | `circ_vmpdf`, `circ_vmrnd`, `circ_vmpar` | `vonmises`          | `dvm`, `rvm`, `vm.ml` | `dvonmises`      | `vonmises`           |
| Wrapped Normal Distribution | -                                        | -                   | -                     | `dwrappednormal` | `wrappednormal_pdf`  |
| Wrapped Cauchy Distribution | -                                        | -                   | -                     | `dwrappedcauchy` | `wrappedcauchy_pdf`  |
| Generalized von Mises (GvM) | -                                        | -                   | -                     | `dgvm`           | `gvm_pdf`            |
| Empirical Distribution      | -                                        | -                   | -                     | `ecdf.circular`  | `circ_empirical`     |
