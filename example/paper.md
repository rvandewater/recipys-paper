---
title: >-
    ReciPys: An Intuitive Machine Learning Transformation Pipeline for Reproducible Data Preprocessing
authors:
  - name: Robin P. van de Water
    email: robin.vandewater@hpi.de
    affiliation: [1, 2]
    orcid: 0000-0002-2895-4872
    corresponding: true
  - name: Hendrik Schmidt
    orcid: 0000-0001-7699-3983
    affiliation: 
    equal-contrib: false
  - name: Patrick Rockenschaub
    orcid: 0000-0002-6499-7933
    affiliation: [1, 3]
    equal-contrib: false
affiliations:
  - index: 1
    name: Hasso Plattner Institute, University of Potsdam, Potsdam, Germany
  - index: 2
    name: Hasso Plattner Institute for Digital Health at Mount Sinai, Icahn School of Medicine at Mount Sinai, New York City, NY, USA
  - index: 3
    name: Innsbruck Medical University, Innsbruck, Austria
date: 2024-07-04
bibliography: paper.bib
repository: https://github.com/rvandewater/ReciPys
tags:
  - reference
  - example
  - markdown
  - publishing
---
<!-- 
Guide:
https://github.com/openjournals/inara/blob/main/example/paper.md  -->
# Summary

Machine learning pipelines often require complicated preprocessing pipelines. Our aim is to simplify this with a python package that can be used to define human-readable and reproducible pipelines for machine learning tasks.

# Statement of Need

Python is the most popular programming language for machine learning. However, preprocessing data for machine learning tasks can be a time-consuming and error-prone process. ReciPys is a python package that aims to simplify this process by providing a fast and intuitive interface for preprocessing data. It can use both a polars [@PolarsPolars2024] backend, which allows for fast python-native data processing, and a traditional Pandas [@mckinney-proc-scipy-2010] backend for use with legacy data tools. 
ReciPys is designed to be easy to use and flexible, allowing users to easily preprocess data for a wide range of machine learning pipelines. 
Moreover, we hide the complexity of the preprocessing pipeline from the user which allows for more reproducible and maintainable code.

# Related Work

The package was inspired by the R package by @kuhnRecipesPreprocessingFeature2024. As such, it is designed to be a python equivalent of the R package, with functionality that is aimed at the ML community. 
The package is designed to be used as part of a pipeline that include machine learning libraries such as Scikit-Learn [@pedregosa_scikit-learn_2011] and PyTorch[@paszkePyTorchImperativeStyle2019].
It is based on the principles of Configuration as Code, which allows for easy reproducibility of scientific ML experiments. 

# Application: ICU Data

The package was specifically aimed towards temporal EHR data and was developed as part of Yet Another ICU Benchmark [@waterAnotherICUBenchmark2023]. We demonstrate the utility of ReciPys by preprocessing a dataset of ICU patients. The dataset contains information about patients in the ICU, including vital signs, lab results, and medications. We show how ReciPys can be used to preprocess this data and prepare it for machine learning tasks.

If we have a dataset, `df` with a label `y`, some features `x1`, `x2`, `x3`, `x4`, an identifier `id`, and a sequential component `time`, we can build a preprocessing pipeline using ReciPys. We first do a train/test split:
``` python
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
roles = {outcomes:["y"], predictors=["x1", "x2", "x3", "x4"], groups=["id"], 
sequences=["time"]}
ing = Ingredients(df_train, roles=roles)
rec = Recipe(ing)
```
We can now add preprocessing steps:
``` python
rec.add_step(StepScale())
rec.add_step(StepSklearn(MissingIndicator(features="all"), sel=has_role("predictor")))
rec.add_step(StepImputeFill(strategy="forward"))
rec.add_step(StepSklearn(LabelEncoder(), sel=has_type("categorical"), columnwise=True))
```

We can now fit the recipe and transform both the train and test set without leakage:
``` python
df_train = rec.prep()
df_test = rec.bake(df_test)
```

# Benchmarks
Below we show benchmarks for ReciPys using both a Pandas and Polars backend. We compare the time taken to preprocess data using ReciPys with the time taken to preprocess data using Pandas and Polars directly. The benchmarks show a mixed picture with the Pandas backend still outperforming Polars on some steps. Presumably this is due to some functionality not being natively available in Polars at the time of writing. The experiments have been performed with 5 seeds and the mean and standard deviation are reported.

![Duration for each step and data size.](images/duration_combined_plots.pdf)
![Memory usage for each step and data size.](images/memory_combined_plots.pdf)

| data_size | step                |   Pandas ± (std)    |   Polars ± (std)    |
| --------: | :------------------ | :-----------------: | :-----------------: |
|       100 | Historical_Count    |    11.77 ± 1.05     |     4.64 ± 0.59     |
|      1000 | Historical_Count    |    29.08 ± 1.64     |     6.80 ± 0.18     |
|     10000 | Historical_Count    |    200.47 ± 2.15    |    23.38 ± 0.45     |
|    100000 | Historical_Count    |   1886.23 ± 38.02   |    185.97 ± 2.36    |
|   1000000 | Historical_Count    |  19135.68 ± 283.49  |   1865.41 ± 36.66   |
|  10000000 | Historical_Count    | 196319.48 ± 1228.98 |  24056.20 ± 289.09  |
|       100 | Historical_Max      |     6.83 ± 0.70     |     4.87 ± 0.46     |
|      1000 | Historical_Max      |     8.14 ± 1.19     |     7.91 ± 0.35     |
|     10000 | Historical_Max      |    20.27 ± 1.88     |    34.48 ± 0.72     |
|    100000 | Historical_Max      |   129.83 ± 28.64    |    289.07 ± 5.06    |
|   1000000 | Historical_Max      |  1396.58 ± 288.93   |   3100.53 ± 74.24   |
|  10000000 | Historical_Max      |  9613.01 ± 222.25   |  35865.05 ± 241.89  |
|       100 | Historical_Mean     |    11.91 ± 0.74     |     5.48 ± 0.82     |
|      1000 | Historical_Mean     |    29.28 ± 1.59     |     7.71 ± 0.47     |
|     10000 | Historical_Mean     |    199.91 ± 3.45    |    31.36 ± 1.47     |
|    100000 | Historical_Mean     |   1863.51 ± 39.57   |    261.96 ± 8.46    |
|   1000000 | Historical_Mean     |  18969.24 ± 346.06  |   2860.31 ± 74.72   |
|  10000000 | Historical_Mean     | 194379.68 ± 1077.68 |  33442.42 ± 394.12  |
|       100 | Historical_Min      |     6.90 ± 0.67     |     5.01 ± 0.41     |
|      1000 | Historical_Min      |     8.36 ± 1.40     |     7.69 ± 0.18     |
|     10000 | Historical_Min      |    20.91 ± 1.73     |    34.24 ± 1.34     |
|    100000 | Historical_Min      |   130.25 ± 29.15    |    290.29 ± 8.26    |
|   1000000 | Historical_Min      |  1386.63 ± 282.46   |   3160.03 ± 14.56   |
|  10000000 | Historical_Min      |  9831.88 ± 171.49   |  35986.32 ± 277.50  |
|       100 | KBinsDiscretizer    |     9.37 ± 0.72     |     7.88 ± 1.61     |
|      1000 | KBinsDiscretizer    |    11.18 ± 1.73     |     7.66 ± 0.77     |
|     10000 | KBinsDiscretizer    |    30.44 ± 2.18     |    19.22 ± 1.97     |
|    100000 | KBinsDiscretizer    |   208.56 ± 40.38    |    132.57 ± 2.05    |
|   1000000 | KBinsDiscretizer    |  2430.03 ± 543.49   |   1495.98 ± 14.09   |
|  10000000 | KBinsDiscretizer    |  14434.75 ± 326.05  |  13772.40 ± 83.03   |
|       100 | MaxAbsScaler        |     7.02 ± 0.64     |     4.99 ± 0.62     |
|      1000 | MaxAbsScaler        |     7.80 ± 0.79     |     6.33 ± 0.52     |
|     10000 | MaxAbsScaler        |    14.05 ± 1.55     |    16.09 ± 0.95     |
|    100000 | MaxAbsScaler        |    77.73 ± 5.22     |    93.50 ± 9.96     |
|   1000000 | MaxAbsScaler        |   731.35 ± 40.15    |   940.63 ± 66.99    |
|  10000000 | MaxAbsScaler        |  7033.08 ± 102.73   |   8695.71 ± 43.69   |
|       100 | MinMaxScaler        |     7.10 ± 0.67     |     5.02 ± 0.59     |
|      1000 | MinMaxScaler        |     7.90 ± 0.68     |     5.98 ± 0.61     |
|     10000 | MinMaxScaler        |    14.39 ± 1.34     |    13.95 ± 0.34     |
|    100000 | MinMaxScaler        |    78.11 ± 10.29    |    79.85 ± 2.99     |
|   1000000 | MinMaxScaler        |   763.72 ± 102.21   |   806.07 ± 10.16    |
|  10000000 | MinMaxScaler        |   6437.05 ± 68.68   |   8101.09 ± 91.64   |
|       100 | MissingIndicator    |     7.17 ± 1.01     |     4.85 ± 0.68     |
|      1000 | MissingIndicator    |     8.32 ± 0.91     |     6.94 ± 4.44     |
|     10000 | MissingIndicator    |    20.15 ± 2.33     |    13.41 ± 0.52     |
|    100000 | MissingIndicator    |   133.30 ± 26.25    |    86.17 ± 6.75     |
|   1000000 | MissingIndicator    |  1340.77 ± 263.58   |   861.31 ± 13.76    |
|  10000000 | MissingIndicator    |  9789.76 ± 248.58   |  8652.58 ± 116.77   |
|       100 | PowerTransformer    |    64.40 ± 1.89     |    63.67 ± 1.95     |
|      1000 | PowerTransformer    |    83.53 ± 2.47     |    82.04 ± 2.44     |
|     10000 | PowerTransformer    |   227.49 ± 12.36    |    215.79 ± 6.78    |
|    100000 | PowerTransformer    |   1780.65 ± 52.37   |   1717.76 ± 18.06   |
|   1000000 | PowerTransformer    |  20199.55 ± 597.75  | 19214.55 ± 1217.01  |
|  10000000 | PowerTransformer    | 223822.38 ± 3796.60 | 341245.64 ± 7683.62 |
|       100 | QuantileTransformer |    12.82 ± 1.03     |    10.69 ± 0.90     |
|      1000 | QuantileTransformer |    18.56 ± 1.32     |    16.11 ± 1.36     |
|     10000 | QuantileTransformer |    71.73 ± 1.03     |    63.20 ± 1.18     |
|    100000 | QuantileTransformer |   444.37 ± 37.04    |    377.84 ± 6.38    |
|   1000000 | QuantileTransformer |  4168.16 ± 353.79   |   3523.79 ± 12.42   |
|  10000000 | QuantileTransformer |  37417.58 ± 265.68  |  39721.94 ± 302.41  |
|       100 | RobustScaler        |    11.94 ± 0.95     |    10.10 ± 0.74     |
|      1000 | RobustScaler        |    14.82 ± 0.95     |    13.14 ± 0.50     |
|     10000 | RobustScaler        |    36.73 ± 1.62     |    36.53 ± 0.51     |
|    100000 | RobustScaler        |   253.41 ± 10.01    |    252.06 ± 3.45    |
|   1000000 | RobustScaler        |  2432.56 ± 122.14   |   2477.61 ± 21.86   |
|  10000000 | RobustScaler        |  22991.53 ± 230.82  |  26018.57 ± 452.09  |
|       100 | SplineTransformer   |    12.56 ± 1.66     |    16.34 ± 1.69     |
|      1000 | SplineTransformer   |    22.76 ± 1.86     |    26.44 ± 2.10     |
|     10000 | SplineTransformer   |   130.17 ± 13.47    |    140.09 ± 8.89    |
|    100000 | SplineTransformer   |  1172.72 ± 123.12   |   1222.58 ± 77.83   |
|   1000000 | SplineTransformer   |  10161.89 ± 170.11  |  13171.53 ± 515.62  |
|  10000000 | SplineTransformer   | 102361.61 ± 2905.30 | 134908.07 ± 6628.31 |
|       100 | StandardScaler      |     7.42 ± 1.04     |     5.54 ± 0.85     |
|      1000 | StandardScaler      |     8.37 ± 0.98     |     6.79 ± 0.79     |
|     10000 | StandardScaler      |    16.18 ± 1.55     |    18.35 ± 1.44     |
|    100000 | StandardScaler      |    97.69 ± 5.19     |   116.77 ± 13.28    |
|   1000000 | StandardScaler      |   956.16 ± 58.34    |  1185.37 ± 110.37   |
|  10000000 | StandardScaler      |  8947.25 ± 123.73   |  10620.27 ± 181.71  |
|       100 | StepImputeFill      |     8.84 ± 0.91     |     4.14 ± 0.40     |
|      1000 | StepImputeFill      |    10.78 ± 1.72     |     5.83 ± 0.36     |
|     10000 | StepImputeFill      |    27.53 ± 2.46     |    17.83 ± 0.15     |
|    100000 | StepImputeFill      |   226.36 ± 17.17    |    150.83 ± 3.17    |
|   1000000 | StepImputeFill      |  2723.33 ± 142.30   |  1729.52 ± 106.81   |
|  10000000 | StepImputeFill      |  31721.51 ± 261.49  |  22858.77 ± 274.37  |


