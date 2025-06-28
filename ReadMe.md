# xLSTMAD

# How to run the experiments?

## Preliminaries

To run the experiments, you need to have the following prerequisites:

- Python 3.11
- Download pytorch and torchvision from [here](https://pytorch.org/get-started/locally/) and install them according to
  your system configuration.
- Install required dependencies by running:
  ```bash
  pip install -r requirements.txt
  ```
- Download the datasets from [here](https://www.thedatum.org/datasets/TSB-AD-M.zip).
- Unzip the downloaded file and place the `TSB-AD-M` and `File_List` directories in the `resources/` directory.

## Running the experiments

To run the experiments, you can use `run_detector_m.py` script. It allows you to run the detector with all datasets
specified by the file list.

Example command to run xLSTMAD-F:

```bash
python scripts/run_detector_m.py --AD_Name xLSTMAD-F
```

The available methods include:

- `xLSTMAD-F`: xLSTMAD in forecasting mode with MSE loss.
- `xLSTMAD-F-DTW`: xLSTMAD in forecasting mode with DTW loss.
- `xLSTMAD-R`: xLSTMAD in reconstruction mode with MSE loss.
- `xLSTMAD-R-DTW`: xLSTMAD in reconstruction mode with DTW loss.
- `RandomModel`: A random model that predicts the mean of the training data.

The details about other available methods can be found in the documentation of the TSB-AD
benchmark ([see here](https://github.com/TheDatumOrg/TSB-AD/tree/main)).

# Datasets

To ensure reproducibility of the results, we use the same datasets as in the TSB-AD benchmark. The datasets are
available at the benchmark original [repository](https://github.com/TheDatumOrg/TSB-AD/tree/main).
Full information about the datasets, splits, and their characteristics can be found in the TSB-AD benchmark
paper - [see here](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3f3c690b7a99fba16d0efd35cb83b2c-Abstract-Datasets_and_Benchmarks_Track.html).

# Results

Table: Summary comparison of our proposed method with 23 competitors across 180 time-series originating from 17 datasets
in terms of multiple metrics.

| Method              | VUS-PR | VUS-ROC | AUC-PR | AUC-ROC | Standard-F1 | PA-F1 | Event-based-F1 | R-based-F1 | Affiliation-F1 |
|---------------------|--------|---------|--------|---------|-------------|-------|----------------|------------|----------------|
| xLSTMAD-F (MSE)     | 0.35   | 0.77    | 0.35   | 0.74    | 0.40        | 0.85  | 0.70           | 0.42       | 0.89           |
| xLSTMAD-F (SoftDTW) | 0.34   | 0.76    | 0.35   | 0.73    | 0.40        | 0.83  | 0.67           | 0.41       | 0.88           |
| xLSTMAD-R (MSE)     | 0.37   | 0.72    | 0.32   | 0.68    | 0.38        | 0.53  | 0.45           | 0.36       | 0.82           |
| xLSTMAD-R (SoftDTW) | 0.36   | 0.72    | 0.31   | 0.68    | 0.37        | 0.57  | 0.48           | 0.35       | 0.82           |
| RandomModel         | 0.10   | 0.59    | 0.05   | 0.50    | 0.09        | 0.71  | 0.10           | 0.10       | 0.69           |
| CNN                 | 0.31   | 0.76    | 0.32   | 0.73    | 0.37        | 0.78  | 0.65           | 0.37       | 0.87           |
| OmniAnomaly         | 0.31   | 0.69    | 0.27   | 0.65    | 0.32        | 0.55  | 0.41           | 0.37       | 0.81           |
| PCA                 | 0.31   | 0.74    | 0.31   | 0.7     | 0.37        | 0.79  | 0.59           | 0.29       | 0.85           |
| LSTMAD              | 0.31   | 0.74    | 0.31   | 0.7     | 0.36        | 0.79  | 0.64           | 0.38       | 0.87           |
| USAD                | 0.3    | 0.68    | 0.26   | 0.64    | 0.31        | 0.53  | 0.4            | 0.37       | 0.8            |
| AutoEncoder         | 0.3    | 0.69    | 0.3    | 0.67    | 0.34        | 0.6   | 0.44           | 0.28       | 0.8            |
| KMeansAD            | 0.29   | 0.73    | 0.25   | 0.69    | 0.31        | 0.68  | 0.49           | 0.33       | 0.82           |
| CBLOF               | 0.27   | 0.7     | 0.28   | 0.67    | 0.32        | 0.65  | 0.45           | 0.31       | 0.81           |
| MCD                 | 0.27   | 0.69    | 0.27   | 0.65    | 0.33        | 0.46  | 0.33           | 0.2        | 0.76           |
| OCSVM               | 0.26   | 0.67    | 0.23   | 0.61    | 0.28        | 0.48  | 0.41           | 0.3        | 0.8            |
| Donut               | 0.26   | 0.71    | 0.2    | 0.64    | 0.28        | 0.52  | 0.36           | 0.21       | 0.81           |
| RobustPCA           | 0.24   | 0.61    | 0.24   | 0.58    | 0.29        | 0.6   | 0.42           | 0.33       | 0.81           |
| FITS                | 0.21   | 0.66    | 0.15   | 0.58    | 0.22        | 0.72  | 0.32           | 0.16       | 0.81           |
| OFA                 | 0.21   | 0.63    | 0.15   | 0.55    | 0.21        | 0.72  | 0.41           | 0.17       | 0.83           |
| EIF                 | 0.21   | 0.71    | 0.19   | 0.67    | 0.26        | 0.74  | 0.44           | 0.26       | 0.81           |
| COPOD               | 0.2    | 0.69    | 0.2    | 0.65    | 0.27        | 0.72  | 0.41           | 0.24       | 0.8            |
| IForest             | 0.2    | 0.69    | 0.19   | 0.66    | 0.26        | 0.68  | 0.41           | 0.24       | 0.8            |
| HBOS                | 0.19   | 0.67    | 0.16   | 0.63    | 0.24        | 0.67  | 0.4            | 0.24       | 0.8            |
| TimesNet            | 0.19   | 0.64    | 0.13   | 0.56    | 0.2         | 0.68  | 0.32           | 0.17       | 0.82           |
| KNN                 | 0.18   | 0.59    | 0.14   | 0.51    | 0.19        | 0.69  | 0.45           | 0.21       | 0.79           |
| TranAD              | 0.18   | 0.65    | 0.14   | 0.59    | 0.21        | 0.68  | 0.4            | 0.21       | 0.79           |
| LOF                 | 0.14   | 0.6     | 0.1    | 0.53    | 0.15        | 0.57  | 0.32           | 0.14       | 0.76           |
| AnomalyTransformer  | 0.12   | 0.57    | 0.07   | 0.52    | 0.12        | 0.53  | 0.33           | 0.14       | 0.74           |

Table: Comparison of our proposed method with its closer baseline, LSTMAD, the best other performing model, CNN, and a
model returning random predictions (Rand). We present the results for the two most comprehensive metrics, VUS-PR and
VUS-AUC. Our method is presented in two variants: reconstruction (noted as **R**) and forecasting (noted as
**F**) with two losses: **MSE** and SoftDTW (noted as **DTW**)

|            | VUS-PR |        |        |        |        |      |      |       | VUS-ROC |        |        |        |      |      |      |
|------------|--------|--------|--------|--------|--------|------|------|-------|---------|--------|--------|--------|------|------|------|
| Dataset    | R(MSE) | R(DTW) | F(MSE) | F(DTW) | LSTMAD | CNN  | Rand | ----- | R(MSE)  | R(DTW) | F(MSE) | F(DTW) | LSTM | CNN  | Rand |
| CATSv2     | 0.10   | 0.10   | 0.04   | 0.04   | 0.04   | 0.08 | 0.03 |       | 0.61    | 0.61   | 0.46   | 0.46   | 0.43 | 0.64 | 0.53 |
| CreditCard | 0.06   | 0.07   | 0.02   | 0.02   | 0.02   | 0.02 | 0.02 |       | 0.75    | 0.81   | 0.87   | 0.87   | 0.87 | 0.86 | 0.84 |
| Daphnet    | 0.50   | 0.46   | 0.31   | 0.31   | 0.31   | 0.21 | 0.06 |       | 0.95    | 0.95   | 0.81   | 0.81   | 0.81 | 0.84 | 0.52 |
| GECCO      | 0.04   | 0.04   | 0.02   | 0.02   | 0.02   | 0.03 | 0.02 |       | 0.50    | 0.54   | 0.44   | 0.44   | 0.44 | 0.65 | 0.64 |
| GHL        | 0.01   | 0.01   | 0.16   | 0.16   | 0.06   | 0.02 | 0.01 |       | 0.46    | 0.45   | 0.69   | 0.68   | 0.63 | 0.55 | 0.52 |
| Genesis    | 0.01   | 0.01   | 0.03   | 0.03   | 0.04   | 0.10 | 0.01 |       | 0.78    | 0.85   | 0.58   | 0.58   | 0.58 | 0.96 | 0.77 |
| LTDB       | 0.40   | 0.34   | 0.34   | 0.34   | 0.31   | 0.33 | 0.19 |       | 0.70    | 0.68   | 0.71   | 0.71   | 0.71 | 0.72 | 0.58 |
| MITDB      | 0.09   | 0.08   | 0.13   | 0.13   | 0.11   | 0.14 | 0.04 |       | 0.66    | 0.66   | 0.69   | 0.68   | 0.67 | 0.69 | 0.54 |
| MSL        | 0.38   | 0.40   | 0.41   | 0.34   | 0.23   | 0.35 | 0.08 |       | 0.81    | 0.81   | 0.81   | 0.77   | 0.73 | 0.77 | 0.63 |
| OPPORT.    | 0.16   | 0.16   | 0.16   | 0.16   | 0.17   | 0.16 | 0.05 |       | 0.28    | 0.28   | 0.65   | 0.65   | 0.65 | 0.61 | 0.53 |
| PSM        | 0.18   | 0.18   | 0.24   | 0.24   | 0.22   | 0.22 | 0.13 |       | 0.61    | 0.62   | 0.74   | 0.73   | 0.72 | 0.70 | 0.54 |
| SMAP       | 0.35   | 0.32   | 0.20   | 0.22   | 0.17   | 0.20 | 0.04 |       | 0.73    | 0.74   | 0.76   | 0.76   | 0.71 | 0.78 | 0.60 |
| SMD        | 0.35   | 0.37   | 0.31   | 0.30   | 0.31   | 0.37 | 0.05 |       | 0.83    | 0.84   | 0.83   | 0.83   | 0.83 | 0.83 | 0.60 |
| SVDB       | 0.26   | 0.19   | 0.20   | 0.20   | 0.15   | 0.19 | 0.06 |       | 0.68    | 0.67   | 0.72   | 0.72   | 0.69 | 0.73 | 0.57 |
| SWaT       | 0.34   | 0.35   | 0.16   | 0.16   | 0.16   | 0.48 | 0.14 |       | 0.71    | 0.71   | 0.50   | 0.50   | 0.47 | 0.74 | 0.53 |
| TAO        | 0.80   | 0.82   | 1.00   | 1.00   | 0.99   | 1.00 | 0.77 |       | 0.88    | 0.91   | 1.00   | 1.00   | 1.00 | 1.00 | 0.94 |