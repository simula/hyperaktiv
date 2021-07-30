# HYPERAKTIV

HYPERAKTIV is a public dataset containing health, activity, andheart rate data from adult patients diagnosed with attention deficit hyperactivity disorder, better known as ADHD. The dataset consists of data collected from 51 patients with ADHD and 52 clinicalcontrols. In addition to the activity and heart rate data, we also include a series of patient attributes such as their age, sex, and information about their mental state, as well as output data from a computerized neuropsychological test.

This repository is structured as follows. The experiments directory contains the code used to perform the baseline experiments presented in the paper. The notebooks directory is supplementary and repeats the baseline experiments using Python notebooks. The scripts directory contains the code used to extract features from the activity data used in the baseline experiments.

The dataset can be downlaoded at the following link: [https://osf.io/3agwr/](https://osf.io/3agwr/)

## Script Usage
This repository contains scritps that may be used to reproduce the experiments presented in the paper. This can be done by either using the notebook included in `notebook` or by using the Pything script `predict_adhd.py` located in `experiments`.

The `predict_adhd.py` script contains the following command-line arguments: 

```
Usage: predict_adhd [OPTIONS]

  Script that runs the baseline experiments.

  Written by Steven Hicks.

Options:
  -x, --x-file-path               Full path to the extracted features (features.csv).
  -y, --y-file-path               Full path to the ground truth (patient_info.csv).
  -o, --output-file-path          Path to the file where the results will be written.
  -k, --k-folds                   The number folds used for cross-validation (training and validation).
  -t, --test-ratio                The ratio between the trainnig and testing data split.
  -s, --random-seed               The seed used to generate random numbers.
```

<!-- ## Cite
If you use this dataset in your research, Please cite the following paper: -->

## Terms of Use
The data is released fully open for research and educational purposes. The use of the dataset for purposes such as competitions and commercial purposes needs prior written permission.
<!-- In all documents and papers that use or refer to the dataset or report experimental results based on HYPERAKTIV, a reference to the related article needs to be added: XXX. -->

## Contact
Please contact steven@simula.no, michael@simula.no, or paalh@simula.no for any questions regarding the dataset.
