# HYPERAKTIV

HYPERAKTIV is a public dataset containing health, activity, andheart rate data from adult patients diagnosed with attention deficit hyperactivity disorder, better known as ADHD. The dataset consists of data collected from 51 patients with ADHD and 52 clinicalcontrols. In addition to the activity and heart rate data, we also include a series of patient attributes such as their age, sex, and information about their mental state, as well as output data from a computerized neuropsychological test.

This repository is structured as follows. The experiments directory contains the code used to perform the baseline experiments presented in the [paper](https://dl.acm.org/doi/10.1145/3458305.3478454). The notebooks directory is supplementary and repeats the baseline experiments using Python notebooks. The scripts directory contains the code used to extract features from the activity data used in the baseline experiments.

The dataset can be downlaoded at the following link: [https://osf.io/3agwr/](https://osf.io/3agwr/)

## Script Usage
This repository contains scritps that may be used to reproduce the experiments presented in the [paper](https://dl.acm.org/doi/10.1145/3458305.3478454). This can be done by either using the notebook included in `notebook` or by using the Pything script `predict_adhd.py` located in `experiments`.

The `predict_adhd.py` script contains the following command-line arguments: 

```
Usage: predict_adhd.py

Script that runs the baseline experiments.

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

As stated in the paper, the experiments were performed with an additional 31 healthy controls taken from our previously published dataset [Psykose](https://datasets.simula.no/psykose/). This slighly modified version of the dataset can be found at the following link: https://drive.google.com/file/d/1z7TX3XO0yRcS4kylrhy91jFaK6kqZjq0/view?usp=sharing

## Cite
If you use this code or the dataset in your research, Please cite the following paper:
```
@inproceedings{10.1145/3458305.3478454,
  author = {Hicks, Steven A. and Stautland, Andrea and Fasmer, Ole Bernt and F\o{}rland, Wenche and Hammer, Hugo Lewi and Halvorsen, P\r{a}l and Mjeldheim, Kristin and Oedegaard, Ketil Joachim and Osnes, Berge and Gi\ae{}ver Syrstad, Vigdis Elin and Riegler, Michael Alexander and Jakobsen, Petter},
  title = {HYPERAKTIV: An Activity Dataset from Patients with Attention-Deficit/Hyperactivity Disorder (ADHD)},
  year = {2021},
  isbn = {9781450384346},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3458305.3478454},
  doi = {10.1145/3458305.3478454},
  abstract = {Machine learning research within healthcare frequently lacks the public data needed to be fully reproducible and comparable. Datasets are often restricted due to privacy concerns and legal requirements that come with patient-related data. Consequentially, many algorithms and models get published on the same topic without a standard benchmark to measure against. Therefore, this paper presents HYPERAKTIV, a public dataset containing health, activity, and heart rate data from patients diagnosed with attention deficit hyperactivity disorder, better known as ADHD. The dataset consists of data collected from 51 patients with ADHD and 52 clinical controls. In addition to the activity and heart rate data, we also include a series of patient attributes such as their age, sex, and information about their mental state, as well as output data from a computerized neuropsychological test. Together with the presented dataset, we also provide baseline experiments using traditional machine learning algorithms to predict ADHD based on the included activity data. We hope that this dataset can be used as a starting point for computer scientists who want to contribute to the field of mental health, and as a common benchmark for future work in ADHD analysis.},
  booktitle = {Proceedings of the 12th ACM Multimedia Systems Conference},
  pages = {314–319},
  numpages = {6},
  keywords = {ADHD, Actigraphy, Artificial Intelligence, Attention-Deficit Hyperactivity Disorder, Dataset, Heart Rate, Machine Learning, Motor Activity},
  location = {Istanbul, Turkey},
  series = {MMSys '21}
}
```

## Terms of Use
The data is released fully open for research and educational purposes. The use of the dataset for purposes such as competitions and commercial purposes needs prior written permission.
<!-- In all documents and papers that use or refer to the dataset or report experimental results based on HYPERAKTIV, a reference to the related article needs to be added: XXX. -->

## Contact
Please contact steven@simula.no, michael@simula.no, or paalh@simula.no for any questions regarding the dataset.
