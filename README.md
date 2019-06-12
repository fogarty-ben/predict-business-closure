# Predicting Chicago Business Closures

Aya Liu, Ben Fogarty, Parth Khare  
11 June 2019  

CAPP 30254: Machine Learning for Public Policy  
Harris School of Public Policy  

## Project overview & requirements

This project's folder contains the following files:

- pipeline_library.py: general functions for a machine learning pipeline (reading data, preprocessing data, generating features, building models, etc.)
- predict_closures.py: specific functions for applying pipeline_library to the Chicago Business Licenses data
- load_data.py: downloads and links all the necessary datasets for this analysis   

- tokens.json: a json file containing Chicago Open Data Portal and US Census Bureau API tokens
to allow for API access to these data sources; the provided file includes formats but not actual tokens for security reasons 
- data_exploration.ipynb: contains code demonstrating basic data exploration
- get_pickle.sh: downloads pickled version of dataset from 10 June 2019 (Linux-specific)
- get_pickle_mac.sh: downloads pickled version of dataset from 10 June 2019 (Mac-specific)
- getfiles_mac.sh: downloads csv files for datasets that could not be obtained through an
API
- mlproject-env.yml: Anaconda environment configuration file for running the project

- configs/: contains json files specifying preprocessing, feature generation, and model specifications to be passed to the predict_closures program

- ethics_aq/: contains files related to the bias and fiarness report
-- ethics_aq/ethics_aequitas.py: code for producing the bias and fairness report

- data/projects_2012-2013.csv: the DonorsChoose dataset
- data/projects_1000.csv: a sample of 1,000 DonorsChoose projects
- data/projects_sample.csv: a sample of 10,000 DonorsChoose projects  



The project was developed using Python 3.7.3 on MacOS Mojave 10.14.4, and results were obtained by running the project on a compute node of the Research Computing Center at the University of Chicago and on a c4.8xlarge, c5.9xlarge, and c5.18xlarge AWS EC2 virtual machines. It requires the following libraries and their dependencies:

| Package        | Version     |
| :------------: | :---------: |
| certifi |  2019.3.9  |
| geopandas |  0.5.0  |
| graphviz | 0.10.1 |
| matplotlib |  3.0.3  |
| numpy |  1.16.2  |
| pandas |  0.24.2  |
| scikit-learn |  0.20.3 |
| seaborn |  0.9.0  |
| shapely |  1.6.4  |
| sodapy |  1.5.2 |
| urllib3 |   1.25.3 |

Alternatively, a conda environment including all of the necessary data libraries is available on Anaconda Cloud at fogarty-ben/mlproject or in the mlproject-env.yml file in the root of the repository.

Helpful documentation and references are cited throughout the docstrings of the code.

To run the program, use the following command:
```
python3 predict_closures.py -f <path to features config JSON file>
-m <path to models config JSON file> [-p <path to optional preprocessing config file>] [-d <path to pickled dataset>] [-s <optional random seed>] [--savefigs (denotes that figures should be saved instead of displayed)] [--savepreds (denotes that predictions from each testing set should be saved)] [--saveeval (denotes that evaluation tables should be saved)]
```
