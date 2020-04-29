# Predicting Chicago Business Closures

Aya Liu, Ben Fogarty, Parth Khare  
12 June 2019  

CAPP 30254: Machine Learning for Public Policy  
Harris School of Public Policy  

## Project overview & requirements

This project's directory contains the following subdirectories and files:

```
modeling/
|- pipeline_library.py: general functions for a machine learning pipeline
|                       (reading data, preprocessing data, generating features,
|                       building models, etc.)
|- predict_closures.py: specific functions for applying pipeline_library to the
|                       Chicago Business Licenses dataset
|- load_data.py: downloads and links all the necessary datasets for this
|                analysis    
|- get_pickle.sh: downloads frozen version of dataset from 10 June 2019 as a
|                 pickle (Linux-specific)
|- get_pickle_mac.sh: downloads frozen version of dataset from 10 June 2019 as a
|                     pickle (Mac-specific)
|- getfiles_mac.sh: downloads csv files for datasets that could not be obtained
|                   through an API (Mac-specific)
|- - tokens.json: a json file containing API tokens for the Chicago Open Data
                  Portal and US Census Bureau website; the provided file includes
                  formats but not actual tokens for security reasons 

aux/
|- data_exploration.ipynb: contains code with basic data exploration

configs/: contains json files specifying preprocessing, feature generation,
|         and model specifications to be passed to the predict_closures.py
|         program  
|- ...


ethics_aq/: contains files related to the bias and fiarness report
|- ethics_aq/ethics_aequitas.py: code for producing the bias and fairness report  
|- ...

requirements.txt: python dependencies list

README.md: README file
```

The project was developed using Python 3.7.3 on MacOS Mojave 10.14.4, and
results were obtained by running the project on compute nodes of the Research
Computing Center at the University of Chicago and on a c4.8xlarge, c5.9xlarge,
and c5.18xlarge AWS EC2 virtual machines. Python package requirements can be
found in the `requirements.txt` file in the root of the project.

To install all the requirements, execute the following command in the root directory
of the project:

```
 pip install -r requirements.txt 
 ```

Helpful documentation and references are cited throughout the docstrings of the
code.

Before running the program for the first time, you'll need to download some datasets
that don't have an API. On Mac, execute the following command from within the
`modeling` directory:

```
sh getfiles_mac.sh
```

On other operating systems, dowload this
[download this archive file](https://uchicago.box.com/shared/static/cgwx1a1e3r0jz48knpy8mymg5etlzdji.zip)
and unzip it within the `modeling` directory. This will create a new subdirectory,
titled `data` within the `modeling` directory.

To run the program, use the following command from within the modelling directory:
```
python3 predict_closures.py -f <path to features config JSON file> -m <path to models config JSON file> [-p <path to optional preprocessing config file>] [-d <path to pickled dataset>] [-s <optional random seed>] [--savefigs (denotes that figures should be saved instead of displayed)] [--savepreds (denotes that predictions from each testing set should be saved)] [--saveeval (denotes that evaluation tables should be saved)]
```

## Presentation and findings

The authors have summarized their findings and saved predictions in a document that
is not included in this repository. For access, please contact one of them.
