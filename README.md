# Working repo for predict business closures


# Aya's files:

Running the pipeline 
---
This is for homework 4, needs modification.

```
python3 main.py <data_filepath> <output_directory> <grid_size> <debug>
```
**grid_size** is `test`, `small`, or `large`. It determines the parametergrid for model building.  
**debug** is `True` or `False`. It determines whether print statements are shown.

Preliminary work
--
`draft-notebook.ipynb` 

Files
---
- `main.py`: to run in bash
- `pipeline.py`: class for a machine learning pipeline and model performance comparison functions
- `model.py`: class for a machine learning model
- `data_preprocess.py`: THIS CURRENTLY HOLDS FUNCTIONS FOR HW 4. ALL THE NOTEBOOK FUNCTIONS SHOULD EVENTUALLY GO HERE. dataset-specific functions to 1) clean the dataset before running the pipeline and 2) to preprocess the training and test sets as part of the pipeline
- `data_explore.py`: functions for exploratory data analysis before running the pipeline.  
