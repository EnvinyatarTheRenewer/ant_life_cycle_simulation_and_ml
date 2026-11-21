# ant_life_cycle_simulation_and_ml

This repository is for a **project that investigate** the ability of various **machine learning models** to **understand the maintenance in a polymorphic** state of a de novo **caste antagonistic allele** on multiple generations in a structured ant population. It will first explore as simpler biological model that is understandable even without machine learning. Then, we will proceed in as second notebook, with **a more complex biological model** adding the male as a third category of antagonism. On a more practical side, this repository has three folders containing the following:

## DatasetGenerators
*This folder contains all the necessary scripts to generate the data sets for the subsequent analysis and machine learning:*

1. **dataset_generator_simple_model.py:** This is the script that has been used to generate the data for the simpler biological model (with only two allele and two castes). If you want to generate data again; all you need to know about how to achieve this in the notebook 1.1-simple-ant-model-analysis.ipynb.
2. **dataset_generator_3way_model_on_CPU.py:** This is the script that has been used to generate the data for more complex biological model with the three way cast antagonism. This version is the CPU friendly one, if by mistake you call the GPU firendly version without having GPUs, it will automatically, fall back to this one.
3. **dataset_generator_3way_model_on_GPU.py** This is the script that has been used to generate the data for more complex biological model with the three way cast antagonism. It is the GPU friendly version.

## Datasets
*This folder contains all the necessary datasets to run the notebooks. If you wish to rerun those, you will get detailled explanation on how to in the related notebook. The dataset of the second more biologically complex model was split in two parts, due to to its consequent size.*

1. **ant_simple_model_dataset.csv:** It is a .csv file, which contains a bit more than 8000 simulations. It is the dataset that has been used in the notebook 1.1-simple-ant-model-analysis.ipynb and was imulated with the dataset_generator_simple_model.py script.
2. **ant_ml_3way_ant_dataset_part1.csv:** It is a .csv file, which contains a bit more than 200,000 simulations. It is the first half of the dataset that is used in the notebook 1.2-3way-ant-model-analysis.ipynb and was imulated with the dataset_generator_3way_model_on_GPU.py script.
3. **ant_ml_3way_ant_dataset_part2.csv:** It is a .csv file, which contains a bit more than 200,000 simulations. It is the second half of the dataset that is used in the notebook 1.2-3way-ant-model-analysis.ipynb and was imulated with the dataset_generator_3way_model_on_GPU.py script.

## DetailedNotebooks
*This folder contains the detailed notebook for the machine learning and data analysis for both biological models:*

1. **1.1-simple-ant-model-analysis.ipynb:** It is the notebook in which all the machine learning has been realised for the simpler biological model. It is both local and collab friendly for running. Detailed explanation about the models explored can be find there.
2. **1.2-3way-ant-model-analysis.ipynb (in DetailedNotebooks):** It is the notebook in which all the machine learning has been realised for the more complex biological model. It is both local and collab friendly for running. Detailed explanation about the models explored can be find there.

## Additionally
*In addition, we have those files helping with the structuring and understanding of the repository:*

1. **README.md:** README.md (this file) contains some general information about this project and about its goal. It is there to help the reader understand the structure of the repository.
2. **.gitignore files:** This one contains useless collateral outputs.

*Now, if you want to try and vizualize a bit more what is happening in term of machine learning models, you may open first the 1.1-simple-ant-model-analysis.ipynb and then the 1.2-3way-ant-model-analysis.ipynb. Those will guide you along each steps of the way in a detailed manner.*