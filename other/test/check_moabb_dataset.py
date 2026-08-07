"""

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import moabb
from moabb import datasets

import pprint

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

dataset_dict = datasets.utils.dataset_dict
dataset_list = datasets.utils.dataset_list

data_dict = {}

for dataset_class in dataset_list :
    print("%%%%%%%%%%%%%%%%%%%%%%")
    print(f"Testing dataset {dataset_class.__name__}...")

    tmp_dataset = dataset_class()
    subj_list = tmp_dataset.subject_list
    try :
        tmp_data = tmp_dataset.get_data(subjects = [subj_list[0]])
        data_dict[tmp_dataset.code] = str(tmp_data)
    except Exception as e :
        data_dict[tmp_dataset.code] = e

# pprint.pprint(dataset_dict)

for dataset_name in data_dict :
    try :
        info = data_dict[dataset_name]
        print(dataset_name[0:10], "\t", info[0:15])
    except :
        print(f"Problem with {dataset_name}")


