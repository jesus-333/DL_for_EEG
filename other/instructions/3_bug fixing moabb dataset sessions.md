At the moment there is a logic `bug` in how train and test dataset are created by the function `get_dataset` of the file `moabb_dataset.py`.

# Current problem

Currently, it creates the train/test dataset based on the name of session inside `moabb`.
E.g. if I pass to the function the parameter `dataset_type = 'train'` it will try to return session with the name like `train`, `training` etc.
Instead, if I pass to the function the parameter `dataset_type = 'test'` it will try to return session with the name like `test`, `val`, `evaluation` etc.
If no such session exist it will always return the last section as test dataset and the rest as training dataset

This approach, even if it works introduce two problems
1) For dataset with only 1 session training and test dataset will be identical. And this is a methodological problem for Deep Learning training.
2) The decision to use the last session as test data it's sensible but arbitrary. Other users could like different approach.

# My proposal 

I want an update that works like that
- Inside the dataset config add the parameter `use_original_split`. It is a boolean parameter and if it is set to `True` it will try to divide the data using the session name (basically the current approach)
- If this division is not available then we will have another parameter to decide how to split the data between training and test.
- This is called `split_train_test`. It is a float that must have value between `0` and `1`
    - It's purpose is to indicate the percentage of data used for training and test. More precisely `split_train_test` will be used for training and `1 - split_train_test` will be used for test.
    - It will be used in the following occasion :
        - When `use_original_split` is set to `False`.
        - When `use_original_split` is set to `True`, but there are no indication which data are training and which are testing.
    - By default, it has value `0.5` (i.e. half of the data for training and the other half for testing)

When the code split the data using `split_train_test` it should work in the following way :
- All the data of all sessions are put together
- So I obtain `N` EEG samples/recording/trials in total (name can change between works but what I mean is the single sample that will be fed into the deep learning models).
- I will create an array with value from `0` to `N-1` (basically indices)
- This array will be split in two part, with size proportional to `split_train_test`.
- The part of size `split_train_test` will contain the indices of the training samples.
- The part of size `1 - split_train_test` will contain the indices of the test samples.

Write an ad-hoc function to create this array. The function should also receive in input a `seed` for reproducibility.

# Final remarks

VERY IMPORTANT. Update all the content inside `llm_wiki_jade` accordingly.
If you need more informations/instructions/clarifications let me know.
