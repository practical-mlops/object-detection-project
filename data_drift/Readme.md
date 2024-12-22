# Data Drift

Before running run_deepchecks.py to compare the data drift between the training and testing datasets, you need to run download.py to download the dataset.

```
python download.py
```
Followed by running modify_brightness.py to modify the brightness of the images in the dataset, and create two directories for the modified and unmodified  images.
```
python modify_brightness.py
```
After this, you can run run_deepchecks.py to compare the data drift between the training and testing datasets.
```
python run_deepchecks.py
```
