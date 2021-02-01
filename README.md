# Installation
Install TensorFlow. The code has been tested with Python 3.7, Tensorflow-1.12.0, scikit-learn 0.20.4, scipy 1.2.1, pillow 7.0.0, CUDA 10.0 and cuDNN 7.4.1 on Centos 7.5

# Usage
To create slices from sMRI:
```
python creat_slices.py
```

To augnment data:
```
python data_aug_and_make_fold.py
```
To train base classifiers corresponded to a 2D slice dataset:
```
python cnnmodel.py
```
After the above training, we can optimize the combination of base classifiers and ensemble:
```
python slices_select_main.py
```

# Data and Code Availability Statement
See data and code availability statement file for details