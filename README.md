# CS663 Assignment-1

This is the first assignment for the course CS663. Our team has the following four members:
- Yash Sanjeev - 180070068
- Shubham Kar - 180070058
- Garaga VVS Krishna Vamsi -  180070020
- Rishav Ranjan - 180070045

Please mount this project on a GitHub repo to look at this README file in its full glory. 

## Installation and Virtual Environment Setup

Enter the following commands one by one into the Windows cmd:

```
git clone https://github.com/sanjivyash/CS-663-Assignment1.git
cd CS-663-Assignment1
pip install virtualenv
virtualenv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

For Linux Distros, follow the commands given below after the usual update and upgrade commands:

```
git clone https://github.com/sanjivyash/CS-663-Assignment1.git
cd CS-663-Assignment1
pip3 install virtualenv
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Multiprocessing and Multithreading

The two files ```2/code/myAHE.py``` and ```2/code/myCLAHE.py``` utilise the multiprocessing library of Python to improve the running time of these calculation intensive algorithms. 
However this functionality is not provided in Windows machines, so they should change the multiprocessing to multithreading. That reduces the efficiency of the algorithm but prevents the machine from freezing.

```python
import os
import math
import numpy as np 
import cv2 as cv
import time
from concurrent.futures import ThreadPoolExecutor as executor
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
```
Please insert these 8 lines at the top of those files instead of the original ones, and you may see that the only change is a small one in line 6.
Linux Distros and Mac users should not have any problems with the original code.

## Structure and Setup

- In Problem 1, the code is present in ```1/code/myMainScript.py``` and the name of the functions follow the submission guidenelines. Since no one is sure of how slow Python is, all the output images have been carefully compiled and stored in ```1/images/``` directory.

- For Problem 2, there are two more scripts, ```2/code/myAHE.py``` and ```2/code/myCLAHE.py``` apart from the usual ```2/code/myMainScript.py``` since they house codes which require multiprocessing. The CLAHE file has two functions ```myExactCLAHE``` and ```myBilinearCLAHE```, the second of which makes use of bilinear interpolation to approximate the CLAHE results. The ```2/images/``` directory also has two subdirectories to classify and separate the results provided by these two functions.

- Images in ```2/images/CLAHE/``` are named ```${window_size}#{threshold_value}.png```, so an image called ```$20#0.02.png``` denotes a window size of 20 and threshold value of 0.02.  

- For Problem 3, the code has been included in ```3/code/myMainScript.py```. The image data is in ```3/data/``` while the results are in ```3/images/```.
