# CS663 Assignment-1

This is the first assignment for the course CS663. Our team has the following four members:
- Yash Sanjeev - 180070068
- Shubham Kar - 180070058
- Garaga VVS Krishna Vamsi -  180070020
- Rishav Ranjan - 180070045

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