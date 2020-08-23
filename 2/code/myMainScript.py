import os
import math
import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt 


def nothing(x):
	pass

def myForegroundMask(path):
	img = cv.imread(path)
	assert isinstance(img, np.ndarray)

	plot = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	mask = np.array(plot > 60, dtype=np.float64)

	plt.figure()
	plt.imshow(plot)
	
	plt.figure()
	plt.imshow(plot * mask / 255)

	plt.figure()
	plt.imshow(mask) 

	plt.show()


if __name__ == '__main__':
	IMG_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
	path = os.path.join(IMG_DIR, 'statue.png')

	myForegroundMask(path)