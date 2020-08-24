import os
import math
import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt 


def extractImage(path):
	img = cv.imread(path)
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	assert isinstance(img, np.ndarray)

	return img


def myForegroundMask(path):
	if isinstance(path, str):
		img = extractImage(path)
	elif isinstance(path, np.ndarray):
		img = path
	else:
		raise TypeError('Provide image or path')

	mask = np.array(img > 60, dtype=np.uint8)

	plt.figure()
	plt.imshow(img)
	
	plt.figure()
	plt.imshow(img * mask)

	plt.figure()
	plt.imshow(mask * 255) 

	plt.show()

	return img * mask


def linearStretch(x):
	assert 0 <= x <= 1

	if x < 0.3:
		return x/5
	if x > 0.7:
		return 1 - (1-x)/5
	return (x-0.3)*0.88/0.4 + 0.06


def myLinearContrastStretching(path):
	if isinstance(path, str):
		img = extractImage(path)
	elif isinstance(path, np.ndarray):
		img = path
	else:
		raise TypeError('Provide image or path')

	m, n = img.shape[:2]
	out = np.zeros(img.shape)

	for i in range(m):
		for j in range(n):
			for k in range(3):
				out[i][j][k] = linearStretch(img[i][j][k]/255)

	plt.figure()
	plt.imshow(img)
	
	plt.figure()
	plt.imshow(out) 

	plt.show()


if __name__ == '__main__':
	IMG_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
	
	path = os.path.join(IMG_DIR, 'statue.png')
	out = myForegroundMask(path)
	myLinearContrastStretching(out)
	
	path = os.path.join(IMG_DIR, 'retina.png')
	myLinearContrastStretching(path)

	for file in os.listdir(IMG_DIR):
		if 'statue' in file or 'retina' in file:
			continue 

		print(file)
		path = os.path.join(IMG_DIR, file)
		myLinearContrastStretching(path)