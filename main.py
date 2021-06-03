import cv2
import numpy as np
import imutils
from mosaic2 import get_chars

def PlateRecognition(image_path):
	answer, img = get_chars(image_path)
	#imgs = get_chars(image_path)
	print(''.join(answer))
	#for img in imgs:
	cv2.imshow('img',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def test():
	image_paths = []
	for image_path in image_paths:
		PlateRecognition(image_path)

if __name__ == "__main__":
	test()