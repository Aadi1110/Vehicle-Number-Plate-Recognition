import cv2
import numpy as np
import imutils
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from mosaic2_test import get_img

def sort_contours(cnts):
	# initialize the reverse flag and sort index
	i=0
	reverse = False
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return cnts

def rot(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	edged = cv2.Canny(gray, 100, 200)
	lines = cv2.HoughLines(edged, 1, np.pi/180, 50)
	temp = 0
	rotated = img.copy()
	if lines is not None:
		line = lines[0]
		theta = line[0,1]
		rotated = imutils.rotate_bound(rotated, 90-theta*180/np.pi)
	'''
	for line in lines:
		for rho,theta in line:
			a = np.cos(theta)
			b = np.sin(theta)
			
			x0 = a*rho
			y0 = b*rho
			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))
			print("rho:",rho)
			print("theta:",theta*180/np.pi-90)
			cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
	'''
	#cv2.imshow('img',img)
	#cv2.waitKey(0)
	return rotated

#img_path = 'indian_plates\Hyundai-Santro-Xing-521162c.jpg_0000_0267_0291_0246_0086.png'
def get_chars(img_path):
	cv2.imshow('Original Image',cv2.imread(img_path))
	img = get_img(img_path)
	#img = rot(img)
	#cv2.imshow('img1',img)

	#for img in imgs:
	h=0
	w=0
	if(img.shape[0]>img.shape[1]):
		h = 480
		w = int(img.shape[1]*h/img.shape[0])
	else:
		w = 480
		h = int(img.shape[0]*w/img.shape[1])
	dsize=(w,h)
	img = cv2.resize(img,dsize)
	
	img_t = img.copy()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	pix = 0
	for i in range(gray.shape[0]):
		for j in range(gray.shape[1]):
			pix += gray[i][j]
	pix = pix/(255*h*w)
	print("PIX:", pix)
	
	alpha = 1.0
	beta = 0
	if(pix<0.4):
		alpha = 1.5
	gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

	#cv2.imshow('gray',gray)
	blur = cv2.fastNlMeansDenoising(gray,2.0,15,25)
	#cv2.imshow('blur', blur)
	#ret,thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 115, 10)
	#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
	#dil = cv2.morphologyEx(thresh2,cv2.MORPH_ERODE,kernel)
	kernel = np.ones((2,2),'uint8')
	dil = cv2.erode(thresh2,kernel,iterations=2)
	#dil = thresh2
	temp_img = dil.copy()
	#dil = thresh
	#cv2.imshow('thresh',thresh)
	#cv2.imshow('dil',dil)
	#cv2.waitKey(0)
	contours,hierarchy = cv2.findContours(dil,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	#contours = sort_contours(contours)
	coords = []
	for cnt in contours:
		(x,y,w,h) = cv2.boundingRect(cnt)
		if(h/w>=0.5 and h/w<=5):
			if(h/img.shape[0]>=0.25 and h/img.shape[0]<0.95):
				cv2.rectangle(img_t, (x,y), (x+w,y+h), (0,255,0), 1)
				cv2.rectangle(temp_img, (int(x+3),int(y+3)), (int(x+w-3),int(y+h-3)), (255,255,255), -1)
				#coords.append((x,y,w,h))
	#cv2.imshow("img_t",img_t)
	#cv2.imshow('temp_img',temp_img)
	#cv2.waitKey(0)
	contours2,_ = cv2.findContours(temp_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	contours2 = sort_contours(contours2)
	for cnt2 in contours2:
		(x,y,w,h) = cv2.boundingRect(cnt2)
		if(h/w>=0.5 and h/w<=5):
			if(h/img.shape[0]>=0.25 and h/img.shape[0]<0.95):
				#cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 1)
				cv2.rectangle(img, (int(x-3),int(y-3)), (int(x+w+3),int(y+h+3)), (0,255,0), 1)
				coords.append((x,y,w,h))

	#cv2.imshow('img',img)
	#cv2.waitKey(0)


	json_file = open('MobileNets_character_recognition.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	model.load_weights("License_character_recognition_weight.h5")

	labels = LabelEncoder()
	labels.classes_ = np.load('license_character_classes.npy')



	num2let = {'0':'O', '1':'I', '2':'Z', '3':'B', '5':'S', '8':'B'}
	let2num = {'B':'8', 'D':'0', 'G':'6', 'I':'1', 'J':'7', 'O':'0', 'Q':'0', 'S':'5', 'T':'7', 'Z':'2'}
	chars = []
	for coord in coords:
		(x,y,w,h) = coord
		chars.append(dil[y:y+h,x:x+w])

	ind = 0
	final = []
	for c in chars:
		ind+= 1
		#c = cv2.resize(c,(76,76))
		hc=0
		wc=0
		if(c.shape[0]>c.shape[1]):
			hc = 76
			wc = int(c.shape[1]*hc/c.shape[0])
		else:
			wc = 76
			hc = int(c.shape[0]*wc/c.shape[1])
		dsizec = (wc,hc)
		c = cv2.resize(c,dsizec)
		#print("\n\nShape",c.shape)
		c = cv2.copyMakeBorder(c, int((80-c.shape[0])/2), 0, int((80-c.shape[1])/2), 0, cv2.BORDER_CONSTANT)
		c = cv2.copyMakeBorder(c, 0, int(80-c.shape[0]), 0, int(80-c.shape[1]), cv2.BORDER_CONSTANT)
		ker = np.ones((2,2),'uint8')
		c = cv2.erode(c,ker,iterations=0)
		c = np.stack((c,)*3, axis=-1)
		prediction = labels.inverse_transform([np.argmax(model.predict(c[np.newaxis,:]))])
		if(len(chars)==10):
			if(ind==3 or ind==7 or ind==8 or ind==9 or ind==10):
				if(prediction[0] in let2num):
					prediction[0] = let2num[prediction[0]]

			elif(ind==1 or ind==2 or ind==5 or ind==6):
				if(prediction[0] in num2let):
					prediction[0] = num2let[prediction[0]]

		if(ind==1):
			if(prediction[0]=='N' or prediction[0]=='H'):
				prediction[0] = 'M'
			if(prediction[0]=='O'):
				prediction[0]='D'
		if(ind==3):
			if(prediction[0]=='O'):
				prediction[0]='0'

		final.append(prediction[0])
		#cv2.imshow('c',c)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()

	#cv2.imshow('img2',img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	return final,img