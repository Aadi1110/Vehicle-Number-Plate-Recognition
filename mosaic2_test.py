
import cv2
import numpy as np
import matplotlib.pyplot as plt
from local_utils import detect_lp
from os.path import splitext, basename
from keras.models import model_from_json

def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)
        
def preprocess_image(image_path,resize=False):
	img = cv2.imread(image_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = img / 255
	return img

def get_plate(image_path, wpod_net, Dmax=640, Dmin=300):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg, cor

def get_img(img_path):
    wpod_net_path = "wpod-net.json"
    wpod_net = load_model(wpod_net_path)


    #test_image = 'indian_plates\Maruti-Suzuki-Swift-Old-522216d.jpg_0036_0684_0320_0110_0032.png'
    #img_path = 'indian_plates\Skoda-Fabia-527446c.jpg_0000_0397_0215_0130_0057.png'
    LpImg,cor = get_plate(img_path,wpod_net)
    img = cv2.imread(img_path)
    bgr = img.copy()
    #bgr = cv2.cvtColor(LpImg[0], cv2.COLOR_RGB2BGR)
    imgs = []
    if(LpImg!=-1):
        #for Lp in LpImg:
        img = LpImg[0]
        img*= 255
        img = img.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #imgs.append(img)
        '''
        bgr = img.copy()
        bgr[:,:,0] = img[:,:,2]
        bgr[:,:,2] = img[:,:,0]
        '''
    #print("\n\nShape:",img.shape,"\n\n")
    #cv2.imshow('img',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return img

