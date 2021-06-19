import cv2
import numpy as np
import pixellib
from pixellib.tune_bg import alter_bg
#image path
imgpath='example.png'
img = cv2.imread(imgpath)
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grey = cv2.medianBlur(grey, 5)
edges = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
#cartoonizing
color = cv2.bilateralFilter(img, 9, 250, 250)
cartoon = cv2.bitwise_and(color, color, mask = edges)
#making a cartoon file
cartoonOutputPath="ddddfs.jpg"
cv2.imwrite(cartoonOutputPath, cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()
change_bg = alter_bg(model_type = "pb")
change_bg.load_pascalvoc_model("xception_pascalvoc.pb")
resultpath='finalimage.jpg'
change_bg.change_bg_img(f_image_path = imgpath ,b_image_path =cartoonOutputPath, output_image_name=resultpath)