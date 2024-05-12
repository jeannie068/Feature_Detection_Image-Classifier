# Import two lib
import cv2
import numpy as np
import os # Read file

# Import images
path = 'image_query' # The folder of images to classifier
images = [] # Build list to store images to classifier
classNames = []
myList = os.listdir(path) # Get the name of images  to classifier
print('total classes detected: ', len(myList))
for class_i in myList:
	imgCur = cv2.imread(f'{path}/{class_i}', 0) # Gray scale
	images.append(imgCur)
	classNames.append(os.path.splitext(class_i)[0]) # Remove file extension: jpg png

# Extract Key Points
orb = cv2.orb_create(nfeature=1000)

def find_des(images):
	desList = [] # Build a list to store descriptors
	for img_i in images:
		kp, des = orb.detectAndCompute(img_i, None) # Having key point
		desList.append(des)
	return desList

#  Find matches
def find_matches(img, desList, threshold=15):
	# Find descriptor in camera img
	kp2, des2 = orb.detectAndCompute(img, None) #having descriptor
	# Match the features
	bf = cv2.BFMatcher()
	matchList = [] # Store各類別的match分數
	final_match_id = -1
	try:
		for des1 in desList:
			matches = bf.knnMatch(des1, des2, k=2) # k個最佳匹配
			# 挑選相匹配的特徵點
			good_match = []
			for m,n in matches:
				# 比例挑選
				if m.distance < 0.75*n.distance:
					good_match.append([m])
				# Show matches result
#			img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_match, None, flag=2)
#			cv2.imshow('Matches Knn', img3)
		matchList.append(len(good_match)) # 將此類別有幾個匹配儲存作為匹配分數
	except:
		pass
	if len(matchList)!=0 and max(matchList)>threshold:
		final_match_id = matchList.index(max(matchList))
	
	return final_match_id

# Video feed
def main() -> None:
	desList = find_des(images)	
	#  camera feed
	cap = cv2.VideoCapture(0)
	while(1):
		success, img_cam = cap.read()
		img_ori = img_cam.copy() # RGB視訊畫面
		img_cam = cv2.cvtColor(img_cam, cv2.COLOR_BGR2GRAY) # 轉視訊畫面呈灰階
		# Classify images
		id = find_matches(img_cam, desList)
		if id != -1:
			cv2.putText(img_ori, classNames[id], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
			cv2.imshow('Image', img_ori)
			if cv2.waitKey(1) == ord('q'):
			    break


if __name__ == '__main__':
    main()
	