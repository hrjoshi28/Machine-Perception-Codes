
#
#  Author: Harshit Joshi
#  Date: 8/4/2017
#
#  Content-based image classification :: Randomized PCA + KNeighborsClassifier  
#
#  

import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

counter=0

def clean_img(img) :
	#prepreocess img : normalize and resize
	try:
		STANDARD_SIZE = (int(img.shape[1]/2), int(img.shape[0]/2))
		resized_img = cv2.resize(img,STANDARD_SIZE, interpolation = cv2.INTER_AREA) # inter_area for shrinking
		cv2.normalize(resized_img,resized_img, 0, 255, cv2.NORM_MINMAX)   #normType= MINMAX #can try other norms
		## ! Is normalizing any good ? 
	except:
		print("error")	
	return resized_img 

def pca_features(X,n_comp) :  # n_comp= no. of components
	# img3=X[3].clone
	# DEEP COPY SHALLOW COPY ???? google !
	
	#X=np.copy(X)  #COPY DEEP ?
	X_flat=[]
	#flatten
	n_img=len(X)

	# for i in range(n_img):
	#  	X_flat.append(X[i].flatten())
	# #!!! Try writing above lines using numpy index in one line(w/o looping) !!! X.reshape(len(X),-1) ?
	# X_flat=np.array(X_flat)
	
	X_flat=X.reshape(len(X),-1)
	print("Shape of X in pca_func",X.shape)
	print("Shape of X_flat in pca_func",X_flat.shape)
	print("\n")

	pca = PCA(n_components=n_comp, svd_solver='randomized') #initialize
	X_new=pca.fit_transform(X_flat)
	return X_new

def stack_features(X) :
	# Stack PCA; VBOW	 n other features ...
	HelloWorld=1

def load_data(path_list) :
	# Function than return numpy arrays : X_train and y  (ready for ML model)
	X_train=[]
	y_train=[]
	global counter
	no_of_folder=len(path_list)
	try:
		for y_label in range(no_of_folder):
			for filename in glob.glob(path_list[y_label]):
				img=cv2.imread(filename)
				
				img_cleaned=clean_img(img)
				if counter==1:            
 				# 	cv2.imshow('img',img)    # ERROR --> GLib-GIO-Message - COZ of cv.imshow ERROR >> REPORT BUG
					plt.subplot(121),plt.imshow(img)
					plt.title('Input Image')
					plt.subplot(122),plt.imshow(img_cleaned)
					plt.title('Processed Image')
					plt.show()

				X_train.append(img_cleaned)
				y_train.append(y_label)
				#print(path_list[i])
				counter+=1
	except:
			print("error")
	
	X_train = np.array(X_train)
	y=np.array(y_train)
	
	#y=y.ravel()    # MxN--(3D) to (M.N)--3d  or  < y.reshape(-1,3) ???? >  
	# ravel flatten ???

	return X_train,y



if __name__ == "__main__":
	# Make Pathlist
	path_list=[]
	path_list.append("crowd/*.jpg")
	path_list.append("pitch/*.jpg")
	path_list.append("Boundary/*.jpg")
	path_list.append("SKy/*.jpg")
	path_list.append("batsman/*.jpg")
	path_list.append("Ground/*.jpg")

	#Load your Data
	X_train,y = load_data(path_list)
	

	print (X_train.shape , y.shape)      
	print("\n")
	
	
	# #############################...... now running PCA+KNN(content based classification) ..........
	X_pca=pca_features(X_train,n_comp=5)
	
	print("X_train shape=",X_train.shape)
	print("X_pca shape=",X_pca.shape)
	
	print("\nvalue of counter=%d (exhaustive debugging :p)\n"%counter)
	print("Final Arg to ml model \n ","X_pca shape=",X_pca.shape)
	print("y.shape",y.shape,"\n")
	

	X_train, X_test, y_train, y_test = train_test_split(X_pca, y, random_state=7)
	#ML-model
	knn = KNeighborsClassifier()
	knn.fit(X_train, y_train)
	# #print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
	print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))



# ##########......... Experiment with other Classifers...   ######################

# 	from sklearn.pipeline import Pipeline
# 	from sklearn.svm import LinearSVC
# 	from sklearn.linear_model import LogisticRegression
# 	from sklearn.naive_bayes import GaussianNB
# 	from sklearn.preprocessing import StandardScaler
# 	X_flat=X_train.reshape(len(X_train),-1)
# 	X_train, X_test, y_train, y_test = train_test_split(X_flat, y, random_state=7)
# 	pipe_knn= Pipeline([ 
# 		##('scl',StandardScaler()), # is it helping  ???
# 		('pca',PCA(n_components=5)),
# 		('clf',KNeighborsClassifier())   # change classifier to experiment
# 		])

# 	pipe_knn.fit(X_train,y_train)
# 	print('Test Accuracy: %.3f'%pipe_knn.score(X_test,y_test))
# ############
