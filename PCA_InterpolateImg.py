# import scipy
# import scipy.ndimage
# import matplotlib.pyplot as plt 
# import numpy as np 
# from PIL import Image

# # IMPORTING IMAGE USING SCIPY AND TAKING R,G,B COMPONENTS

# a = scipy.ndimage.imread(r"E:\NCR\TestImages\FOrPCA_Demo.jpg")
# a_np = np.array(a)
# a_r = a_np[:,:,0]
# a_g = a_np[:,:,1]
# a_b = a_np[:,:,2]

# def comp_2d(image_2d): # FUNCTION FOR RECONSTRUCTING 2D MATRIX USING PCA
# 	cov_mat = image_2d - np.mean(image_2d , axis = 1)
# 	eig_val, eig_vec = np.linalg.eigh(np.cov(cov_mat)) # USING "eigh", SO THAT PROPRTIES OF HERMITIAN MATRIX CAN BE USED
# 	p = np.size(eig_vec, axis =1)
# 	idx = np.argsort(eig_val)
# 	idx = idx[::-1]
# 	eig_vec = eig_vec[:,idx]
# 	eig_val = eig_val[idx]
# 	numpc = 100 # THIS IS NUMBER OF PRINCIPAL COMPONENTS, YOU CAN CHANGE IT AND SEE RESULTS
# 	if numpc <p or numpc >0:
# 		eig_vec = eig_vec[:, range(numpc)]
# 	score = np.dot(eig_vec.T, cov_mat)
# 	recon = np.dot(eig_vec, score) + np.mean(image_2d, axis = 1).T # SOME NORMALIZATION CAN BE USED TO MAKE IMAGE QUALITY BETTER
# 	recon_img_mat = np.uint8(np.absolute(recon)) # TO CONTROL COMPLEX EIGENVALUES
# 	return recon_img_mat

# a_r_recon, a_g_recon, a_b_recon = comp_2d(a_r), comp_2d(a_g), comp_2d(a_b) # RECONSTRUCTING R,G,B COMPONENTS SEPARATELY
# recon_color_img = np.dstack((a_r_recon, a_g_recon, a_b_recon)) # COMBINING R.G,B COMPONENTS TO PRODUCE COLOR IMAGE
# recon_color_img = Image.fromarray(recon_color_img)
# recon_color_img.show()




# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# import cv2
# from scipy.stats import stats
# import matplotlib.image as mpimg



# img = cv2.cvtColor(cv2.imread(r"E:\NCR\TestImages\rose.jpg"), cv2.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.show()


# #Splitting into channels
# blue,green,red = cv2.split(img)
# # Plotting the images
# fig = plt.figure(figsize = (15, 7.2)) 
# fig.add_subplot(131)
# plt.title("Blue Channel")
# plt.imshow(blue)
# fig.add_subplot(132)
# plt.title("Green Channel")
# plt.imshow(green)
# fig.add_subplot(133)
# plt.title("Red Channel")
# plt.imshow(red)
# plt.show()

# blue_temp_df = pd.DataFrame(data = blue)
# blue_temp_df



# df_blue = blue/255
# df_green = green/255
# df_red = red/255


# pca_b = PCA(n_components=50)
# pca_b.fit(df_blue)
# trans_pca_b = pca_b.transform(df_blue)
# pca_g = PCA(n_components=50)
# pca_g.fit(df_green)
# trans_pca_g = pca_g.transform(df_green)
# pca_r = PCA(n_components=50)
# pca_r.fit(df_red)
# trans_pca_r = pca_r.transform(df_red)


# print(trans_pca_b.shape)
# print(trans_pca_r.shape)
# print(trans_pca_g.shape)



# print(f"Blue Channel : {sum(pca_b.explained_variance_ratio_)}")
# print(f"Green Channel: {sum(pca_g.explained_variance_ratio_)}")
# print(f"Red Channel  : {sum(pca_r.explained_variance_ratio_)}")

# fig = plt.figure(figsize = (15, 7.2)) 
# fig.add_subplot(131)
# plt.title("Blue Channel")
# plt.ylabel('Variation explained')
# plt.xlabel('Eigen Value')
# plt.bar(list(range(1,51)),pca_b.explained_variance_ratio_)
# fig.add_subplot(132)
# plt.title("Green Channel")
# plt.ylabel('Variation explained')
# plt.xlabel('Eigen Value')
# plt.bar(list(range(1,51)),pca_g.explained_variance_ratio_)
# fig.add_subplot(133)
# plt.title("Red Channel")
# plt.ylabel('Variation explained')
# plt.xlabel('Eigen Value')
# plt.bar(list(range(1,51)),pca_r.explained_variance_ratio_)
# plt.show()

# b_arr = pca_b.inverse_transform(trans_pca_b)
# g_arr = pca_g.inverse_transform(trans_pca_g)
# r_arr = pca_r.inverse_transform(trans_pca_r)
# print(b_arr.shape, g_arr.shape, r_arr.shape)


# img_reduced= (cv2.merge((b_arr, g_arr, r_arr)))
# print(img_reduced.shape)


# fig = plt.figure(figsize = (10, 7.2)) 
# fig.add_subplot(121)
# plt.title("Original Image")
# plt.imshow(img)
# fig.add_subplot(122)
# plt.title("Reduced Image")
# plt.imshow(img_reduced)
# plt.show()




# # Import important libraries
# import cv2
# import numpy as np
# import copy
# # Load the image
# imgpath = r"E:\NCR\TestImages\noisy10degline.jpg"
# imgpath2 = r"E:\NCR\TestImages\noisy45degline.jpg"



# def PCA_Image(FilePath):
#     img = cv2.imread(FilePath, 0)
#     img =cv2.resize(img,(600,600))

#     # Calculating the mean columnwise
#     M = np.mean(img.T, axis=1)

#     # Sustracting the mean columnwise
#     C = img - M

#     # Calculating the covariance matrix
#     V = np.cov(C.T)

#     # Computing the eigenvalues and eigenvectors of covarince matrix
#     values, vectors = np.linalg.eig(V)

#     p = np.size(vectors, axis =1)

#     # Sorting the eigen values in ascending order
#     idx = np.argsort(values)
#     idx = idx[::-1]

#     # Sorting eigen vectors
#     vectors = vectors[:,idx]
#     values = values[idx]

#     # PCs used for reconstruction (can be varied)
#     num_PC = len(vectors)

#     # Cutting the PCs
#     if num_PC <p or num_PC >0:
#         vectors = vectors[:, range(num_PC)]

#     # Reconstructing the image with PCs
#     #score = np.dot(vectors.T, C)
#     #constructed_img = np.dot(vectors, score) + M
#     #constructed_img = np.uint8(np.absolute(constructed_img))

#     # Show reconstructed image
#     #cv2.imshow("Reconstructed Image", constructed_img)
#     #cv2.waitKey(0)
#     #cv2.destroyAllWindows()

#     return vectors,M,C

# def Reconstruct(vectors,M,C):
#     # Reconstructing the image with PCs
#     score = np.dot(vectors.T, C)
#     constructed_img = np.dot(vectors, score) + M
#     constructed_img = np.uint8(np.absolute(constructed_img))

#     # Show reconstructed image
#     cv2.imshow("Reconstructed Image", constructed_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# vectors1,M1,C1=PCA_Image(imgpath)
# vectors2,M2,C2=PCA_Image(imgpath2)

# #make sure vectors same length
# MaxLength=min(len(vectors1),len(vectors2))
# vectors1=vectors1[:MaxLength]
# vectors2=vectors2[:MaxLength]

# #average mean 
# res=[]
# for Indexer, Item in enumerate(M1):
#     res.append((M1[Indexer]+M2[Indexer])/2)

# print(len(vectors1))
# print(len(vectors2))

# AverageVector=copy.deepcopy(vectors1)

# for indexer, ArrayElemOuter in enumerate(vectors1):
#     for innerIndex, ArrayElemInner in enumerate(ArrayElemOuter):
#         #will die if dont exist
#         AverageVector[indexer,innerIndex]=(vectors1[indexer,innerIndex]+vectors2[indexer,innerIndex])/2

# AverageM=copy.deepcopy(M1)
# for indexer, ArrayElem in enumerate(M1):
#     AverageM[indexer]=(M1[indexer]+M2[indexer])/2


# AverageC=copy.deepcopy(C1)
# for indexer, ArrayElemOuter in enumerate(C1):
#     for innerIndex, ArrayElem in enumerate(ArrayElemOuter):
#         AverageC[indexer,innerIndex]=(C1[indexer,innerIndex]+C2[indexer,innerIndex])/2


# #Recostruct(AverageVector,M2,AverageC)
# #Reconstruct(AverageVector,AverageM,AverageC)
# #Reconstruct(AverageVector,AverageM,AverageC)

# #Reconstruct(vectors2,M2,C2)
# Reconstruct(vectors1,M1,C1)
# Reconstruct(AverageVector,AverageM,AverageC)
# Reconstruct(vectors2,M1,C2)










# import cv2
# import numpy as np
# import sklearn.datasets, sklearn.decomposition

# imgpath = r"E:\NCR\TestImages\Anything\20210324_125733.jpg"
# img = cv2.imread(imgpath, 0)
# img =cv2.resize(img,(600,600))

# X = img#sklearn.datasets.load_iris().data
# mu = np.mean(img, axis=0)

# pca = sklearn.decomposition.PCA()
# pca.fit(X)

# nComp = 2
# Xhat = np.dot(pca.transform(X)[:,:nComp], pca.components_[:nComp,:])
# Xhat += mu

# print(Xhat[0,])





import matplotlib.image as mplib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import copy

def getPCA(ImageFile):
    img=mplib.imread(ImageFile)
    print(img.shape)
    img_re=np.reshape(img,(img.shape[0],img.shape[1]*3))
    PCA_Object=PCA(400).fit(img_re)
    img_trans=PCA_Object.transform(img_re)


    return PCA_Object,img_trans,img.shape



#
def InversePCA(PCA_Object,img_trans,img_shape):
    img_Inv=PCA_Object.inverse_transform(img_trans)
    img=np.reshape(img_Inv,img_shape)
    plt.axis('off')
    plt.imshow(img.astype('uint8'))
    plt.show()

#Load the image
imgpath = r"E:\NCR\TestImages\noisy45degline.jpg"
imgpath2 = r"E:\NCR\TestImages\noisy10degline.jpg"


PCA_Object1,img_trans1,img_shape1=getPCA(imgpath)
PCA_Object2,img_trans2,img_shape2=getPCA(imgpath2)

Averageimg_trans=copy.deepcopy(img_trans1)
for indexer, ArrayElemOuter in enumerate(img_trans1):
    for innerIndex, ArrayElem in enumerate(ArrayElemOuter):
        Averageimg_trans[indexer,innerIndex]=(img_trans1[indexer,innerIndex]+img_trans2[indexer,innerIndex])/2



InversePCA(PCA_Object1,Averageimg_trans,img_shape1)
InversePCA(PCA_Object2,img_trans2,img_shape2)



# #https://www.section.io/engineering-education/image-compression-using-pca/
# # Importing necessary libraries
# import numpy as np
# import matplotlib.pyplot as plt
# from numpy import linalg as LA
# import os

# dirname = '/content/gdrive/My Drive/Kaggle'
# fileName = 'olivetti_faces.npy'
# faces = np.load(os.path.join(dirname,fileName))
# plt.imshow(faces[1], cmap='gray')
# plt.axis('off')



# avgFace = np.average(faces, axis=0)
# plt.imshow(avgFace, cmap='gray')
# plt.axis('off')



# X = faces
# X = X.reshape((X.shape[0], X.shape[1]**2)) #flattening the image 
# X = X - np.average(X, axis=0) #making it zero centered

# #printing a sample image to show the effect of zero centering
# plt.imshow(X[0].reshape(64,64), cmap='gray')
# plt.axis('off')


# cov_mat = np.cov(X, rowvar = False)

# #now calculate eigen values and eigen vectors for cov_mat
# eigen_values, eigen_vectors  = np.linalg.eig(cov_mat)

# #sort the eigenvalues in descending order
# sorted_index = np.argsort(eigen_values)[::-1]
 
# sorted_eigenvalue = eigen_values[sorted_index]
# #similarly sort the eigenvectors 
# sorted_eigenvectors = eigen_vectors[:,sorted_index]

# n_components = 100
# eigenvector_subset = sorted_eigenvectors[:,0:n_components]
# print(eigenvector_subset.shape)


# fig = plt.figure(figsize=[25,25])
# for i in range(16):
#     if(i%4==0):
#         fig.add_subplot(4,4,i+1)
#         plt.imshow(eigenvector_subset.T[i].reshape(64,64) , cmap= 'gray')
#         plt.axis('off')
#         plt.show()

# x_reduced = np.dot(eigenvector_subset.transpose(),X.transpose()).transpose()
# print(x_reduced.shape)


# # Reconstructing the first image
# temp= np.matmul(eigenvector_subset,x_reduced[1])
# temp.shape
# temp=temp.real
# plt.imshow(temp.reshape(64,64) , cmap= 'gray')
# plt.axis('off')

