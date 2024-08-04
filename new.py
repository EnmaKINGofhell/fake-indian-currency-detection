import numpy as np import cv2 import matplotlib.pyplot as plt import joblib import cvutils import os import sys import tkinter as tk from tkinter import filedialog
# Extracting features from training images def trainproc():
train_imgs1 = cvutils.imlist("train_images\\500") train_imgs2 = cvutils.imlist("train_images\\2000") k = 0 for tr in train_imgs1: pth = tr
# Reading the image out = "train\\500" img = cv2.imread(pth)
# resizing
img = cv2.resize(img, (1200, 512), interpolation=cv2.INTER_LINEAR)
# Denoising image
img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
# Converting to grayscale
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# compute the median of the single channel pixel intensities v = np.median(img) sigma = 0.33
# apply automatic Canny edge detection using the computed median
lower = int(max([ 0, (1.0 - sigma) * v ])) upper = int(min([ 255, (1.0 + sigma) * v ])) img = cv2.Canny(img, lower, upper)
# Extracting features
id1 = img[ 195:195 + 170, 190:190 + 85 ]
id2 = img[ 330:330 + 105, 720:720 + 105 ] id3 = img[ 320:320 + 90, 865:865 + 205 ] id4 = img[ 250:250 + 40, 1120:1120 + 40 ] id5 = img[ 5:5 + 405, 660:660 + 40 ] id6 = img[ 284:284 + 132, 1090:1090 + 90 ]
# Saving the features ids = [ id1, id2, id3, id4, id5, id6 ] out1 = "\\demo" + str(k) + ".jpg" d = 1 for i in ids:
cv2.imwrite(out + "\\id%d" % d + out1, i) d = d + 1
k = k + 1
k = 0 for tr in train_imgs2: pth = tr
# Reading the image out = "train\\2000" img = cv2.imread(pth)
# resizing
img = cv2.resize(img, (1200, 512), interpolation=cv2.INTER_LINEAR)
# Denoising image
img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
# Converting to grayscale
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# compute the median of the single channel pixel intensities v = np.median(img) sigma = 0.33
# apply automatic Canny edge detection using the computed median
lower = int(max([ 0, (1.0 - sigma) * v ])) upper = int(min([ 255, (1.0 + sigma) * v ])) img = cv2.Canny(img, lower, upper)
# Extracting features
id1 = img[ 195:195 + 165, 225:225 + 55 ] id2 = img[ 330:330 + 95, 760:760 + 90 ] id3 = img[ 335:335 + 80, 890:890 + 205 ] id4 = img[ 255:255 + 25, 1105:1105 + 53 ] id5 = img[ 10:10 + 480, 726:726 + 35 ] id6 = img[ 280:280 + 140, 1100:1100 + 75 ]
# Saving the features ids = [ id1, id2, id3, id4, id5, id6 ] out1 = "\\demo" + str(k) + ".jpg" d = 1 for i in ids:
cv2.imwrite(out + "\\id%d" % d + out1, i) d = d + 1
k = k + 1
# Extracting features from test image# Extracting features from test image def testproc(): root = tk.Tk() pth = tk.filedialog.askopenfilename() root.destroy() out1 = "test\\ids1" out2 = "test\\ids2"
# Reading the image img = cv2.imread(pth) cv2.imshow('qq', img) cv2.waitKey(0) cv2.destroyAllWindows()
# resizing
img = cv2.resize(img, (1200, 512), interpolation=cv2.INTER_LINEAR) cv2.imshow('qq', img) cv2.waitKey(0) cv2.destroyAllWindows()
# Denoising image
img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
# Converting to grayscale
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imshow('qq', img) cv2.waitKey(0) cv2.destroyAllWindows()
# compute the median of the single channel pixel intensities
v = np.median(img) sigma = 0.33
# apply automatic Canny edge detection using the computed median
lower = int(max([ 0, (1.0 - sigma) * v ])) upper = int(min([ 255, (1.0 + sigma) * v ])) img = cv2.Canny(img, lower, upper) cv2.imshow('qq', img) cv2.waitKey(0) cv2.destroyAllWindows()
# Extracting features id1 = img[ 195:195 + 170, 190:190 + 85 ] id2 = img[ 330:330 + 105, 720:720 + 105 ] id3 = img[ 320:320 + 90, 865:865 + 205 ] id4 = img[ 250:250 + 40, 1120:1120 + 40 ] id5 = img[ 5:5 + 405, 660:660 + 40 ] id6 = img[ 284:284 + 132, 1090:1090 + 90 ] ids1 = [ id1, id2, id3, id4, id5, id6 ]
# Saving the features
id1 = img[ 195:195 + 165, 225:225 + 55 ] id2 = img[ 330:330 + 95, 760:760 + 90 ] id3 = img[ 335:335 + 80, 890:890 + 205 ] id4 = img[ 255:255 + 25, 1105:1105 + 53 ] id5 = img[ 10:10 + 480, 726:726 + 35 ] id6 = img[ 280:280 + 140, 1100:1100 + 75 ] ids2 = [ id1, id2, id3, id4, id5, id6 ]
d = 1 for i in ids1:
cv2.imwrite(out1 + "\\test%d.jpg" % d, i) d = d + 1
d = 1 for i in ids2: cv2.imwrite(out2 + "\\test%d.jpg" % d, i) d = d + 1
# Displaying th features for i in range(6):
plt.subplot(2, 3, i + 1) plt.imshow(ids1[ i ]) plt.xticks([ ]) plt.yticks([ ]) plt.show()
for i in range(6):
plt.subplot(2, 3, i + 1) plt.imshow(ids2[ i ]) plt.xticks([ ]) plt.yticks([ ]) plt.show()
# Main procedure while True:
x = input("Enter 0 to start extracting features from training images or 1 fortesting the image\n") if x == '0': trainproc() print("Features extracted!!!!!!") os.system("perform-training.py")
break if x == '1': testproc() print("Features extracted!!!!!!") os.system("perform-testing.py")
break if x == 'exit':
print("Exited!!!") break else:
print("Enter correct key") continue
Training :
#For image processing import cv2
# To performing path manipulations import os
# Local Binary Pattern function
from skimage.feature import local_binary_pattern
# For plotting
import matplotlib.pyplot as plt # For array manipulations import numpy as np # For saving histogram values import joblib # Utility Package import cvutils
# Store the path of training images in train_images train_images500 = [] for d in range(6):
i = d+1 ti = cvutils.imlist("train/500/id%i"%i) train_images500.append(ti)
n = len(train_images500[0])
train_images2000 = [] for d in range(6):
i = d+1 ti = cvutils.imlist("train/2000/id%i"%i) train_images2000.append(ti)
n = len(train_images2000[0])
X_test500 = []
X_test2000 = []
# For each image in the training set calculate the LBP histogram
# and update X_test, X_name and y_test for train_image in train_images500: # Read the image
X_temp = []
for i in range(n):
im = cv2.imread(train_image[i])
# Convert to grayscale as LBP works on grayscale image im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
radius = 3
# Number of points to be considered as neighbours no_points = 8 * radius # Uniform LBP is used
lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
# Calculate the histogram n_bins = int(lbp.max() + 1)
hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
X_temp.append(hist)
# Append histogram to X_test X_test500.append(X_temp)
for train_image in train_images2000:
# Read the image X_temp = []
for i in range(n):
im = cv2.imread(train_image[i])
# Convert to grayscale as LBP works on grayscale image im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#im_gray = cv2.resize(im_gray, (100, 100), interpolation=cv2.INTER_LINEAR) radius = 3
# Number of points to be considered as neighbours no_points = 8 * radius # Uniform LBP is used
lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
# Calculate the histogram n_bins = int(lbp.max() + 1)
hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
X_temp.append(hist)
# Append histogram to X_test X_test2000.append(X_temp)
# Dump the  data
joblib.dump((X_test500, X_test2000,n), "lbp.pkl", compress=3)
print("Images are been trained") os.system("preprocessing.py")
Testing :
# For image processing import cv2
# To performing path manipulations import os
# Local Binary Pattern function
from skimage.feature import local_binary_pattern
# For plotting import matplotlib.pyplot as plt # For array manipulations import numpy as np # For saving histogram values import joblib # Utility Package import cvutils
# Displaying the fake result image def fake_img(): pth = "fake.jpg" img = cv2.imread(pth) cv2.imshow('FAKE!!!!', img) cv2.waitKey(0) cv2.destroyAllWindows()
# Displaying the genuine result image def genuine_img(): pth = "genuine.jpg" img = cv2.imread(pth) cv2.imshow('GENUINE!!!!', img) cv2.waitKey(0) cv2.destroyAllWindows()
# Load the List for storing the LBP Histograms, address of images and the corresponding label
X_test500, X_test2000, n = joblib.load("lbp.pkl")
# Store the path of testing images in test_images test_images500 = cvutils.imlist("test/ids1") test_images2000 = cvutils.imlist("test/ids2")
# Dict containing scores results_all500 = {} results_all2000 = {}
# total scores tot500 = 0 tot2000 = 0
for i in range(6):
# Read the image
im = cv2.imread(test_images500[ i ], 0)
radius = 3
# Number of points to be considered as neighbourers no_points = 8 * radius # Uniform LBP is used
lbp = local_binary_pattern(im, no_points, radius, method='uniform')
# Calculate the histogram n_bins = int(lbp.max() + 1)
hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
# Display the query image
results = [ ] scores = 0
# For each image in the training dataset
# Calculate the chi-squared distance and the sort the values for index, x in enumerate(X_test500[ i ]):
score = cv2.compareHist(np.array(x, dtype=np.float32), np.array(hist,
dtype=np.float32), cv2.HISTCMP_CHISQR)
#        print(score) scores += score
scores = scores / 3 results.append(round(scores, 3)) results_all500[ "id%i" % i ] = results tot500 += results[ 0 ]
#   print(results_all)
for i in range(6):
# Read the image
im = cv2.imread(test_images2000[ i ], 0)
radius = 3
# Number of points to be considered as neighbourers no_points = 8 * radius # Uniform LBP is used
lbp = local_binary_pattern(im, no_points, radius, method='uniform')
# Calculate the histogram n_bins = int(lbp.max() + 1)
hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
# hist = x[:, 1]/sum(x[:, 1]) # Display the query image
results = [ ] scores = 0
# For each image in the training dataset
# Calculate the chi-squared distance and the sort the values for index, x in enumerate(X_test2000[ i ]):
score = cv2.compareHist(np.array(x, dtype=np.float32), np.array(hist,
dtype=np.float32), cv2.HISTCMP_CHISQR)
#        print(score) scores += score
scores = scores / 3 results.append(round(scores, 3)) results_all2000[ "id%i" % i ] = results tot2000 += results[ 0 ] #   print(results_all)
buff1 = 0 buff2 = 0
while True:
if results_all500[ 'id0' ][ 0 ] > 0.006: break if results_all500[ 'id1' ][ 0 ] > 0.02: break if results_all500[ 'id2' ][ 0 ] > 0.008: break if results_all500[ 'id3' ][ 0 ] > 0.06: break if results_all500[ 'id4' ][ 0 ] > 0.02: break if results_all500[ 'id5' ][ 0 ] > 0.07:
break else: buff1 = 1 break
while True:
if results_all2000[ 'id0' ][ 0 ] > 0.006: break if results_all2000[ 'id1' ][ 0 ] > 0.02: break if results_all2000[ 'id2' ][ 0 ] > 0.008: break if results_all2000[ 'id3' ][ 0 ] > 0.06: break if results_all2000[ 'id4' ][ 0 ] > 0.02: break if results_all2000[ 'id5' ][ 0 ] > 0.07:
break else: buff2 = 1 break
if buff1 == 0 and buff2 == 0:
fake_img() print("1")
elif buff1 and buff2: print("2") if tot2000 > tot500:
print("500 CURRENCY NOTE") else:
print("2000 CURRENCY NOTE")
genuine_img() elif buff1: print("3")
print("500 CURRENCY NOTE") genuine_img() else:
print("4")
print("2000 CURRENCY NOTE") genuine_img() os.system("preprocessing.py")
