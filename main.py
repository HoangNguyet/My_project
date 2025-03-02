import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

#Đường dẫn tới tập dữ liệu và file csv
data_dir = "Dataset" #đường dẫn tới thư mục chứa các hình ảnh
label_file = "signnames.csv" # đường dẫn tới filr csv chứa các nhãn
batch_size = 32 #kích thước batch trong quá trình huấn luyện
epochs = 10 # số vòng lặp huấn luyện
imageDimesions = (32,32,3) #kích thước đầu vào của hình ảnh
testRatio = 0.2 #tỉ lệ dữ liệu kiểm thử
validationRatio = 0.2 # tỉ lệ dữ liệu xác thực

count = 0 #đếm số lượng thư mục con
images = [] #danh sách lưu trữ hình ảnh đã đọc
classNo = []  #danh sách nhãn tương ứng với từng hình ảnh
myList = os.listdir(data_dir) #lấy danh sách tất cả thư mục con trong thư mục dataset
print("Total classes Detected: ", len(myList))

#len(mylist): đếm số lượng thư mục(lớp) phát hiện được

for x in range (0, len(myList)):
    myPicList = os.listdir(data_dir + "/" + str(count)) # lấy danh sách các ảnh trong từng thư mục qua mỗi vòng lặp
    #duyệt qua từng ảnh trong danh sách picList



 

