import numpy as np #xử lý mảng số học
import matplotlib.pyplot as plt 
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd #đọc file dữ liệu
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

#Đường dẫn tới tập dữ liệu và file csv
path = "Dataset" #đường dẫn tới thư mục chứa các hình ảnh
label_file = "signnames.csv" # đường dẫn tới filr csv chứa các nhãn
batch_size = 32 #kích thước batch trong quá trình huấn luyện
epochs = 10 # số vòng lặp huấn luyện
imageDimesions = (32,32,3) #kích thước đầu vào của hình ảnh
testRatio = 0.2 #tỉ lệ dữ liệu kiểm thử
validationRatio = 0.2 # tỉ lệ dữ liệu xác thực

count = 0 #đếm số lượng thư mục con
images = [] #danh sách lưu trữ hình ảnh đã đọc
classNo = []  #danh sách nhãn tương ứng với từng hình ảnh
myList = os.listdir(path) #lấy danh sách tất cả thư mục con trong thư mục dataset
print("Total classes Detected: ", len(myList))

#len(mylist): đếm số lượng thư mục(lớp) phát hiện được

#Đọc và lưu ảnh vào mảng
for x in range (0, len(myList)):
    myPicList = os.listdir(path + "/" + str(count)) # lấy danh sách các ảnh trong từng thư mục qua mỗi vòng lặp
    #duyệt qua từng ảnh trong thư mục hiện tại
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(count) + "/" + y) #y là tên file ảnh
        curImg = cv2.resize(curImg, (32,32))  # (width, height)
        # print(curImg.shape)
        # print("CurImg: ", curImg)
        images.append(curImg) #danh sách chứa tất cả các ảnh
        classNo.append(count) #Danh sách chứa nhãn của hình ảnh hiện tại hay chính là số thứ tự của thư mục hiện tại
    # print(count, end="") #end: giữ các giá trị trên cùng một dòng (không xuống dòng)
    count += 1

images = np.array(images) #chuyển đổ sang mảng numpy
classNo = np.array(classNo)
#tách tập dữ liệu thành tập train (80%) và test 20%
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
#tách tập train thành 2 phần
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

print("Data Shapes")
print("Train",end = "");print(X_train.shape,y_train.shape)
print("Validation",end = "");print(X_validation.shape,y_validation.shape)
print("Test",end = "");print(X_test.shape,y_test.shape)

data = pd.read_csv(label_file)
# print("data shapes", data.shape, type(data))

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img = cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img) #chuyển ảnh màu về ảnh xám
    img = equalize(img) # cân bằng histogram
    img = img /255 # chuẩn hóa về đoạn [0,1]
    return img

#cấu trúc hàm map: map(function, iterable (đối tượng có thể lặp))
#Tạo ra 1 đối tượng ánh xạ mapping để áp dụng cho từng phần tử trong X_train. Sau dó list chuyển kết quả từ map thành danh sách các ảnh đã xử lý. Cuối cùng np.array chuyển đổi danh sách nay thành mảng numpy
X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))
# Kiểm tra một vài ảnh đã tiền xử lý
plt.figure(figsize=(10, 5))
for i in range(40):
    plt.subplot(1, 40, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.axis('off')
plt.show()

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)


dataGen= ImageDataGenerator(width_shift_range=0.1,   
                            height_shift_range=0.1,
                            zoom_range=0.2,  
                            shear_range=0.1,  
                            rotation_range=10)  
dataGen.fit(X_train)
batches= dataGen.flow(X_train,y_train,batch_size=20)
X_batch,y_batch = next(batches)


numofClass = len(myList)
#Chuyển ảnh về dạng one-hot encoding
y_train = to_categorical(y_train, numofClass)
y_validation = to_categorical(y_validation, numofClass)
y_test = to_categorical(y_test, numofClass)

def myModel():
    model = Sequential() #tạo mô hình tuần tự +> các lớp sau nối liền lớp trước phù hợp cho mô hình cnn
    #Các lớp tích chập
    model.add((Conv2D(60, (5,5), input_shape = (32,32,1), activation = "relu")))
    model.add((Conv2D(60, (5, 5), activation = 'relu')))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Conv2D(30, (3, 3), activation = 'relu'))
    model.add(Conv2D(30, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(500, activation = 'relu')) #lớp full connected: densen layer. Có 500 nơ ron kết nối đầy đủ
    model.add(Dropout(0.5)) #thêm dropout để tránh quá khớp
    model.add(Dense(numofClass, activation = 'softmax'))
    model.compile(Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

model = myModel()
print(model.sumary()) #là phương thức của lớp Sequential dùng để: Hiển thị tóm tắt kiến trúc mô hình, thông tin các lớp trong mô hình, số lượng tham số cần huấn luyện, kích thước đầu vào, ra của mỗi lớp
    
history=model.fit_generator(dataGen.flow(X_train,y_train,batch_size=32),steps_per_epoch=len(X_train)//32,epochs=epochs,validation_data=(X_validation,y_validation),shuffle=1) 

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
score =model.evaluate(X_test,y_test,verbose=0)
print('Test Score:',score[0])
print('Test Accuracy:',score[1])
 
model.save("model.h5")
