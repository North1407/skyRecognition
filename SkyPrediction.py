import numpy as np
import cv2
import math
import pandas as pd
from  tkinter import messagebox as mbox
from tkinter.filedialog import *
from PIL import ImageTk, Image
from tkinter import filedialog,Tk,Button,simpledialog,Label
import matplotlib.image as mpimg
import os, sys
from imutils import paths
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import glob



def convert_rgb_to_bgr(img):
    # Chuyển đổi ảnh từ chế độ RGB sang BGR
    img_bgr = img[:, :, ::-1]
    return img_bgr

def resize_image(img, size):
    # Chuyển đổi ảnh thành mảng NumPy
    img_array = np.array(img)

    # Resize ảnh với kích thước cụ thể
    img_resized = Image.fromarray(img_array)
    img_resized = img_resized.resize(size)

    return img_resized

def preImg(img):
        
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        img = resize_image(img,(224, 224))
        
        
        return img

def loadData():
    data = []
    labels = []
    fileName =[]
    for path in glob.glob('skyDataset/*/**.jpg'):
            _, brand, fn = path.split('\\')
            # _ : datasetSky2, brand : sunset, fn : fileName
            # tiền xử lý Dl
            #dùng cv2 sẽ ra định dạng BGR thay vì RGB
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img,dsize=(224,224))
            # img = Image.open(path)
            fileName.append(path)
            # trich rút đặc trung
            features = featueVector(img)
            # gán nhãn cho data
            data.append(features)
            labels.append(brand)

    return data,labels,fileName


def writeData(str,data,target,fileName):

    dataFrame = []
    for i in range(len(data)):
        arr = []
    # arr.append(data[i])
        for value in data[i]:
            arr.append(value)
    
        arr.append(target[i])
        arr.append(fileName[i])
        dataFrame.append(arr)

    #print(dataFrame[0:10])

    df = pd.DataFrame(dataFrame,columns= ['Trung bình B', 'Trung bình G',
    'Trung bình R',"Độ lệch chuẩn B","Độ lệch chuẩn G","Độ lệch chuẩn R","Nhãn",'URL'])
    df.to_csv (str,index = False, header=True,encoding='utf_8_sig')
    print("Ghi dữ liệu thành công!")
    

def readData(str):
    data = pd.read_csv(str)
    data = data[1:]  # Xóa header
    data = np.array(data)  # Chuyển đổi thành ma trận numpy
    np.random.shuffle(data)  # Xáo trộn dữ liệu

    # Chia tập train và test theo tỉ lệ 80:20
    trainSet, testSet = train_test_split(data, test_size=0.2, random_state=42)

    return trainSet, testSet


#tính toán trung bình màu và độ lệch chuẩn của ảnh
def featueVector(img):
  
        img = np.array(img)
        B = img[:, :, 0]
        G = img[:, :, 1]
        R = img[:, :, 2]
       
        tb1 =np.sum(B)/(img.shape[0]*img.shape[1])
        tb2 =np.sum(G)/(img.shape[0]*img.shape[1])
        tb3 =np.sum(R)/(img.shape[0]*img.shape[1])

        feature = []
        feature.append(tb1)
        feature.append(tb2)
        feature.append(tb3)
        sum=0
        sum2=0
        sum1=0

        G=G.flatten()

        for i in G:
                sum += ((i-tb2)*(i-tb2))

        SSG= sum/(img.shape[0]*img.shape[1]-1)

        B=B.flatten()

        for i in B:
                sum1 += ((i-tb1)*(i-tb1))
        SSB= sum1/(img.shape[0]*img.shape[1]-1)       
        R=R.flatten()
        for i in R:
                sum2 += ((i-tb3)*(i-tb3))
        SSR= sum2/(img.shape[0]*img.shape[1]-1)

        feature.append(np.sqrt(SSB))
        feature.append(np.sqrt(SSG))
        feature.append(np.sqrt(SSR))

        return feature


def calcDistancs(pointA, pointB, numOfFeature=6):
    tmp = 0
    for i in range(numOfFeature):
        tmp += (float(pointA[i]) - float(pointB[i])) ** 2
    return math.sqrt(tmp)

def kNearestNeighbor(trainSet, point, k):
    distances = []
    for item in trainSet:
        distances.append({
            "label": item[-2], # get label
            "value": calcDistancs(item, point), # get valude dist
            "fileName" : item[-1]
        }) 
    # thuoc list cac dict gom key = label, value = dist
    distances.sort(key=lambda x: x["value"]) # sort Asc by value

    labels = [item["label"] for item in distances] # duyet list da sort lay ra k label
    fileNames = [item["fileName"] for item in distances]
    dist = [item["value"] for item in distances]
    # labels.insert(0,labels[0])
    print(dist[:k])
    return labels[:k],fileNames[:k] # return k point nearest

def findMostOccur(arrLabel): # arr is list k  first labels
   
    print("label  ",arrLabel)
    labels = set(arrLabel) # filter key only == set in java
    print(' ',labels)
    ans = ""
    maxOccur = 0
    for label in labels:
        num = arrLabel.count(label) # dem so phan tu value == label in arr
        print(label , " = ", num)
        if num > maxOccur:
            maxOccur = num
            ans = label
    return ans

#huấn luyện và kiểm thử thuật toán knn
def fit(img):
        try:
            train,test= readData(img)
            numOfRightAnwser = 0
            
            for item in test:
               
                knn,knn1 = kNearestNeighbor(train, item, 5)
                print(type(item))
                answer = findMostOccur(knn)
           
                numOfRightAnwser += item[-2] == answer
                print(item[-1]) # get last index arr
                print("label: {} -> predicted: {}".format(item[-2], answer))
                    
            print("Accuracy", numOfRightAnwser/len(test)*100)
            mbox.showinfo( "Accuracy", numOfRightAnwser/len(test)*100)
        except :
            mbox.showerror( "Message", "lỗi đường dẫn!")

#dự đoán và đưa ra nhãn kết quả + link ảnh giống nhất
def predict(img,train):
#   img = cv2.imread(anh,cv2.IMREAD_UNCHANGED)
    global anh_like 
    #tiền xử lí
    img = preImg(img)

    featue = featueVector(img)
    print("Đặc trưng của hình: ",featue)
    
    pred_label,img_path = kNearestNeighbor(train,featue,5)
    answer = findMostOccur(pred_label)
    print(answer,img_path)
    anh_like = img_path[0]
    return answer,img_path[0]

#trích rút đặc trưng từ bộ dữ liệu
def saveFeature():
      
    data,target,fileName = loadData()
    choose =  mbox.askquestion("Question" , "Lưu đặc trưng ??")
    if choose == 'yes':
            nameFile =  simpledialog.askstring(title="Lưu file",prompt="Hãy nhập tên: ")
            
            if nameFile is not None and nameFile != ''  :
                    nameFile = nameFile+".csv"
                    
                    writeData(nameFile,data,target,fileName)
                    fit(nameFile)
                    # mbox.showinfo( "Message", "Thành công!!!")
            else:
                    mbox.showerror( "Message", "Nhập lại tên file")
        
        
def findkNeast():
        gray = plt.imread(anh_like, cv2.IMREAD_UNCHANGED)


        import matplotlib.image as mpimg
        plt.figure(figsize = (12, 4)) # chia khoảng cho 2 ảnh
        plt.subplot(1, 2, 1) # tạo 2 ô gồm 1 dòng 2 cột ảnh 1 ở cột 1
        img = mpimg.imread(path_img_predict)
        img = resize_image(img,(224,224))
        imgplot = plt.imshow(img)

        plt.title("Ảnh được dự đoán : ")
        plt.subplot(1, 2, 2)
        gray = mpimg.imread(anh_like)
        plt.imshow(gray)
        plt.title("Ảnh tương đồng trong bộ dữ liệu : ")
        plt.show()   


def open_img():
    global panel
     # Select path  img
    global path_img_predict
    
    # Select path  img
    x = openfilename()

    path_img_predict = x
    if x:
        print('x= ',x)
        
        # mở đường dẫn ảnh
        
        img = Image.open(x)
        print('img',img)
        
        # chỉnh kích thước và tăng chất lượng ảnh 
        img = img.resize((250, 250), Image.ANTIALIAS)
        
        # đưa hình vào widgets
        img = ImageTk.PhotoImage(img)
        
        # cập nhật nhãn cho ảnh mới
        if panel:
            panel.config(image=img)
            panel.image = img
        else:
            panel = Label(root, image=img)
            panel.image = img
            panel.pack(side="top", fill="both", expand=True)
        
        # doc anh 
        image = mpimg.imread(x)
        # image = Image.open(x)
        
        train,test= readData("export_dataframe.csv")
        arrData = []
        arrData = np.array(train)
        arrData = np.append(arrData,test,axis = 0)
        print("*"*30)
        print(arrData.shape)
        #img_Like chỉ để chứa, ngăn k lỗi
        str,img_Like = predict(image,arrData)
        res = Label(root,text="Thời điểm bầu trời dự đoán - "+ str +"        " )
        res.pack(side="top", fill="both", expand=True)
        res.place(x=150, y=190)

def markCenter(root):
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height //2)
    print(root.winfo_screenwidth())
    print(root.winfo_screenheight())
    print(width,height)
    print(x,y)
    root.geometry('{}x{}+{}+{}'.format(width,height,x,y))

def openfilename():
      
    # mở ra đường dẫn ảnh cần đọc
    filename = filedialog.askopenfilename(title ='"Open')
    return filename




#btn = Button(root, text ='open image', command = open_img).grid( row = 1, columnspan = 4)
# *********************************************
root = Tk()
root.title("Phân loại ảnh bầu trời")
root.resizable(height=True,width=True)
root.minsize(height=500,width=500)

# Button Huấn luyện mô hình
btnfeature = Button(root, text="Huấn luyện mô hình", command=saveFeature).pack(side="top", fill="both", expand=True)

# Button dự đoán ảnh
btnpredict = Button(root, text="Dự đoán ảnh", command=open_img).pack(side="top", fill="both", expand=True)

# Label hiển thị ảnh
panel = Label(root)
panel.pack(side="bottom", fill="both", expand=True)

# Button hiện ảnh tương đồng
# btnshow = Button(root, text="Hiện ảnh tương đồng", command=findkNeast).pack(side="bottom", fill="both", expand=True)



markCenter(root)
root.mainloop()

