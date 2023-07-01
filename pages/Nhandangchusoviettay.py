import streamlit as st
from PIL import ImageTk, Image
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import model_from_json 
from keras.optimizers import SGD 
import cv2

model_architecture = "digit_config.json"
model_weights = "digit_weight.h5"
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights) 

optim = SGD()
model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"]) 

mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_test_image = X_test

RESHAPED = 784

X_test = X_test.reshape(10000, RESHAPED)
X_test = X_test.astype('float32')

#normalize in [0,1]
X_test /= 255
index = np.random.randint(0, 9999, 150)
def create_image():

    digit_random = np.zeros((10*28, 15*28), dtype = np.uint8)
    for i in range(0, 150):
        m = i // 15
        n = i % 15
        digit_random[m*28:(m+1)*28, n*28:(n+1)*28] = X_test_image[index[i]] 
    cv2.imwrite('digit_random.jpg', digit_random)
    return Image.open('digit_random.jpg')

def predict(image):
    X_test_sample = np.zeros((150,784), dtype = np.float32)
    for i in range(0, 150):
        X_test_sample[i] = X_test[index[i]] 
    prediction = model.predict(X_test_sample)
    s = ''
    for i in range(0, 150):
        ket_qua = np.argmax(prediction[i])
        s = s + str(ket_qua) + ' '
        if (i+1) % 15 == 0:
            s = s + '\n'
    return s

def main():
    st.set_page_config(page_title='Nhan dang chu so viet tay', page_icon='🔢')
    st.title('Nhan dang chu so viet tay')
    
    image = create_image()
    st.image(image, width=421, caption='Ảnh đầu vào')
    st.button('Tạo ảnh mới', key='create_image')

    if st.button('Nhận dạng', key='predict'):
        result = predict(image)
        st.text(result)

if __name__ == '__main__':
    main()
