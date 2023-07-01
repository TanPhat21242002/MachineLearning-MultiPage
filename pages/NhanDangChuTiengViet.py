import streamlit as st
import numpy as np
import os
import cv2
# import our model, different layers and activation function 
from keras.layers import Input, Conv2D, MaxPool2D,  Activation, Add, BatchNormalization, LSTM, Lambda, Bidirectional,  Dense
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K

# Input
inputs = Input(shape=(118,2167,1))
 
# layer 1
model = Conv2D(64, (3,3), padding='same')(inputs)
model = MaxPool2D(pool_size=3, strides=3)(model)
model = Activation('relu')(model)

# layer 2
model = Conv2D(128, (3,3), padding='same')(model)
model = MaxPool2D(pool_size=3, strides=3)(model)
model = Activation('relu')(model)

# layer 3
model = Conv2D(256, (3,3), padding='same')(model)
model = BatchNormalization()(model)
model = Activation('relu')(model)
x = model

# layer 4
model = Conv2D(256, (3,3), padding='same')(model)
model = BatchNormalization()(model)
model = Add()([model, x])
model = Activation('relu')(model)

# layer 5
model = Conv2D(512, (3,3), padding='same')(model)
model = BatchNormalization()(model)
model = Activation('relu')(model)
x2 = model

# layer 6
model = Conv2D(512, (3,3), padding='same')(model)
model = BatchNormalization()(model)
model = Add()([model, x2])
model = Activation('relu')(model)

# layer 7
model = Conv2D(1024, (3,3), padding='same')(model)
model = BatchNormalization()(model)
model = MaxPool2D(pool_size=(3, 1))(model)
model = Activation('relu')(model)
model = MaxPool2D(pool_size=(3, 1))(model)
 
# remove the first dimension(1, 31, 512) to (31, 512) 
squeezed = Lambda(lambda x: K.squeeze(x, 1))(model)
 
# RNN
blstm_1 = Bidirectional(LSTM(512, return_sequences=True, dropout = 0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(512, return_sequences=True, dropout = 0.2))(blstm_1)

# this is our softmax character proprobility with timesteps 
outputs = Dense(141, activation = 'softmax')(blstm_2)
# model to be used at test time
act_model = Model(inputs, outputs)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

# Demo
st.title('Nhan dien chu tieng Viet')
upload_file = st.file_uploader('Chon File')
if upload_file is not None:
    # Load char list
    string =" #'()+,-./0123456789:ABCDEFGHIJKLMNOPQRSTUVWXYabcdeghiklmnopqrstuvwxyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ"
    char_list = [*string]
    #Demo img
    demo_img = []
    demo_txt = []
    demo_input_length = []
    demo_label_length = []
    demo_orig_txt = []
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    # preprocess
    img = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    height, width = img.shape
    img = cv2.resize(img,(2167,118))
    
    img = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    img = np.expand_dims(img , axis = 2)
    img = img/255.
    demo_img.append(img)
    demo_img = np.array(demo_img)
    # load the saved best model weights
    act_model.load_weights(os.path.join('./checkpoint_weights.hdf5'))
    prediction = act_model.predict(demo_img)
    # use CTC decoder
    decode = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1], greedy=True)[0][0])
    # Display
    st.subheader("Hinh anh:")
    st.image(upload_file)
    st.subheader("Sau khi Scan")
    st.image(img)
    st.write('Chuoi ket qua: ')
    # predict
    # see the results
    i = 0
    for l in decode:
        print("Dong chu ket qua = ", end = '')
        predict = ""
        for j in l:  
            if int(j) != -1:
                predict += char_list[int(j)]
        st.write(predict)
        i+=1
