import tkinter as tk
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


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Nhan dang chu so viet tay')
        self.geometry('520x550')
        self.index = None
        self.image_tk = None
        self.cvs_digit = tk.Canvas(self, width = 421, height = 285, relief = tk.SUNKEN, border = 1)
        self.lbl_ket_qua = tk.Label(self, width = 42, height = 11, relief = tk.SUNKEN, border = 1, font = ('Consolas', 14))

        btn_tao_anh = tk.Button(self, text = 'Tao anh', width = 9, command = self.btn_tao_anh_click)
        btn_nhan_dang = tk.Button(self, text = 'Nhan dang', width = 9, command = self.btn_nhan_dang_click)

        self.cvs_digit.place(x = 5, y = 5)
        self.lbl_ket_qua.place(x = 6, y = 300)
        btn_tao_anh.place(x = 440, y = 6)
        btn_nhan_dang.place(x = 440, y = 41)

        self.cvs_digit.update()

    def btn_tao_anh_click(self):
        self.index = np.random.randint(0, 9999, 150)
        digit_random = np.zeros((10*28, 15*28), dtype = np.uint8)
        for i in range(0, 150):
            m = i // 15
            n = i % 15
            digit_random[m*28:(m+1)*28, n*28:(n+1)*28] = X_test_image[self.index[i]] 
        cv2.imwrite('digit_random.jpg', digit_random)

        image = Image.open('digit_random.jpg') 
        self.image_tk = ImageTk.PhotoImage(image)
        self.cvs_digit.create_image(0, 0, anchor = tk.NW, image = self.image_tk)
        self.cvs_digit.update()
        self.lbl_ket_qua.configure(text = '')


    def btn_nhan_dang_click(self):
        X_test_sample = np.zeros((150,784), dtype = np.float32)
        for i in range(0, 150):
            X_test_sample[i] = X_test[self.index[i]] 

        prediction = model.predict(X_test_sample)
        s = ''
        for i in range(0, 150):
            ket_qua = np.argmax(prediction[i])
            s = s + str(ket_qua) + ' '
            if (i+1) % 15 == 0:
                s = s + '\n'

        self.lbl_ket_qua.configure(text = s)


if __name__ == "__main__":
    app = App()
    app.mainloop()
