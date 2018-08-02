import tensorflow as tf
import numpy as np 
import xlrd
import pandas as pd

# read xls to set the dataset
print("start read xlsx")
filename = 'arrival.xlsx'
book = xlrd.open_workbook(filename)
sheel_1 = book.sheet_by_index(0)
ROWS = sheel_1.nrows
#COLS = sheel_1.ncols
COLS = 8
datalist = []
Loacation = sheel_1.cell_value(rowx=0,colx=COLS)
for i in range(1,ROWS):
    datalist.append(sheel_1.cell_value(rowx=i,colx=COLS))
#datalist = [datalist]
#datalist = [int(i) for i in datalist]
print("...take data OK...")
print(Loacation)
#print(datalist)

data = datalist
#TestData
TestData = []
for i in range(12):
    TestData = np.append(TestData,data[-24+i:-24+i+12])
print(len(TestData))
testdata = TestData.reshape(12,12)

#TrainData
TrainData = np.array(data[:-12])
X=[]
Y=[]
for i in range(0,len(TrainData)-12):
    Y=np.append(Y,TrainData[i+12])
    X=np.append(X,[TrainData[i:i+12]])
x_train=X.reshape(180, 12).astype('float32')
y_train = Y.reshape(180,1).astype('float32')
print("x_train")
print(x_train)
print("y_train")
print(y_train)

tf.reset_default_graph()

num_periods = 180
input_n = 12
hidden = 10
output_n = 1
LearningRate = 0.001

x = tf.placeholder(tf.float32,[num_periods,input_n])
y = tf.placeholder(tf.float32,[num_periods,output_n])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden ,activation=tf.nn.relu)
rnn_output ,states = tf.nn.dynamic_rnn(basic_cell, x, dtype=tf.float32)

stack_rnn_outputs = tf.reshape(rnn_output,[-1,hidden])
stack_outputs = tf.layers.dense(stack_rnn_outputs,output_n)
outputs = tf.reshape(stack_outputs,[-1,num_periods,output_n])

loss = tf.reduce_sum(tf.square(outputs-y))
optimizer = tf.train.AdamOptimizer(learning_rate = LearningRate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

epochs = 1000

with tf.Session() as sess:
    init.run()
    for ep in range(epochs):
        #sess.run(init)
        sess.run(training_op,feed_dict={x:x_train ,y:y_train })
        if ep %10 == 0:
            mes = loss.eval(feed_dict={x:x_train ,y:y_train })
            #mes = loss.eval()
    y_pred = sess.run(outputs,feed_dict={x:testdata})
    print(y_pred)




