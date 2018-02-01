"""
Deep Learning with tensorflow (Intro - 1)
Number puzzle solving using Neural Network
wai yan nyein naing

Introduction to Tensorflow & Deep Learning
Linear Regression

"""


import tensorflow as tf


#parameter intialization
W = tf.Variable([.3],tf.float32)                  #intialize as random    
b = tf.Variable([-.3],tf.float32)                 #intialize as random 

#Input / Target
x = tf.placeholder(tf.float32)             #input vector
y = tf.placeholder(tf.float32)             #output vector

#predict model
predict = (W*x) + b                        #Linear algebra model
           
#Calculate error & loss function
square_error = tf.square(predict - y)
loss = tf.reduce_sum(square_error)     

#Calculate the optimizer (Using Gradient Descent)
optimizer = tf.train.GradientDescentOptimizer(0.0001)           # learning rate = 0.01
train = optimizer.minimize(loss)

#Session and Intializer
init = tf.global_variables_initializer()  #important (need to initialize of all input data)
sess = tf.Session()
sess.run(init)

#Training
for i in range (10000):                  
        sess.run(train,{x:[4,9,15],y:[2,8,15]})
        print("error rate is %s" % (sess.run(loss,{x:[4,9,15],y:[2,8,15]})))
        print("updated W is %s and updated b is %s " % (sess.run(W),sess.run(b)))
    
#Run Prediction
print("Predict output is %s" % sess.run(predict,{x:[22]}))
