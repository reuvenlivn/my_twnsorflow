# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 18:11:03 2019

@author: reuve
"""
#2. Now we can start using TensorFlow. Don’t forget to import it before use:

import tensorflow as tf

tf.reset_default_graph()

#3. Create two constant nodes (node1,node2) with the values 5 and 11
node1 = tf.constant(5) 
node2 = tf.constant(11)

#4. Add the two nodes and put the value in ndoe3

node3 = node1+node2

#5. Create a TensorFlow Session and run it to produce the answer
sess = tf.Session() 
res1 = sess.run(node3)
print ('node3={}'.format(res1))

#6. Create two placeholders (X,Y), add them using a feed_dict in the session run function
x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)
z = x + y
sess = tf.Session() 
print ('z={}'.format(sess.run(z, feed_dict= {x:10.0,y:16.0})))

#
#7. Create two variables (W,b) and initialize their values to 7,8, add them and run a session,
#don’t forget to initialize all variables before running the session
w=tf.get_variable("w1",initializer=tf.constant(7))
b=tf.get_variable("b1",initializer=tf.constant(8))

#8. Please write three lines about the difference between a placeholder and a variable.
sess = tf.Session() 
sess.run(tf.global_variables_initializer())
res2 = sess.run(w+b)
print ('res2={}'.format(res2))

#9. Build a node called linear_model, with the calculation: Wx+b
linear_model=w*x+b

#10. Run it with the value for x of [1,2,3,4]. Hint: use feed_dict
sess.run(linear_model,feed_dict={x:[1,2,3,4]})
#
#11. Add a placeholder y, and then add a node containing the squred difference between
#linear_model and y. (Hint: simple algebra and the function tf.square)
sqr_diff=tf.square(y-linear_model)
#
#12. Add a node called loss with the sum of all these squared values using reduce_sum
sess.run(sqr_diff,feed_dict={x:[1,2,3,4],y:[6,9,10,12]})
loss=tf.reduce_sum(sqr_diff)

#13. Run the session and find the loss value with y = [ 0 ,- 1 ,- 2 ,- 3 ] and x = [ 1 , 2 , 3 , 4 ]
aaa = sess.run(loss,feed_dict={y:[0,-1,-2,-3],x:[1,2,3,4]})
#
#14. Assign W = -1 and b = 1 and run the session again and compare the loss values
sess.run(tf.assign(w,-1))
sess.run(tf.assign(b,1))

bbb = sess.run(loss,feed_dict={y:[0,-1,-2,-3],x:[1,2,3,4]})


#15. Build a gradient descent optimizer

tf.reset_default_graph()
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
w2 = tf.get_variable("w2",initializer=tf.constant(22.0,dtype=tf.float32),dtype=tf.float32)
b2 = tf.get_variable("b2",initializer=tf.constant(24.0,tf.float32),dtype=tf.float32)

sess = tf.Session() 
sess.run(tf.global_variables_initializer())
linear_model = w2 * x + b2
sqr_diff=tf.square(y-linear_model)
loss = tf.reduce_sum(sqr_diff)
optimizer = tf.train.GradientDescentOptimizer(0.01)

mini=optimizer.minimize(loss)

x_set = [1.0, 2.0, 3.0, 4.0]
y_set = [3.0, -1.0, -2.0, -3.0]
print("w2={}, b2={}, loss={}:".format(sess.run(w2), sess.run(b2), sess.run(loss,{x:x_set, y:y_set})))

for i in range(1000):
    sess.run(mini,{x:x_set, y:y_set})
    
print("epoch={}, w={}, b={}, loss={}:".format(i,sess.run(w2), sess.run(b2), sess.run(loss,{x:x_set, y:y_set})))

# quadric equation

tf.reset_default_graph()

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
a = tf.Variable(2.0,name="a",dtype=tf.float32)
b = tf.Variable(4.0, name="b",dtype=tf.float32)
c = tf.Variable(1.0, name="c",dtype=tf.float32)

sess = tf.Session() 
sess.run(tf.initialize_all_variables())

solution = a*tf.square(x) + b*x + c
eq_error = tf.square(y-solution)
loss = tf.reduce_sum(eq_error)

optimizer=tf.train.GradientDescentOptimizer(0.001)
mini=optimizer.minimize(loss)

x_set = [1.0, 0.0, 4.0, -1.0]
y_set = [8.0, 2.0, 86.0, 6.0]
print("a={}, b={}, c={}, loss={}:".format(sess.run(a), sess.run(b), sess.run(c), sess.run(loss,{x:x_set, y:y_set})))

#17. Run the train step 1000 times (use a loop)
for i in range(1000):
    sess.run(mini,{x:x_set, y:y_set})
    
#18. Print the current W, b and loss values after these 1000 iterations of training
print("epoch={},a={}, b={}, c={}, loss={}:".format(i,sess.run(a), sess.run(b), sess.run(c), sess.run(loss,{x:x_set, y:y_set})))

# Add ops to save and restore all the variables.
tf.train.write_graph(sess.graph_def, './save/', 'test1.pb')
saver = tf.train.Saver()
saver.save(sess, './save/test1')

