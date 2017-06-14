import tensorflow as tf

gt = tf.constant([[[[0], [0], [0], [0]],
                   [[0], [0], [0], [0]],
                   [[0], [1], [1], [0]],
                   [[0], [1], [0], [0]]],
                  [[[0], [0], [0], [0]],
                   [[0], [0], [0], [0]],
                   [[0], [1], [1], [0]],
                   [[0], [1], [0], [0]]]
                  ])

reduceSum = tf.reduce_sum(gt, axis=-1)
zero = tf.constant(0, dtype=tf.int32)
where = tf.not_equal(reduceSum, zero)
indices = tf.where(where)
print indices

#num_row, num_clos = indices.get_shape().as_list()
num_pos = tf.to_int32(tf.shape(indices)[0])
print num_pos
num_row = num_pos.get_shape().as_list()
#num_pos = num_pos.as_list()
print num_row
#sizeForSlice = tf.constant(1, shape=[num_pos])
# slice = tf.slice(reduceSum, indices, sizeForSlice)
with tf.Session() as sess:
    print ('indices of no-zero elements:')
    print (sess.run(indices))
    print (indices.get_shape())
    #print (sess.run(num_pos))
    print ('number of no-zero elements:')
    print (sess.run(tf.shape(indices))[0])
    print ('----:')
    #print (sess.run(slice))
    # print (sess.run(posSubpos))



