#
import tensorflow as tf

y_true = tf.constant([
    [
        [
            [4, 4, 4, 1],
            [3, 3, 3, 0],
            [2, 2, 2, 0],
            [1, 1, 1, 0]
        ],
        [
            [4, 4, 4, 1],
            [3, 3, 3, 0],
            [2, 2, 2, 0],
            [1, 1, 1, 0]
        ],
        [
            [4, 4, 4, 1],
            [3, 3, 3, 0],
            [2, 2, 2, 0],
            [1, 1, 1, 0]
        ],
        [
            [4, 4, 4, 1],
            [3, 3, 3, 0],
            [2, 2, 2, 0],
            [1, 1, 1, 0]
        ]
    ],
    [   [
            [4, 4, 4, 1],
            [3, 3, 3, 0],
            [2, 2, 2, 0],
            [1, 1, 1, 0]
        ],
        [
            [4, 4, 4, 1],
            [3, 3, 3, 0],
            [2, 2, 2, 0],
            [1, 1, 1, 0]
        ],
        [
            [4, 4, 4, 1],
            [3, 3, 3, 0],
            [2, 2, 2, 0],
            [1, 1, 1, 0]
        ],
        [
            [4, 4, 4, 1],
            [3, 3, 3, 0],
            [2, 2, 2, 0],
            [1, 1, 1, 0]
        ]
    ]
])

y_pred = tf.constant([
    [
        [
            [1, 1, 1],
            [3, 3, 3],
            [2, 2, 2],
            [1, 1, 1]
        ],
        [
            [1, 1, 1],
            [3, 3, 3],
            [2, 2, 2],
            [1, 1, 1]
        ],
        [
            [1, 1, 1],
            [3, 3, 3],
            [2, 2, 2],
            [1, 1, 1]
        ],
        [
            [1, 1, 1],
            [3, 3, 3],
            [2, 2, 2],
            [1, 1, 1]
        ]
    ],
        [
        [
            [1, 1, 1],
            [3, 3, 3],
            [2, 2, 2],
            [1, 1, 1]
        ],
        [
            [1, 1, 1],
            [3, 3, 3],
            [2, 2, 2],
            [1, 1, 1]
        ],
        [
            [1, 1, 1],
            [3, 3, 3],
            [2, 2, 2],
            [1, 1, 1]
        ],
        [
            [1, 1, 1],
            [3, 3, 3],
            [2, 2, 2],
            [1, 1, 1]
        ]
    ]
])

y_true = tf.to_float(y_true)
y_pred = tf.to_float(y_pred)

y_true_lastAxis_lastElem = y_true[:, :, :, 3]
y_true_expand = tf.expand_dims(y_true_lastAxis_lastElem, axis=3)


abs_rel = tf.abs(y_true[:, :, :, 0:3] - y_pred)
smooth = tf.where(tf.greater(abs_rel, 1),
                  abs_rel - 0.5,
                  0.5 * abs_rel ** 2)
loss = y_true[:, :, :, 3:6] * smooth
loss_sum = tf.reduce_sum(loss, axis=-1)
loss_mean = tf.reduce_mean(loss, axis=-1)
loss_all_mean = tf.reduce_mean(loss)
for i in xrange(2):
    y_true = tf.concat([y_true, y_true_expand], axis=3)
with tf.Session() as sess:
    print (sess.run(loss_sum))
    print (sess.run(loss_mean))
    print (sess.run(loss_mean / tf.to_float(tf.shape(y_true)[0])))
    print (sess.run(loss_all_mean))
#    print (sess.run(y_true_lastAxis_lastElem))
#    print (sess.run(y_true_expand))
#    print (sess.run(y_true))

"""
reduceSum = tf.reduce_sum(gt, axis=-1)
zero = tf.constant(0, dtype=tf.int32)
where = tf.not_equal(reduceSum, zero)
indices = tf.where(where)
print indices

# num_row, num_clos = indices.get_shape().as_list()
num_pos = tf.to_int32(tf.shape(indices)[0])
print num_pos
num_row = num_pos.get_shape().as_list()
# num_pos = num_pos.as_list()
print num_row
# sizeForSlice = tf.constant(1, shape=[num_pos])
# slice = tf.slice(reduceSum, indices, sizeForSlice)
with tf.Session() as sess:
    print ('indices of no-zero elements:')
    print (sess.run(indices))
    print (indices.get_shape())
    # print (sess.run(num_pos))
    print ('number of no-zero elements:')
    print (sess.run(tf.shape(indices))[0])
    print ('----:')
    # print (sess.run(slice))
    # print (sess.run(posSubpos))
"""



