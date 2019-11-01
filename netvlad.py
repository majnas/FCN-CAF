
import tensorflow as tf

def vlad_pooling_layer(inputs, K, centers=None):
  D = inputs.get_shape().as_list()[3]
  
  filters = tf.Variable(tf.truncated_normal([1, 1, D, K], 0.0, 0.001), name = "nvlad_filters") #[1, 1, D, K]
  biases = tf.Variable(tf.truncated_normal([K], 0.0, 0.001), name = "nvlad_biases")
  if centers is None:
    centers = tf.Variable(tf.truncated_normal([D, K], 0.0, 0.001), name = "nvlad_centers")
    print('learn VLAD centers from scratch.')
  else: 
    #centers = centers[:D, :] # for test
    assert (centers.shape[0]==D)
    centers = tf.Variable(centers, name = "nvlad_centers")
    print('loading VLAD centers from npz file.')
    
  
  descriptor = tf.nn.l2_normalize(inputs, axis = [1,2])                     # descriptor = [B, H, W, D]
  conv_1x1 = tf.nn.convolution(descriptor, filters, padding = 'VALID')      # conv_vlad = [B, H, W, K]
  conv_vlad = tf.nn.bias_add(conv_1x1, biases)                              # conv_vlad = [B, H, W, K]
  a_k = tf.nn.softmax(conv_vlad, axis = -1, name = "vlad_softmax")          # a_k  = [B, H, W, K]
  
  ak_i = tf.split(value=a_k, num_or_size_splits=K, axis=3) # K * [B, H, W, 1]
  V1 = []
  for ak in ak_i:   # ak = [B, H, W, 1]
      v1 = tf.multiply(inputs, ak)        # [B, H, W, D] x [B, H, W, 1] = [B, H, W, D]
      v1 = tf.reduce_sum(v1, axis=[1,2])  # [B, D]
      v1 = tf.expand_dims(v1, axis=2)     # [B, D, 1]
      V1.append(v1)
  V1 = tf.concat(V1, axis=2) # [B, K, D]
  
  v2 = tf.reduce_sum(a_k, axis = [1,2], keepdims = True) # v2 = [B, 1, 1, K] 
  V2 = tf.multiply(v2, centers)       # [B, 1, 1, K] * [D, K] = [B, 1, D, K]
  V2 = tf.squeeze(V2, axis=1)         # [B, D, K]
  V = tf.subtract(V1, V2)             # [B, D, K]
  
  V_intra_norm = tf.nn.l2_normalize(V, axis = 1)   #intra-normalization
  V_intra_norm_flat = tf.reshape(V_intra_norm, shape = [-1, D*K]) # [B, DxK]
  output = tf.nn.l2_normalize(V_intra_norm_flat, axis = 1)  #l2 normalization [B, DxK]
  
  return output

