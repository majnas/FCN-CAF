from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from utils import preprocessing
from utils import util
import fcn8_vgg

def model_fn(features, labels, mode, params):
  images = tf.cast(
      tf.map_fn(preprocessing.mean_image_addition, features),
      tf.uint8)

  model = fcn8_vgg.FCN8VGG(vgg16_ckpt_path= params['vgg16_ckpt_path'],
                           vgg16_npy_path= params['vgg16_npy_path'])
  
  model.build(rgb= features, 
              keep_prob= params['keep_prob'],
              is_CAF = params['is_CAF'],
              Netvlad_K= params['Netvlad_K'],
              Netvlad_centers= params['Netvlad_centers'],
              is_training= (mode == tf.estimator.ModeKeys.TRAIN),
              num_classes= params['num_classes'], 
              random_init_fc8= True)
  
  logits = model.upscore
  #init_fn = model.init_fn
  #print(logits)
  #print(labels)

  pred_classes = tf.expand_dims(tf.argmax(logits, axis=3, output_type=tf.int32), axis=3) # [b, 385, 385, 1] int32
  pred_decoded_labels = tf.py_func(preprocessing.decode_labels,
                                   [pred_classes, params['batch_size'], params['num_classes']],
                                   tf.uint8) # [b, 385, 385, 3] rgb int

  predictions = {
      'logits': logits,
      'classes': pred_classes,   
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
      'decoded_labels': pred_decoded_labels
  }
  
  #predictions.update(debug_tensors)
  

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Delete 'decoded_labels' from predictions because custom functions produce error when used with saved_model
    predictions_without_decoded_labels = predictions.copy()
    del predictions_without_decoded_labels['decoded_labels']

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'preds': tf.estimator.export.PredictOutput(
                predictions_without_decoded_labels)
        })

  gt_decoded_labels = tf.py_func(preprocessing.decode_labels,
                                 [labels, params['batch_size'], params['num_classes']], tf.uint8)

  labels = tf.squeeze(labels, axis=3)  # reduce the channel dimension. [b, 385, 385]
  
  labels_oh = tf.one_hot(labels, params['num_classes'])
  tf.identity(labels_oh, name='labels')
  #print(labels_oh)

  logits_by_num_classes = tf.reshape(logits, [-1, params['num_classes']]) #[b*385*385, 21]
  labels_flat = tf.reshape(labels, [-1, ]) #[b*385*385]

  valid_indices = tf.to_int32(labels_flat <= params['num_classes'] - 1) #[b*385*385] - (pixels==255)
  valid_logits = tf.dynamic_partition(logits_by_num_classes, valid_indices, num_partitions=2)[1]
  valid_labels = tf.dynamic_partition(labels_flat, valid_indices, num_partitions=2)[1]

  preds_flat = tf.reshape(pred_classes, [-1, ]) #[b*385*385]
  valid_preds = tf.dynamic_partition(preds_flat, valid_indices, num_partitions=2)[1]
  confusion_matrix = tf.confusion_matrix(valid_labels, valid_preds, num_classes=params['num_classes'])

  predictions['valid_preds'] = valid_preds
  predictions['valid_labels'] = valid_labels
  predictions['confusion_matrix'] = confusion_matrix

  cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      logits=valid_logits, labels=valid_labels)
  
  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)
  
  train_var_list = [v for v in tf.trainable_variables()]

  # Add weight decay to the loss.
  with tf.variable_scope("total_loss"):
    loss = cross_entropy + params.get('weight_decay', params['weight_decay']) * tf.add_n(
        [tf.nn.l2_loss(v) for v in train_var_list])
  #loss = tf.losses.get_total_loss()  # obtain the regularization losses as well
  #loss = loss_fn.loss(logits=logits, labels=labels_oh, num_classes=params['num_classes'], head=None)

  if mode == tf.estimator.ModeKeys.TRAIN:
    
#    tf.summary.image('images',
#                     tf.concat(axis=2, values=[images, gt_decoded_labels, pred_decoded_labels]),
#                     max_outputs=params['tensorboard_images_max_outputs'])  # Concatenate row-wise.

    global_step = tf.train.get_or_create_global_step()

    if params['learning_rate_policy'] == 'piecewise':
      # Scale the learning rate linearly with the batch size. When the batch size
      # is 128, the learning rate should be 0.1.
      initial_learning_rate = 0.1 * params['batch_size'] / 128
      batches_per_epoch = params['num_train'] / params['batch_size']
      # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
      boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 150, 200]]
      values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
      learning_rate = tf.train.piecewise_constant(
          tf.cast(global_step, tf.int32), boundaries, values)
    elif params['learning_rate_policy'] == 'poly':
      learning_rate = tf.train.polynomial_decay(
          params['initial_learning_rate'],
          tf.cast(global_step, tf.int32) - params['initial_global_step'],
          params['max_iter'], params['end_learning_rate'], power=params['power'])
    elif params['learning_rate_policy'] == 'constant':
      learning_rate = params['initial_learning_rate']
    else:
      raise ValueError('Learning rate policy must be "piecewise" or "poly"')

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)


    encoder_trainable = train_var_list
    #encoder_trainable = [v for v in train_var_list if 'conv' not in v.name and 'fc' not in v.name]

    #assert(len(train_var_list) == len(encoder_trainable) + len(decoder_aspp_trainable))
    
    #opt_encoder = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=params['momentum'])
    opt_encoder = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #opt_decoder_aspp = tf.train.MomentumOptimizer(learning_rate=learning_rate * 1, momentum=params['momentum'])

    grads = tf.gradients(loss, encoder_trainable)
    #grads = tf.gradients(loss, encoder_trainable + decoder_aspp_trainable)
    grads_encoder = grads[:len(encoder_trainable)]
    #grads_decoder_aspp = grads[len(encoder_trainable) : (len(encoder_trainable) + len(decoder_aspp_trainable))]
    
    train_op_encoder = opt_encoder.apply_gradients(zip(grads_encoder, encoder_trainable), global_step=global_step)
    #train_op_decoder_aspp = opt_decoder_aspp.apply_gradients(zip(grads_decoder_aspp, decoder_aspp_trainable), global_step=global_step)

    # Batch norm requires update ops to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      #train_op = optimizer.minimize(loss, global_step, var_list=train_var_list)  
      train_op = tf.group(train_op_encoder)
      #train_op = tf.group(train_op_encoder, train_op_decoder_aspp)
  else:
    train_op = None

  accuracy = tf.metrics.accuracy(valid_labels, valid_preds)
  mean_iou = tf.metrics.mean_iou(valid_labels, valid_preds, params['num_classes'])
  metrics = {'px_accuracy': accuracy, 'mean_iou': mean_iou}

  # Create a tensor named train_accuracy for logging purposes
  tf.identity(accuracy[1], name='train_px_accuracy')
  tf.summary.scalar('train_px_accuracy', accuracy[1])

  def compute_mean_iou(total_cm, name='mean_iou'):
    """Compute the mean intersection-over-union via the confusion matrix."""
    sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
    sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
    cm_diag = tf.to_float(tf.diag_part(total_cm))
    denominator = sum_over_row + sum_over_col - cm_diag

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = tf.reduce_sum(tf.cast(
        tf.not_equal(denominator, 0), dtype=tf.float32))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = tf.where(
        tf.greater(denominator, 0),
        denominator,
        tf.ones_like(denominator))
    iou = tf.div(cm_diag, denominator)

    for i in range(params['num_classes']):
      tf.identity(iou[i], name='train_iou_class{}'.format(i))
      tf.summary.scalar('train_iou_class{}'.format(i), iou[i])

    # If the number of valid entries is 0 (no classes) we return 0.
    result = tf.where(
        tf.greater(num_valid_entries, 0),
        tf.reduce_sum(iou, name=name) / num_valid_entries,
        0)
    return result

  train_mean_iou = compute_mean_iou(mean_iou[1])

  tf.identity(train_mean_iou, name='train_mean_iou')
  tf.summary.scalar('train_mean_iou', train_mean_iou)

  util.count_params()
  
  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)
