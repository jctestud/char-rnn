# adapted from: https://github.com/kuza55/keras-extras
# with Keras 2.x fix: https://github.com/kuza55/keras-extras/pull/19
# and issue #23 (Model Saving/Checkpointing incompatibilities)

from keras.layers.merge import Concatenate
from keras.layers.core import Lambda
from keras.models import Model

import tensorflow as tf

def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([ shape[:1] // parts, shape[1:] ],axis=0)
        stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ],axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    #Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)                

                outputs = model(inputs)
                
                if not isinstance(outputs, list):
                    outputs = [outputs]
                
                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(Concatenate(axis=0)(outputs))

        # update model saving scheme to save underlying model rather than parallel
        new_model = Model(inputs=model.inputs, outputs=merged)
        save_model_function = type(model.save)

        def save_old_model(self_, model_path, overwrite=True):
            model.save(model_path, overwrite)
        new_model.save = save_model_function(save_old_model, new_model)
        # update weight saving scheme to save underlying model weights

        save_weights_function = type(model.save_weights)

        def save_old_weights(self_, weights_path, overwrite=True):
            model.save_weights(weights_path, overwrite)
        
        new_model.save_weights = save_weights_function(save_old_weights, new_model)
        
        return new_model
