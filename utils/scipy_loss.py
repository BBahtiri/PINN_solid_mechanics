#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import tracemalloc
import sys 


def scipy_function_factory(model, *args):
    """A factory to create a function required by scipy.opimizer.
    Based on the example from https://stackoverflow.com/questions/59029854/use-scipy-optimizer-with-tensorflow-2-0-for-neural-network-training
    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        *args: arguments to be passed to model.get_grads method        
        
    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """

    #Run first the model to get trainable_variables
    #If you do not forward pass before the trainable variables will be []
    # model.bounds(*args)

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = [] # stitch indices
    part = [] # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n

    part = tf.constant(part)

    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.

        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    # now create a function that will be returned by this factory
    def f(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.

        This function is created by function_factory.

        Args:
           params_1d [in]: a 1D tf.Tensor.

        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """
        assign_new_model_parameters(params_1d)        
        loss_value, grads = model.get_grad(*args)
        grads = tf.dynamic_stitch(idx, grads)
        l1,l2,l3,l4 = model.get_all_losses(*args)
        # ab,ad = model.obtain_adaptive_const(*args)
        # model.update_adaptive_const(*args)
        # print out iteration & loss
        f.iter.assign_add(1)
        if f.iter%model.print_epoch == 0:
            tf.print("Iter:", f.iter, "l1:", l1, "l2", l2, "l3", l3,"l4",l4)
            # tf.print("Iter", f.iter, "ab:", ab, "ad:",ad)

        return loss_value.numpy(), grads.numpy()

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters

    return f
