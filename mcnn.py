#!/usr/bin/python

import os
import sys, getopt
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression 
from mlp import HiddenLayer
from logistic_vote import LogisticRegressionVote

from load_data import load_data
from load_data import batch_downsample
from load_data import batch_movingavrg
from load_data import shared_dataset
from load_data import shared_data_x

__autor__ = 'Zhicheng Cui'

def reLU(x): #activation function
    return T.switch(x<0,0,x)

class ShapeletPoolLayer(object):
    """Pool Layer of a convolutional network, support learning shapelet """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 1), W = None, previous_layer = None, trans = 'euc', active_func=T.tanh):

        assert image_shape[1] == filter_shape[1]
        self.input = input
        if W == None:
			fan_in = numpy.prod(filter_shape[1:])
			fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
					   numpy.prod(poolsize))
			# initialize weights with random weights
			W_bound = numpy.sqrt(6. / (fan_in + fan_out))
			new_W_bound = numpy.sqrt(1. / (fan_in ))
			W = theano.shared(
				numpy.asarray(
					rng.uniform(low=-new_W_bound, high=new_W_bound, size=filter_shape),
					dtype=theano.config.floatX
				),
				borrow=True
			)
        self.W = W
        self.dummy_w = theano.shared(
            numpy.ones(filter_shape, dtype=theano.config.floatX
            ),
            borrow=True
        )

        conv_x_sqrd = conv.conv2d(
            input = (self.input ** 2),
            filters = self.dummy_w,
            filter_shape = filter_shape,
            image_shape = image_shape
        )
        self.w_sqrd = (T.sum( (self.W ** 2),axis=2))
        self.w_sqrd = self.w_sqrd.reshape((1,filter_shape[0],1,1))
        self.w_sqrd = T.tile(self.w_sqrd,(image_shape[0], 1,image_shape[2] - filter_shape[2] + 1, 1))
        w_sqrd = self.w_sqrd
        conv_out = conv.conv2d(
            input=self.input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )
        if trans == 'euc':
            conv_out = (w_sqrd + conv_x_sqrd - 2 * conv_out)
        elif trans == 'reuc':
            conv_out = 1 / (w_sqrd + conv_x_sqrd - 2 * conv_out)
        elif trans == 'sqrteuc':
            conv_out = T.sqrt((w_sqrd + conv_x_sqrd - 2 * conv_out))
        elif trans == 'sqrtreuc':
            conv_out = T.sqrt( 1 / (w_sqrd + conv_x_sqrd - 2 * conv_out) )
        elif trans == 'conv':
            conv_out = conv_out

        else:
            print "No such trans func. Please choose from the following options: <euc | reuc | sqrteuc | sqrtreuc>"
            sys.exit(2)

        # downsample each feature map individually, using minpooling
        pooled_out = pool.pool_2d(
            input=-conv_out,
            ws=poolsize,
            ignore_border=True
        )
        pooled_out = -pooled_out

        if previous_layer == None:
            self.previous_layer = (pooled_out)
        else:
            self.previous_layer = previous_layer
        self.output = active_func(pooled_out)

        # store parameters of this layer, no bias here
        self.params = [self.W]

def evaluate(init_learning_rate=0.1, n_epochs=200,
                    datasets='Trace' ,nkerns=[256, 256], n_train_batch=10,
                    trans='euc', active_func=T.tanh, window_size = 0.2, 
                    ada_flag = False, pool_factor = 2, slice_ratio = 1
                    ):

    rng = numpy.random.RandomState(23455) #set random seed
    learning_rate = theano.shared(numpy.asarray(init_learning_rate,dtype=theano.config.floatX))
    #used for learning_rate decay
 
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    ori_len = datasets[3]
    slice_ratio = datasets[4]

    valid_num = valid_set_x.shape[0]
    increase_num = ori_len - int(ori_len * slice_ratio) + 1 #this can be used as the bath size
    print "increase factor is ", increase_num, ', ori len', ori_len
    valid_num_batch = valid_num / increase_num

    test_num = test_set_x.shape[0]
    test_num_batch = test_num / increase_num

    length_train = train_set_x.shape[1] #length after slicing.
    num_of_categories = int(train_set_y.max()) + 1
 
    window_size = int(length_train * window_size) if window_size < 1 else int(window_size)

    #*******set up the ma and ds********#
    ma_base,ma_step,ma_num   = 5, 6, 0
    ds_base,ds_step, ds_num  = 2, 1, 4

    ds_num_max = length_train / (pool_factor * window_size)
    ds_num = min(ds_num, ds_num_max)
    
    #*******set up the ma and ds********#

    (ma_train, ma_valid, ma_test , ma_lengths) = batch_movingavrg(train_set_x,
                                                    valid_set_x, test_set_x,
                                                    ma_base, ma_step, ma_num)
    (ds_train, ds_valid, ds_test , ds_lengths) = batch_downsample(train_set_x,
                                                    valid_set_x, test_set_x,
                                                    ds_base, ds_step, ds_num)
 
    #concatenate directly
    data_lengths = [length_train] 
    #downsample part:
    if ds_lengths != []:
        data_lengths +=  ds_lengths
        train_set_x = numpy.concatenate([train_set_x, ds_train], axis = 1)
        valid_set_x = numpy.concatenate([valid_set_x, ds_valid], axis = 1)
        test_set_x = numpy.concatenate([test_set_x, ds_test], axis = 1)

    #moving average part
    if ma_lengths != []:
        data_lengths += ma_lengths
        train_set_x = numpy.concatenate([train_set_x, ma_train], axis = 1)
        valid_set_x = numpy.concatenate([valid_set_x, ma_valid], axis = 1)
        test_set_x = numpy.concatenate([test_set_x, ma_test], axis = 1)

    train_set_x, train_set_y = shared_dataset(train_set_x, train_set_y)
    
    valid_set_x = shared_data_x(valid_set_x)
    test_set_x = shared_data_x(test_set_x)

    #compute number of minibatches for training, validation and testing
    n_train_size = train_set_x.get_value(borrow=True).shape[0]
    n_valid_size = valid_set_x.get_value(borrow=True).shape[0]
    n_test_size = test_set_x.get_value(borrow=True).shape[0]
    batch_size = n_train_size / n_train_batch
    n_train_batches = n_train_size / batch_size
    data_dim = train_set_x.get_value(borrow=True).shape[1]
    print 'train size', n_train_size, ',valid size', n_valid_size, ' test size', n_test_size
    print 'batch size ', batch_size
    print 'n_train_batches is ', n_train_batches
    print 'data dim is ', data_dim
    print '---------------------------'

    # allocate symbolic variables for the data
    index = T.lscalar('index')  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   
    y = T.ivector('y')  
                       
    x_vote = T.matrix('xvote')   # the data is presented as rasterized images
    #y_vote = T.ivector('y_vote')  # the labels are presented as 1D vector of

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print 'building the model...'

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = []
    inputs = x.reshape((batch_size, 1, data_dim, 1))
    
    layer0_input_vote = []
    inputs_vote = x_vote.reshape((increase_num, 1, data_dim, 1))
    ind = 0
    for i in xrange(len(data_lengths)):
        layer0_input.append(inputs[:,:,ind : ind + data_lengths[i],:])
        layer0_input_vote.append(inputs_vote[:,:,ind : ind + data_lengths[i],:])
        ind += data_lengths[i]

    layer0 = []
    layer0_vote = []
    feature_map_size = 0

    for i in xrange(len(layer0_input)):
        pool_size = (data_lengths[i] - window_size + 1) / pool_factor 
        feature_map_size += (data_lengths[i] - window_size + 1) / pool_size

        layer0.append(ShapeletPoolLayer(
            numpy.random.RandomState(23455 + i),
            input=layer0_input[i],
            image_shape=(batch_size, 1, data_lengths[i], 1),
            filter_shape=(nkerns[0], 1, window_size, 1),
            poolsize=(pool_size , 1),
            trans = trans,
            active_func=active_func
        ))
        layer0_vote.append(ShapeletPoolLayer(
            numpy.random.RandomState(23455 + i),
            input=layer0_input_vote[i],
            image_shape=(increase_num, 1, data_lengths[i], 1),
            filter_shape=(nkerns[0], 1, window_size, 1),
            poolsize=(pool_size , 1),
			W = layer0[i].W,
            trans = trans,
            active_func=active_func
        ))

    layer1_input = layer0[0].output.flatten(2)
    layer1_vote_input = layer0_vote[0].output.flatten(2)
    for i in xrange(1, len(data_lengths)):
        layer1_input = T.concatenate([layer1_input, layer0[i].output.flatten(2)], axis = 1)
        layer1_vote_input = T.concatenate([layer1_vote_input, layer0_vote[i].output.flatten(2)], axis = 1)

    # construct a fully-connected sigmoidal layer
    layer1 = HiddenLayer(
        rng,
        input=layer1_input,
        n_in=nkerns[0] * feature_map_size,
        n_out=nkerns[1],
        activation=active_func,
        previous_layer = None
    )
    # construct a fully-connected sigmoidal layer for prediction
    layer1_vote = HiddenLayer(
        rng,
        input=layer1_vote_input,
        n_in=nkerns[0] * feature_map_size,
        n_out=nkerns[1],
        activation=active_func,
        previous_layer = None,
        W = layer1.W,
        b = layer1.b
    )

    # classify the values of the fully-connected sigmoidal layer
    layer2 = LogisticRegression(input=layer1.output, n_in=nkerns[1], n_out= num_of_categories , previous_layer = None)
    layer2_vote = LogisticRegressionVote(input=layer1_vote.output, n_in=nkerns[1], n_out= num_of_categories , previous_layer = None, W = layer2.W, b = layer2.b)

    # the cost we minimize during training is the NLL of the model
    cost = layer2.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer2_vote.prediction(),
        givens={
            x_vote : test_set_x[index * (increase_num) : (index + 1) * (increase_num)]
        }
    )
    # function for validation set. Return the prediction value
    validate_model = theano.function(
        [index],
        layer2_vote.prediction(),
        givens={
            x_vote : valid_set_x[index * (increase_num) : (index + 1) * (increase_num)]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer2.params + layer1.params
    for i in xrange(len(layer0_input)):
        params += layer0[i].params

    # Adagradient part
    grads = T.grad(cost, params)
    import copy
    G = [] 
    for i in xrange(2 + len(layer0_input)):
        G.append( theano.shared(
            numpy.zeros(params[i].shape.eval(), dtype=theano.config.floatX
            ),
            borrow=True
        ))

    # parameter update methods
    if ada_flag == True:
        updates = [
            (param_i, param_i -  learning_rate * (grad_i / (T.sqrt(G_i) + 1e-5) ))
            for param_i, grad_i, G_i in zip(params, grads, G)
        ]
    else:
        updates = [
            (param_i, param_i -  learning_rate * grad_i )
            for param_i, grad_i in zip(params, grads)
        ]
 
    update_G = theano.function(inputs=[index], outputs = G,
            updates=[(G_i, G_i  + T.sqr(grad_i) )
            for G_i, grad_i in zip(G,grads)],
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]
            }
            )
    reset_G = theano.function(inputs=[index],outputs = G,
            updates=[(G_i, grad_i - grad_i) 
            for G_i, grad_i in zip(G,grads)],
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]
            }
            )       

    #Our training function, return value: NLL cost and training error
    train_model = theano.function(
        [index],
        [cost, layer2.errors(y)],
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    decrease_learning_rate = theano.function(inputs=[], outputs = learning_rate,
            updates={learning_rate: learning_rate * 1e-4})
    
    ###############
    # TRAIN MODEL #
    ###############
    print 'training...'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    best_test_iter = 0
    best_test_loss = numpy.inf
    test_patience = 200
    valid_loss = 0.
    test_loss = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    last_train_err = 1
    last_avg_err = float('inf')
    first_layer_prev = 0
    num_no_update_epoch = 0
    epoch_avg_cost = float('inf')
    epoch_avg_err = float('inf')

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        epoch_train_err = 0.
        epoch_cost = 0.
        if ada_flag:
            reset_G(0)
        num_no_update_epoch += 1
        if num_no_update_epoch == 500:
            break
        for minibatch_index in xrange(n_train_batches):

            iteration = (epoch - 1) * n_train_batches + minibatch_index

            if ada_flag:
                update_G(minibatch_index)
            
            [cost_ij,train_err] = train_model(minibatch_index)
            
            epoch_train_err = epoch_train_err + train_err
            epoch_cost = epoch_cost + cost_ij
            
            if (iteration + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                # validation set loss
                valid_results = [validate_model(i) for i in xrange(valid_num_batch)]
                valid_losses = []
                for i in xrange(valid_num_batch):
                    y_pred = valid_results[i]
                    label = valid_set_y[i * increase_num]
                    unique_value, sub_ind, correspond_ind, count = numpy.unique(y_pred, True, True, True)
                    unique_value = unique_value.tolist()
                    curr_err = 1.
                    if label in unique_value:
                        target_ind = unique_value.index(label)
                        count = count.tolist()
                        sorted_count = sorted(count)
                        if count[target_ind] == sorted_count[-1]:
                            if len(sorted_count) > 1 and sorted_count[-1] == sorted_count[-2]:
                                curr_err = 0.5 #tie
                            else:
                                curr_err = 0.
                    valid_losses.append(curr_err)
                valid_loss = sum(valid_losses) / float(len(valid_losses)) 

                print('...epoch %i, valid err: %.5f |' %
                      (epoch, valid_loss)),

                # if we got the best validation score until now
                if valid_loss <= best_validation_loss:
                    num_no_update_epoch = 0

                    #improve patience if loss improvement is good enough
                    if valid_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iteration * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = valid_loss
                    best_iter = iteration

                    # test it on the test set
                    test_results = [test_model(i) for i in xrange(test_num_batch)]
                    test_losses = []
                    for i in xrange(test_num_batch):
                        y_pred = test_results[i]
                        label = test_set_y[i * increase_num]
                        unique_value, sub_ind, correspond_ind, count = numpy.unique(y_pred, True, True, True)
                        unique_value = unique_value.tolist()
                        curr_err = 1
                        if label in unique_value:
                            target_ind = unique_value.index(label)
                            count = count.tolist()
                            sorted_count = sorted(count)
                            if count[target_ind] == sorted_count[-1]:
                                if len(sorted_count) > 1 and sorted_count[-1] == sorted_count[-2]:
                                    curr_err = 0.5 # tie
                                else:
                                    curr_err = 0.
                        test_losses.append(curr_err)
                    test_loss = sum(test_losses) / float(len(test_losses)) 
                    print(('test err: %.5f |') %
                          (test_loss)),

                    best_test_loss = test_loss
                    test_patience = 200

            #test_patience -= 1 
            #if test_patience <= 0:
            #    break
            
            if patience <= iteration:
                done_looping = True
                break

        epoch_avg_cost = epoch_cost/n_train_batches
        epoch_avg_err = epoch_train_err/n_train_batches
        #curr_lr = decrease_learning_rate()
        last_avg_err = epoch_avg_cost
 
        print ('train err %.5f, cost %.4f' %(epoch_avg_err,epoch_avg_cost))
        if epoch_avg_cost == 0:
             break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test error: %f %%' %
          (best_validation_loss * 100., best_iter + 1, best_test_loss * 100.))
    print('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    return best_validation_loss

if __name__ == '__main__':
    init_learning_rate = 0.1
    n_epochs = 200
    dataset = 'Trace'
    nkerns =  [256, 256]
    n_train_batch = 10
    trans = 'conv'
    active_func = T.nnet.sigmoid
    window_size = 0.2 
    ada_flag = False
    pool_factor = 5
    slice_ratio = 0.90
    normalization_flag = False
    idx_valid = 1 #0-5, default 0: no split on train set
    if len(sys.argv) == 1:
        pass
    else:
        try:
            opts, args = getopt.getopt(sys.argv[1:], 'hamt:f:l:d:w:n:b:p:s:',
                                       ['help','trans=','activeFunc=','learning_rate=',
                                        'filename=','window_size=','n_epochs=',
                                        'n_train_batch=','pool_factor=','slice_ratio='])
        except getopt.GetoptError as err:
            print str(err)
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print 'mcnn.py \n\t-t <euc | reuc | sqrteuc | sqrtreuc | conv> \
                    \n\t-f <tanh | sigmoid | reLU> \n\t-l <learning_rate> \n\t-d <filename> \
                    \n\t-n <n_epochs> \n\t-b <n_train_batch> \n\t-a [addgradient flag] \
                    \n\t-m [normalization_flag] \n\t-p <pool_factor> \n\t-s <slice ratio>'
                sys.exit()
            elif opt in ('-t', '--trans'):
                trans = arg
            elif opt in ('-f', '--activeFunc'):
                if arg == 'tanh':
                    active_func = T.tanh
                elif arg == 'sigmoid':
                    active_func = T.nnet.sigmoid
                else:
                    active_func = reLU
            elif opt in ('-l','--learning_rate'):
                init_learning_rate = float(arg)
            elif opt in ('-d','--filename'):
                dataset = arg
            elif opt in ('-w','--window_size'):
                window_size = float(arg)
            elif opt in ('-n','--n_epochs'):
                n_epochs = float(arg)
            elif opt in ('-b','--n_train_batch'):
                n_train_batch = int(arg)
            elif opt in ('-p', '--pool_factor'):
                pool_factor = int(arg)
            elif opt == '-a':
                ada_flag = True
            elif opt == '-s':
                slice_ratio = float(arg)
            elif opt == '-m':
                normalization_flag = True
            else:
                print 'param error'
                sys.exit(2)

    datasets = load_data(dataset, normalization_flag, slice_ratio, idx_valid) 
    evaluate(init_learning_rate=init_learning_rate, n_epochs = n_epochs,
                    datasets=datasets, nkerns=nkerns, n_train_batch=n_train_batch,
                    trans=trans, active_func=active_func,window_size=window_size,
                    ada_flag=ada_flag, pool_factor=pool_factor,slice_ratio=slice_ratio
                    )

