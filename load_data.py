#!/usr/bin/python
import sys
import numpy as np
import theano
import theano.tensor as T

database_path = './'
def load_data(dirname, normalization = False, slice_ratio = 1, idx_valid = 0):

    rng = np.random.RandomState(23455)
    train_file = database_path + dirname + '/' + dirname + '_TRAIN'
    test_file  = database_path + dirname + '/' + dirname + '_TEST'

    #load train set
    data = np.loadtxt(train_file, dtype = np.str)
    train_x = data[:,1:].astype(np.float32)
    train_y = np.int_(data[:,0].astype(np.float32)) - 1 #label starts from 0
    
    len_data = train_x.shape[1]

    #restrict slice ratio when data lenght is too large
    if len_data > 500:
        slice_ratio = slice_ratio if slice_ratio > 0.98 else 0.98

    #shuffle for splitting train set and dataset
    n = train_x.shape[0]
    ind = np.arange(n)
    rng.shuffle(ind) #shuffle the train set
 
    #split train set into train set and validation set
    if idx_valid > 0: 
        valid_x = train_x[ind[int(0.2 * (idx_valid-1)*n):int(0.2 * (idx_valid)*n)]]
        valid_y = train_y[ind[int(0.2 * (idx_valid-1)*n):int(0.2 * (idx_valid)*n)]]

        ind = np.delete(ind, (range(int(0.2 * (idx_valid-1)*n),int(0.2 * (idx_valid)*n))))

        train_x = train_x[ind] 
        train_y = train_y[ind]
        #remove rows
    else:
        train_x = train_x[ind] 
        train_y = train_y[ind]

        valid_x = train_x.copy()
        valid_y = train_y.copy()
    
    train_x, train_y = slice_data(train_x, train_y, slice_ratio)
    valid_x, valid_y = slice_data(valid_x, valid_y, slice_ratio)

    #shuffle again
    n = train_x.shape[0]
    ind = np.arange(n)
    rng.shuffle(ind) #shuffle the train set
 
    #load test set
    data = np.loadtxt(test_file, dtype = np.str)
    test_x = data[:,1:].astype(np.float32)
    test_y = np.int_(data[:,0].astype(np.float32)) - 1 

    test_x, test_y = slice_data(test_x, test_y, slice_ratio)

    #z-normalization
    if normalization == True:
        mean_x = train_x.mean(axis = 0)
        std_x  = train_x.std(axis = 0)
        train_x = (train_x - mean_x) / std_x
        valid_x = (valid_x - mean_x) / std_x
        test_x = (test_x - mean_x) / std_x

    return [(train_x,train_y), (valid_x, valid_y), (test_x, test_y), (len_data), (slice_ratio)]

def slice_data(data_x, data_y, slice_ratio = 1): 
    #return the sliced dataset
    if slice_ratio == 1:
        return data_x, data_y
    n = data_x.shape[0]
    length = data_x.shape[1]
    length_sliced = int(length * slice_ratio)
   
    increase_num = length - length_sliced + 1 #if increase_num =5, it means one ori becomes 5 new instances.
    n_sliced = n * increase_num
    #print "*increase num", increase_num
    #print "*new length", n_sliced, "orginal len", n

    new_x = np.zeros((n_sliced, length_sliced))
    new_y = np.zeros((n_sliced))
    for i in xrange(n):
        for j in xrange(increase_num):
            new_x[i * increase_num + j, :] = data_x[i,j : j + length_sliced]
            new_y[i * increase_num + j] = np.int_(data_y[i].astype(np.float32))
    return new_x, new_y


def _downsample(data_x, sample_rate, offset = 0):
    num = data_x.shape[0]
    length_x = data_x.shape[1]
    last_one = 0
    if length_x % sample_rate > offset:
        last_one = 1
    new_length = int(np.floor( length_x / sample_rate)) + last_one
    output = np.zeros((num, new_length))
    for i in xrange(new_length):
        output[:,i] = np.array(data_x[:,offset + sample_rate * i])

    return output

def _movingavrg(data_x, window_size):
    num = data_x.shape[0]
    length_x = data_x.shape[1]
    output_len = length_x - window_size + 1
    output = np.zeros((num, output_len))
    for i in xrange(output_len):
        output[:,i] = np.mean(data_x[:, i : i + window_size], axis = 1)
    return output

def movingavrg(data_x, window_base, step_size, num):
    if num == 0:
        return (None, [])
    out = _movingavrg(data_x, window_base)
    data_lengths = [out.shape[1]]
    for i in xrange(1, num):
        window_size = window_base + step_size * i
        if window_size > data_x.shape[1]:
            continue
        new_series = _movingavrg(data_x, window_size)
        data_lengths.append( new_series.shape[1] )
        out = np.concatenate([out, new_series], axis = 1)
    return (out, data_lengths)

def batch_movingavrg(train,valid,test, window_base, step_size, num):
    (new_train, lengths) = movingavrg(train, window_base, step_size, num)
    (new_valid, lengths) = movingavrg(valid, window_base, step_size, num)
    (new_test, lengths) = movingavrg(test, window_base, step_size, num)
    return (new_train, new_valid, new_test, lengths)

def downsample(data_x, base, step_size, num):
    if num == 0:
        return (None, [])
    out = _downsample(data_x, base,0)
    data_lengths = [out.shape[1]]
    #for offset in xrange(1,base): #for the base case
    #    new_series = _downsample(data_x, base, offset)
    #    data_lengths.append( new_series.shape[1] )
    #    out = np.concatenate( [out, new_series], axis = 1)
    for i in xrange(1, num):
        sample_rate = base + step_size * i 
        if sample_rate > data_x.shape[1]:
            continue
        for offset in xrange(0,1):#sample_rate):
            new_series = _downsample(data_x, sample_rate, offset)
            data_lengths.append( new_series.shape[1] )
            out = np.concatenate( [out, new_series], axis = 1)
    return (out, data_lengths)

def batch_downsample(train,valid,test, window_base, step_size, num):
    (new_train, lengths) = downsample(train, window_base, step_size, num)
    (new_valid, lengths) = downsample(valid, window_base, step_size, num)
    (new_test, lengths) = downsample(test, window_base, step_size, num)
    return (new_train, new_valid, new_test, lengths)

def shared_data_x(data_x, borrow=True):
    shared_x = theano.shared(np.asarray(data_x,
                                    dtype = T.config.floatX),
                                    borrow = borrow)
    return shared_x

def my_shared_datasets(x,y,z, borrow = True):
    shared_x = theano.shared(np.asarray(x,
                                    dtype = T.config.floatX),
                                    borrow = borrow)
    shared_y = theano.shared(np.asarray(y,
                                    dtype = T.config.floatX),
                                    borrow = borrow)
    shared_z = theano.shared(np.asarray(z,
                                    dtype = T.config.floatX),
                                    borrow = borrow)
    return (shared_x, shared_y, shared_z)

def shared_dataset(data_x, data_y,  borrow=True):
    shared_x = theano.shared(np.asarray(data_x,
                                    dtype = T.config.floatX),
                                    borrow = borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                    dtype = T.config.floatX),
                                    borrow = borrow)
    return shared_x, T.cast(shared_y, 'int32')


