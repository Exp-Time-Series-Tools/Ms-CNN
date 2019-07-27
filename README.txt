#########################################################
#                    Python packages                   ##
#########################################################
0. numpy (version I use: 1.10.4)
1. Theano (version I use: 0.7.0)

Useful link for installing theano:
http://deeplearning.net/software/theano/install.html

#########################################################
#                    How To Run                       ##
#########################################################
mcnn.py trains on one of the ucr dataset, Trace, with 
multiscale convolutional neural network (MSCNN). 
It can run on both CPU and GPU.

1. run 'python mcnn.py' 
## This command trains a MSCNN with default parameters. 
## You should be able to get zero error on train set, 
## validation set as well as test set.

2. run "THEANO_FLAGS='blas.ldflags=-lblas -lgfortran,mode=FAST_RUN,
          cuda.root=/usr/local/cuda,device=gpu,floatX=float32,
          lib.cnmem=1' python mcnn.py" 
## This command trains exactly the same MSCNN as the first one. 
## However, You can enjoy 10x speedup if you have GPU with 
## CUDA installed using this command.

3. run 'python mcnn.py -h' for more information.

#########################################################
#                    Standard Output                   ##
#########################################################
increase factor is  29 , ori len 275
train size 2320 ,valid size 580  test size 2900
batch size  232
n_train_batches is  10
data dim is  371
---------------------------
building the model...
training...
...epoch 1, valid err: 0.75000 | test err: 0.43000 | train err 0.73621, cost 2.2027
...epoch 2, valid err: 0.75000 | test err: 0.43000 | train err 0.65603, cost 1.4417
...epoch 3, valid err: 0.75000 | test err: 0.43000 | train err 0.54526, cost 1.0825
...epoch 4, valid err: 0.75000 | test err: 0.43000 | train err 0.48319, cost 0.9583
  
#########################################################
Please contact me if you have any question.
