# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:43:36 2017

@author: Administrator
"""

# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import timeit
import numpy

import theano
import theano.tensor as T

from Logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d


class FeatureLayer(object):
    
    
    def __init__(self,input,ne,de,cs):
        

        self.input = input
        idx = T.cast(input,'int32')
        W_values = 0.2 * numpy.random.uniform(-1.0, 1.0,
                                 (ne+1, de)).astype(theano.config.floatX)
        #self.W = theano.shared(W_values)
        self.emb = theano.shared(W_values)
        self.output = self.emb[idx].reshape((input.shape[0], de,cs))
        self.params = [self.emb]
       
class ConvPoolLayer(object):

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):

        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray( 
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        # pool each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input
       
def test_nn_location(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,nkerns=[20, 30],
              batch_size=20, n_hidden=400):
    
    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    #print('test_nn',type(train_set_x[0:20]))
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
    
    index = T.iscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    rng = numpy.random.RandomState(1234)
    #featurelayer_input = x.reshape((batch_size,11)) 
    featurelayer = FeatureLayer(
                input=x,
                ne=2190,
                de=50,
                cs=11
                )
    convpoolayer_input = featurelayer.output.reshape((batch_size,1,11,50))
    
    convpoolayer0 = ConvPoolLayer(
        rng,
        input=convpoolayer_input,
        image_shape=(batch_size, 1, 11, 50),
        filter_shape=(nkerns[0], 1, 2, 3),
        poolsize=(2, 2)
    )
    convpoolayer1 = ConvPoolLayer(
        rng,
        input=convpoolayer0.output,
        image_shape=(batch_size, nkerns[0], 5, 24),
        filter_shape=(nkerns[1], nkerns[0], 2, 3),
        poolsize=(2, 2)
    )
    hiddenlayerinput = convpoolayer1.output.flatten(2)
    hiddenlayer = HiddenLayer(
            rng=rng,
            input=hiddenlayerinput,
            n_in=22*nkerns[1],#(5-2+1)/2*(24-3+1)/2=22*kerl[1]
            n_out=n_hidden,
            activation=T.tanh
        )
    logRegressionLayer = LogisticRegression(
            input=hiddenlayer.output,
            n_in=n_hidden,
            n_out=2190
        )
    L1 = (
            abs(featurelayer.emb).sum()+abs(hiddenlayer.W).sum()
            + abs(logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
    L2_sqr = (
                (featurelayer.emb**2).sum()
            +(hiddenlayer.W ** 2).sum()
            + (logRegressionLayer.W ** 2).sum()
        )
        
    params = featurelayer.params+convpoolayer0.params+convpoolayer1.params+hiddenlayer.params+logRegressionLayer.params
    #print ('classifier',type(x))
    cost = (
        logRegressionLayer.negative_log_likelihood(y)
        + L1_reg * L1
        + L2_reg * L2_sqr
    )
    
    test_model = theano.function(
        inputs=[index],
        outputs=logRegressionLayer.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    validate_model = theano.function(
        inputs=[index],
        outputs=logRegressionLayer.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    
    gparams = [T.grad(cost, param) for param in params]


    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(params, gparams)
    ]


    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    print('... training')

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):#分片操作

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index#迭代的次数

            if (iter + 1) % validation_frequency == 0:#验证频率，当迭代次数满足要求时则进行验证
			#验证集上的操作
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

if __name__ == '__main__':
    
    test_nn_location()
    
                                   
    
        
        


        
            
            
        
        



