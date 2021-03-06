# -*- coding: utf_8 -*-
import numpy as np
from sklearn.base import BaseEstimator

from chainer import Variable, cuda
import chainer.functions as F


class NNmanager (BaseEstimator):
    def __init__(self, model, optimizer, lossFunction, gpu=True, **params):
        # CUDAデバイスの設定
        self.gpu = gpu
        # 学習器の初期化
        # ネットワークの定義
        if gpu:
            self.model = model.to_gpu()
        else:
            self.model = model
        # オプティマイザの設定
        self.optimizer = optimizer
        self.optimizer.setup(self.model)
        # 損失関数の設定
        self.lossFunction = lossFunction
        # epochの設定
        self.epoch = params['epoch'] if 'epoch' in params else 20
        # バッチサイズの設定
        self.batchsize = params['batchsize'] if 'batchsize' in params else 100
        # ロギングの設定
        self.testing_cycle  = params['testing_cycle'] if 'testing_cycle' in params else 1
        self.logging = params['logging'] if 'logging' in params else False
        self.train_logFormat = "[%d epoch] mean loss: %f, mean accuracy: %f"
        self.testing_logFormat = "[%d epoch] mean loss: %f, mean accuracy: %f, testing loss: %f, testing accuracy: %f"
        # テストデータの設定
        self.x_test = None
        self.y_test = None
        self.showTestingMode = None

    def fit(self, x_train, y_train):
        if self.showTestingMode:
            if self.x_test is None or self.y_test is None:
                raise RuntimeError("先にテストデータを登録してください")
            self.runEpoch(x_train, y_train, self.x_test, self.y_test)
        else:
            self.runEpoch(x_train, y_train)
        return self

    def registTestingData(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test

    def showTesting(self, mode):
        self.showTestingMode = mode

    def predict(self, x_test):
        if self.gpu:
            # GPU向け実装
            x_test = cuda.to_gpu(x_test)
            output = self.forward(x_test, train=False)
            output.data = cuda.to_cpu(output.data)
        else:
            # CPU向け実装
            output = self.forward(x_test, train=False)
        return self.trimOutput(output)

    def trimOutput(self, output):
        # 結果を整形したいときなど。
        return output.data

    # 順伝播・逆伝播
    def forward(self, x_data, train):
        # x = Variable(x_data)
        # h1 = F.relu(self.model.l1(x))
        # h2 = F.relu(self.model.l2(h1))
        # y_predict = self.model.l3(h2)
        # return y_predict
        raise NotImplementedError("`forward` method is not implemented.")

    def backward(self, y_predict, y_data):
        y = Variable(y_data)
        loss = self.lossFunction(y_predict, y)
        accuracy = F.accuracy(y_predict, y)
        loss.backward()
        return loss, accuracy

    def setLogger(self, logging):
        self.logging = logging

    def setTrainLogFormat(self, logFormat):
        self.train_logFormat = logFormat

    def setTestingLogFormat(self, logFormat):
        self.testing_logFormat = logFormat

    def runEpoch(self, x_train, y_train, x_test=None, y_test=None):
        if (x_test is None) and (y_test is not None):
            raise RuntimeError("x_testとy_testの片方のみの指定は許されません")
        if (x_test is not None) and (y_test is None):
            raise RuntimeError("x_testとy_testの片方のみの指定は許されません")
        testing = (x_test is not None)

        for epoch in xrange(self.epoch):
            mean_loss, mean_accuracy = self.epochProcess(x_train, y_train)

            mode_train_only = not testing or (epoch % self.testing_cycle > 0)
            mode_train_test = testing and (epoch % self.testing_cycle == 0)

            if mode_train_only and self.logging:
                # 訓練データのMean_Loss, Mean_Accuracyを表示
                print self.train_logFormat % (epoch, mean_loss, mean_accuracy)

            elif mode_train_test and self.logging:
                # 訓練データとテストデータのMean_Loss, Mean_Accuracyを表示
                if self.gpu:
                    # GPU向け実装 ToDo: バッチ分割回りがepochProcess()と似ているので、まとめる
                    testsize = len(y_test)
                    indexes = np.random.permutation(testsize)
                    sum_loss = 0.0
                    sum_accuracy = 0.0
                    for i in xrange(0, testsize, self.batchsize):
                        x_batch = x_test[indexes[i: i + self.batchsize]]
                        y_batch = y_test[indexes[i: i + self.batchsize]]
                        x_batch = cuda.to_gpu(x_batch)
                        y_batch = cuda.to_gpu(y_batch)
                        y_predict = self.forward(x_batch, train=False)
                        loss, accuracy = self.backward(y_predict, y_batch)
                        sum_loss += loss.data * self.batchsize
                        sum_accuracy += accuracy.data * self.batchsize
                    testing_loss = sum_loss / testsize
                    testing_accuracy = sum_accuracy / testsize
                else:
                    # CPU向け実装 一括処理
                    y_predict = self.forward(x_test, train=False)
                    loss, accuracy = self.backward(y_predict, y_test)
                    testing_loss = loss.data
                    testing_accuracy = accuracy.data
                print self.testing_logFormat % (epoch, mean_loss, mean_accuracy, testing_loss, testing_accuracy)


    def epochProcess(self, x_train, y_train):
        trainsize = len(y_train)
        indexes = np.random.permutation(trainsize)
        sum_loss = 0
        sum_accuracy = 0

        for i in xrange(0, trainsize, self.batchsize):
            x_batch = x_train[indexes[i: i + self.batchsize]]
            y_batch = y_train[indexes[i: i + self.batchsize]]
            if self.gpu:
                x_batch = cuda.to_gpu(x_batch)
                y_batch = cuda.to_gpu(y_batch)
            self.optimizer.zero_grads()
            y_predict = self.forward(x_batch, train=True)
            loss, accuracy = self.backward(y_predict, y_batch)
            self.optimizer.update()
            sum_loss += loss.data * self.batchsize
            sum_accuracy += accuracy.data * self.batchsize

        mean_loss = sum_loss / trainsize
        mean_accuracy = sum_accuracy / trainsize
        return mean_loss, mean_accuracy