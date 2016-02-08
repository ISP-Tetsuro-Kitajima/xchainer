#Overview
##About
Xchainer is an expansion tool of [Chainer](http://chainer.org/).

## Aim
1. To use functions for model evaluation of [Scikit-learn](http://scikit-learn.org/stable/).
2. To simplify coding of NN learning process. 

##Quick Start
###Install
```shell
git clone https://github.com/recruit-tech/xchainer.git
cd xchainer
pip install -r requirements.txt
pip install .
```
###Test
```shell
$ python -m unittest discover -s tests
```

###Examples (mnist)
```shell
$ python ./examples/mnist_simple.py
```

It perform 2-fold cross validation test of simple NN for mnist data (10 class).

```shell
$ python ./examples/mnist_complex.py
```

It perform 2-fold cross validation test of complex NN for mnist data (10 class).

##License
MIT License (See a License file)

#日本語ドキュメント
##About
Scikit-learnとのアダプターなどを提供するChainerの拡張モジュールです。
本モジュールの目的は、Chainerにおける学習プロセスの記述の簡略化及び評価・手段の拡充です。
Scikit-learnの評価モジュールを利用するために、Scikit-learnの学習器としてChainerをラップしています。
Chainerの基本的な使い方につきましては、公式のチュートリアルをご参照ください。

* [Chainer Tutorial](http://docs.chainer.org/en/latest/tutorial/index.html)

##Coding style
本モジュールのコードは、python標準のPEP8に則って開発しています。

* [pep8 日本語ドキュメント](http://pep8-ja.readthedocs.org/ja/latest/)

#Documentation
##NNmanager
`NNmanager`は、学習プロセスのパラメータ化により、必要最低限の記述によるネットワークの定義を可能にします。
また、`NNmanager`はScikit-learnの学習器として拡張されているため、交差確認法やAUC評価など、Scikit-learnから提供されている様々な評価モジュールを利用することができます。


###Start with Example
`NNmanager`は学習器の枠組みを提供するインタフェースです。`NNmanager`を継承し、目的に応じて拡張することで、学習器を作ることができます。
継承の際必要になるのは、ネットワーク構造`model`、最適化関数`optimizer`、損失関数`lossFunction`の三つです。ここで、`model`は`chainer.FunctionSet`クラスのインスタンスで、ネットワークのパラメータを全てまとめて管理する役目を持ちます。`optimizer`は`chainer.optimizers`で提供される最適化関数、`lossFunction`は`chainer.functions`で提供される損失関数です。
詳しくは[chainerのリファレンスマニュアル](http://docs.chainer.org/en/latest/reference/index.html)をご参照ください。

* [chainer FunctionSet](http://chainer.readthedocs.org/en/latest/reference/core/function_set.html)
* [chainer optimizers](http://chainer.readthedocs.org/en/latest/reference/optimizers.html)
* [chainer functions](http://chainer.readthedocs.org/en/latest/reference/functions.html)

これらに加えて、オプションとして`params`を渡すことができます。`params`はdict型です。設定できる項目は、エポック数`epoch`、バッチサイズ`batchsize`、学習ログ表示フラグ`logging`です。
拡張の際に必要になるのは、`forward`メソッドと`trimOutput`メソッドの定義です。これにより、学習器を具体化します。

ここでは、例として手書き文字認識のデータを対象にしたネットワークをあげます。

```python
from xchainer import NNmanager
import numpy as np
from chainer import FunctionSet, Variable, optimizers
import chainer.functions as F
from sklearn.base import ClassifierMixin

# NNmanagerとClassifierMixinの継承
class TestNNM(NNmanager, ClassifierMixin):
    def __init__(self, logging=False):
        # ネットワーク構造の定義
        model = FunctionSet(
            l1=F.Linear(784, 100),
            l2=F.Linear(100, 100),
            l3=F.Linear(100,  10)
        )
        # 最適化手法の選択
        optimizer = optimizers.SGD()
        # 損失関数の選択
        lossFunction = F.softmax_cross_entropy
        # パラメータの設定
        params = {'epoch': 20, 'batchsize': 100, 'logging': logging, 'gpu': True}
        NNmanager.__init__(self, model, optimizer, lossFunction, **params)

    def trimOutput(self, output):
        y_trimed = output.data.argmax(axis=1)
        return np.array(y_trimed, dtype=np.int32)

    def forward(self, x_batch, train):
        x = Variable(x_batch)
        h1 = F.relu(self.model.l1(x))
        h2 = F.relu(self.model.l2(h1))
        output = F.relu(self.model.l3(h2))
        return output
```

今回扱う手書き文字認識は、1~10までの10種類の数字を判別する10クラスの分類問題なので、`ClassifierMixin`を利用しています。回帰問題を対象とする場合には、`RegressorMixin`を使います。

* [Scikit-learn ClassifierMixin](http://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html)
* [Scikit-learn RegressorMixin](http://scikit-learn.org/stable/modules/generated/sklearn.base.RegressorMixin.html)

####forward
NNmanagerでは順伝播`forward`メソッドを定義すれば、ニューラルネットワークの学習過程を構築できます。
ニューラルネットワークにおける学習は、ネットワーク構造に強く依存します。Chainerでは、基本的にネットワーク構造に即した定義が必要なのは順方向の伝播だけで、その他の過程は一般化することができます。
`forward`メソッドは、ネットワークの入力層への入力`x_batch`を受け取り、出力層からの出力`output`を返します。ここで、`output`は`chainer.Variable`クラスのインスタンスです。`train`はネットワークの学習フラグで、`fit`の際には`True`、`predict`の際には`False`が入ります。


```python
# 上略
    def forward(self, x_batch, **options):
        x = Variable(x_batch)
        h1 = F.relu(self.model.l1(x))
        h2 = F.relu(self.model.l2(h1))
        output = F.relu(self.model.l3(h2))
        return output
```

####trimOutput
`trimOutput`メソッドは、`forward`メソッドの結果である`output`を受け取り、ネットワークの出力値をラベル（被説明変数）と比較可能な形で取り出します。Scikit-learnの評価モジュールを使う際には、`chainer.Variable`型のままでは扱えないためです。
`trimOutput`メソッドは、デフォルトで`output.data`を取り出して返すので、回帰問題の際にはメソッド・オーバーライドは必要ありません。今回は10クラスの分類問題であるため、10次元列ベクトルの出力値の中で最も大きな値を持つ行番号をラベル値として取得しています。

```python
# 上略
    def trimOutput(self, output):
        y_trimed = output.data.argmax(axis=1)
        return np.array(y_trimed, dtype=np.int32)
```

###Try Example
上記のサンプルコードは、`./examples/mnist_simple.py`で試すことができます

```shell
$ python ./examples/mnist_simple.py
```



#Test
テストは以下のコマンドで実行できます

```shell
$ pwd 
# /path/to/xchainer
$ python -m unittest discover -s tests
```

このテストでは、各機能についての動作検証を主な目的としているため、学習の反復数(epoch)が`5`と非常に短い設定になっています。実際に利用する際には、少なくとも20epoch以上の学習を行います。
  
```
Loading MNIST data for test. This could take a while...
...done

===Test `fit` method===
This could take a while...
...done

.===Test `predict` method===
This could take a while...
...done

.===Test learning with `cross_val_score` of sklearn===
logging learning process below...
[0 epoch] mean loss: 2.287327, mean accuracy: 0.175805
[1 epoch] mean loss: 2.246331, mean accuracy: 0.278081
[2 epoch] mean loss: 2.188395, mean accuracy: 0.365838
[3 epoch] mean loss: 2.113845, mean accuracy: 0.438593
[4 epoch] mean loss: 2.025454, mean accuracy: 0.490570
[0 epoch] mean loss: 2.353900, mean accuracy: 0.154853
[1 epoch] mean loss: 2.318040, mean accuracy: 0.278290
[2 epoch] mean loss: 2.270524, mean accuracy: 0.389367
[3 epoch] mean loss: 2.209837, mean accuracy: 0.466798
[4 epoch] mean loss: 2.139440, mean accuracy: 0.500224
[ 0.49895723  0.50941509]
...complete

.Loading MNIST data for test. This could take a while...
...done

===Test `fit` method with GPU===
This could take a while...
...done

.===Test `predict` method with GPU===
This could take a while...
...done

.===Test learning with `cross_val_score` of sklearn with GPU===
logging learning process below...
[0 epoch] mean loss: 2.295167, mean accuracy: 0.114053
[1 epoch] mean loss: 2.259959, mean accuracy: 0.227072
[2 epoch] mean loss: 2.207839, mean accuracy: 0.349437
[3 epoch] mean loss: 2.144566, mean accuracy: 0.442651
[4 epoch] mean loss: 2.074051, mean accuracy: 0.490887
[0 epoch] mean loss: 2.360087, mean accuracy: 0.156977
[1 epoch] mean loss: 2.320490, mean accuracy: 0.257635
[2 epoch] mean loss: 2.277127, mean accuracy: 0.344599
[3 epoch] mean loss: 2.218657, mean accuracy: 0.409803
[4 epoch] mean loss: 2.149638, mean accuracy: 0.450038
[ 0.50464246  0.47524073]
...complete

.===Test `execute` method===
...done

.===Test `getFunctions` method===
...done

.===Test `insource` method===
...done

.===Test `outsource` method===
...done

.
----------------------------------------------------------------------
Ran 10 tests in 18.704s

OK

```

###Error in Loading MNIST data
このテストではScikit-learnのMNISTデータを利用していますが、お使いのマシンに古いMNISTデータがキャッシュされていると、データの読み込み時にエラーが発生する可能性があります。その際には、古いデータをマシンから削除してもう一度お試しください。

```shell
rm ~/scikit_learn_data/mldata/mnist-original.mat
```
