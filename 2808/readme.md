# Using Bert for Emotion Classification with IMDB Dataset

#### Describe
&emsp;&emsp;In order to analyze the popularity of movies, 
most researchers have explored the techniques of comment analysis.
In this project, I used a pretrained language model
called 'bert' to make emotion classification for
imdb dataset including 50,000 comments of which 25,000
are used as training set. Each comment is
marked as positive or negative.

&emsp;&emsp;For convenient, I used a third-party tool called transformers from huggingface 
which has implemented many model architectures including 'bert'. 
It’s remarkable that a model was instantiated by 'BertForSequenceClassification' 
which was added a sequence classification head on top of 'bert' in this project.
Besides, I used pytorch, tqdm and sklearn for training, showing progress bar and metrics
respectively. In terms of data preprocessing, I remove all punctuation and tag '</br/>'
from each text.

&emsp;&emsp;At each end of training epoch, evaluation is executed to determine whether the model 
performance has improved and whether to save the model. 

#### Download dataset
&emsp;&emsp;Unzip it and put it in the directory 'data'

    http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
#### Download pretrained model
    https://huggingface.co/bert-base-cased/tree/main

#### Train
    python train.py

#### Evaluation result
|Metric|Value|
|:---:|---|
|f1|0.880519|
|recall|0.880526|
|precision|0.88052|

#### Reference
* https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForSequenceClassification
* https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py





























