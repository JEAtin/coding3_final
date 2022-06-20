# coding3_final Final project



YiTing Jia
21031402
MSc CCI Modular



About my final project.
1. This project takes the sentiment analysis task in natural language processing as the starting point to classify the sentiment polarity of the existing movie review dataset imdb. In terms of model selection, in order to extract richer textual information, the project uses the pre-trained language model bert as backbone, and adds a fully-connected layer as classifier in the top layer for binary classification, and computes the output results with labels for cross-entropy loss.


2. In terms of data processing, this project filters the punctuation marks of the samples and removes <br />tags. To take advantage of the pre-trained model, this project uses bert-base-cased weights for initialization; in the process of forward propagation, the top-level output corresponding to the [CLS] tag of the sample is selected as the sample feature and fed into the classifier. In the validation phase, this project selects f1 as the evaluation metric, and saves the model under the folderoutput when the metric improves from the previous evaluation result.


3. This project uses the pytorch deep learning framework for training, and refers to the official tutorials, handwritten training and validation process; in the model, this project uses the third-party library transformers released by huggingface, and instantiates the " BertForSequenceClassification" object, which adds classifiers to the top layer of bert and does not require human to build the model. In the validation process, this project calls the sklearn.metrics.classification_report interface to calculate the metrics.


4. For model reproduction, set the random seed to 42. If you want to change the training parameters such as sentence length, you can adjust them in the args.py file.
