## ensemble learning

### bagging (bootstrap aggregation)

train many classifiers to get the mean
example: random forest: random samples, random features
&emsp; can handle data with many features
&emsp; without selecting features, can get the importance of features
&emsp; parallel computing
&emsp; visualization

### boosting

reinforce from a weak classifier, a model would be added only if it can make fewer overall loss
example: Adaboost, Xgboost
&emsp; for classification, times the samples with wrong classification 1 + learning rate;
&emsp; for regression, subtraction can make the next model focus on the samples with wrong prediction;
&emsp; in Adaboost, if a data is classified wrongly, it would be given a bigger weight in next model; the final model would be the weighted sum of models according to their accuracy.


### stacking

aggregation of many classifiers or regressors

