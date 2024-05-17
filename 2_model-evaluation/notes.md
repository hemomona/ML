overfitting:

- accuracy = 1 - error rate
- training error / empirical error <-> generalization error
- since $P \neq NP$,we can only reduce the risk of overfitting, not avoid.

evaluation: training data + testing data

- hold-out
  - stratified sampling + multi random repeat
  - small training data -> big bias
  - small testing data -> big variation
- cross validation
  - $D = D_1 \cup D_2 \cup ... \cup D_k, while D_i \cap D_j = \emptyset (i \neq j)$,k-1 for training, also called k-fold cross validation
  - if k = number of data in D -> Leave-One-Out (LOO)
- bootstrapping
  - hold-out & cross validation import the bias of sample scale difference
  - bootstrap sampling = sampling with replacement, also called out-of-bag estimate
  - sampling *m* times, the probability of never being sampled is $\lim_{m \to \infty} (1-1/m)^m \mapsto 1/e \approx 0.368$
  - sample scale of training & testing dataset equals
  - bootstrapping imports the bias of sample distribution difference, it suits when the dataset is too small
- parameter tuning
  - after algorithms and parameters are determined, we should re-train the model using all of the data.

performance measure:

- regression task: mean square error (MSE）
- classification task:

  - error rate + accuracy = 1
  - precision = TP / (TP + FP) & Recall = TP / (TP + FN)
    - sort the dataset from the one most likely to be positive to the one most likely negative according the prediction, and see the sample as positive one by one, for calculating the P & R to draw a P-R figure
    - Break-Event Point (BEP) is where P = R
    - $F_1 = \frac{2 \times P \times R}{P + R} = \frac{2 \times TP}{total + TP - TN}$
    - $F_\beta = \frac{(1 + \beta^2) \times P \times R}{\beta^2 \times P + R}$,$\beta > 1$ -> R more important;$\beta < 1$ -> P more important
    - macro-P, macro-R, marco-F1, individual PR then average
    - micro-P, micro-R, micro-F1, average TP\FP\TN\FN at first
- Receiver Operating Characteristic (ROC)

  - horizontal axis - False Positive Rate = FP / (TN + FP)
  - vertical axis - True Positive Rate = TP / (TP + FN)
  - Area Under ROC Curve (AUC), $AUC = 1 - l_rank$
- unequal cost, cost matrix, cost-sensitive error rate, cost curve

  - area under all curves of (0, FPR) - (1, FNR)

hypothesis test:

- binomial test & t-test -> paired t-tests
- McNemar test -> Friedman test & Nemenyi test

$E(f;D) = bias^2(x) + var(x) + \varepsilon^2$

- bias-variance dilemma: not trained enough -> bias; trained enough -> variance.
