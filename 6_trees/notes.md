## decision tree

entropy: $H(X) = - \sum_i{p_i * logp_i}$
information gain: feature X makes class Y less chaotic.

ID3 algorithm: information gain, not suitable when the feature has too many values, like index or ID number.

C4.5 algorithm: information gain ratio = information gain / entropy of the feature itself, so like regularization, punishing the use of the feature with too many values.
&emsp; pros:
&emsp; - use information gain ratio;
&emsp; - can handle continuous values by Bisection Method;
&emsp; - can handle missing data by seeing (has data / no data) as a feature or assigning the samples without data to different nodes weightedly;
&emsp; - import regularization to reduce overfitting.
&emsp; cons:
&emsp; - log computation consumes time;
&emsp; - inefficient multi-branches tree;
&emsp; - likely to overfit.

CART (Classification And Regression Tree):
Classification Tree: use Gini Impurity

$$
Gini(p) = 1 - \sum_{k=1}^K{p_k^2}
$$

Regression Tree: use Least Squares Method to get the minimum of the sum of variances.

### pruning

pre-pruning: depth, leaf node number, the sample number in a leaf node, information gain et al.
post-pruning: Reduced-Error Pruning (REP), Pesimistic-Error Pruning (PEP), Cost-Complexity Pruning (CCP). lower risk of underfit, better generalization than pre-pruning, but more time-consuming.

