logistic regression error:

$$
J(\theta) = \frac{1}{2} \sum_{i=1}^{m} (h_\theta (x^i) - y^i)^2 = \frac{1}{2} (X\theta - y)^T(X\theta - y)
$$

$$
\begin{aligned}
\nabla_\theta J(\theta) & = \nabla_\theta [\frac{1}{2} (X\theta - y)^T(X\theta - y)] \\
& = \nabla_\theta [\frac{1}{2} (\theta^T X^T - y^T)(X\theta - y)] \\
& = \nabla_\theta [\frac{1}{2} (\theta^T X^T X \theta - \theta^T X^T y - y^T X \theta + y^T y)] \\
& = \frac{1}{2} [2 X^T X \theta - X^T y - (y^T X)^T] \\
& = X^T X \theta - X^T y
\end{aligned}
$$

$$
\nabla_\theta J(\theta) = 0 => \theta = (X^T X)^{-1} X^T y
$$

