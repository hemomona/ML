# target function

computing the distance between a point $x$ and a hyperplane $w^Tx+b=0$:

$$
distance = \frac{|w^Tx+b|}{||w||}
$$

where w is the normal vector, and b is bias

considering a line ax + by + c = 0 and a point (0, y0), can simplify the issue a lot:
&emsp; the slope = $-\frac{a}{b}$, so its normal vector is $(a,b)$ and draw a line crossing the point and perpendicular to it, this slope is $\frac{b}{a}$
&emsp; draw a line which is parallel to ax + by + c = 0 while crossing the point, ax + by + c = B. the vertical distance between two line is $\frac{B-c}{b} - (-\frac{c}{b}) = \frac{B}{b}$
&emsp; $\frac{distance}{\frac{B}{b}} = \frac{b}{\sqrt{a^2+b^2}} \rightarrow distance = \frac{|w^Tx0+b|}{||w||}$

support vectors are those sample points which are closest to the hyperplane in each class.
margin is the sum of distance between two support vectors belonging to different classes and the hyperplane.
to find the hyperplane with the maximum margin, we need to find $w, b$ to minimize $||w||^2$, that is,

$$
min_{w,b} \frac{1}{2}||w||^2 \\
suject\ to\ y_i(w^Tx_i + b) \geq 1 \\
$$

# Lagrange function

convert to Lagrange function, get the dual problem:

$$
max_\alpha\ \sum_{i=1}^{m}{\alpha_i} - \frac{1}{2}\sum_{i=1}^{m} \sum_{j=1}^{m}{\alpha_i \alpha_j y_i y_j x_i^T x_j} \\
s.t.\ \sum_{i=1}^{m}{\alpha_i y_i} = 0, \\
\alpha_i \geq 0,\ i=1,2,...,m
$$

after sloving $\alpha$:

$$
f(x) = w^Tx+b = \sum_{i=1}^{m}{\alpha_i y_i x_i^T x + b}
$$

according to KKT:
&emsp; if $\alpha_i = 0$, the sample won't affect the model;
&emsp; if $\alpha_i > 0$, then $y_if(x_i) = 0$, the sample is on the border, i.e. support vector.
***After training, the SVM model would only be related to support vectors***
Caution: $y_i$ is the classification.

ref: `https://www.cnblogs.com/mo-wang/p/4775548.html`
Lagrange Multiplier and KKT:Lagrange Multiplier and KKT:
$for\ minf(x)\ s.t.\ h_k(x) = 0,\ k=1,2,...K$
Lagrange Function:
$F(x, \lambda) = f(x) + \sum_{k=1}^{K}{\lambda_kh_k(x)}$
when $F(x, \lambda)$ minimize, $f(x)$ and $h(x)$ share same gradient.

# soft margin

some noise samples can affect the model severely.
slack variables:

$$
min_{w,b,\xi_i} \  \frac{1}{2}||w||^2 + C\sum_{i=1}^{m}{\xi_i} \\
suject\ to\ y_i(w^Tx_i + b) \geq 1 - \xi_i \\
\xi_i \geq 0, i = 1,2,...,m
$$

the bigger C, the more rigorous

# kernel function

for some tasks, there is no hyperplane that can divide the samples, e.g. XOR samples.
so we can map the samples from original space to a feature space with more dimensions, there must be a feature space with more dimensions can make the sample divisible.
However, after mapping, the $\phi(x_i)^T \phi(x_j)$ can be very hard and time-consuming.
Therefore, $\phi(x_i)^T \phi(x_j) = \kappa(x_i, x_j)$, the $\kappa$ is the kernel function can make the computation much more easy in original space.

linear kernel $\kappa(x_i, x_j) = x_i^Tx_j$
polynomial kernel $\kappa(x_i, x_j) = (x_i^Tx_j)^d$
gauss kernel $\kappa(x_i, x_j) = exp(-\frac{||x_i - x_j||^2}{2\sigma^2})$
laplacian kernel $\kappa(x_i, x_j) = exp(-\frac{||x_i - x_j||}{\sigma})$
sigmoid kernel $\kappa(x_i, x_j) = tanh(\beta x_i^Tx_j + \theta)$



