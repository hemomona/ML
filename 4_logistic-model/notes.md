$$
\begin{aligned}
Cost(h_\theta(x), y) & =
\begin{cases}
-log(h_\theta(x)) & \text{if y = 1} \\
-log(1-h_\theta(x)) & \text{if y = 0} \\
\end{cases} \\
& = -[ylog(h_\theta(x)) + (1-y)log(1-h_\theta(x))]
\end{aligned}
$$

