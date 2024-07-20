naive bayes: features are independent;
when comparing the probability, P(D) won't change the size relationship.

$$
P(s+|D) = P(s+) * P(D|s+) / P(D) \\
\begin{aligned}
log(P(s+|D)\ \alpha&\ log(P(s+) * P(D|s+)) \\
=&\ log(P(s+)) + log(P(D|s+)) \\
=&\ log(P(s+)) + log(P(D_1|s+) * P(D_2|s+) * ... * P(D_n|s+)) \\
=&\ log(P(s+)) + \sum_i{log(P(D_i|s+))} \\
=&\ log(s+ prior\ prob) + \sum{log(word\ prob\ in\ the\ class)}
\end{aligned}
$$

