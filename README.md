# var-ids
Variance IDS algorithm for MS&amp;E 338 project

We use Equation (17) in Reinforcement Learning, Bit by Bit to compute the information ratio:
$$
\begin{equation}
\min_{\nu\in\Delta_{\mathcal{A}}}\frac{\mathbb{E}\left[\max_{a\in\mathcal{A}}\hat{Q}_{*,t}(S_t,a) - \hat{Q}_{*,t}(S_t,\tilde{A}_t)\right]^2}{\mathbb{E}\left[\text{tr}\left(\text{Cov}\left[\hat{Q}_{\dagger,t}(S_t,\tilde{A}_t)\mid X_t,\tilde{A}_t,\hat{\pi}_t(\cdot\mid S_t)\right]\mid X_t,\tilde{A}_t\right)\mid X_t\right]}
\end{equation}
$$
Using convexity analysis in Russo & Van Roy (2014a, 2018), one can show that there exists a policy that randomizes between at most two actions. Therefore, we only need to iterate over pairs of actions $a_1,a_2$, and find the 
$\alpha\in[0,1]$ such that $\nu$, where $\nu(a_1)=\alpha, \nu(a_2)=1-\alpha$, minimizes the above expression. For simplicity of notation, we define
$$
\begin{align*}
A_1 &= \max_{a\in\mathcal{A}}\hat{Q}_{*,t}(S_t,a) - \hat{Q}_{*,t}(S_t,a_1)\\
A_2 &= \max_{a\in\mathcal{A}}\hat{Q}_{*,t}(S_t,a) - \hat{Q}_{*,t}(S_t,a_2)\\
C_1 &= \text{tr}\left(\text{Cov}\left[\hat{Q}_{\dagger,t}(S_t,a_1)\right]\right)\\
C_2 &= \text{tr}\left(\text{Cov}\left[\hat{Q}_{\dagger,t}(S_t,a_2)\right]\right)
\end{align*}
$$
Therefore, we want to find the optimal $\alpha\in[0,1]$ that minimizes
$$
\begin{equation}
f(\alpha) = \frac{\mathbb{E}[\alpha A_1 + (1-\alpha)A_2]^2}{\alpha C_1 + (1-\alpha) C_2}.
\end{equation}
$$
Using algebra, we get that 
$$
\begin{equation}
f'(\alpha_0) = 0\quad\text{when}~\alpha_0=\frac{A_2C_1 - 2A_1C_2 + A_2C_2}{(C_1-C_2)(A_1-A_2)}
\end{equation}
$$
In particular, we only need to compare the following values
$$
\begin{align*}
f(0) &= \frac{A_2^2}{C_2}\\
f(1) &= \frac{A_1^2}{C_1}\\
f(\alpha_0) &= \frac{4(A_1-A_2)(A_2C_1-A_1C_2)}{(C_1-C_2)^2}
\end{align*}
$$

## For the case of pessimism
We have objective
$$
\begin{equation}
f(\alpha) = \frac{\left(\alpha A_1 + (1-\alpha)A_2\right)^2 + \epsilon}{\alpha C_1 + (1-\alpha) C_2}.
\end{equation}
$$

The derivative has roots
$$
\alpha = \dfrac{\left(A_2-A_1\right)C_2}{\left(A_1-A_2\right)\left(C_1-C_2\right)} \pm \dfrac{\sqrt{\left(C_2 - C_1\right)^2\epsilon+(A_1C_2-A_2C_1)^2}}{\left(A_1-A_2\right)\left(C_1-C_2\right)}
$$
which gives
$$
f(\alpha) = \dfrac{\left(A_2C_1-A_1C_2\pm\sqrt{\left(C_2 - C_1\right)^2\epsilon+(A_1C_2-A_2C_1)^2}\right)^2/(C_1-C_2)^2+\epsilon}{\pm\sqrt{\left(C_2 - C_1\right)^2\epsilon+(A_1C_2-A_2C_1)^2}/(A_1-A_2)}
$$
Also 
$$
\begin{align*}
f(0) &= \frac{A^2_2+\epsilon}{C_2}\\
f(1) &= \frac{A_1^2+\epsilon}{C_1}
\end{align*}
$$

## Alternative approach
We propose several modifications of the variance-IDS agent. The first one is var-ids-exp, implemented by taking the exponential of the numerator of the information ratio. The objective becomes
$$
\begin{equation*}
f(\alpha)=\frac{\alpha \mathrm{e}^{A_1} + (1-\alpha)\mathrm{e}^{A_2}}{\alpha C_1+(1-\alpha)C_2}
\end{equation*}
$$

The derivative
$$
f'(\alpha)=\dfrac{\mathrm{e}^{A_1}C_2-\mathrm{e}^{A_2}C_1}{\left(\left(C_2-C_1\right){\alpha}-C_2\right)^2}
$$
is either always positive or always negative. Thus it suffices to consider deterministic actions.

$$
\begin{align*}
f(0) &= \frac{\mathrm{e}^{A_2}}{C_2}\\
f(1) &= \frac{\mathrm{e}^{A_1}}{C_1}
\end{align*}
$$

The second modification is var-ids-outside-exp, implemented by taking the exponential of the expected shortfall. The objective becomes
$$
f(\alpha)=\frac{e^{\alpha A_1+(1-\alpha)A_2}}{\alpha C_1+(1-\alpha)C_2}
$$

The derivative has a root
$$
\alpha = \dfrac{\left(A_2-A_1-1\right)C_2+C_1}{\left(A_2-A_1\right)C_2+\left(A_1-A_2\right)C_1}
$$

In this case, we only need to compare
$$
\begin{align*}
f(0) &= \frac{\mathrm{e}^{A_2}}{C_2}\\
f(1) &= \frac{\mathrm{e}^{A_1}}{C_1}\\
f(\alpha) &= f\left(\dfrac{\left(A_2-A_1-1\right)C_2+C_1}{\left(A_2-A_1\right)C_2+\left(A_1-A_2\right)C_1}\right)
\end{align*}
$$