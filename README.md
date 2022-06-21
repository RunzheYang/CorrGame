# A General Framework for Unsupervised Learning with Hebbian/Anti-Hebbian Plasticity

## A Unified Framework

Given multiple input steams (can be 1) $\{X_i \in \mathbb R^{n_i \times T}\}$, the goal is to learn $k$-dimensional outputs $Y \in \mathbb R^{k\times T}$ such that

$$\max_{Y} \sum_{i} \Phi^\star_i \left(\frac{YX_i^{\top}}{T}\right) - \frac{1}{2}\Psi^\star \left(\frac{YY^{\top}}{T}\right),$$

where $\Phi^\star(\cdot)$ and $\Psi^\star(\cdot)$ are Legendre transforms of strictly convex functions $\Phi(\cdot)$ and $\Psi(\cdot)$ (so that the optimal dual variables exist)

$$\Phi^\star(C_{YX}):= \max_{W \in \mathcal D_{W}} \text{Tr}(WC_{YX}^{\top}) - \Phi(W),$$

$$\Psi^\star(C_{YY}):= \max_{M \in \mathcal D_{M}} \text{Tr}(MC_{YY}^{\top}) - \Psi(M).$$

The these functions are non-decreasing when the optimal dual variables $W^*$ and $M^*$ are non-negative. Then, we can interpret the objective as *maximizing the correlation between inputs and outputs, while minimizing the auto-correlation of outputs*.

#### Neural Network Algorithms

We can use projected gradient descent / ascent to solve a dual problem of the above problem in an online fasion, which leads to a bio-plausible neural network algorithm.

The steady activities of output neurons can be solve by offline projected gradient ascent
$$Y \leftarrow \text{proj}_{\mathcal D_{\bf Y}}\left[{\bf Y} + \eta_Y\frac{\partial \cdots}{\partial Y}\right] =  \text{proj}_{\mathcal D_{\bf Y}}\left[{\bf Y} + \frac{\eta_Y}{T}(WX - MY)\right],$$
or via online updates
$${\bf y}_t \leftarrow \text{proj}_{\mathcal D_{\bf y}}\left[{\bf y} + \eta_y(W{\bf x} - M{\bf y})\right].$$

The synaptic learning rules are
$$W\leftarrow \text{proj}_{\mathcal D_W} \left[W + \eta_W \frac{\partial \cdots}{\partial W}\right] = \text{proj}_{\mathcal D_W} \left[W + \eta_W\left( \frac{YX^{\top} }{T}- \Phi'(W)\right)\right];$$

and 

$$M\leftarrow \text{proj}_{\mathcal D_M} \left[M + \eta_M \frac{\partial \cdots}{\partial M}\right] = \text{proj}_{\mathcal D_M} \left[M + \frac{\eta_M}{2}\left( \frac{XX^{\top} }{T}- \Psi'(M)\right)\right].$$

We can also write the synaptic learning rule as the online updates

$$W\leftarrow \text{proj}_{\mathcal D_W} \left[W + \eta_W\left({\bf y}_t{\bf x}_t^{\top}- \Phi'(W)\right)\right],$$

$$M\leftarrow \text{proj}_{\mathcal D_M} \left[M + \frac{\eta_M}{2}\left({\bf y}_t{\bf y}_t^{\top}- \Psi'(M)\right)\right].$$

#### Choices of $\Phi$ and $\Psi$
We can implemented different algorithms from similarity matching famility by specifying different $\Phi$ and $\Psi $, and constraints on $Y$, $W$, and $M$.
For example:

When $W,M,Y$ are unbounded, and $$\Phi(W) = \frac{1}{2}\text{Tr}WW^{\top}, \text{  and  } \Psi(M) = \frac{1}{2}\text{Tr}MM^{\top},$$
we get the *similarity matching principle*. We can simply create a similarity matching instance with the following code
```
sim_match = CorrGame(n=n, k=k, 
                Phi=lambda W, X: (W*W).sum()/2,
                Psi=lambda M, X: (M*M).sum()/2,
                dPhi = lambda W, X:W,
                dPsi = lambda M, X:M,
                constraints = {'Y': lambda x:x,
                               'W': lambda x:x,
                               'M': lambda x:x},
                eta= {'Y': eta_Y, 'W': eta_W, 'M': eta_M},
                device=device)
 ```
where the derivatives `dPhi` and `dPsi` are optional. The algorithm will be more efficient if the analytical derivatives are provided, otherwise they will be approximated solved by auto-grad. We can also specify closed-form solutions to the online or the offline trainer to speed up convergence.

When $Y$ is non-negative, and we have two input streams $X_1$ and $X_2$, by setting $$\Phi_i(W_i) = \frac{1}{2}\text{Tr}W_iC_{X_i X_i}W_i^{\top}, i\in \\{1,2\\} \text{  and  } \Psi(M) = \frac{1}{2}\text{Tr}MM^{\top},$$
we get the *nonnegative cannonical correlation analysis*. We can simply create a NCCA instance with the following code
```
cca = CorrGame(n=[n1, n2], k=top_k, 
                Phi=lambda W, X: (W.mm(X.mm(X.t())/X.size(1))*W).sum()/2,
                Psi=lambda M, X: (M*M).sum()/2,
                dPhi = lambda W, X:W.mm(X.mm(X.t())/X.size(1)),
                dPsi = lambda M, X:M,
                constraints = {'Y': F.relu,
                               'W': lambda x:x,
                               'M': lambda x:x},
                eta= {'Y': eta_Y, 'W': eta_W, 'M': eta_M},
                device=device)
 ```


Please see [`general-correlation-game.ipynb`]() and [`multi-source-correlation-game.ipynb`]() for more examples.

## Reference
...
