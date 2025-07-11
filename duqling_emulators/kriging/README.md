<h2 style="text-align: center;"><ins>GP Design</ins></h2>



#### Kernels

To choose the best kernel for emulation, we brute force search the following kernels and select the one that maximizes the predictive log-marginal likelihood:

- [RBF](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html)

<!-- $$
k(x, x') = \mathrm{exp}\left(-\frac{\lVert x - x' \rVert^2}{2l^2} \right)
$$ -->

- [Matern](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html) (for $\nu \in \{0.5, 1.5, 2.5\}$)

<!-- $$
k(x, x') = \frac{1}{\Gamma(\nu)2^{\nu-1}} \left(\frac{\sqrt{2\nu}}{l}\lVert x - x' \rVert\right)^\nu K_\nu \left(\frac{\sqrt{2\nu}}{l}\lVert x - x' \rVert\right)
$$ -->

- [Rational Quadratic](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RationalQuadratic.html)

<!-- $$
k(x, x') = \left(1+\frac{\lVert x - x' \rVert^2}{2\alpha l^2} \right)^{-\alpha}
$$ -->

- [RBF](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html) + [Linear](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.DotProduct.html)

<!-- $$
k(x, x') = k_{\mathrm{RBF}}(x,x') + \sigma_0^2 + x \cdot x'
$$ -->


