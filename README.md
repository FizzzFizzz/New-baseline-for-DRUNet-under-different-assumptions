# New-baseline-for-DRUNet-under-different-assumptions
This is a new baseline for DRUNet under different assumptions. 
----


We are currently retraining each denoisers (MMO, NE-DRUNet, SPC-DRUNet with different $k$, and PC-DRUNet) with power iterative method and a modified power iterative method. PC denotes pseudo-contractive, SPC denotes strictly pseudo-contractive.

A new spectral term for PC-DRUNet
----

In this repo, different from the original paper [ICML](https://openreview.net/forum?id=G0vZ5ENrJQ&noteId=G0vZ5ENrJQ), each method is trained with the (modified) power iterative method with 15 iterations. The trainset are DIV2k, contains 800 images. The loss functions in the paper [ICML](https://openreview.net/forum?id=G0vZ5ENrJQ&noteId=G0vZ5ENrJQ) for SPC-DRUNet and PC-DRUNet are 

$`\mathbb{E}\|D_\sigma(x+\xi_\sigma;\theta)-x\|_1+r\max\{\|kI+(1-k)J\|_*,1-\epsilon\},`$ and $`  \mathbb{E}\|D_\sigma(x+\xi_\sigma;\theta)-x\|_1+r\max\{\|(S-2I)^{-1}S\|_*,1-\epsilon\}, `$ repectively.

We use a different holomorphic transformation for PC-DRUNet, to stablize the training procedure. The new loss function is 

$`  \mathbb{E}\|D_\sigma(x+\xi_\sigma;\theta)-x\|_1+r\max\{\|(J-10I)^{-1}(J+8I)\|_*,1-\epsilon\} + s \|J^\mathrm{T}-J\|_*`$.

The last term $`\|J^\mathrm{T}-J\|_*`$ is to encouraging a symmetric Jacobian, and therefore, encouraging a conversative denoiser $`D_\sigma`$. The second term $` \max\{\|(J-10I)^{-1}(J+8I)\|_*,1-\epsilon\} `$ is to encourage that the real part of any eigenvalue of $`J`$ is no larger than $`1`$. To see this, consider the transformation $`f:\mathbb{C}\rightarrow \mathbb{C}`$, $`f(z)=\frac{z+8}{z-10}`$. Then obviously, 

$`f(\{z\in\mathbb{C}: real(z)\le 1\})=\{z\in \mathbb{C}: |z|\le 1\}.`$

This new loss term $` \max\{\|(J-10I)^{-1}(J+8I)\|_*,1-\epsilon\} `$ has the advantage that it stablize the training, because now, the denominator is unlikely to be zero. The real part of any eigenvalue of $`J`$ is typically far from $`10`$.

Parameter settings 
----

We set $`\epsilon=0.1`$ for each assumption. We set Patchsize = 64 for PC-DRUNet, and 128 for others. The average value and the standard derivations of the spectral norms (such as $`||2J-I||_*`$ for MMO, $`||J||_*`$ for NE-DRUNet) evaluated at different images after training is denoted by Mean and Std, respectively. The balancing parameter r in the loss function is chosen to be large enough, such that, $`Mean + 3  Std \le 1,`$ and small enough, such that the denoising performace is least compromised.

Conclusion
----
Overall, this repo provides a more accurate evaluation for each denoiser. We will continue updating the denoising performance results, and the PnP restoration results. This will serve as a new baseline. 

Denoising performance on CBSD68
---

$`\sigma`$ | DRUNet  | MMO | NE-DRUNet| SPC-DRUNet ($`k=0.5`$) | SPC-DRUNet ($`k=0.6`$) | SPC-DRUNet ($`k=0.7`$) |  SPC-DRUNet ($`k=0.8`$) |  SPC-DRUNet ($`k=0.9`$) |  PC-DRUNet ($`k=1`$)
---- | ----- | ------ | ---- | ----- | ------  |---- | ----- | ------  | ------ 
15 | 34.14 |                   |32.54|
25 | 31.54|                    |30.12|
40| 29.33 |                    |28.00|
Spectral Term| \   | $`||2J-I||_*`$ | $`||J||`$ | 
$`Mean`$ | \ |      | 0.933 | 
$`Std`$ | \  |      | 0.01967|

Update 20240623
----
We will upload the codes and the pretrained models gradually.
