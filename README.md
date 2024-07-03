# New-baseline-for-DRUNet-under-different-assumptions
This is a new baseline for DRUNet under different assumptions. 
----


We are currently retraining each denoisers (MMO, NE-DRUNet, SPC-DRUNet with different $k$, and PC-DRUNet) with power iterative method and a modified power iterative method. PC denotes pseudo-contractive, SPC denotes strictly pseudo-contractive.


# Pseudo-Contractive Denoisers

Let $`V`$ be a real Hilbert space with inner product $`\langle\cdot,\cdot\rangle`$. $`\|\cdot\|`$ is the induced norm. $`I`$ is the identity mapping. A mapping $`D:V\rightarrow V`$ is pseudo-contractive, if there exists $`k\in [0,1]`$, such that $`\forall x,y\in V`$, we have

$`\|D(x)-D(y)\|^2\le \|x-y\|^2 + k\| (I-D)(x)-(I-D)(y)\|^2. `$

When $`k<1`$, $`D`$ is said to be $`k`$-strictly pseudo-contractive. Please check Lemma 2.1 to Theorem 3.3 in [ICML](https://openreview.net/forum?id=G0vZ5ENrJQ&noteId=G0vZ5ENrJQ). 

A new spectral term for PC-DRUNet
----

In this repo, different from the original paper [ICML](https://openreview.net/forum?id=G0vZ5ENrJQ&noteId=G0vZ5ENrJQ), each method is trained with the (modified) power iterative method with 15 iterations. The trainset are DIV2k, containing 800 images. The loss functions in the paper [ICML](https://openreview.net/forum?id=G0vZ5ENrJQ&noteId=G0vZ5ENrJQ) for SPC-DRUNet and PC-DRUNet are 

$`\mathbb{E}\|D_\sigma(x+\xi_\sigma;\theta)-x\|_1+r\max\{\|kI+(1-k)J\|_*,1-\epsilon\},`$ and $`  \mathbb{E}\|D_\sigma(x+\xi_\sigma;\theta)-x\|_1+r\max\{\|(S-2I)^{-1}S\|_*,1-\epsilon\}, `$ repectively.

We use a different holomorphic transformation for PC-DRUNet, to stablize the training procedure. The new loss function is 

$`  \mathbb{E}\|D_\sigma(x+\xi_\sigma;\theta)-x\|_1+r\max\{\|(J-10I)^{-1}(J+8I)\|_*,1-\epsilon\} + s \|J^\mathrm{T}-J\|_*`$.

The last term $`\|J^\mathrm{T}-J\|_*`$ is to encouraging a symmetric Jacobian, and therefore, encouraging a conversative denoiser $`D_\sigma`$. The second term $` \max\{\|(J-10I)^{-1}(J+8I)\|_*,1-\epsilon\} `$ is to encourage that the real part of any eigenvalue of $`J`$ is no larger than $`1`$. To see this, consider the transformation $`f:\mathbb{C}\rightarrow \mathbb{C}`$, $`f(z)=\frac{z+8}{z-10}`$. Then obviously, 

$`f(\{z\in\mathbb{C}: real(z)\le 1\})=\{z\in \mathbb{C}: |z|\le 1\}.`$

This new loss term $` \max\{\|(J-10I)^{-1}(J+8I)\|_*,1-\epsilon\} `$ has the advantage that it stablizes the training. Because now, the denominator is unlikely to be zero. The real part of any eigenvalue of $`J`$ is typically far from $`10`$.

Parameter settings 
----

We set $`\epsilon=0.1`$ for each assumption. We set Patchsize = 64 for PC-DRUNet, and 128 for others. The average value and the standard derivations of the spectral norms (such as $`||2J-I||_*`$ for MMO, $`||J||_*`$ for NE-DRUNet) evaluated at different images after training is denoted by Mean and Std, respectively. The balancing parameter r in the loss function is chosen to be large enough, such that, $`Mean + 3  Std \le 1,`$ and small enough, such that the denoising performace is least compromised.

Results
----


Table 1. Denoising performance on CBSD68 in PSNR values
---

$`\sigma`$ | DRUNet  | MMO | NE-DRUNet| SPC-DRUNet ($`k=0.5`$) | SPC-DRUNet ($`k=0.6`$) | SPC-DRUNet ($`k=0.7`$) |  SPC-DRUNet ($`k=0.8`$) |  SPC-DRUNet ($`k=0.9`$) |  PC-DRUNet ($`k=1`$)
---- | ----- | ------ | ---- | ----- | ------  |---- | ----- | ------  | ------ 
15 | 34.14 |         32.21          |32.54| 33.30|33.43|33.48|33.66|33.67|33.96
25 | 31.54|         29.99           |30.12| 30.55|30.62|30.94|31.05|31.36|31.44
40| 29.33 |           27.87         |28.00| 28.43|28.43|28.70|28.84|28.96|29.14
Spectral Term |  \ | $` \|\|2J-I\|\|_* `$ | $`\|\|J\|\|_*`$ | $`\|\|0.5I+0.5J\|\|_*`$ |  $`\|\|0.6I+0.4J\|\|_*`$ |$`\|\|0.7I+0.3J\|\|_*`$ |$`\|\|0.8I+0.2J\|\|_*`$ |$`\|\|0.9I+0.1J\|\|_*`$ | $`\|\|(J-10I)^{-1}(J+8I)\|\|_*`$ 
$`r`$ | \ | 0.01 | 0.02|0.02|0.02|0.02|0.02|0.01|0.01
$`Mean`$ | \ |  0.932    | 0.933 | 0.956   | 0.950 | 0.951 | 0.953| 0.951|0.955
$`Std`$ |  \ |     0.0454| 0.0197| 0.0137| 0.0148  |0.0133 |0.0112|0.0136|0.0150


Table 2. Deblurring performance on CBSD68 in PSNR and SSIM values with Levin's kernels
---

\ |Noise level |$`\sigma=12.75`$|  \    |$`\sigma=17.85`$ | \
----| ---- |---- |---- |---- |----
Converge? | Measurement| PSNR           | SSIM | PSNR            | SSIM
 $`\surd`$ | MMO-FBS| 26.03 | 0.6871 | 25.30 | 0.6424
 $`\surd`$ | NE-PGD | 26.16| 0.6977| 25.37| 0.6525
 $`\surd`$ | Prox-DRS| 26.64| 0.7200 |25.99 |0.6900 
 $`\surd`$ | PnPI-GD / REDI  ($`k=1.0`$)  | 26.54 | 0.6984 | 25.80 | 0.6679
 $`\surd`$ | PnPI-FBS ($`k=0.5`$) | 26.70 | 0.7176 | 25.74 | 0.6732 
 $`\surd`$ | PnPI-FBS ($`k=0.7`$) | 26.72 | 0.7264 | 25.91 | 0.6753  
 $`\times`$ | PnPI-FBS ($`k=0.9`$) | 26.90 | 0.7252 | 25.98 | 0.6787  
 $`\times`$| PnPI-FBS ($`k=1.0`$) | 26.90 | 0.7342 | 26.04 | 0.6889 
$`\surd`$ | PnPI-HQS ($`k=0.5`$) |27.09 | 0.7494|26.23 |0.7110 
$`\times`$ | PnPI-HQS ($`k=0.7`$) | 27.28| 0.7559| 26.38|0.7191 
$`\times`$ | PnPI-HQS ($`k=0.9`$) | 27.57|0.7616 | 26.57| 0.7200
$`\times`$ | PnPI-HQS ($`k=1.0`$) | 27.65| 0.7704|26.66 |0.7306 
 $`\surd`$ | REDI-Prox($`k=0.9`$) | 27.59 | 0.7682 | 26.63 | 0.7296




It can be seen in Table 1 that, when $`k`$ gets larger, the assumption gets weaker, and the denoising performance gets better. When $`k\ge0.9`$, the denoisers have a satisfying denoising performance.


Overall, this repo provides a more accurate evaluation for each denoiser. We will continue updating the denoising performance results, and the PnP restoration results. This will serve as a new baseline. 


Update 20240623
----
We will upload the codes and the pretrained models gradually.
