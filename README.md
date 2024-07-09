# New-baseline-for-DRUNet-under-different-assumptions
This is a new baseline for DRUNet under different assumptions. 
----


We are currently retraining each denoisers ([MMO](https://github.com/matthieutrs/LMMO_lightning), NE-DRUNet, SPC-DRUNet with different $k$, and PC-DRUNet) with power iterative method and a modified power iterative method. PC denotes pseudo-contractive, SPC denotes strictly pseudo-contractive. Please note that in this repo, the results may be different from the repo [pseudo-contractive denoisers](https://github.com/FizzzFizzz/Learning-Pseudo-Contractive-Denoisers-for-Inverse-Problems).


A new strategy in plug-and-play Ishikawa algorithms
----

Different from the original paper [ICML](https://openreview.net/forum?id=G0vZ5ENrJQ&noteId=G0vZ5ENrJQ), we replace the denoiser $` D_\sigma`$ in PnPI-GD, PnPI-HQS, PnPI-FBS with $` \tilde{D}_\sigma = 0.2D_\sigma+ 0.8I`$. Then, when $`D_\sigma `$ is pseudo-contractive, we can see that $`\tilde{D}_\sigma`$ is also pseudo-contractive. When $`D_\sigma`$ is 0.9-strictly pseudo-contractive, one can prove that $`\tilde{D}`$ is 0.5-strictly pseudo-contractive. Therefore, according to our theory, the three methods converge. In experiments, we find that this new strategy is very powerful, that it stablizes the PnP restoration procedure, and provides better restoration results.


# Quick Start!

How to test?
----
If you want to test it on your own, it would be beneficial if you are familiar with [DPIR](https://github.com/cszn/DPIR)/[KAIR](https://github.com/cszn/KAIR), [LMMO](https://github.com/basp-group/PnP-MMO-imaging), and [Prox-PnP](https://github.com/samuro95/Prox-PnP). The code is based on these pioneer projects.

Step 1: Create env according to [KAIR](https://github.com/cszn/KAIR).

Step 2: Download this code, along with the pretrained models at [pretrained baseline models link1](https://drive.google.com/drive/folders/1-FC9koWoKar7RDJEjU154_K6GTs8NfMO?usp=drive_link), [pretrained baseline models link2](https://drive.google.com/drive/folders/1-FC9koWoKar7RDJEjU154_K6GTs8NfMO?usp=drive_link).

Step 3: Download the trainsets and testsets as you wish.

Step 4: Run any code starting with 'new_PnP_main_xxx.py'. You might need to create some folders in './log'.
For example, try 'new_PnP_main_09SPC+REDIPROX_deblur_color.py' to deblur with REDI-Prox.

> Please note that, in Line 500 of 'new_PnP_main_09SPC+REDIPROX_deblur_color.py', plot_psnr(level,lambda, sigma) is to process single image, and the results will be stored in the folder './images'. 'level' is the denoising level of the denoiser, 'lambda' is the balancing parameter, 'sigma' is the Gaussian noise level.
> 
> In Line 233, you can change different test images.
> 
> In Line 234, you can change different kernels.

> In Line 488, search_args() is to process a test set.

> In Line 311, you can change the sigma value, in Line 326, you can change the test set.

> You can fine tune the parameters in Lines 342-386. For example, in Line 342, 'search_range[0.1] = [3.5] # 27.3472, 0.7582'. 0.1 is the denoising level, and 3.5 is the balancing parameter.

> In Line 112, you can modify the iteration number by changing 'nb=200'.

> If you are testing images on a testset, you may need to delete or annotate the codes in Line 203-210.

> You can write your own code, or modify the PnP algorithm, in about Line 151-220.

> Other files have very similar structrures.


How to train?
----
Step : Train the denoiser with spectral regularization terms to make it pseudo-contractive. You can achieve this by editting a 'xxx.json' file. The parameters need to be fine tuned very carefully. You may mainly check 'train_drunet_k_small_1_color_2024.json' and 'train_drunet_k=1_color_2024.json', for $`k`$-strictly pseudo-contractive denoisers pseudo-contractive denoisers ($`k=1`$) respectively. The training results will be saved in the folder './denoising'.

> Please note that a larger $k$ typically means a better denoising performance.

> You can modify the spectral regularization term in the file 'loss_jacobian.py' in Lines 29, and 316-322, or just define a new regularizer.

> In the file 'train_drunet_k_small_1_color_2024.json', in Lines 70-77, we give the parameters for the iterative power method. For $`k<1`$, there is no inner iterations, and thus 'jacobian_dt' and 'jacobian_inner_step' are useless. In the file 'train_drunet_k=1_color_2024.json', we set 'jacobian_dt' to be 0.01, and 'jacobian_inner_step' to be 10, empirically.

> 'jacobian_start_step' means where to start. For example, when it's 10000, it means that after the first 10000 iterations, we starting using the spectral regularizations.

> 'jacobian_loss_weight' is the weighting parameter $`r`$ in the loss functions. It balances the denoising loss and the spectral regularization loss. When it gets larger, the denoiser is more likely to satisfy the assumptions you want, but often has worse denoising performance.

> 'jacobian_checkpoint' means the frequency of the spectral regularizations. When it is 1, it means that in the training procedure, we regularize the denoiser for each training image.

> 'jacobian_step' is the iteration number for the iterative power method.

> 'jacobian_eps' is the $`\epsilon`$ in the loss.

> 'checkpoint_test' is the frequency to test the images. I typically set it to be 10000, which means that we test the denoiser every 10000 iterations.

> 'checkpoint_save' is the frequency to save the model. I typically set it to be 10000, which means that we test the denoiser every 10000 iterations.

> 'checkpoint_print' is the frequency to print the loss. I typically set it to be 100, which means that we print the loss every 100 iterations.


 
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

$`\sigma`$ | DRUNet  | MMO | NE-DRUNet| SPC-DRUNet ($`k=0.5`$)  | SPC-DRUNet ($`k=0.7`$)  |  SPC-DRUNet ($`k=0.9`$) |  PC-DRUNet ($`k=1`$)
---- | ----- | ------ | ---- | ----- | ------  |---- | ----- 
15 | 34.14 |         32.21          |32.54| 33.30|33.48|33.67|34.01
25 | 31.54|         29.99           |30.12| 30.55|30.94|31.36|31.44
40| 29.33 |           27.87         |28.00| 28.43|28.70|28.96|29.14
Spectral Term |  \ | $` \|\|2J-I\|\|_* `$ | $`\|\|J\|\|_*`$ | $`\|\|0.5I+0.5J\|\|_*`$  |$`\|\|0.7I+0.3J\|\|_*`$ |$`\|\|0.9I+0.1J\|\|_*`$ | $`\|\|(J-10I)^{-1}(J+8I)\|\|_*`$ 
$`r`$ | \ | 0.01 | 0.02|0.02|0.02|0.01|0.01
$`Mean`$ | \ |  0.932    | 0.933 | 0.956    | 0.951 | 0.951|0.955
$`Std`$ |  \ |     0.0454| 0.0197| 0.0137| 0.0133 |0.0136|0.0150



<!--
Table 1. Denoising performance on CBSD68 in PSNR values
---

$`\sigma`$ | DRUNet  | MMO | NE-DRUNet| SPC-DRUNet ($`k=0.5`$) | SPC-DRUNet ($`k=0.6`$) | SPC-DRUNet ($`k=0.7`$) |  SPC-DRUNet ($`k=0.8`$) |  SPC-DRUNet ($`k=0.9`$) |  PC-DRUNet ($`k=1`$)
---- | ----- | ------ | ---- | ----- | ------  |---- | ----- | ------  | ------ 
15 | 34.14 |         32.21          |32.54| 33.30|33.43|33.48|33.66|33.67|34.01
25 | 31.54|         29.99           |30.12| 30.55|30.62|30.94|31.05|31.36|31.44
40| 29.33 |           27.87         |28.00| 28.43|28.43|28.70|28.84|28.96|29.14
Spectral Term |  \ | $` \|\|2J-I\|\|_* `$ | $`\|\|J\|\|_*`$ | $`\|\|0.5I+0.5J\|\|_*`$ |  $`\|\|0.6I+0.4J\|\|_*`$ |$`\|\|0.7I+0.3J\|\|_*`$ |$`\|\|0.8I+0.2J\|\|_*`$ |$`\|\|0.9I+0.1J\|\|_*`$ | $`\|\|(J-10I)^{-1}(J+8I)\|\|_*`$ 
$`r`$ | \ | 0.01 | 0.02|0.02|0.02|0.02|0.02|0.01|0.01
$`Mean`$ | \ |  0.932    | 0.933 | 0.956   | 0.950 | 0.951 | 0.953| 0.951|0.955
$`Std`$ |  \ |     0.0454| 0.0197| 0.0137| 0.0148  |0.0133 |0.0112|0.0136|0.0150
-->






It can be seen in Table 1 that, when $`k`$ gets larger, the assumption gets weaker, and the denoising performance gets better. When $`k\ge0.9`$, the denoisers have a satisfying denoising performance.

<!--
 Table 2. Deblurring performance on CBSD68 in PSNR and SSIM values with Levin's kernels
---
\ | \ |Noise level |$`\sigma=12.75`$|  \    |$`\sigma=17.85`$ | \  
----| ---- |-----|---- |---- |---- |---- 
Converge? | Denoiser| Measurement| PSNR           | SSIM | PSNR            | SSIM 
 $`\surd`$ | MMO.pth | MMO-FBS| 26.03 | 0.6871 | 25.30 | 0.6424
 $`\surd`$ | NE.pth |NE-PGD | 26.16| 0.6977| 25.37| 0.6525
 $`\surd`$ | Prox-DRUNet | Prox-DRS| 26.64| 0.7200 |25.99 |0.6900 
 $`\surd`$ | PC.pth | PnPI-GD ($`k=1.0`$)  | 27.55 | 0.7694 | 26.63 | 0.7307
 $`\surd`$ | 05SPC.pth |PnPI-FBS ($`k=0.5`$) | 26.70 | 0.7176 | 25.74 | 0.6732 
 $`\surd`$ | 07SPC.pth | PnPI-FBS ($`k=0.7`$) | 26.72 | 0.7264 | 25.91 | 0.6753  
 $`\times`$ | 09SPC.pth | PnPI-FBS ($`k=0.9`$) | 26.90 | 0.7252 | 25.98 | 0.6787  
 $`\times`$| PC.pth | PnPI-FBS ($`k=1.0`$) | 26.90 | 0.7342 | 26.04 | 0.6889 
$`\surd`$ | 05SPC.pth | PnPI-HQS ($`k=0.5`$) |27.09 | 0.7494|26.23 |0.7110 
$`\times`$ | 07SPC.pth | PnPI-HQS ($`k=0.7`$) | 27.28| 0.7559| 26.38|0.7191 
$`\times`$ | 09SPC.pth | PnPI-HQS ($`k=0.9`$) | 27.57|0.7616 | 26.57| 0.7200
$`\times`$ | PC.pth | PnPI-HQS ($`k=1.0`$) | 27.65| 0.7704|26.66 |0.7306 
$`\surd`$  | 09SPC.pth | PnPI-AHQS ($`k=0.9`$) | 27.60 | 0.7697 | 26.63 | 0.7304
$`\surd`$  | GS-DRUNet | SNORE | 26.94 | 0.7225 | 25.77 | 0.6546
$`\surd`$  | GS-DRUNet |SNORE-Prox | 26.94 | 0.7226 | 25.78 | 0.6548
 $`\surd`$ | 09SPC.pth | REDI-Prox ($`k=0.9`$) | 27.59 | 0.7682 | 26.63 | 0.7296
 $`?`$     | 09SPC.pth |Diff-REDI-Prox ($`k=0.9`$) | 27.56 | 0.7685 | 26.60 | 0.7289
-->


Table 2. Deblurring performance on CBSD68 by different convergent PnP methods in PSNR and SSIM values with Levin's kernels
---

\ | Noise level |$`\sigma=12.75`$|  \    |$`\sigma=17.85`$ | \  
----|-----|---- |---- |---- |---- 
 Pub | Measurement| PSNR           | SSIM | PSNR            | SSIM 
SIIMS 2021 |  [MMO-FBS](https://github.com/matthieutrs/LMMO_lightning) | 26.03 | 0.6871 | 25.30 | 0.6424
Neurips 2021 | NE-PGD | 26.16| 0.6977| 25.37| 0.6525
ICML 2022 | [Prox-DRS](https://github.com/samuro95/Prox-PnP) | 26.64| 0.7200 |25.99 |0.6900 
ICML 2024 |PnPI-GD ($`k=1.0`$)  | 27.55 | 0.7694 | 26.63 | 0.7307
ICML 2024 |PnPI-HQS ($`k=0.9`$) | 27.60 | 0.7697 | 26.63 | 0.7304
ICML 2024 | PnPI-FBS ($`k=0.9`$) | 27.58 | 0.7699 | 26.68 | 0.7333
ICML 2024 | [SNORE](https://github.com/Marien-RENAUD/SNORE) | 26.94 | 0.7225 | 25.77 | 0.6546
ICML 2024 | [SNORE-Prox](https://github.com/Marien-RENAUD/SNORE) | 26.94 | 0.7226 | 25.78 | 0.6548
TBD | REDI-Prox ($`k=0.9`$) | 27.59 | 0.7682 | 26.63 | 0.7296

Table 3. Super-resolution performance on CBSD68 by different convergent PnP methods in PSNR and SSIM values with a Gaussian blur kernel, different scales, and different noise levels.
---
scale | \ | s=2 | \ | \ | s=4 | \ | 
----|-----|---- |---- |---- |---- |---- 
$`\sigma`$ | 0 | 2.55 | 7.65 | 0 | 2.55 | 7.65
[MMO-FBS](https://github.com/matthieutrs/LMMO_lightning) | 27.02 | 26.16 | 25.28 | 25.30 | 25.17 | 24.51 
\ | 0.7719 | 0.7142 | 0.6604 | 0.6692 | 0.6602  | 0.6285
NE-PGD | 27.02 | 26.23 | 25.27 | 25.34 | 25.21 | 24.54 
\ | 0.7822 | 0.7197 | 0.6622 | 0.6719 | 0.6632 | 0.6311
[Prox-DRS](https://github.com/samuro95/Prox-PnP) | 30.26 | 26.63 |25.57 | 25.49 | 25.23 | 24.48 
\ | 0.8874 | 0.7364 | 0.6805 | 0.7007 | 0.6716| 0.6280
PnPI-HQS ($`k=0.9`$) | 30.36 | 26.98 | 25.89 | 25.80 | 25.48 | 24.78
\                    | 0.8822 | 0.7487 | 0.6977 | 0.7083 | 0.6822 | 0.6453
PnPI-FBS ($`k=0.9`$) | \ | 26.98 | 25.91 | \ | 25.42 | 24.76 
\  | \ | 0.7453 | 0.6995 | \  |0.6734 | 0.6449







Overall, this repo provides a more accurate evaluation for each denoiser. We will continue updating the denoising performance results, and the PnP restoration results. This will serve as a new baseline. 


Update 20240623
----
We will upload the codes and the pretrained models gradually.

Update 20240703
----
We have uploaded the denoisers and the codes. The explanations for each algorithm will be uploaded.
