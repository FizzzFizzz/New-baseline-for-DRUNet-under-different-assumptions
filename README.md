# New-baseline-for-DRUNet-under-different-assumptions
This is a new baseline for DRUNet under different assumptions. 



 We are currently retraining each denoisers (MMO, NE-DRUNet, SPC-DRUNet with different $k$, and PC-DRUNet) with power iterative method and a modified power iterative method, in this repo. 

 In this new repo, different from the original paper [ICML](https://openreview.net/forum?id=G0vZ5ENrJQ&noteId=G0vZ5ENrJQ), each method is trained with the (modified) power iterative method with 15 iterations. The average value and the standard derivations of the spectral norms (such as ||2J-I||_* for MMO, ||J||_* for NE-DRUNet) evaluated at different images after training is denoted by Mean and Std, respectively. The balancing parameter r in the loss function is chosen to be large enough, such that,

  Mean + 3 * Std <=1. 

 Therefore, this repo provides a more accurate evaluation for each denoiser. We will continue updating the denoising performance results, and the PnP restoration results. This will serve as a new baseline.
