import torch
import cv2
import os
def _jacobian_vec(y, x, v):
    w = torch.ones_like(x, requires_grad=True)
    t = torch.autograd.grad(y, x, w, create_graph=True)[0]
    return torch.autograd.grad(t, w, v, create_graph=True)[0]

def _jacobian_transpose_vec(y, x, v):
    return torch.autograd.grad(y, x, v, create_graph=True)[0]

class JacobianLoss(torch.nn.Module):
    def __init__(self, dt=0.4, iters=5, inner_iters=10, loss_type='max', eps=1e-8):
        super(JacobianLoss, self).__init__()
        self.dt = dt
        self.iters = iters
        self.inner_iters = inner_iters
        self.loss_type = loss_type
        self.eps = eps

    def forward(self, img1, img2, net):
        # jacobian_norm = self.jacobian_GS05(img1[0:1], img2[0:1], net, interpolation=False, training=True)
        # jacobian_norm = self.jacobian_spectral_norm_a(img1[0:1], img2[0:1], net, interpolation=False, training=True) #a=1, 0.5-SPC-DRUNet.

        # PCnew, ||(J+8I)/(J-10I)||_*. We should also require J=J^T.
        # jacobian_norm = self.jacobian_PCnew(img1[0:1], img2[0:1], net, interpolation=False, training=True)

        # jacobian_norm = self.jacobian_spectral_norm_selfajoint(img1[0:1], img2[0:1], net, interpolation=False, training=True)
        jacobian_norm = self.jacobian_k05(img1[0:1], img2[0:1], net, interpolation=False, training=True)
        # jacobian_norm = self.jacobian_res_k05(img1[0:1], img2[0:1], net, interpolation=False, training=True)
        # jacobian_norm = self.jacobian_k0(img1[0:1], img2[0:1], net, interpolation=False, training=True)
        # jacobian_norm = self.jacobian_LMMO(img1[0:1], img2[0:1], net, interpolation=False, training=True)
        
        # jacobian_norm = self.jacobian_res_nonexpansive(img1[0:1], img2[0:1], net, interpolation=False, training=True)

        # jacobian_norm = self.jacobian_res0414(img1[0:1], img2[0:1], net, interpolation=False, training=True)
        
        
        # jacobian_norm = self.jacobian_spectral_norm_cp(img1[0:1], img2[0:1], net, interpolation=False, training=True)
        # jacobian_norm = self.jacobian_spectral_norm_cp2(img1[0:1], img2[0:1], net, interpolation=False, training=True)
        # jacobian_norm = self.jacobian_spectral_norm_a(img1[0:1], img2[0:1], net, interpolation=False, training=True)
        # jacobian_norm = self.jacobian_spectral_norm_yin(img1[0:1], img2[0:1], net, interpolation=False, training=True)
        # jacobian_norm = self.jacobian_spectral_norm(img1[0:1], img2[0:1], net, interpolation=False, training=True)
        # jacobian_norm, fenmu = self.jacobian_symmetric(img1[0:1], img2[0:1], net, interpolation=False, training=True)

        # jacobian_norm = self.jacobian_save(img1[0:1], img2[0:1], net, interpolation=False, training=False)

        # self.log('train/jacobian_norm_max', jacobian_norm.max(), prog_bar=True)

        if self.loss_type == 'max':
            # jacobian_loss = torch.maximum(jacobian_norm, 0*torch.ones_like(jacobian_norm) ) # only for symmetric Jacobian
            jacobian_loss = torch.maximum(jacobian_norm, torch.ones_like(jacobian_norm) - self.eps)

            # jacobian_loss = torch.maximum(jacobian_norm, 0.414*torch.ones_like(jacobian_norm) - self.eps)
            # jacobian_loss = torch.maximum(jacobian_norm, 1000*torch.ones_like(jacobian_norm) - self.eps)
            # jacobian_loss = torch.minimum(jacobian_loss, 5 * torch.ones_like(jacobian_norm)) # added by Wei Deliang
        elif self.loss_type == 'exp':
            jacobian_loss = self.eps * torch.exp(jacobian_norm - (1 + self.eps))  / self.eps
        else:
            print("jacobian loss not available")

        jacobian_loss = torch.clip(jacobian_loss, 0, 1e3)
        # self.log('train/jacobian_loss_max', jacobian_loss.max(), prog_bar=True)

        # print(jacobian_norm.detach())
        return jacobian_loss.mean(), jacobian_norm.detach()
        # return jacobian_loss.mean(), jacobian_norm.detach()/fenmu.detach() # only for printing relative symmetric measure.


    def jacobian_spectral_norm(self, y_in, x_hat, net, interpolation=True, training=False):
        '''
        Jacobian spectral norm from Pesquet et al; computed with a power iteration method.
        Given a denoiser J, computes the spectral norm of Q = 2J-I where J is the denoising model.

        Inputs:
        :y_in: point where the jacobian is to be computed, typically a noisy image (torch Tensor)
        :x_hat: denoised image (unused if interpolation = False) (torch Tensor)
        :sigma: noise level
        :interpolation: whether to compute the jacobian only at y_in, or somewhere on the segment [x_hat, y_in].
        :training: set to True during training to retain grad appropriately
        Outputs:
        :z.view(-1): the square of the Jacobian spectral norm of (2J-Id)

        Beware: reversed usage compared to the original Pesquet et al code.
        '''

        (y_in, noise_map) = torch.split(y_in, [y_in.shape[1]-1, 1], dim=1)
        if interpolation:
            eta = torch.rand(y_in.size(0), 1, 1, 1, requires_grad=True).to(y_in.device)
            x = eta * y_in.detach() + (1 - eta) * x_hat.detach()
            x = x.to(y_in.device)
        else:
            x = y_in

        x.requires_grad_()
        x_hat = net(torch.cat((x, noise_map), dim=1))

        # y = x_hat  # Beware notation : y_in = input, x_hat = output network
        y = 2 * x_hat - y_in  # Beware notation : y_in = input, x_hat = output network

        u = torch.randn_like(x)
        u = u / torch.norm(u, p=2)

        z_old = torch.zeros(u.shape[0])

        for it in range(self.iters):

            w = torch.ones_like(y, requires_grad=True)  # Double backward trick. From https://gist.github.com/apaszke/c7257ac04cb8debb82221764f6d117ad
            v = torch.autograd.grad(torch.autograd.grad(y, x, w, create_graph=True), w, u, create_graph=training)[0]  # Ju

            v, = torch.autograd.grad(y, x, v, retain_graph=True, create_graph=True)  # vtJt

            z = torch.matmul(u.reshape(u.shape[0], 1, -1), v.reshape(v.shape[0], -1, 1)) / torch.matmul(
                u.reshape(u.shape[0], 1, -1), u.reshape(u.shape[0], -1, 1))

            # if it > 0:
            #     rel_var = torch.norm(z - z_old)
            #     if rel_var < self.eps:
            #         break
            z_old = z.clone()

            u = v / torch.norm(v, p=2)  # Modified

            # if self.eval:
            #     w.detach_()
            #     v.detach_()
            #     u.detach_()

        return z.view(-1)

    


    def jacobian_res_nonexpansive(self, y_in, x_hat, net, interpolation=True, training=False):

        (y_in, noise_map) = torch.split(y_in, [y_in.shape[1]-1, 1], dim=1)
        if interpolation:
            eta = torch.rand(y_in.size(0), 1, 1, 1, requires_grad=True).to(y_in.device)
            x = eta * y_in.detach() + (1 - eta) * x_hat.detach()
            x = x.to(y_in.device)
        else:
            x = y_in

        x.requires_grad_()
        x_hat = net(torch.cat((x, noise_map), dim=1))
        # Beware notation : y_in = input, x_hat = output network
        y = x_hat - y_in
        u = torch.randn_like(x)
        u = u / torch.norm(u, p=2)

        z_old = torch.zeros(u.shape[0])

        for it in range(self.iters):

            w = torch.ones_like(y, requires_grad=True)  # Double backward trick. From https://gist.github.com/apaszke/c7257ac04cb8debb82221764f6d117ad
            v = torch.autograd.grad(torch.autograd.grad(y, x, w, create_graph=True), w, u, create_graph=training)[0]  # Ju

            v, = torch.autograd.grad(y, x, v, retain_graph=True, create_graph=True)  # vtJt

            z = torch.matmul(u.reshape(u.shape[0], 1, -1), v.reshape(v.shape[0], -1, 1)) / torch.matmul(
                u.reshape(u.shape[0], 1, -1), u.reshape(u.shape[0], -1, 1))

            z_old = z.clone()

            u = v / torch.norm(v, p=2)  # Modified


        return z.view(-1)
    

    def jacobian_res0414(self, y_in, x_hat, net, interpolation=True, training=False):

        (y_in, noise_map) = torch.split(y_in, [y_in.shape[1]-1, 1], dim=1)
        if interpolation:
            eta = torch.rand(y_in.size(0), 1, 1, 1, requires_grad=True).to(y_in.device)
            x = eta * y_in.detach() + (1 - eta) * x_hat.detach()
            x = x.to(y_in.device)
        else:
            x = y_in

        x.requires_grad_()
        x_hat = net(torch.cat((x, noise_map), dim=1))
        # Beware notation : y_in = input, x_hat = output network
        y = x_hat - y_in
        y = y / 0.414
        u = torch.randn_like(x)
        u = u / torch.norm(u, p=2)

        z_old = torch.zeros(u.shape[0])

        for it in range(self.iters):

            w = torch.ones_like(y, requires_grad=True)  # Double backward trick. From https://gist.github.com/apaszke/c7257ac04cb8debb82221764f6d117ad
            v = torch.autograd.grad(torch.autograd.grad(y, x, w, create_graph=True), w, u, create_graph=training)[0]  # Ju

            v, = torch.autograd.grad(y, x, v, retain_graph=True, create_graph=True)  # vtJt

            z = torch.matmul(u.reshape(u.shape[0], 1, -1), v.reshape(v.shape[0], -1, 1)) / torch.matmul(
                u.reshape(u.shape[0], 1, -1), u.reshape(u.shape[0], -1, 1))

            z_old = z.clone()

            u = v / torch.norm(v, p=2)  # Modified


        return z.view(-1)

    




    


    def jacobian_LMMO(self, y_in, x_hat, net, interpolation=True, training=False):

        (y_in, noise_map) = torch.split(y_in, [y_in.shape[1]-1, 1], dim=1)
        if interpolation:
            eta = torch.rand(y_in.size(0), 1, 1, 1, requires_grad=True).to(y_in.device)
            x = eta * y_in.detach() + (1 - eta) * x_hat.detach()
            x = x.to(y_in.device)
        else:
            x = y_in

        x.requires_grad_()
        x_hat = net(torch.cat((x, noise_map), dim=1))

        # y = x_hat  # Beware notation : y_in = input, x_hat = output network
        # y = 2 * x_hat - y_in  # Beware notation : y_in = input, x_hat = output network
        y = 2*x_hat- y_in
        u = torch.randn_like(x)
        u = u / torch.norm(u, p=2)

        z_old = torch.zeros(u.shape[0])

        for it in range(self.iters):

            w = torch.ones_like(y, requires_grad=True)  # Double backward trick. From https://gist.github.com/apaszke/c7257ac04cb8debb82221764f6d117ad
            v = torch.autograd.grad(torch.autograd.grad(y, x, w, create_graph=True), w, u, create_graph=training)[0]  # Ju

            v, = torch.autograd.grad(y, x, v, retain_graph=True, create_graph=True)  # vtJt

            z = torch.matmul(u.reshape(u.shape[0], 1, -1), v.reshape(v.shape[0], -1, 1)) / torch.matmul(
                u.reshape(u.shape[0], 1, -1), u.reshape(u.shape[0], -1, 1))

            z_old = z.clone()

            u = v / torch.norm(v, p=2)  # Modified


        return z.view(-1)



    def jacobian_k0(self, y_in, x_hat, net, interpolation=True, training=False):

        (y_in, noise_map) = torch.split(y_in, [y_in.shape[1]-1, 1], dim=1)
        if interpolation:
            eta = torch.rand(y_in.size(0), 1, 1, 1, requires_grad=True).to(y_in.device)
            x = eta * y_in.detach() + (1 - eta) * x_hat.detach()
            x = x.to(y_in.device)
        else:
            x = y_in

        x.requires_grad_()
        x_hat = net(torch.cat((x, noise_map), dim=1))

        # y = x_hat  # Beware notation : y_in = input, x_hat = output network
        # y = 2 * x_hat - y_in  # Beware notation : y_in = input, x_hat = output network
        y = x_hat
        u = torch.randn_like(x)
        u = u / torch.norm(u, p=2)

        z_old = torch.zeros(u.shape[0])

        for it in range(self.iters):

            w = torch.ones_like(y, requires_grad=True)  # Double backward trick. From https://gist.github.com/apaszke/c7257ac04cb8debb82221764f6d117ad
            v = torch.autograd.grad(torch.autograd.grad(y, x, w, create_graph=True), w, u, create_graph=training)[0]  # Ju

            v, = torch.autograd.grad(y, x, v, retain_graph=True, create_graph=True)  # vtJt

            z = torch.matmul(u.reshape(u.shape[0], 1, -1), v.reshape(v.shape[0], -1, 1)) / torch.matmul(
                u.reshape(u.shape[0], 1, -1), u.reshape(u.shape[0], -1, 1))

            z_old = z.clone()

            u = v / torch.norm(v, p=2)  # Modified


        return z.view(-1)




    def jacobian_k05(self, y_in, x_hat, net, interpolation=True, training=False):
    

        (y_in, noise_map) = torch.split(y_in, [y_in.shape[1]-1, 1], dim=1)
        if interpolation:
            eta = torch.rand(y_in.size(0), 1, 1, 1, requires_grad=True).to(y_in.device)
            x = eta * y_in.detach() + (1 - eta) * x_hat.detach()
            x = x.to(y_in.device)
        else:
            x = y_in

        x.requires_grad_()
        x_hat = net(torch.cat((x, noise_map), dim=1))

        # y = x_hat  # Beware notation : y_in = input, x_hat = output network
        # y = 2 * x_hat - y_in  # Beware notation : y_in = input, x_hat = output network
        # y = 0.5*x_hat - y_in # old, before 20240618. This seems trying to ensure I-D to be 0.5-strictly pseudo-contractive.
        # y = 0.5*x_hat + 0.5* y_in # new, 20240618

        # # only for test! Remember to change to right
        y = 0.1*x_hat + 0.9* y_in # 20240701, k=0.9, that is, 0.9-strictly pseudo-contractive denoiser.
        # y = 0.2*x_hat + 0.8* y_in # 20240701, k=0.8, that is, 0.8-strictly pseudo-contractive denoiser.
        # y = 0.3*x_hat + 0.7* y_in # 20240701, k=0.7, that is, 0.7-strictly pseudo-contractive denoiser.
        # y = 0.4*x_hat + 0.6* y_in # 20240701, k=0.6, that is, 0.6-strictly pseudo-contractive denoiser.
        # y = 0.5*x_hat + 0.5* y_in # 20240701, k=0.5, that is, 0.5-strictly pseudo-contractive denoiser.
        # y = 1*x_hat + 0* y_in # 20240701, k=0.0, that is, non-expansive denoiser.
        u = torch.randn_like(x)
        u = u / torch.norm(u, p=2)

        z_old = torch.zeros(u.shape[0])

        for it in range(self.iters):

            w = torch.ones_like(y, requires_grad=True)  # Double backward trick. From https://gist.github.com/apaszke/c7257ac04cb8debb82221764f6d117ad
            v = torch.autograd.grad(torch.autograd.grad(y, x, w, create_graph=True), w, u, create_graph=training)[0]  # Ju

            v, = torch.autograd.grad(y, x, v, retain_graph=True, create_graph=True)  # vtJt

            z = torch.matmul(u.reshape(u.shape[0], 1, -1), v.reshape(v.shape[0], -1, 1)) / torch.matmul(
                u.reshape(u.shape[0], 1, -1), u.reshape(u.shape[0], -1, 1))

            z_old = z.clone()

            u = v / torch.norm(v, p=2)  # Modified


        return z.view(-1)
    


    def jacobian_res_k05(self, y_in, x_hat, net, interpolation=True, training=False):
    

        (y_in, noise_map) = torch.split(y_in, [y_in.shape[1]-1, 1], dim=1)
        if interpolation:
            eta = torch.rand(y_in.size(0), 1, 1, 1, requires_grad=True).to(y_in.device)
            x = eta * y_in.detach() + (1 - eta) * x_hat.detach()
            x = x.to(y_in.device)
        else:
            x = y_in

        x.requires_grad_()
        x_hat = net(torch.cat((x, noise_map), dim=1))

        # y = x_hat  # Beware notation : y_in = input, x_hat = output network

        y = 0.5*x_hat - y_in # ensure I-D to be strictly pseudo-contractive.

        u = torch.randn_like(x)
        u = u / torch.norm(u, p=2)

        z_old = torch.zeros(u.shape[0])

        for it in range(self.iters):

            w = torch.ones_like(y, requires_grad=True)  # Double backward trick. From https://gist.github.com/apaszke/c7257ac04cb8debb82221764f6d117ad
            v = torch.autograd.grad(torch.autograd.grad(y, x, w, create_graph=True), w, u, create_graph=training)[0]  # Ju

            v, = torch.autograd.grad(y, x, v, retain_graph=True, create_graph=True)  # vtJt

            z = torch.matmul(u.reshape(u.shape[0], 1, -1), v.reshape(v.shape[0], -1, 1)) / torch.matmul(
                u.reshape(u.shape[0], 1, -1), u.reshape(u.shape[0], -1, 1))

            if it > 0:
                rel_var = torch.norm(z - z_old)
                if rel_var < self.eps:
                    break

            z_old = z.clone()

            u = v / torch.norm(v, p=2)  # Modified


        return z.view(-1)

    

    
    def jacobian_spectral_norm_yin(self, y_in, x_hat, net, interpolation=True, training=False):

        (y_in, noise_map) = torch.split(y_in, [y_in.shape[1]-1, 1], dim=1)
        if interpolation:
            eta = torch.rand(y_in.size(0), 1, 1, 1, requires_grad=True).to(y_in.device)
            x = eta * y_in.detach() + (1 - eta) * x_hat.detach()
            x = x.to(y_in.device)
        else:
            x = y_in

        x.requires_grad_()
        x_hat = net(torch.cat((x, noise_map), dim=1))

        # y = x_hat  # Beware notation : y_in = input, x_hat = output network
        y = x_hat - y_in  # Beware notation : y_in = input, x_hat = output network

        u = torch.randn_like(x)
        u = u / torch.norm(u, p=2)

        z_old = torch.zeros(u.shape[0])

        for it in range(self.iters):

            w = torch.ones_like(y, requires_grad=True)  # Double backward trick. From https://gist.github.com/apaszke/c7257ac04cb8debb82221764f6d117ad
            v = torch.autograd.grad(torch.autograd.grad(y, x, w, create_graph=True), w, u, create_graph=training)[0]  # Ju

            v, = torch.autograd.grad(y, x, v, retain_graph=True, create_graph=True)  # vtJt

            z = torch.matmul(u.reshape(u.shape[0], 1, -1), v.reshape(v.shape[0], -1, 1)) / torch.matmul(
                u.reshape(u.shape[0], 1, -1), u.reshape(u.shape[0], -1, 1))

            if it > 0:
                rel_var = torch.norm(z - z_old)
                if rel_var < self.eps:
                    break
            z_old = z.clone()

            u = v / torch.norm(v, p=2)  # Modified

        return z.view(-1)
    



    def jacobian_spectral_norm_a(self, y_in, x_hat, net, interpolation=True, training=False):


        (y_in, noise_map) = torch.split(y_in, [y_in.shape[1]-1, 1], dim=1)
        if interpolation:
            eta = torch.rand(y_in.size(0), 1, 1, 1, requires_grad=True).to(y_in.device)
            x = eta * y_in.detach() + (1 - eta) * x_hat.detach()
            x = x.to(y_in.device)
        else:
            x = y_in

        x.requires_grad_()
        x_hat = net(torch.cat((x, noise_map), dim=1))

        # y = x_hat  # Beware notation : y_in = input, x_hat = output network
        a = 1
        y = a * y_in + x_hat  # Beware notation : y_in = input, x_hat = output network
        y = y/(a+1)

        u = torch.randn_like(x)
        u = u / torch.norm(u, p=2)

        z_old = torch.zeros(u.shape[0])

        for it in range(self.iters):

            w = torch.ones_like(y, requires_grad=True)  # Double backward trick. From https://gist.github.com/apaszke/c7257ac04cb8debb82221764f6d117ad
            v = torch.autograd.grad(torch.autograd.grad(y, x, w, create_graph=True), w, u, create_graph=training)[0]  # Ju

            v, = torch.autograd.grad(y, x, v, retain_graph=True, create_graph=True)  # vtJt

            z = torch.matmul(u.reshape(u.shape[0], 1, -1), v.reshape(v.shape[0], -1, 1)) / torch.matmul(
                u.reshape(u.shape[0], 1, -1), u.reshape(u.shape[0], -1, 1))

            if it > 0:
                rel_var = torch.norm(z - z_old)
                if rel_var < self.eps:
                    break
            z_old = z.clone()

            u = v / torch.norm(v, p=2)  # Modified

            # if self.eval:
            #     w.detach_()
            #     v.detach_()
            #     u.detach_()

        return z.view(-1)



    
    def jacobian_spectral_norm_cp(self, y_in, x_hat, net, interpolation=True, training=False):
        (y_in, noise_map) = torch.split(y_in, [y_in.shape[1]-1, 1], dim=1)
        if interpolation:
            eta = torch.rand(y_in.size(0), 1, 1, 1, requires_grad=True).to(y_in.device)
            x = eta * y_in.detach() + (1 - eta) * x_hat.detach()
            x = x.to(y_in.device)
        else:
            x = y_in

        x.requires_grad_()
        x_hat = net(torch.cat((x, noise_map), dim=1))
        
        z_all = torch.zeros(self.iters).to(y_in.device)
        for i in range(self.iters):
            v = torch.randn_like(x)
            v /= torch.norm(v, p=2)

            Jv = _jacobian_vec(x_hat, x, v)
            z_all[i] = torch.linalg.norm(Jv) / (torch.linalg.norm(Jv - 2 * v) + self.eps)

        return z_all.max().view(-1)
    

    def jacobian_spectral_norm_cp2(self, y_in, x_hat, net, interpolation=True, training=False):

        (y_in, noise_map) = torch.split(y_in, [y_in.shape[1]-1, 1], dim=1)
        if interpolation:
            eta = torch.rand(y_in.size(0), 1, 1, 1, requires_grad=True).to(y_in.device)
            x = eta * y_in.detach() + (1 - eta) * x_hat.detach()
            x = x.to(y_in.device)
        else:
            x = y_in

        x.requires_grad_()
        x_hat = net(torch.cat((x, noise_map), dim=1))

        y = x_hat  # Beware notation : y_in = input, x_hat = output network


        q = torch.randn_like(x)
        q /= torch.linalg.norm(q)

        z0 = _jacobian_vec(y, x, q)

        for _ in range(self.iters):

            # gradient step for solving z
            for _ in range(self.inner_iters):
                z0_old = z0.detach()
                temp_vec = _jacobian_vec(y, x, z0 - q) - 2 * z0
                temp_grad = _jacobian_transpose_vec(y, x, temp_vec) - 2 * temp_vec
                z0 = z0 - self.dt * temp_grad

                # if torch.linalg.norm(z0_old - z0) < self.eps:
                #     break
                if torch.linalg.norm(z0_old-z0)/torch.linalg.norm(z0) < 0.05:
                    # print('Yes!!!')
                    break
            q_k = q.clone()
            q = z0 / (torch.linalg.norm(z0)+self.eps)  # Modified


        # return ((z.view(-1)/(2-z.view(-1)))**2)**(0.5)
        # return (torch.sum(q * z0)).view(-1)

        # t = (torch.linalg.norm(z0)).view(-1)
        # return t
        return (torch.linalg.norm(z0)).view(-1)
    

    def jacobian_symmetric(self, y_in, x_hat, net, interpolation=True, training=False):
        
        (y_in, noise_map) = torch.split(y_in, [y_in.shape[1]-1, 1], dim=1)
        if interpolation:
            eta = torch.rand(y_in.size(0), 1, 1, 1, requires_grad=True).to(y_in.device)
            x = eta * y_in.detach() + (1 - eta) * x_hat.detach()
            x = x.to(y_in.device)
        else:
            x = y_in
        x.requires_grad_()
        x_hat = net(torch.cat((x, noise_map), dim=1))
        y = x_hat  
        q = torch.randn_like(x)
        q /= torch.linalg.norm(q)

        z0 = _jacobian_vec(y,x,q)-_jacobian_transpose_vec(y,x,q)
        for _ in range(self.iters):
            z0_old = z0.clone()
            vec_z1   = _jacobian_vec(y, x, z0)
            vec_z2   = _jacobian_transpose_vec(y,x,z0)
            z0    = vec_z1 - vec_z2


            if torch.linalg.norm(z0_old-z0)/torch.linalg.norm(z0) < 0.05:
                # print('Yes!!!')
                break
            q = z0 / (torch.linalg.norm(z0))  # Modified
        

        q = torch.randn_like(x)
        q /= torch.linalg.norm(q)
        t = _jacobian_vec(y,x,q)
        for _ in range(self.iters):
            t_old = z0.clone()
            t   = _jacobian_vec(y, x, t)
            q = t / (torch.linalg.norm(t))  # Modified

        return (torch.linalg.norm(z0)).view(-1), (torch.linalg.norm(t)).view(-1)



    def jacobian_spectral_norm_selfajoint(self, y_in, x_hat, net, interpolation=True, training=False):
        
        (y_in, noise_map) = torch.split(y_in, [y_in.shape[1]-1, 1], dim=1)
        if interpolation:
            eta = torch.rand(y_in.size(0), 1, 1, 1, requires_grad=True).to(y_in.device)
            x = eta * y_in.detach() + (1 - eta) * x_hat.detach()
            x = x.to(y_in.device)
        else:
            x = y_in
        x.requires_grad_()
        x_hat = net(torch.cat((x, noise_map), dim=1))
        y = x_hat  
        q = torch.randn_like(x)
        q /= torch.linalg.norm(q)
        # z0_1 = _jacobian_vec(y, x, q)
        # z0_2 = _jacobian_transpose_vec(y, x, q)
        # z0 = z0_1 + z0_2
        z0 = _jacobian_vec(y,x,q) * 0.5 + _jacobian_transpose_vec(y,x,q) * 0.5
        for _ in range(self.iters):
            # gradient step for solving z
            for _ in range(self.inner_iters):
                z0_old = z0.detach()
                vec_z1   = _jacobian_vec(y, x, z0)
                vec_z2   = _jacobian_transpose_vec(y,x,z0)
                vec_z    = vec_z1 + vec_z2 - 4 * z0

                vec_q1   = _jacobian_vec(y, x, q)
                vec_q2   = _jacobian_transpose_vec(y,x,q)
                vec_q    = vec_q1 + vec_q2
                vec      = vec_z - vec_q

                grad_1     = _jacobian_vec(y, x, vec)
                grad_2     = _jacobian_transpose_vec(y, x, vec)
                grad       = grad_1 + grad_2 - 4 * vec
                z0 = z0 - self.dt * grad


                if torch.linalg.norm(z0_old-z0)/torch.linalg.norm(z0) < 0.05:
                    # print('Yes!!!')
                    break
            q_k = q.clone()
            q = z0 / (torch.linalg.norm(z0))  # Modified
        return (torch.linalg.norm(z0)).view(-1)



    def jacobian_PCnew(self, y_in, x_hat, net, interpolation=True, training=False):
        # 20240623
        # to encourage that ||S/(S-2I)||_*<=1. But consider x>1, ||(J-(2-x)I)/(J-xI)||_*<=1. Set x = 10, ||(J+8I)/(J-10I)||_*<=1. More stable.
        # need to require J=J^T.

        (y_in, noise_map) = torch.split(y_in, [y_in.shape[1]-1, 1], dim=1)
        if interpolation:
            eta = torch.rand(y_in.size(0), 1, 1, 1, requires_grad=True).to(y_in.device)
            x = eta * y_in.detach() + (1 - eta) * x_hat.detach()
            x = x.to(y_in.device)
        else:
            x = y_in
        x.requires_grad_()
        x_hat = net(torch.cat((x, noise_map), dim=1))
        y = x_hat  
        q = torch.randn_like(x)
        q /= torch.linalg.norm(q)
        z0 = _jacobian_vec(y,x,q)
        for _ in range(self.iters):
            # gradient step for solving z
            for _ in range(self.inner_iters):
                z0_old = z0.detach()
                vec_z1   = _jacobian_vec(y, x, z0)
                vec_z    = vec_z1 - 10 * z0

                vec_q1   = _jacobian_vec(y, x, q)
                vec_q    = vec_q1 + 8 * q

                vec      = vec_z - vec_q

                grad_2     = _jacobian_transpose_vec(y, x, vec)
                grad       = grad_2 - 10 * vec
                z0 = z0 - self.dt * grad


            q_k = q.clone()
            q = z0 / (torch.linalg.norm(z0))  # Modified
        return (torch.linalg.norm(z0)).view(-1)
    


    



    def jacobian_save(self, y_in, x_hat, net, interpolation=False, training=False):
        
        (y_in, noise_map) = torch.split(y_in, [y_in.shape[1]-1, 1], dim=1)
        
        x = y_in

        x.requires_grad_()
        x_hat = net(torch.cat((x, noise_map), dim=1))
        y = x_hat  
        q = torch.randn_like(x)
        q = 0 * q # now, q is zero vector.
        # q = q.detach()
        
        if not os.path.exists('wdl_try_SPC'):
            map_ = noise_map.detach()
            print(torch.max(map_))
            os.makedirs('wdl_try_SPC')  
            aa = y_in.detach()
            aa = aa.squeeze()
            aa = aa.squeeze()
            aa = aa.cpu().numpy()
            cv2.imwrite('temp_input.png', aa*255)
            bb = x_hat.detach()
            bb = bb.squeeze()
            bb = bb.squeeze()
            bb = bb.cpu().numpy()
            cv2.imwrite('temp_output.png',bb*255)
            for i in range(64):
                for j in range(64):
                    p = 0*q
                    p[0,0,i,j] = 255
                    temp = _jacobian_vec(y,x,p)
                    
                    # p = temp + 127.5
                    p = temp + 63.75
                    p = p.detach()
                    p = p.squeeze()
                    p = p.squeeze()
                    if torch.max(p)>255:
                        print(torch.max(p))
                    # print(torch.min(p))
                    pp = p.cpu().numpy()
                    
                    cv2.imwrite('wdl_try_SPC/' + '{}_{}.png'.format(i,j), pp)
            print('Done!!!!!!')
                
        print(q.shape)

        z0 = _jacobian_vec(y,x,q)

        return (torch.linalg.norm(z0)).view(-1)
    



















    def jacobian_k05_away(self, y_in, x_hat, net, interpolation=True, training=False):
    

        (y_in, noise_map) = torch.split(y_in, [y_in.shape[1]-1, 1], dim=1)
        if interpolation:
            eta = torch.rand(y_in.size(0), 1, 1, 1, requires_grad=True).to(y_in.device)
            x = eta * y_in.detach() + (1 - eta) * x_hat.detach()
            x = x.to(y_in.device)
        else:
            x = y_in

        x.requires_grad_()
        x_hat = net(torch.cat((x, noise_map), dim=1))

        # y = x_hat  # Beware notation : y_in = input, x_hat = output network
        # y = 2 * x_hat - y_in  # Beware notation : y_in = input, x_hat = output network
        y = 0.5*x_hat - y_in
        u = torch.randn_like(x)
        u = u / torch.norm(u, p=2)

        z_old = torch.zeros(u.shape[0])

        for it in range(self.iters):

            w = torch.ones_like(y, requires_grad=True)  # Double backward trick. From https://gist.github.com/apaszke/c7257ac04cb8debb82221764f6d117ad
            v = torch.autograd.grad(torch.autograd.grad(y, x, w, create_graph=True), w, u, create_graph=training)[0]  # Ju

            v, = torch.autograd.grad(y, x, v, retain_graph=True, create_graph=True)  # vtJt

            z = torch.matmul(u.reshape(u.shape[0], 1, -1), v.reshape(v.shape[0], -1, 1)) / torch.matmul(
                u.reshape(u.shape[0], 1, -1), u.reshape(u.shape[0], -1, 1))

            # if it > 0:
            #     rel_var = torch.norm(z - z_old)
            #     if rel_var < self.eps:
            #         break
            z_old = z.clone()

            u = v / torch.norm(v, p=2)  # Modified

            # if self.eval:
            #     w.detach_()
            #     v.detach_()
            #     u.detach_()

        return z.view(-1)








