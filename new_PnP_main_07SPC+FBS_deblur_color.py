import torch.nn as nn
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from tqdm import tqdm
import logging
import os
import sys 
import cv2
from scipy import ndimage
from PIL import Image
sys.path.append("..") 


import utils.utils_image as util
from utils import utils_logger
from utils import utils_option as option
from utils import utils_deblur as deblur
from utils.utils_dist import get_dist_info, init_dist


    
from models.network_unet import UNetRes as Net


    





model_path = "denoising/drunet_k=07_color_2024/models/12000_G.pth"
n_channels = 3



model = Net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose", bias=False)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.cuda()
device = 'cuda'
for k, v in model.named_parameters():
    v.requires_grad = False













class Drunet_running(torch.nn.Module):
    def __init__(self):
        super(Drunet_running, self).__init__()
        # self.models = {}
        # for level in models:
        #     self.models[level] = models[level]
        #     self.models[level].eval()
        self.models = model
        self.models.eval()
    
    def to(self, device):
        
        self.models.to(device)    

    def forward(self, x, sigma):
        x = np.array(x)
        x = torch.tensor(x, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        x = x.to(device)
        sigma = float(sigma)
        # sigma_div_255 = torch.FloatTensor([sigma/255.]).repeat(1, 1, x.shape[2], x.shape[3]).cuda()
        sigma_div_255 = torch.FloatTensor([sigma/255.]).repeat(1, 1, x.shape[2], x.shape[3]).to(device)
        x = torch.cat((x, sigma_div_255), dim=1)
        return self.models(x)



def run_model(x, sigma):       
    '''
        x is image in [0, 1]
        simga in [0, 255]
    '''
    # print(x.size())
    sigma = float(sigma)
    sigma_div_255 = torch.FloatTensor([sigma]).repeat(1, 1, x.shape[2], x.shape[3]).to(device)
    # sigma_div_255 = 0*x + sigma
    x = torch.cat((x, sigma_div_255), dim=1)

    return model(x)
# # #









def print_line(y, pth, label):
    x = range(len(y))
    plt.plot(x, y, '-', alpha=0.8, linewidth=1.5, label=label)
    plt.legend(loc="upper right")
    plt.xlabel('iter')
    plt.ylabel(label)
    plt.savefig(pth)
    plt.close()    

# nb: default 50.
class PnP_ADMM(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nb=100, act_mode='R'):
        super(PnP_ADMM, self).__init__()
        self.nb = nb

        self.net = Drunet_running()
        # self.net = run_model()

        # only test
        self.res = {}
        self.res['psnr'] = [0] * nb
        self.res['ssim'] = [0] * nb
        self.res['image'] = [0]* nb

    def IRL1(self, f, u, v, b2, sigma, lamb, sigma2, k=10, eps=1e-5):
        for j in range(k):
            fenzi = lamb * (v-f)/(sigma**2+(v-f)**2)+(v-u-b2)
            fenmu = lamb * (sigma**2-(v-f)**2)/(sigma**2+(v-f)**2)**2+1
            v = v - fenzi / fenmu
            v = torch.clamp(v, min=0, max=255.)



        return v

    def get_psnr_i(self, u, clean, i):
        pre_i = torch.clamp(u / 255., 0., 1.)
        img_E = util.tensor2uint(pre_i)
        img_H = util.tensor2uint(clean)
        psnr = util.calculate_psnr(img_E, img_H, border=0)
        # print(psnr)
        ssim = util.calculate_ssim(img_E, img_H, border=0)
        self.res['psnr'][i] = psnr
        self.res['ssim'][i] = ssim
        #if i > 0:
        #    if self.res['psnr'][i] == max(self.res['psnr']):
        self.res['image'][i] = ToPILImage()(pre_i[0])

    def forward(self, kernel, initial_uv, f, clean, sigma=25.5, lamb=690, sigma2=1.0, denoisor_sigma=25, irl1_iter_num=10, eps=1e-5): 
        # model(kernel, initial_uv, img_L, img_H, sigma, lamb, sigma2, denoisor_level, 5, 1e-5)
        # init
        f *= 255
        u  = f

        K = kernel

        # a = 0.3
        # b = 0.15

        # a = 0.2
        # b = 0.1

        a = 0.1
        b = 0.05


        fft_k = deblur.p2o(K, u.shape[-2:])
        fft_kH = torch.conj(fft_k)

        # fft_kH = deblur.cconj(fft_k)
        abs_k = fft_kH * fft_k
        for k in range(self.nb):
            # u = f
            self.get_psnr_i(torch.clamp(u, min = -0., max =255.), clean, k)

            alpha = 1/( (k+1) **a )
            beta  = 1/( (k+1) **b )

            tt = torch.real(deblur.ifftn(abs_k*deblur.fftn(u) - fft_kH * deblur.fftn(f)))
            t = u - lamb*tt
            # t = torch.clamp(t, min = -0., max =255.) 
            t = t.type(torch.cuda.FloatTensor)
            v = (1-beta)*u+beta*(    0.5*(run_model(t/255,denoisor_sigma) * 255) + 0.5*t ) 
            tt = torch.real(deblur.ifftn(abs_k*deblur.fftn(v) - fft_kH * deblur.fftn(f)))
            t = v - lamb*tt
            # t = torch.clamp(t, min = -0., max =255.)
            t = t.type(torch.cuda.FloatTensor)
            u = (1-alpha)*u+alpha*(  0.5*(run_model(t/255,denoisor_sigma) * 255) + 0.5*t )



            
            
            
        
        return torch.clamp(u, min = -0., max =255.) # HBS/HQS

def plot_psnr(denoisor_level, lamb, sigma):
    device = 'cuda'
    model = PnP_ADMM()
    model.to(device)
    model.net.to(device)
    model.eval()
    model.net.eval()
    
    # i = 4
    sigma2 = 1.0
    # fp = "/home/dlwei/Documents/Rician_PnP_0/testsets/brain4/Brain{}.png".format(i)
    # img_H = util.imread_uint(fp, 1)
    fp = '/home/dlwei/Documents/pnp_jacobian/trainsets/CBSD68_cut8/0037.png'
    kernel_fp = '/home/dlwei/Documents/pnp_jacobian/kernels/kevin_2.png'
    kernel = util.imread_uint(kernel_fp,1)
    kernel = util.single2tensor3(kernel).unsqueeze(0) / 255.
    kernel = kernel / torch.sum(kernel)
    img_H = util.imread_uint(fp, 3)
    img_H = util.single2tensor3(img_H).unsqueeze(0) /255.
    initial_uv, img_L, img_H = gen_data(img_H, sigma,kernel)
    

    # img_H = util.uint2single(img_H)
    # img_L = util.uint2single(img_L)
    # initial_uv = util.uint2single(initial_uv)
    # print(img_H.shape)
    # print(img_L.shape)
    # print(initial_uv.shape)
    # img_L = util.single2tensor3(img_L).unsqueeze(0)
    # img_H = util.single2tensor3(img_H).unsqueeze(0)
    # initial_uv = util.single2tensor3(initial_uv).unsqueeze(0)

    initial_uv = initial_uv.to(device)
    img_L = img_L.to(device)
    img_H = img_H.to(device)
    kernel = kernel.to(device)




    with torch.no_grad():
        img_L, img_H = img_L.to(device), img_H.to(device)
        kernel = kernel.to(device)
        # model(img_L, img_H, sigma, lamb, sigma2, denoisor_level, 10, 1e-5)
        model(kernel, initial_uv, img_L, img_H, sigma, lamb, sigma2, denoisor_level, 5, 1e-5)

    savepth = 'images/'
    for j in range(len(model.res['image'])):
        # model.res['image'][j].save(savepth + 'result_Brain{}_{}.png'.format(i, j))
        model.res['image'][j].save(savepth + 'result_{}.png'.format(j))

    y = model.res['psnr']
    # print(y)
    print(y[-1])
    x = range(len(y))
    plt.plot(x, y, '-', alpha=0.8, linewidth=1.5)
    # plt.legend(loc="upper right")
    plt.xlabel('iter')
    plt.ylabel('PSNR')
    # plt.show()
    plt.savefig('PSNR_level{}_lamb{}.png'.format(denoisor_level, lamb))






def gen_data(img_clean_uint8, sigma,kernel):
    img_H = img_clean_uint8
    img_L = img_clean_uint8
    fft_k = deblur.p2o(kernel, img_L.shape[-2:])
    temp = fft_k * deblur.fftn(img_L)
    img_L = torch.abs(deblur.ifftn(temp))
    np.random.seed(seed=0)
    noise = np.random.normal(0, 1, img_L.shape)*sigma / 255
    # img_L = np.float32(np.random.poisson(img_L * sigma) / sigma)
    img_L += noise





    initial_uv = img_L
    return initial_uv, img_L, img_H






def search_args():
    sigma = 12.75
    # sigma = 17.85
    utils_logger.logger_info('rician', log_path='log/sigma_{}/logger.log'.format(sigma))
    logger = logging.getLogger('rician')
    device = 'cuda'

    logger.info('sigma = {}'.format(sigma))

    model = PnP_ADMM()
    model.to(device)
    model.net.to(device)
    model.eval()
    model.net.eval()


    dataset_root = '/home/dlwei/Documents/pnp_jacobian/trainsets/CBSD68_cut8/'


    max_psnr   = -1
    max_level  = -1
    max_lamb   = -1
    max_sigma2 = -1

    search_range = {}

    kernel_fp = '/home/dlwei/Documents/pnp_jacobian/kernels/kevin_8.png'
    # kevin's 8 kernels, CBSD68, 12.75 noise.
    # kernel 01, 
    search_range[0.04] = [1.2] # 26.3225, 0.7094
    # kernel 02, 
    # search_range[0.04] = [1.4] # 26.1396, 0.7027
    # kernel 03, 
    # search_range[0.04] = [1.3] # 26.8968, 0.7323
    # kernel 04, 
    # search_range[0.04] = [1.4] # 25.9320, 0.6897
    # kernel 05,
    # search_range[0.04] = [1.1] # 27.7880, 0.7686
    # kernel 06,
    # search_range[0.04] = [1.1] # 27.4420, 0.7553
    # kernel 07,
    # search_range[0.04] = [1.1] # 26.8058, 0.7353
    # kernel 08,
    # search_range[0.04] = [1.2] # 26.4084, 0.7181

    
    # kevin's 8 kernels, CBSD68, 17.85 noise.
    # kernel 01, 
    # search_range[0.05] = [1.0] # 25.5233, 0.6567
    # kernel 02, 
    # search_range[0.05] = [1.2] # 25.3607, 0.6523
    # kernel 03, 
    # search_range[0.05] = [1.0] # 26.1269, 0.6904
    # kernel 04, 
    # search_range[0.05] = [1.2] # 25.1497, 0.6385
    # kernel 05,
    # search_range[0.05] = [0.9] # 26.8919, 0.6946
    # kernel 06,
    # search_range[0.05] = [0.9] # 26.5301, 0.7067 
    # kernel 07,
    # search_range[0.05] = [0.9] # 26.0200, 0.6911
    # kernel 08,
    # search_range[0.05] = [1.0] # 25.6415, 0.6724



    


    kernel = util.imread_uint(kernel_fp,1)
    kernel = util.single2tensor3(kernel).unsqueeze(0) / 255.
    kernel = kernel / torch.sum(kernel)


    search_level = [0.04]

    psnr_save_root  = 'log/' + 'sigma_' + str(sigma) + '/psnr'
    ssim_save_root  = 'log/' + 'sigma_' + str(sigma) + '/ssim'
    image_save_root = 'log/' + 'sigma_' + str(sigma) + '/image'
    if not os.path.exists(psnr_save_root):
        os.makedirs(psnr_save_root)    
    if not os.path.exists(ssim_save_root):
        os.makedirs(ssim_save_root)   
    if not os.path.exists(image_save_root):
        os.makedirs(image_save_root)

    for denoisor_level in search_level:
        logger.info('========================================')
        logger.info('denoisor_level: {}'.format(denoisor_level))
        logger.info('========================================')
        for sigma2 in [1.]: 
            # for lamb in range(*search_range[denoisor_level]):
            for lamb in search_range[denoisor_level]:
                logger.info('==================')
                logger.info('lamb: {}'.format(lamb))

                dataset_psnr = None
                dataset_ssim = None
                image_paths = util.get_image_paths(dataset_root)
                # image_number=12
                image_number = len(image_paths)
                # image_number=5
                for ii in range(0,image_number):
                    fp = image_paths[ii]
                    # print('it is the image ')
                    # print(fp)
                    kernel = util.imread_uint(kernel_fp,1)
                    kernel = util.single2tensor3(kernel).unsqueeze(0) / 255.
                    kernel = kernel / torch.sum(kernel)
                    img_H = util.imread_uint(fp, 3)
                    img_H = util.single2tensor3(img_H).unsqueeze(0) /255.
                    initial_uv, img_L, img_H = gen_data(img_H, sigma,kernel)
                    
                    initial_uv = initial_uv.to(device)
                    img_L = img_L.to(device)
                    img_H = img_H.to(device)
                    kernel = kernel.to(device)

                    with torch.no_grad():
                        img_L, img_H = img_L.to(device), img_H.to(device)
                        kernel = kernel.to(device)
                        model(kernel, initial_uv, img_L, img_H, sigma, lamb, sigma2, denoisor_level, 5, 1e-5)

                    cur_psnr = np.array(model.res['psnr'])
                    # print(ii+1)
                    print(np.max(cur_psnr))
                    cur_ssim = np.array(model.res['ssim'])
                    if dataset_psnr is None:
                        dataset_psnr = cur_psnr
                        dataset_ssim = cur_ssim
                    else:
                        dataset_psnr += cur_psnr
                        dataset_ssim += cur_ssim
                # dataset_psnr /= len(image_paths)
                # dataset_ssim /= len(image_paths)
                dataset_psnr /= image_number
                dataset_ssim /= image_number
                print(dataset_psnr.shape)

                cur_avg_psnr = np.max(dataset_psnr)
                cur_avg_ssim = np.max(dataset_ssim)
                logger.info("PSNR: {:.4f}".format(cur_avg_psnr))
                logger.info("SSIM: {:.4f}".format(cur_avg_ssim))
                psnr_save_pth = psnr_save_root + '/level' + str(denoisor_level) + '_lamb' + str(lamb) + '_psnr' + str(cur_avg_psnr)[:7] + '.png'
                ssim_save_pth = ssim_save_root + '/level' + str(denoisor_level) + '_lamb' + str(lamb) + '_psnr' + str(cur_avg_psnr)[:7] + '.png'
                # image_save_pth = image_save_root + '/level' + str(denoisor_level) + '_lamb' + str(lamb) + '_psnr' + str(cur_psnr)[:7] + '.png'
                print_line(dataset_psnr, psnr_save_pth, "PSNR")
                print_line(dataset_ssim, ssim_save_pth, "SSIM")

                if cur_avg_psnr > max_psnr:
                    max_psnr   = cur_avg_psnr
                    max_level  = denoisor_level
                    max_lamb   = lamb
                    max_ssim   = cur_avg_ssim
                    # max_sigma2 = sigma2
                    # print(model.res['l'])


    logger.info('========================================')
    logger.info('========================================')
    logger.info('max_psnr: {:.4f}'.format(max_psnr))
    logger.info('max_ssim: {:.4f}'.format(max_ssim))
    logger.info('level: {:.4f}'.format(max_level))
    logger.info('lamb: {:.4f}'.format(max_lamb))
    return max_psnr, max_level, max_lamb






# PnPI-FBS

max_psnr, max_level, max_lamb = search_args()

# plot(level, lambda, sigma)


# PnPI-FBS 0026.png, kernel 6.
# plot_psnr(0.07, dsafas, 17.85) 


