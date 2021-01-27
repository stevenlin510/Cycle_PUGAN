import os,sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', '-e', type=str, required=True, help='experiment name')
parser.add_argument('--debug', action='store_true', help='specify debug mode')
parser.add_argument('--batch_size',type=int,default=16)
parser.add_argument('--gpu',type=str,default='0')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
sys.path.append('../')
import torch
from network.networks import Generator,Discriminator,Downsampler
from data.data_loader import PUNET_Dataset
import time
from option.train_option import get_train_options
from utils.Logger import Logger
from torch.utils import data
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from loss.loss import Loss
import datetime
import torch.nn as nn
from cyclegan.utils import ReplayBuffer

criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

def xavier_init(m):
    classname = m.__class__.__name__
    #print('classname: ',classname)
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('Linear')!=-1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def train(args):
    start_t=time.time()
    params=get_train_options()
    params["exp_name"]=args.exp_name
    params["patch_num_point"]=1024
    params["batch_size"]=args.batch_size

    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

    if args.debug:
        params["nepoch"]=2
        params["model_save_interval"]=3
        params['model_vis_interval']=3

    log_dir=os.path.join(params["model_save_dir"],args.exp_name)
    if os.path.exists(log_dir)==False:
        os.makedirs(log_dir)
    tb_logger=Logger(log_dir)

    trainloader=PUNET_Dataset(h5_file_path=params["dataset_dir"],split_dir=params['train_split'])
    #print("params['dataset_dir']: ",params["dataset_dir"])
    num_workers=4
    train_data_loader=data.DataLoader(dataset=trainloader,batch_size=params["batch_size"],shuffle=True,
                                      num_workers=num_workers,pin_memory=True,drop_last=True)
    device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')

    ##########################################
    # Initialize generator and discriminator #
    ##########################################
    G_AB=Generator(params)
    G_AB.apply(xavier_init)
    G_AB=torch.nn.DataParallel(G_AB).to(device)

    G_BA=Downsampler(params)
    G_BA.apply(xavier_init)
    G_BA=torch.nn.DataParallel(G_BA).to(device)


    D_A=Discriminator(params,in_channels=3)
    D_A.apply(xavier_init)
    D_A=torch.nn.DataParallel(D_A).to(device)

    D_B=Discriminator(params,in_channels=3)
    D_B.apply(xavier_init)
    D_B=torch.nn.DataParallel(D_B).to(device)


    ########################################
    #Optimizers and Learning Rate scheduler#
    ########################################

    optimizer_D_A=Adam(D_A.parameters(),lr=params["lr_D"],betas=(0.9,0.999))
    optimizer_D_B=Adam(D_B.parameters(),lr=params["lr_D"],betas=(0.9,0.999))

    optimizer_G_AB=Adam(G_AB.parameters(),lr=params["lr_G"],betas=(0.9,0.999))
    optimizer_G_BA=Adam(G_BA.parameters(),lr=params["lr_G"],betas=(0.9,0.999))

    D_A_scheduler = MultiStepLR(optimizer_D_A,[50,80],gamma=0.2)
    G_AB_scheduler = MultiStepLR(optimizer_G_AB,[50,80],gamma=0.2)
    D_B_scheduler = MultiStepLR(optimizer_D_A,[50,80],gamma=0.2)
    G_BA_scheduler = MultiStepLR(optimizer_G_AB,[50,80],gamma=0.2)

    Loss_fn=Loss()

    print("preparation time is %fs" % (time.time() - start_t))
    iter=0
    for e in range(params["nepoch"]):

        for batch_id,(input_data, gt_data, radius_data) in enumerate(train_data_loader):

            G_AB.train()
            G_BA.train()
            D_A.train()
            D_B.train()

            optimizer_G_AB.zero_grad()
            optimizer_D_A.zero_grad()
            optimizer_G_BA.zero_grad()
            optimizer_D_B.zero_grad()

            input_data=input_data[:,:,0:3].permute(0,2,1).float().cuda()
            gt_data=gt_data[:,:,0:3].permute(0,2,1).float().cuda()
            start_t_batch=time.time()

            output_point_cloud_high=G_AB(input_data)
            output_point_cloud_low=G_BA(gt_data)

            #####################################
            #               Loss                #
            #####################################
            repulsion_loss_AB = Loss_fn.get_repulsion_loss(output_point_cloud_high.permute(0, 2, 1))
            uniform_loss_AB = Loss_fn.get_uniform_loss(output_point_cloud_high.permute(0, 2, 1))
            repulsion_loss_BA = Loss_fn.get_repulsion_loss(output_point_cloud_low.permute(0, 2, 1))
            uniform_loss_BA = Loss_fn.get_uniform_loss(output_point_cloud_low.permute(0, 2, 1))
            emd_loss_AB = Loss_fn.get_emd_loss(output_point_cloud_high.permute(0, 2, 1), gt_data.permute(0, 2, 1))
            #emd_loss_Ba = Loss_fn.get_emd_loss(output_point_cloud_low.permute(0, 2, 1), input_data.permute(0, 2, 1))

            # Identity loss
            loss_id_A = criterion_identity(G_BA(input_data), input_data)
            loss_id_B = criterion_identity(G_AB(gt_data), gt_data)
            loss_identity = (loss_id_A + loss_id_B) // 2

            #GAN loss
            fake_pred_B = D_A(output_point_cloud_high.detach())
            d_A_loss_fake = Loss_fn.get_discriminator_loss_single(fake_pred_B,label=False)
            d_A_loss_fake.backward()
            optimizer_D_A.step()

            real_pred_B = D_A(gt_data.detach())
            d_A_loss_real = Loss_fn.get_discriminator_loss_single(real_pred, label=True)
            d_A_loss_real.backward()
            optimizer_D_A.step()

            d_A_loss=d_A_loss_real+d_A_loss_fake

            fake_pred=D_A(output_point_cloud_high)
            g_AB_loss=Loss_fn.get_generator_loss(fake_pred)
            #print(repulsion_loss,uniform_loss,emd_loss)
            total_G_AB_loss=params['uniform_w']*uniform_loss+params['emd_w']*emd_loss+ \
            repulsion_loss*params['repulsion_w']+ g_AB_loss*params['gan_w']

            

            total_G_AB_loss.backward()
            optimizer_G_AB.step()

            current_lr_D_A=optimizer_D_A.state_dict()['param_groups'][0]['lr']
            current_lr_G_AB=optimizer_G_AB.state_dict()['param_groups'][0]['lr']

            tb_logger.scalar_summary('repulsion_loss', repulsion_loss.item(), iter)
            tb_logger.scalar_summary('uniform_loss', uniform_loss.item(), iter)
            tb_logger.scalar_summary('emd_loss', emd_loss.item(), iter)
            
            tb_logger.scalar_summary('d_A_loss', d_A_loss.item(), iter)
            tb_logger.scalar_summary('g_AB_loss', g_AB_loss.item(), iter)
            tb_logger.scalar_summary('Total_G_AB_loss', total_G_AB_loss.item(), iter)
            tb_logger.scalar_summary('lr_D', current_lr_D_A, iter)
            tb_logger.scalar_summary('lr_G', current_lr_G_AB, iter)

            msg="{:0>8},{}:{}, [{}/{}], {}: {},{}:{}".format(
                str(datetime.timedelta(seconds=round(time.time() - start_t))),
                "epoch",
                e,
                batch_id + 1,
                len(train_data_loader),
                "total_G_loss",
                total_G_AB_loss.item(),
                "iter time",
                (time.time() - start_t_batch)
            )
            print(msg)

            iter+=1

        D_A_scheduler.step()
        G_AB_scheduler.step()
        
        if (e+1) % params['model_save_interval'] == 0 and e > 0:
            model_save_dir = os.path.join(params['model_save_dir'], params['exp_name'])
            if os.path.exists(model_save_dir) == False:
                os.makedirs(model_save_dir)
            D_A_ckpt_model_filename = "D_A_iter_%d.pth" % (e)
            G_AB_ckpt_model_filename = "G_AB_iter_%d.pth" % (e)
            D_A_model_save_path = os.path.join(model_save_dir, D_A_ckpt_model_filename)
            G_AB_model_save_path = os.path.join(model_save_dir, G_AB_ckpt_model_filename)
            torch.save(D_A_model.module.state_dict(), D_A_model_save_path)
            torch.save(G_AB_model.module.state_dict(), G_AB_model_save_path)





            



if __name__=="__main__":
    import colored_traceback
    train(args)