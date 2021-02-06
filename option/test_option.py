import os

def get_train_options():
    opt = {}

    opt['project_dir'] = "/steven/home/cycle_pugan"
    opt['model_save_dir'] = opt['project_dir'] + '/checkpoints'
    opt["test_save_dir"]=opt['project_dir'] + '/test_results'
    opt['test_log_dir']=opt['project_dir'] + '/log_results'
    opt['dataset_dir'] = os.path.join(opt["project_dir"],"Patches_noHole_and_collected.h5")
    opt['isTrain']=False
    opt['batch_size'] = 1
    opt["patch_num_point"]=1024
    opt['lr_D_A']=1e-4
    opt['lr_G_AB']=1e-3
    opt['lr_D_B']=1e-4
    opt['lr_G_BA']=1e-3
    opt['emd_w_AB']=100.0
    opt['uniform_w_AB']=10.0
    opt['gan_w_AB']=0.5
    opt['repulsion_w_AB']=5.0
    opt['uniform_w_BA']=10.0
    opt['gan_w_BA']=0.5
    opt['repulsion_w_BA']=5.0
    return opt
