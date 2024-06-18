import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from bruges.filters import wavelets
import os
import torch
from os.path import join
from core.forward_models import forward_model
from core.utils import *
from core.datasets import SeismicDataset1D
from torch.utils.data import DataLoader
from torch.autograd import Variable
from core.model import Nash_MTL_STCN
import scipy.stats as stats
import errno
import argparse
import math
from skimage import metrics
from methods.Nash import NashMTL

def preprocess(no_wells, data_flag='marmousi'):
    
    data_dic = np.load(join('data','train_data.npy'), allow_pickle=True).item()    
    seismic = data_dic["synth_seismic_15db_noise"].squeeze()
    model = data_dic["parameter"].squeeze()[:,:,::6]

        
    seismic_mean = np.mean(seismic)
    seismic_std = np.std(seismic)
    seismic_normalization = standardize(mean_val=seismic_mean, std_val=seismic_std)
    seismic = seismic_normalization.normalize(seismic)
 
    train_indices = (np.linspace(0, len(model)-1, no_wells, dtype=np.int_))
    
    Vp_mean = np.mean(model[train_indices,0,:])
    Vp_std = np.std(model[train_indices,0,:])   
    Vp_normalization = standardize(mean_val=Vp_mean, std_val=Vp_std)
    
    Vs_mean = np.mean(model[train_indices,1,:])
    Vs_std = np.std(model[train_indices,1,:])   
    Vs_normalization = standardize(mean_val=Vs_mean, std_val=Vs_std)
    
    Den_mean = np.mean(model[train_indices,2,:])
    Den_std = np.std(model[train_indices,2,:])   
    Den_normalization = standardize(mean_val=Den_mean, std_val=Den_std)
    
    model[:,0,:] = Vp_normalization.normalize(model[:,0,:])
    model[:,1,:] = Vs_normalization.normalize(model[:,1,:])
    model[:,2,:] = Den_normalization.normalize(model[:,2,:])    

    return seismic, model, seismic_normalization,Vp_normalization, Vs_normalization, Den_normalization 


def get_models(args):
    
    wavelet, wavelet_time = wavelets.ormsby(args.wavelet_duration, args.dt,args.f, return_t=True)
    wavelet = torch.tensor(wavelet).unsqueeze(dim=0).unsqueeze(dim=0).float()
    forward_net = forward_model(wavelet=wavelet)
    forward_net.cuda()

    return  forward_net


def train(**kwargs):
    # obtain data
    seismic, model, seismic_normalization, Vp_normalization, Vs_normalization, Den_normalization  = preprocess(kwargs['no_wells'], kwargs['data_flag'])
    forward_net = get_models(args)  


    traces_seam_train = np.linspace(0, len(model)-1, kwargs['no_wells'], dtype=int)
    traces_seam_unlabel = np.linspace(1500, 1600, 101, dtype=int)
    
    traces_seam_validation = np.linspace(0, len(model)-1, 13, dtype=int)
    
    
    seam_train_dataset = SeismicDataset1D(seismic, model, traces_seam_train)
    seam_train_loader = DataLoader(seam_train_dataset, batch_size=args.batch_size)
    
    seam_unlabel_dataset = SeismicDataset1D(seismic, model, traces_seam_unlabel)   
    seam_unlabel_loader = DataLoader(seam_unlabel_dataset, batch_size=args.batch_size)
    
    seam_val_dataset = SeismicDataset1D(seismic, model, traces_seam_validation)
    seam_val_loader = DataLoader(seam_val_dataset, batch_size=args.batch_size)
    # seam_val_loader = DataLoader(seam_val_dataset, batch_size = len(seam_val_dataset))
    
    
    # define device for training
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")

    # set up models 
    model_seam = Nash_MTL_STCN().to(device)
    
    """load low-frequence model"""
    pretrained_model_state = torch.load("saved_models/model_low/marmousi2.pth") 
    model_seam.load_state_dict(pretrained_model_state)
    
    """load NashMTL"""
    method = NashMTL(n_tasks=3, device=device)
    
    """"Set up loss"""
    class AndrewsLoss(torch.nn.Module):
        def __init__(self, delta):
            super(AndrewsLoss, self).__init__()
            self.delta = delta
    
        def forward(self, input, target):
            abs_diff = torch.abs(input - target)
            mask = (abs_diff < self.delta).float()
            loss = torch.log(1 + torch.pow(abs_diff / self.delta, 2)) * mask + (abs_diff - 0.5 * self.delta) * (1 - mask)
            return torch.mean(loss)
    delta = 1.0
    criterion = AndrewsLoss(delta)
    
    optimizer_seam = torch.optim.Adam(model_seam.parameters(),
                                      weight_decay=0.0001,
                                      lr=0.001)
    
    for epoch in range(kwargs['epochs']):
    # for epoch in tqdm(range(args.epochs)):
    
      model_seam.train()
      optimizer_seam.zero_grad()
      
      for x,y in seam_train_loader:
        y_pred = model_seam(x)
        vp_train = criterion(y_pred[:,0,:], y[:,0,:]) 
        vs_train = criterion(y_pred[:,1,:], y[:,1,:]) 
        rhob_train = criterion(y_pred[:,2,:], y[:,2,:]) 
        

        supervised_weight = math.exp(-epoch / 1200) 
        unsupervised_weight = 1 - supervised_weight
        if args.beta!=0:
            #loading unlabeled data
            try:
                x_u = next(unlabeled)[0]
            except:
                unlabeled = iter(seam_unlabel_loader)
                x_u = next(unlabeled)[0]
                y_u = next(unlabeled)[1]
            y_u_pred = model_seam(x_u)
            y_u_pred[:,0,:] = Vp_normalization.unnormalize(y_u_pred[:,0,:])
            y_u_pred[:,1,:] = Vs_normalization.unnormalize(y_u_pred[:,1,:])
            y_u_pred[:,2,:] = Den_normalization.unnormalize(y_u_pred[:,2,:])
            y_u[:,0,:] = Vp_normalization.unnormalize(y_u[:,0,:])
            y_u[:,1,:] = Vs_normalization.unnormalize(y_u[:,1,:])
            y_u[:,2,:] = Den_normalization.unnormalize(y_u[:,2,:])   
            
            x_u_rec = forward_net(y_u_pred) 
            x_u_rec = seismic_normalization.normalize(x_u_rec)
            seismic_loss = criterion(x_u_rec,x_u)
            if not math.isnan(seismic_loss):
                vp_train = supervised_weight*vp_train + unsupervised_weight*seismic_loss/args.factor 
                vs_train = supervised_weight*vs_train + unsupervised_weight*seismic_loss/args.factor 
                rhob_train = supervised_weight*rhob_train + unsupervised_weight*seismic_loss/args.factor  
            
        
        """Nash Equilibrium"""
        losses = torch.stack((vp_train, vs_train, rhob_train))         
        property_loss, _ = method.backward(
            losses=losses,
            shared_parameters=list(model_seam.shared_parameters()),
            task_specific_parameters=list(model_seam.task_specific_parameters()),
            last_shared_parameters=list(model_seam.last_shared_parameters()),)    
       
      for x, y in seam_val_loader:     
        model_seam.eval()
        y_pred = model_seam(x)
        val_loss = criterion(y_pred, y)
        vp_val = criterion(y_pred[:,0,:], y[:,0,:]) 
        vs_val = criterion(y_pred[:,1,:], y[:,1,:]) 
        rhob_val = criterion(y_pred[:,2,:], y[:,2,:]) 
      loss_train = property_loss

      
      optimizer_seam.step()
      print('Epoch: {} | Train Loss: {:0.4f} | vp_train:{:0.4f} | vs_train:{:0.4f} | rhob_train:{:0.4f} | Val Loss: {:0.4f} \
        '.format(epoch, loss_train.item(),vp_train.item(), vs_train.item(), rhob_train.item(), val_loss.item()))
             
    
    # save trained models
    if not os.path.isdir('saved_models'):  # check if directory for saved models exists
        os.mkdir('saved_models')
    torch.save(model_seam.state_dict(), 'saved_models/model.pth')
    

def test(**kwargs):
    """Function tests the trained network on SEAM and Marmousi sections and 
    prints out the results"""
    
    # obtain data
    seismic, model, seismic_normalization, Vp_normalization, Vs_normalization, Den_normalization  = preprocess(kwargs['no_wells'], kwargs['data_flag'])
                                                                          
    
    # define device for training
    device = torch.device("cuda" )
    # specify pseudolog positions for testing 
    traces_seam_test = np.arange(len(model), dtype=int)
    seam_test_dataset = SeismicDataset1D(seismic, model, traces_seam_test)
    seam_test_loader = DataLoader(seam_test_dataset, batch_size = 8)
    # load saved models
    if not os.path.isdir('saved_models'):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 'saved_models')  
    # set up models
    model_seam = Nash_MTL_STCN().to(device)
    model_seam.load_state_dict(torch.load('saved_models/model.pth'))
    
    # infer on SEAM
    print("\nInferring ...")
    
    
    x, y = seam_test_dataset[0]  # get a sample

    AI_pred = torch.zeros((len(seam_test_dataset), y.shape[0], y.shape[-1])).float().to(device)
    AI_act = torch.zeros((len(seam_test_dataset), y.shape[0], y.shape[-1])).float().to(device)
    
    
    mem = 0
    with torch.no_grad():
        for i, (x,y) in enumerate(seam_test_loader):
          model_seam.eval()
          y_pred  = model_seam(x)
          # print(y_pred.shape)
          AI_pred[mem:mem+len(x)] = y_pred.squeeze().data
          AI_act[mem:mem+len(x)] = y.squeeze().data
          mem += len(x)
          del x, y, y_pred 
    
    vmin, vmax = AI_act.min(), AI_act.max()
    AI_pred = AI_pred.detach().cpu().numpy()
    AI_act = AI_act.detach().cpu().numpy()
    AI_pred[:,0,:] = Vp_normalization.unnormalize(AI_pred[:,0,:])
    AI_pred[:,1,:] = Vs_normalization.unnormalize(AI_pred[:,1,:])
    AI_pred[:,2,:] = Den_normalization.unnormalize(AI_pred[:,2,:])
    AI_act[:,0,:] = Vp_normalization.unnormalize(AI_act[:,0,:])
    AI_act[:,1,:] = Vs_normalization.unnormalize(AI_act[:,1,:])
    AI_act[:,2,:] = Den_normalization.unnormalize(AI_act[:,2,:])

    vmax = [ 5000, 3000, 2.65]
    vmin = [ 1000, 0, 1.90]    
    cols = ['{}'.format(col) for col in ['Predicted parameter','True parameter', 'Absolute difference']] 
    rows = ['{}'.format(row) for row in ['P-wave','S-wave', 'Density']] 
    colorbar_label = ['Vp ($m/s$)','Vs ($m/s$)','Density ($g/cm^3$)']
    fig, axes = plt.subplots(nrows=AI_act.shape[1], ncols=3)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(AI_act.shape[1]):
        im = axes[i][0].imshow(AI_pred[:,i,:].T, extent=(0,17000,3500,400), cmap='jet',aspect=3, vmin=vmin[i], vmax=vmax[i])
        cbar = axes[i][0].figure.colorbar(im ,fraction=0.05, pad=0.05)  
        cbar.set_label(colorbar_label[i],rotation=-90 )
        cbar.ax.yaxis.set_label_coords(7.2, 0.5)
        axes[i][0].set_xticks(np.linspace(0,17000,6))
        axes[i][0].set_yticks(np.linspace(500,3500,4))  
        axes[i][0].tick_params(axis='both', which='major', labelsize=10) 
        axes[i][0].set_ylabel('Depth(m)')
        axes[i][0].xaxis.tick_top()
        axes[i][0].xaxis.set_label_position("top")
        im = axes[i][1].imshow(AI_act[:,i,:].T, extent=(0,17000,3500,400), cmap='jet',aspect=3,vmin=vmin[i], vmax=vmax[i])
        cbar = axes[i][1].figure.colorbar(im ,fraction=0.05, pad=0.05)
        cbar.set_label(colorbar_label[i],rotation=-90 )
        cbar.ax.yaxis.set_label_coords(7.2, 0.5)
        axes[i][1].set_xticks(np.linspace(0,17000,6))
        axes[i][1].set_yticks(np.linspace(500,3500,4))  
        axes[i][1].tick_params(axis='both', which='major', labelsize=10) 
        axes[i][1].set_ylabel('Depth(m)')
        axes[i][1].xaxis.tick_top()
        axes[i][1].xaxis.set_label_position("top")
        im = axes[i][2].imshow(abs(AI_act[:,i,:].T-AI_pred[:,i,:].T), extent=(0,17000,3500,400) ,cmap='gray',aspect=3)
        cbar = axes[i][2].figure.colorbar(im ,fraction=0.05, pad=0.05)
        cbar.set_label(colorbar_label[i],rotation=-90 )
        cbar.ax.yaxis.set_label_coords(7.2, 0.5)
        axes[i][2].set_xticks(np.linspace(0,17000,6))
        axes[i][2].set_yticks(np.linspace(500,3500,4))  
        axes[i][2].tick_params(axis='both', which='major', labelsize=10) 
        axes[i][2].set_ylabel('Depth(m)')
        axes[i][2].xaxis.tick_top()
        axes[i][2].xaxis.set_label_position("top")
        for j in range(3):
            axes[i][j].xaxis.tick_top()  
            axes[i][j].xaxis.set_label_position("top") 

    
    pad = 50 # in points
    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row,xy=(0,0.5), xytext=(-(pad+20),0), xycoords='axes fraction', textcoords='offset points', ha='right', va='center', fontsize=12)
    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points', ha='center', va='baseline', fontsize=12)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparams')
    
    parser.add_argument('--epochs', nargs='?', type=int, default=500,
                        help='Number of epochs. Default = 500')
    parser.add_argument('--no_wells', nargs='?', type=int, default=24,
                        help='Number of sampled pseudologs for seismic section. Default = 24.')
    parser.add_argument('--data_flag', type=str, default='marmousi',
                        help='Data flag to specify the dataset used to train the model')
    parser.add_argument('-beta', type=float, default=1, help="weight of seismic loss term")
    parser.add_argument('-batch_size', type=int, default=100,help="Batch size for training")
    parser.add_argument('-factor', type=int, default=20,help="factor for unsupervised.  Default = 10")
    parser.add_argument('-dt', type=float, default=1e-3, help='Time resolution in seconds')
    parser.add_argument('-wavelet_duration',  type=float, default=0.2, help='wavelet duration in seconds')
    parser.add_argument('-f', default="5, 10, 60, 80", help="Frequency of wavelet. if multiple frequencies use , to seperate them with no spaces, e.g., -f \"5,10,60,80\"", type=lambda x: np.squeeze(np.array(x.split(",")).astype(float)))

    args = parser.parse_args()
    train(no_wells=args.no_wells, epochs=args.epochs, data_flag=args.data_flag)
    test(no_wells=args.no_wells, epochs=args.epochs, data_flag=args.data_flag)
    
    