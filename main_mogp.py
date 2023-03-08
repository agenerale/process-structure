import os
import scipy.io as sio
import numpy as np
import math
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import gpytorch
from sklearn import preprocessing
from scipy import interpolate
import argparse
from matplotlib import pyplot as plt
plt.rc('xtick',labelsize=30)
plt.rc('ytick',labelsize=30)
font = {'family' : 'normal','weight' : 'normal','size'   : 32}
plt.rc('font', **font)

parser = argparse.ArgumentParser(description="Deep Probabilistic Inverse Microstructure Training beta-VAE")
parser.add_argument("--train", default=True, type=bool, help="train (True) cuda")
parser.add_argument("--load", default=False, type=bool, help="load trained model")
parser.add_argument("--batch", default=1024, type=int, help="minibatch training size")
parser.add_argument("--num_latent", default=20, type=int, help="# latent GPs")
parser.add_argument("--num_inducing", default=100, type=int, help="# inducing points")
parser.add_argument("--num_epochs", default=2000, type=int, help="# training epochs")
parser.add_argument("--lr_init", default=1e-2, type=float, help="init. learning rate")
parser.add_argument("--lr_end", default=0, type=float, help="end learning rate")
parser.add_argument("--pc", default=0, type=int, help="model for PC # [0 - 9]")
parser.add_argument("--curve_points", default=20, type=float, help="# discretization points")
args = parser.parse_args()

device = torch.device("cuda" if (torch.cuda.is_available() and args.train) else "cpu")
device = torch.device("cuda")
params_train = np.load('params_train.npy')
params_test = np.load('params_test.npy')
scores_train = np.load('scores_train.npy')
scores_test = np.load('scores_test.npy')

###############################################################################       
efinal = 99
e11 = np.arange(0,efinal+1e-10,efinal/(args.curve_points-1))

scores_train_interp = np.zeros((scores_train.shape[0],args.curve_points,10))
scores_test_interp = np.zeros((scores_test.shape[0],args.curve_points,10))        

for i in range(scores_train.shape[0]):
    for j in range(10):
        fs = interpolate.interp1d(np.arange(0,100,1),scores_train[i,:,j])
        scores_train_interp[i,:,j] = fs(e11)
        
        if i < scores_test.shape[0]:
            fs = interpolate.interp1d(np.arange(0,100,1),scores_test[i,:,j])
            scores_test_interp[i,:,j] = fs(e11)
    print(i)
        
fig = plt.figure(figsize=(10, 8))
for i in range(100):
    plt.plot(np.arange(0,100,1),scores_train[i,:,0],'k--',linewidth=3)
    plt.plot(e11,scores_train_interp[i,:,0])
plt.title('Interpolate to ' + str(args.curve_points) + ' points')
###############################################################################
train_x = np.hstack((params_train,scores_train_interp[:,0,:]))
test_x = np.hstack((params_test,scores_test_interp[:,0,:]))
train_x = torch.from_numpy(train_x).float().to(device)
test_x = torch.from_numpy(test_x).float().to(device)

train_y = torch.from_numpy(scores_train_interp[:,:,args.pc]).float().to(device)
test_y = torch.from_numpy(scores_test_interp[:,:,args.pc]).float().to(device)

# Standard scaling
m = train_x.mean(0, keepdim=True)
s = train_x.std(0, unbiased=False, keepdim=True)
train_x -= m
train_x /= s 

test_x -= m
test_x /= s 

#mo = train_y.mean(0, keepdim=True)
#so = train_y.std(0, unbiased=False, keepdim=True)
#train_y -= mo
#train_y /= so

#test_y -= mo
#test_y /= so 

train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)

train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch, drop_last=False, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch, drop_last=False, shuffle=True)

# Specify MOGP model
class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, num_latents, num_tasks, num_inducing, input_dims):
        
        inducing_points = torch.randn(num_latents, num_inducing, input_dims)
        
        batch_shape = torch.Size([num_latents])

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=batch_shape, mean_init_std = 1e-6,
        )

        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1
        )

        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))

        #self.covar_module = gpytorch.kernels.ScaleKernel(
        #    gpytorch.kernels.RBFKernel(batch_shape=batch_shape,ard_num_dims=input_dims),
        #    batch_shape=batch_shape, ard_num_dims=None
        #)     
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=0.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)   

num_latents = args.num_latent
num_tasks = args.curve_points
num_inducing = args.num_inducing
input_dims = train_x.shape[1]

model = MultitaskGPModel(num_latents,num_tasks,num_inducing,input_dims).to(device)
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks).to(device)

print('Model Parameters: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
print('Lieklihood Parameters: ' + str(sum(p.numel() for p in likelihood.parameters() if p.requires_grad)))

if args.load:
    state_dict_model = torch.load('mogp_model_lmc_'+str(args.pc)+'_state.pth', map_location=device)
    state_dict_likelihood = torch.load('mogp_likelihood_lmc_'+str(args.pc)+'_state.pth', map_location=device)
    model.load_state_dict(state_dict_model)
    likelihood.load_state_dict(state_dict_likelihood)

if args.train:
    
    model.train()
    likelihood.train()

    #optimizer = torch.optim.Adam([
    #    {'params': model.parameters()},
    #    {'params': likelihood.parameters()},
    #    ], lr=args.lr_init)

    optimizer = torch.optim.RMSprop([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=args.lr_init)

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.num_epochs,args.lr_end)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, int(args.num_epochs/2), 1, args.lr_end)
    
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))    # loss object ELBO
    loss_list = []
    for i in range(args.num_epochs):
        batch_losses = []
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.cpu().detach())

        scheduler.step()
        loss_mean = np.mean(batch_losses)
    
        if (i + 1) % 5 == 0:
            print(f"epoch: {(i+1):}, loss: {loss_mean:.5f}, lr: {scheduler.get_last_lr()[0]:.5f}")
            loss_list.append(loss_mean)
            
        if (i + 1) % int(args.num_epochs/5) == 0:
            # Save model
            torch.save(model.state_dict(), 'mogp_model_lmc_'+str(args.pc)+'_state.pth')
            torch.save(likelihood.state_dict(), 'mogp_likelihood_lmc_'+str(args.pc)+'_state.pth')
  

# Make predictions
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    train_mean = torch.zeros((train_y.shape))
    trainlower = torch.zeros((train_y.shape))
    trainupper = torch.zeros((train_y.shape))
    for i in range(int(train_y.shape[0])):
        train_predictions = likelihood(model(train_x[i:(i+1),:]))
        train_mean[i:(i+1),:] = train_predictions.mean.cpu()
        trainlower[i:(i+1),:], trainupper[i:(i+1),:] = train_predictions.confidence_region()
        trainupper[i:(i+1),:] = trainupper[i:(i+1),:].cpu()
        trainlower[i:(i+1),:] = trainlower[i:(i+1),:].cpu()
        
    test_mean = torch.zeros((test_y.shape))
    testlower = torch.zeros((test_y.shape))
    testupper = torch.zeros((test_y.shape))
    for i in range(int(test_y.shape[0])):
        test_predictions = likelihood(model(test_x[i:(i+1),:]))
        test_mean[i:(i+1),:] = test_predictions.mean.cpu()
        testlower[i:(i+1),:], testupper[i:(i+1),:] = test_predictions.confidence_region()
        testupper[i:(i+1),:] = testupper[i:(i+1),:].cpu()
        testlower[i:(i+1),:] = testlower[i:(i+1),:].cpu() 
    
train_mean = train_mean.detach().cpu().numpy()
test_mean = test_mean.detach().cpu().numpy()
#train_y *= so
#train_y += mo 

#test_y *= so
#test_y += mo 

train_y = train_y.detach().cpu().numpy()
test_y = test_y.detach().cpu().numpy()
   
mae_train = np.mean(abs(train_mean - train_y),axis=0)
mae_test = np.mean(abs(test_mean - test_y),axis=0)
print(mae_train)
print(mae_test)

# Plotting stuff
fig = plt.figure(figsize=(15, 12.5))
plt.plot(e11,mae_train,linestyle='-', marker='o',)
plt.plot(e11,mae_test,linestyle='-', marker='o',)
plt.ylabel(r'$NMAE$') 
plt.legend(['Train','Test'])
plt.savefig('MAE_'+str(args.pc)+'.png', bbox_inches='tight')

# Plot all curves
fig = plt.figure(figsize=(15, 12.5))
for i in range(0,10,1):
    plt.plot(e11,train_mean[i,:],'k')
    plt.plot(e11,train_y[i,:],'--',color='tab:red')
    plt.fill_between(e11, trainupper[i,:], trainlower[i,:], alpha=0.1, color='tab:blue')
plt.ylabel(r'$\alpha_{'+str(args.pc)+'}$')
plt.legend(['Pred','Actual'], loc='upper left')
plt.title('Train')
plt.savefig('Train_'+str(args.pc)+'.png', bbox_inches='tight')

fig = plt.figure(figsize=(15, 12.5))
for i in range(0,10,1):
    plt.plot(e11,test_mean[i,:],'k')
    plt.plot(e11,test_y[i,:],'--',color='tab:red')
    plt.fill_between(e11, testupper[i,:], testlower[i,:], alpha=0.1, color='tab:blue')
plt.ylabel(r'$\alpha_{'+str(args.pc)+'}$')
plt.legend(['Pred','Actual'], loc='upper left')
plt.title('Test')
plt.savefig('Test_'+str(args.pc)+'.png', bbox_inches='tight')

# Parity plots every 10 points
inc = 5
fig, axes = plt.subplots(1,int(args.curve_points/inc), figsize=(int(inc*args.curve_points/2), 8), sharex=False)
for i in range(int(args.curve_points/inc)):
    ax = axes[i]
    ax.errorbar(train_y[:,inc*(i+1)-1],train_mean[:,inc*(i+1)-1],yerr=(trainupper[:,inc*(i+1)-1]-trainlower[:,inc*(i+1)-1]),fmt='o',capsize=2)
    ax.errorbar(test_y[:,inc*(i+1)-1],test_mean[:,inc*(i+1)-1],yerr=(testupper[:,inc*(i+1)-1]-testlower[:,inc*(i+1)-1]),fmt='o',capsize=2)
    ax.legend(['Train','Test'])
    ax.plot([np.min(train_y[:,inc*(i+1)-1]),np.max(train_y[:,inc*(i+1)-1])],
                 [np.min(train_y[:,inc*(i+1)-1]),np.max(train_y[:,inc*(i+1)-1])],'k--',linewidth=3)
    ax.set_xlabel(r'Actual $\alpha_{'+str(args.pc)+'}$')
    ax.set_ylabel(r'Predicted $\alpha_{'+str(args.pc)+'}$')
    ax.set_title('#'+str(inc*(i+1)))
plt.savefig('Points_'+str(args.pc)+'.png', bbox_inches='tight')  
    