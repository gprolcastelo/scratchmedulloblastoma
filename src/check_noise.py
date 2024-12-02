import datetime
import matplotlib.pyplot as plt
from src.models.train_model import set_seed
from src.models.my_model import VAE
plt.style.use('ggplot')
# Custom functions:
from src.utils import apply_VAE, fun_decode, get_hyperparams
plt.style.use('ggplot')
device = 'cpu'
today = datetime.datetime.now().strftime("%Y%m%d")


from src.data_processing.classification import *
from src.visualization.visualize import plot_umap
from src.data_processing.shap import *
from src.data_augmentation import synth_data_generation

# Importing the data:

full_data_path = "data/interim/20241115_preprocessing/cavalli_maha.csv"
clinical_data_path = "data/raw/GEO/cavalli_subgroups.csv"


# Load data:
rnaseq = pd.read_csv(full_data_path, index_col=0)
print(rnaseq.shape)
clinical = pd.read_csv(clinical_data_path, index_col=0)
clinical = clinical['Sample_characteristics_ch1']
clinical.replace({'Group3':'Group 3','Group4':'Group 4'},inplace=True)
print(clinical.shape)

# Importing VAE
model_here = VAE
model_path = 'models/20241115_VAE/20241115_VAE_idim14349_md1024_feat64_lr0.0001.pth'
# Hyperparameters:
idim, md, feat = get_hyperparams(model_path)
# One hot encode the clinical data:
clinical_onehot = pd.get_dummies(clinical)
# Importing the model:
set_seed(2023)
model_vae = model_here(idim, md, feat)  # Initialize the model
model_vae.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load the state dictionary
model_vae.eval()  # Set the model to evaluation mode
# Apply AE or VAE to all data:
reconstruction_x, _, _, z, scaler = apply_VAE(
    torch.tensor(rnaseq.values).to(torch.float32),
    model_vae,
    y=None)
# data_reconstruction = pd.DataFrame(data_reconstruction.T, index=data.index, columns=data.columns)
df_reconstruction_x = pd.DataFrame(reconstruction_x, index=rnaseq.index, columns=rnaseq.columns)
df_z = pd.DataFrame(z, index=rnaseq.index)

# Reconstruction network
from src.adjust_reconstruction import NetworkReconstruction as model_net_here
network_model_path = "models/20241115_adjust_reconstruction/network_reconstruction.pth"
hyperparam_path = "data/interim/20241115_adjust_reconstruction/best_hyperparameters.csv"
# Hyperparameters:
hyper = pd.read_csv(hyperparam_path,index_col=0)
hyper = hyper.values.tolist()[0]
hyper=[idim]+hyper+[idim]
# Importing the model:
set_seed(2023)
model_net = model_net_here(hyper)  # Initialize the model
model_net.load_state_dict(torch.load(network_model_path, map_location=torch.device('cpu')))  # Load the state dictionary
model_net.eval()  # Set the model to evaluation mode

# Apply reconstruction network:
rec_tensor = torch.tensor(reconstruction_x).to(torch.float32)
net_output=model_net(rec_tensor)
df_net_output = pd.DataFrame(net_output.detach().numpy(), index=rnaseq.index, columns=rnaseq.columns)
net_output.shape, df_net_output.shape

# Checking noise with UMAP

save_dir = f'reports/figures/{today}_noise_umaps/'
os.makedirs(save_dir,exist_ok=True)
save_fig = True
mu = 0
std = 1
dict_umap = {'SHH': '#b22222', 'WNT': '#6495ed', 'Group 3': '#ffd700', 'Group 4': '#008000', 'In Between': '#db7093', 'Synthetic': '#808080'}

dict_umap_augment = {'SHH': '#b22222', 'WNT': '#6495ed', 'Group 3': '#ffd700', 'Group 4': '#008000',
                     'synthetic_SHH': 'coral', 'synthetic_WNT': 'cyan', 'synthetic_Group 3': 'yellow', 'synthetic_Group 4': 'limegreen',
                    }

group_to_augment = ['SHH','WNT','Group 3','Group 4']

# Real space
space='realspace'
for n_synth in [50,100,200,500]:
    for noise_ratio in [0.1,0.2,0.3,0.5]:
        title = f'UMAP with n_synth = {n_synth}, noise_ratio = {noise_ratio}, mu = {mu}, std = {std}'
        file_name = f'umap_{space}_nsynth{n_synth}_noiseratio{noise_ratio}_mu{mu}_std{std}'
        save_as = os.path.join(save_dir,space,file_name)
        os.makedirs(os.path.join(save_dir,space),exist_ok=True)
        df_augment, metadata_augment = synth_data_generation(df_to_augment=rnaseq, metadata=clinical, group_to_augment=group_to_augment, mu=0, std=1, n_synth=n_synth, rand_state=2023, noise_ratio=noise_ratio)
        plot_umap(data=df_augment.T, clinical=metadata_augment, colors_dict=dict_umap_augment, n_components=2, save_fig=save_fig, save_as=save_as, seed=None, title= f'{title}',show=False)

# Noise in Latent Space, then Decode + Post-Processing

# augment data in l.s., then plot umap with synthetic (reconstructed) and real patients
df_here = df_z
space='latentspace_to_recons_plus_real'
include_real_recons = False
for n_synth in [50,100,200,500]:
    for noise_ratio in [0.1,0.2,0.3,0.5]:
        title = f'UMAP in {space} with n_synth = {n_synth}, noise_ratio = {noise_ratio}, mu = {mu}, std = {std}'
        file_name = f'umap_{space}_nsynth{n_synth}_noiseratio{noise_ratio}_mu{mu}_std{std}'
        save_as = os.path.join(save_dir,space,file_name)
        os.makedirs(os.path.join(save_dir,space),exist_ok=True)
        # augment data in l.s.
        df_augment, metadata_augment = synth_data_generation(df_to_augment=df_here, metadata=clinical, group_to_augment=group_to_augment, mu=0, std=1, n_synth=n_synth, rand_state=2023, noise_ratio=noise_ratio)
        
        # l.s. to real space with decoder
        ## select synthetic patients:
        synth_pats=[i.startswith('synth') for i in df_augment.index]

        print('A) Not including real data')
        df_z_augment = df_augment.loc[synth_pats]
        ## apply decoder and inverse scaler to synth patients in l.s:
        df_recons_augment = fun_decode(df_z_augment,model=model_vae,scaler=scaler)
        print("type(df_recons_augment):",type(df_recons_augment))
        ## apply postprocessing network:
        model_net.eval()  # Set the model to evaluation mode
        rec_tensor_i = torch.tensor(df_recons_augment).to(torch.float32)
        df_recons_augment = model_net(rec_tensor_i).detach().numpy()
        df_recons_augment = pd.DataFrame(df_recons_augment,index=df_augment.index[synth_pats],columns=rnaseq.columns)
        ## concat with real patients:
        df_recons_augment=pd.concat([rnaseq,df_recons_augment])
        
        print("df_recons_augment.shape = ", df_recons_augment.shape)
        # plot synth (decoded) and real patients together
        plot_umap(data=df_recons_augment, clinical=metadata_augment, colors_dict=dict_umap_augment, n_components=2, save_fig=save_fig, save_as=save_as, seed=None, title= f'{title}',show=False)