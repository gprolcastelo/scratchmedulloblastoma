'''
Code to explain classification of medulloblastoma subgroups in the latent space.
Explainability is performed using SHAP values in the latent space, and then in the original space.
This ay, first we map subgroups --> latent variables, and then latent variables --> genes, to find the mapping of subgroups --> genes.
Author: Guillermo Prol-Castelo
Date: 2024-10-31
License: Apache 2.0
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch, os, sys, datetime, re, shap, pickle, csv
import xgboost as xgb
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score
from src.models.train_model import set_seed
from src.models.my_model import VAE, CVAE
from src.data_processing.shap import *
from src.data_processing.classification import *
from src.utils import apply_VAE, get_hyperparams
plt.style.use('ggplot')
def str_or_list(value):
    if isinstance(value, str):
        return [item.strip() for item in value.split(',')]
    elif isinstance(value, list):
        return value
    else:
        raise argparse.ArgumentTypeError("Value must be a string or a list")

def xgboost_shap(embeddings,xgb_model):
    # SHAP Tree Explainer to get the importance of the features
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(embeddings)  # we want to explain the whole dataset

    return explainer, shap_values

def vae_shap(data, model_vae):
    # MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(data)
    data2explain = scaler.transform(data)
    data2explain = torch.tensor(data2explain).float()
    # SHAP Deep Explainer to get the importance of the features
    deep_explainer = shap.DeepExplainer(model_vae, data2explain)
    deep_shap_values = deep_explainer.shap_values(data2explain)

    return deep_explainer, deep_shap_values

def sum_shap_values(shap_values,q_here,cols,idx):
    # Get the sum of SHAP value magnitudes over all samples
    sum_shap_values = np.sum(np.abs(shap_values), axis=0)
    # Apply quantile to the sum of SHAP values to get the most important features
    q_shap_values = np.quantile(sum_shap_values,q_here,axis=0)
    # Get the most important features (genes)
    tf_shap_values = sum_shap_values>q_shap_values
    important_features = pd.DataFrame(tf_shap_values,columns=cols,index=idx)
    return important_features, sum_shap_values

# We will only use the encoder and reparametrization trick from the VAE
class EncoderWithReparam(torch.nn.Module):
    def __init__(self, vae_model, selected_lv=None):
        super().__init__()
        self.vae_model = vae_model
        self.selected_lv = selected_lv

    def forward(self, x):
        mu_logvar = self.vae_model.encoder(x).view(-1, 2, self.vae_model.features)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.vae_model.reparametrize(mu, logvar)
        if self.selected_lv is not None:
            z = z[:, self.selected_lv]
        return z
#######################################################################

class EncoderOnly(torch.nn.Module):
    def __init__(self, vae_model, selected_lv=None):
        super().__init__()
        self.vae_model = vae_model
        self.selected_lv = selected_lv

    def forward(self, x):
        mu_logvar = self.vae_model.encoder(x).view(-1, 2, self.vae_model.features)
        if self.selected_lv is not None:
            # print('Inside EncoderOnly, and if self.selected_lv is not None')
            mu = mu_logvar[:, 0, self.selected_lv]
            logvar = mu_logvar[:, 1, self.selected_lv]
            # print('mu.shape = ', mu.shape)
            # print('logvar.shape = ', logvar.shape)
            concat = torch.cat((mu, logvar), dim=1)
            # print('concat.shape = ', concat.shape)
        else:
            mu = mu_logvar[:, 0, :]
            logvar = mu_logvar[:, 1, :]
            concat = torch.cat((mu, logvar), dim=1)
        return concat


def final_shaps(shap_values, use_abs=True):
    array_abs = np.abs(shap_values) if use_abs else np.array(shap_values)  # get absolute values
    print(array_abs.shape)
    sum_class = np.nansum(array_abs, axis=-1)  # sum over classes
    print(sum_class.shape)
    mean_samples = np.nanmean(sum_class, axis=1)  # mean over samples
    print(mean_samples.shape)
    mean_seed = np.nanmean(mean_samples, axis=0)  # mean over repetitions

    return mean_seed

def main(args):
    # Importing the data:
    full_data_path = args.data_path
    clinical_data_path = args.clinical_path
    data = pd.read_csv(full_data_path, index_col=0)
    clinical = pd.read_csv(clinical_data_path, index_col=0).squeeze()

    print('data.shape = ', data.shape)
    print('clinical.shape = ', clinical.shape)
    # Make sure data first dimension is the number of samples
    if data.shape[0] != clinical.shape[0]:
        data = data.T
    if data.shape[0] != clinical.shape[0]:
        raise ValueError("Data and clinical data do not have the same number of samples")

    print('clinical.value_counts() = ', clinical.value_counts())
    if args.group_to_analyze[0] != 'all':
        print('args.group_to_analyze = ', args.group_to_analyze)
        clinical = clinical[clinical.isin(args.group_to_analyze)]
        print('after selection,\nclinical.value_counts() = ', clinical.value_counts())


    # Try to extract the samples from the data
    try:
        data = data.loc[clinical.index]
    except KeyError as e:
        print(f"KeyError with indices: {e}")
        try:
            data = data[clinical.index]
        except KeyError as e:
            print(f"KeyError with columns: {e}")
            sys.exit(1)


    model_here = VAE
    model_path = args.model_path

    # Hyperparameters:
    idim, md, feat = get_hyperparams(model_path)

    # One hot encode the clinical data:
    clinical_onehot = pd.get_dummies(clinical)

    # Importing the model:
    # We do not use a seed to take into account stochasticity into SHAP values
    model_vae = model_here(idim, md, feat)  # Initialize the model
    model_vae.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load the state dictionary
    model_vae.eval()  # Set the model to evaluation mode

    # Apply VAE to all data:
    reconstruction_x, _, _, z, scaler_vae = apply_VAE(torch.tensor(data.values).to(torch.float32),
                                                      model_vae,
                                                      y=None)
    df_reconstruction_x = pd.DataFrame(reconstruction_x, index=data.index, columns=data.columns)
    df_z = pd.DataFrame(z, index=data.index)


    # Shap: groups --> latent space

    # SHAP values from the XGBoost model in the latent space

    (bagging_shap_values, metrics, all_params, seeds) = bagging_shap_pipeline(
        n_bags=args.n_shap,
        X_data=data,
        y_data=clinical,
        z_data=df_z,
        classification_type=args.classification_type,
        test_size=args.test_size,
        n_br=args.n_br,
        num_classes=args.num_classes,
        num_threads=args.n_threads,
        n_trials=args.n_trials,
        save_path=args.save_path
    )

    # Mean of shap values
    # 1. SHAP values from the XGBoost model in the latent space
    # Calculate the sum over classes and mean over samples
    sum_mean_shap = final_shaps(bagging_shap_values, use_abs=True)
    print('sum_mean_shap= ', sum_mean_shap)
    # Save sum_mean_shap
    np.save(os.path.join(args.save_path, 'sum_mean_shap.npy'), sum_mean_shap)

    # Shap: latent space --> genes
    # take those variables that are more explainable than average
    selected_lv = np.where(sum_mean_shap > np.mean(sum_mean_shap))[0]
    selected_lv = np.sort(selected_lv)  # this is to avoid problem with negative strides
    print('selected_lv = ', selected_lv)
    ## Create an instance of the new model: encoder + reparametrization trick + selected latent variables
    # model_to_explain = EncoderOnly(model_vae, selected_lv=selected_lv)
    model_to_explain = EncoderWithReparam(model_vae, selected_lv=selected_lv)
    model_to_explain.eval()

    (deep_bagging_shap_values, seeds) = deep_bagging_shap_pipeline(
        n_bags = args.n_shap,
        X_data = data,
        deep_model = model_to_explain,
        scaler = scaler_vae,
        save_path=args.save_path)
    print('deep_bagging_shap_values.shape =', np.array(deep_bagging_shap_values).shape)
    # # 2. SHAP values from the VAE model in the latent space
    sum_deep_shap = final_shaps(deep_bagging_shap_values, use_abs=True)
    # Save sum_deep_shap
    np.save(os.path.join(args.save_path, 'sum_deep_shap.npy'), sum_deep_shap)
    # Select most important genes
    # get top n% of genes with highest shap values
    # Ensure sorted_sum_deep_shap contains indices
    sorted_indices = np.argsort(sum_deep_shap)[::-1]
    top_n = int(0.05* len(sum_deep_shap))
    top_n_indices = sorted_indices[:top_n]
    selected_genes = data.columns[top_n_indices].to_list()
    df_selected_genes = pd.DataFrame(selected_genes, columns=['selected_genes'])
    print('df_selected_genes.shape=',df_selected_genes.shape)
    df_selected_genes.to_csv(os.path.join(args.save_path,'selected_genes.csv'))
    # plot distribution of sum_deep_shap with top n%
    plt.figure(figsize=(8, 6))
    plt.hist(sum_deep_shap, bins=100, alpha=0.75, color='cornflowerblue', edgecolor='black')
    # plot vertical line for the top n% of genes
    plt.axvline(sum_deep_shap[sorted_indices[top_n - 1]], color='black', linestyle='dashed', linewidth=1)
    plt.legend(['Cutoff'], loc='upper right', fontsize=12)
    plt.xlabel('Sum of SHAP values', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.savefig(os.path.join(args.save_path, 'sum_deep_shap_distribution.png'),dpi=600)
    plt.savefig(os.path.join(args.save_path, 'sum_deep_shap_distribution.pdf'))
    plt.savefig(os.path.join(args.save_path, 'sum_deep_shap_distribution.svg'))
    print('Done :D')

if __name__=='__main__':
    import os
    # Change the limit for the number of threads used by NumExpr
    os.environ['NUMEXPR_MAX_THREADS'] = '112'
    today = datetime.datetime.now().strftime("%Y%m%d")
    import argparse
    parser = argparse.ArgumentParser(description='Medulloblastoma classification and SHAP values')
    parser.add_argument('--n_shap',
                        type=int,
                        default=1000,
                        help='Number of times to repeat shap analysis')
    parser.add_argument('--qval',
                        type=float,
                        default=0.95,
                        help='Quantile value for feature selection')
    parser.add_argument('--data_path',
                        type=str,
                        default="data/interim/20240301_Mahalanobis/cavalli.csv",
                        help='Transcriptomics data path')
    parser.add_argument('--clinical_path',
                        type=str,
                        default="data/cavalli_subgroups.csv",
                        help='Clinical data path')
    parser.add_argument('--model_path',
                        type=str,
                        default='models/20240417_cavalli_maha/20240417_VAE_idim12490_md2048_feat16mse_relu.pth',
                        help='Model path')
    parser.add_argument('--classification_type',
                        type=str,
                        default='weighted',
                        help='Classification type: weighted or unbalanced')
    parser.add_argument('--num_classes',
                        type=int,
                        default=4,
                        help='Number of classes')
    parser.add_argument('--test_size',
                        type=float,
                        default=0.2,
                        help='Test size')
    parser.add_argument('--n_br',
                        type=int,
                        default=100,
                        help='Number of boosting rounds for XGBoost')
    parser.add_argument('--n_trials',
                        type=int,
                        default=100,
                        help='Number of optuna trials')
    parser.add_argument('--optimization_optuna',
                        type=bool,
                        default=True, choices=[True, False],
                        help='Optimization with Optuna. True or False')
    parser.add_argument('--tree_method',
                        type=str,
                        default='exact', choices=['exact', 'hist'],
                        help='Tree method for XGBoost')
    parser.add_argument('--n_threads',
                        type=int,
                        default=112,
                        help='Number of threads')
    parser.add_argument('--save_path',
                        type=str,
                        default=f'data/interim/{today}_shap',
                        help='Path to save the results')
    parser.add_argument('--group_to_analyze',
                        type=str_or_list,
                        default='all',
                        help='Group to augment')
    args = parser.parse_args()
    device = 'cpu'

    # Create a directory to save the results
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    # generate report in csv with parser parameters used
    with open(os.path.join(save_path, 'params.csv'), 'w') as f:
        writer = csv.writer(f)
        for key, value in vars(args).items():
            writer.writerow([key, value])
    main(args)