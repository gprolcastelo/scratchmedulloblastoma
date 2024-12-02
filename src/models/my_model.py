import torch
import torch.nn as nn
torch_dtype = torch.float32
torch.set_default_dtype(torch_dtype)
class VAE(nn.Module):
    '''
    Variational Autoencoder (VAE) class.
    This class represents a VAE, which is a type of autoencoder that uses variational inference to train.

    Attributes:
    input_dim (int): The dimension of the input data.
    mid_dim (int): The dimension of the hidden layer.
    features (int): The number of features in the latent space.
    output_layer (nn.Module): The output layer function.

    Methods:
    reparametrize(mu, log_var): Reparameterization trick to sample from the latent space.
    forward(x): Forward pass through the network.
    '''
    def __init__(self, input_dim, mid_dim, features, output_layer=nn.ReLU):
        super().__init__()
        self.input_dim = input_dim
        self.mid_dim = mid_dim
        self.features = features
        self.output_layer = output_layer
        # print('input_dim =\t',input_dim)
        # print('mid_dim =\t',mid_dim)
        # print('features =\t',features)
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.mid_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.mid_dim, out_features=self.features * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.features, out_features=self.mid_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.mid_dim, out_features=self.input_dim),
            # Output activation layer function:
            # activation_layer()
            output_layer()
        )

    def reparametrize(self, mu, log_var):

        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            sample = mu + (eps * std)
        else:
            sample = mu
        return sample

    def forward(self, x):

        mu_logvar = self.encoder(x).view(-1, 2, self.features)
        mu = mu_logvar[:, 0, :]
        log_var = mu_logvar[:, 1, :]

        z = self.reparametrize(mu, log_var)
        reconstruction = self.decoder(z)
        # print('Inside VAE forward')
        # print('reconstruction.shape =\t', reconstruction.shape)
        # print('mu.shape =\t', mu.shape)
        # print('log_var.shape =\t', log_var.shape)
        # print('z.shape =\t', z.shape)
        return reconstruction, mu, log_var, z
# Autoencoder, with the same architecture as the VAE, but without the reparametrization trick
class AE(nn.Module):
    def __init__(self, input_dim, mid_dim, features, output_layer=nn.ReLU):
        super().__init__()
        self.input_dim = input_dim
        self.mid_dim = mid_dim
        self.features = features
        self.output_layer = output_layer

        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.mid_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.mid_dim, out_features=self.features)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.features, out_features=self.mid_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.mid_dim, out_features=self.input_dim),
            # Output activation layer function:
            # activation_layer()
            output_layer()
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, None, None, z

class VAE_plus_bias(nn.Module):
    def __init__(self, my_pretrained_model,input_dim):
        super(VAE_plus_bias,self).__init__()
        self.pretrained = my_pretrained_model
        self.input_dim = input_dim
        # Modify the decoder to include the bias layer
        # 8 extra layers
        # self.final_layer = nn.Sequential(
        #     nn.Linear(in_features=self.input_dim, out_features=self.input_dim//2),
        #     nn.ReLU(),
        #     nn.Linear(in_features=self.input_dim//2, out_features=self.input_dim // 4),
        #     nn.ReLU(),
        #     nn.Linear(in_features=self.input_dim // 4, out_features=self.input_dim // 8),
        #     nn.ReLU(),
        #     nn.Linear(in_features=self.input_dim // 8, out_features=self.input_dim // 16),
        #     nn.ReLU(),
        #     nn.Linear(in_features=self.input_dim // 16, out_features=self.input_dim//8),
        #     nn.ReLU(),
        #     nn.Linear(in_features=self.input_dim//8, out_features=self.input_dim//4),
        #     nn.ReLU(),
        #     nn.Linear(in_features=self.input_dim//4, out_features=self.input_dim//2),
        #     nn.ReLU(),
        #     nn.Linear(in_features=self.input_dim//2, out_features=self.input_dim),
        #     nn.ReLU(),
        # )

        # 4 extra layers
        self.final_layer = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.input_dim//4),
            nn.ReLU(),
            nn.Linear(in_features=self.input_dim//4, out_features=self.input_dim // 16),
            nn.ReLU(),
            nn.Linear(in_features=self.input_dim // 16, out_features=self.input_dim // 4),
            nn.ReLU(),
            nn.Linear(in_features=self.input_dim // 4, out_features=self.input_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        reconstruction, mu, log_var, z = self.pretrained(x)
        reconstruction = self.final_layer(reconstruction)
        return reconstruction, mu, log_var, z


class VAE_3layer(nn.Module):
    def __init__(self, input_dim, features, output_layer=nn.ReLU):
        super().__init__()
        self.input_dim = input_dim
        self.features = features
        self.output_layer = output_layer
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.input_dim//2),
            nn.ReLU(),
            nn.Linear(in_features=self.input_dim//2, out_features=self.input_dim//4),
            nn.ReLU(),
            nn.Linear(in_features=self.input_dim//4, out_features=self.input_dim//8),
            nn.ReLU(),
            nn.Linear(in_features=self.input_dim//8, out_features=self.features * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.features, out_features=self.input_dim//8),
            nn.ReLU(),
            nn.Linear(in_features=self.input_dim//8, out_features=self.input_dim//4),
            nn.ReLU(),
            nn.Linear(in_features=self.input_dim//4, out_features=self.input_dim//2),
            nn.ReLU(),
            nn.Linear(in_features=self.input_dim//2, out_features=self.input_dim),
            # Output activation layer function:
            # activation_layer()
            output_layer()
        )

    def reparametrize(self, mu, log_var):

        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            sample = mu + (eps * std)
        else:
            sample = mu
        return sample

    def forward(self, x):

        mu_logvar = self.encoder(x).view(-1, 2, self.features)
        mu = mu_logvar[:, 0, :]
        log_var = mu_logvar[:, 1, :]

        z = self.reparametrize(mu, log_var)
        reconstruction = self.decoder(z)
        # print('Inside VAE forward')
        # print('reconstruction.shape =\t', reconstruction.shape)
        # print('mu.shape =\t', mu.shape)
        # print('log_var.shape =\t', log_var.shape)
        # print('z.shape =\t', z.shape)
        return reconstruction, mu, log_var, z


class VAE_clinical(VAE):
    '''
    Variational Autoencoder (VAE) class with an additional classifier for clinical data.

    This class extends the VAE class and includes a classifier for a feature that is independent of the numeric input data.
    The classifier is applied to the output data of the VAE.

    Attributes:
    input_dim (int): The dimension of the input data.
    mid_dim (int): The dimension of the hidden layer.
    features (int): The number of features in the latent space.
    num_classes (int): The number of classes for the independent feature.
    output_layer (nn.Module): The output layer function.

    Methods:
    forward(x): Forward pass through the network. Returns the reconstructed output, the mean and log variance of the latent space, the latent space representation, and the class probabilities for the reconstructed output.
    '''

    def __init__(self, input_dim, mid_dim, features, num_classes, output_layer=nn.ReLU):
        '''
        Initialize the VAE_clinical class.

        Args:
        input_dim (int): The dimension of the input data.
        mid_dim (int): The dimension of the hidden layer.
        features (int): The number of features in the latent space.
        num_classes (int): The number of classes for the independent feature.
        output_layer (nn.Module): The output layer function.
        '''
        super().__init__(input_dim, mid_dim, features, output_layer)

        # Classifier
        self.input_dim = input_dim
        self.features = features
        self.num_classes = num_classes
        # print('Inside VAE_clinical: input_dim =',input_dim)
        # print('Inside VAE_clinical: num_classes =',num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=input_dim//4),
            nn.ReLU(),
            nn.Linear(in_features=input_dim//4, out_features=input_dim//4//4),
            nn.ReLU(),
            nn.Linear(in_features=input_dim//4//4, out_features=input_dim//4//4//4),
            nn.ReLU(),
            nn.Linear(in_features=input_dim//4//4//4, out_features=num_classes),
            nn.Softmax(dim=1),
            # nn.ReLU(),
        )
        self.classifier_latent = nn.Sequential(
            nn.Linear(in_features=features, out_features=features // 2),
            nn.ReLU(),
            nn.Linear(in_features=features // 2, out_features=features // 4),
            nn.ReLU(),
            nn.Linear(in_features=features // 4, out_features=num_classes),
            nn.Softmax(dim=1),
            # nn.ReLU(),
        )
    def forward(self, x, clinical_x):
        '''
        Forward pass through the network.

        Args:
        x (Tensor): The input data.

        Returns:
        reconstruction (Tensor): The reconstructed output.
        mu (Tensor): The mean of the latent space.
        log_var (Tensor): The log variance of the latent space.
        z (Tensor): The latent space representation.
        class_probs (Tensor): The class probabilities for the reconstructed output.
        '''
        # print('Inside VAE_clinical forward')
        # print('x =\t',x)
        if clinical_x is None:
            # print('clinical_x is None')
            reconstruction, mu, log_var, z = super().forward(x)
            # Check if outputs are NaN
            assert not torch.isnan(reconstruction).all(), "NaN value found in output"
            return reconstruction, mu, log_var, z
        else:
            reconstruction, mu, log_var, z = super().forward(x)
            class_probs = self.classifier_latent(z)
            # Check if outputs are NaN
            assert not torch.isnan(reconstruction).any(), "NaN value found in output"
            # Replace nan with zeros:
            if torch.isnan(class_probs).any():
                # print("NaN values found in class_probs!")
                class_probs = torch.where(torch.isnan(class_probs), torch.zeros_like(class_probs), class_probs)

            # print('class_probs.shape =\t',class_probs.shape)
            return reconstruction, mu, log_var, z, class_probs

class CVAE(VAE):
    def __init__(self, input_dim, mid_dim, features, num_classes, output_layer=nn.ReLU):
        super().__init__(input_dim, mid_dim, features, output_layer)
        self.num_classes = num_classes

        # Modify the encoder and decoder to accept the condition
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.input_dim + self.num_classes, out_features=self.mid_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.mid_dim, out_features=self.features * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.features + self.num_classes, out_features=self.mid_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.mid_dim, out_features=self.input_dim),
            output_layer()
        )

    def forward(self, x, condition):
        # Concatenate the input data with the condition.
        # This is the encoder input
        x = torch.cat([x, condition], dim=1)
        # Encoder
        mu_logvar = self.encoder(x).view(-1, 2, self.features)
        mu = mu_logvar[:, 0, :]
        log_var = mu_logvar[:, 1, :]
        # Reparametrization trick
        z = self.reparametrize(mu, log_var)
        # Concatenate the latent space with the condition.
        # This is the decoder input
        z = torch.cat([z, condition], dim=1)
        # Decoder
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var, z

class BiasLayer(nn.Module):
    '''
    Bias layer class.
    '''
    def __init__(self, input_dim, requires_grad=True):
        super().__init__()
        # self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=requires_grad)
        # Define additional fully connected layers
        self.fc1 = nn.Linear(input_dim, input_dim*2)
        self.fc2 = nn.Linear(input_dim*2, input_dim*4)
        self.fc3 = nn.Linear(input_dim*4, input_dim*2)
        self.fc4 = nn.Linear(input_dim*2, input_dim)
        self.relu = nn.ReLU()  # Activation function
    def forward(self, input):
        x = self.fc1(input)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.relu(x)
        return x

# class VAE_plus_bias(VAE):
#     def __init__(self, input_dim, mid_dim, features, output_layer=nn.ReLU, requires_grad=True):
#         super().__init__(input_dim, mid_dim, features, output_layer)
#
#         # Modify the decoder to include the bias layer
#         self.decoder = nn.Sequential(
#             nn.Linear(in_features=self.features, out_features=self.mid_dim),
#             nn.ReLU(),
#             nn.Linear(in_features=self.mid_dim, out_features=self.input_dim),
#             output_layer(),
#             BiasLayer(self.input_dim, requires_grad)  # Add the bias layer here
#         )
#
#     def forward(self, x):
#         reconstruction, mu, log_var, z = super().forward(x)
#
#         return reconstruction, mu, log_var, z

class n_VAE(nn.Module):
    """ VAE with an arbitrary number of layers layers_encoder
    """
    def __init__(self, input_dim, features, encoder_dims, decoder_dims):
        super().__init__()
        self.input_dim = input_dim
        self.features = features
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims

        # Create encoder layers
        encoder_layers = []
        # Calculate the dimensions of the intermediate layers
        # Strategy 1: Linearly interpolate between the input and output dimensions for each layer
        # encoder_layer_dims = torch.linspace(input_dim, features * 2, layers_encoder + 1).int().tolist()
        # Strategy 2: Use half the input dimension for the second layer and the output dimension for the remaining layers
        # encoder_layer_dims = [input_dim] + [input_dim // 2] + [features * 2] * (layers_encoder - 1)
        # Strategy 3: use n/2 layers with input_dim//2 and remaining n//2 layers with features*2
        # encoder_layer_dims = [input_dim] + [input_dim//2] * (layers_encoder//2) + [features * 2] * (layers_encoder//2)
        # # Create the encoder layers
        # for i in range(layers_encoder):
        #     encoder_layers.append(nn.Linear(encoder_layer_dims[i], encoder_layer_dims[i + 1]))
        #     if i != layers_encoder - 1:  # Don't apply ReLU after the last layer
        #         encoder_layers.append(nn.ReLU())
        # Strategy 4: Use pre-defined layer dimensions
        # here layers_encoder is a list of dimensions of each layer, not a scalar

        previous_dim = input_dim
        for hidden_dim in encoder_dims:
            encoder_layers.append(nn.Linear(previous_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            previous_dim = hidden_dim
        encoder_layers.append(nn.Linear(previous_dim, features * 2))  # Final layer


        # Create decoder layers
        decoder_layers = []
        # Calculate the dimensions of the intermediate layers for the decoder
        # Strategy 1: Linearly interpolate between the output and input dimensions for each layer
        # decoder_layer_dims = torch.linspace(features, input_dim, layers_encoder + 1).int().tolist()
        # Strategy 2: Use features dimension for n/2 - 1 layers and the input dimension for the last layer
        # decoder_layer_dims = [features] * (layers_encoder - 1) + [input_dim // 2]  + [input_dim]
        # Strategy 3: use n//2 layers with features and remaining n//2 layers with input_dim
        # decoder_layer_dims = [features] * (layers_encoder//2) + [input_dim//2] * (layers_encoder//2) + [input_dim]
        # Create the decoder layers
        # decoder_layers = []
        # for i in range(layers_encoder):
        #     decoder_layers.append(nn.Linear(decoder_layer_dims[i], decoder_layer_dims[i + 1]))
        #     decoder_layers.append(nn.ReLU())

        # Strategy 4: Use pre-defined layer dimensions
        # here layers_encoder is a list of dimensions of each layer, not a scalar
        # decoder_layer_dims = layers_encoder[::-1] # Reverse the list
        previous_dim = features
        for hidden_dim in decoder_dims:
            decoder_layers.append(nn.Linear(previous_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            previous_dim = hidden_dim
        decoder_layers.append(nn.Linear(previous_dim, input_dim))
        decoder_layers.append(nn.ReLU())

        # Define encoder and decoder
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)


    def reparametrize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            sample = mu + (eps * std)
        else:
            sample = mu
        return sample

    def forward(self, x):
        mu_logvar = self.encoder(x).view(-1, 2, self.features)
        mu = mu_logvar[:, 0, :]
        log_var = mu_logvar[:, 1, :]

        z = self.reparametrize(mu, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var, z

