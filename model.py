import torch.nn as nn

device = "cuda:0"
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim)
        )

    def forward(self, x):
        latent_z = self.encoder(x)
        rec_feature = self.decoder(latent_z)
        return latent_z, rec_feature


class DigitClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, latent_dim):
        super(DigitClassifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.classifier = nn.Linear(input_dim, num_classes)
        self.autoencoder = AutoEncoder(input_dim, latent_dim)

    def forward(self, x):
        encoded_feature = self.encoder(x)
        logits = self.classifier(encoded_feature)
        latent_z, rec_feature = self.autoencoder(encoded_feature)
        return logits, encoded_feature, rec_feature
    
def compute_loss(logits, labels, encoded_feature, rec_feature):
    classification_loss = nn.CrossEntropyLoss()(logits, labels)
    reconstruction_loss = nn.MSELoss()(encoded_feature, rec_feature)
    return classification_loss, reconstruction_loss