import torch
import models.vae_mnist as vae_mnist
from diffclassification.smallnet import LinearNet, Net, split_dataset
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from torchvision import transforms, datasets
from torch import nn, optim



PTH = 'conv_vae_mnist.pth'
DEVICE = 'cuda'
SIZE_PER_CLASS = 2
EPOCHS = 90

activations_encoder = {
    'layer1' : None,
    'layer2' : None,
    'layer3' : None,
    'layer4' : None,
}

activations_decoder = {
    'layer1' : None,
    'layer2' : None,
    'layer3' : None,
    'layer4' : None,
}
# activations_encoder = {}
def get_activation_foo(name, activations):
    def hookFoo(model, input, output):
        # if activations[name] == None:
        activations[name] = output.detach()
        # else:
        #     activations[name] = torch.cat((activations[name], output.detach()), 0)
    return hookFoo

def get_activation_foo_input(name, activations):
    def hookFoo(model, input, output):
        # if activations[name] == None:
        activations[name] = input[0].detach()
        # else:
        #     activations[name] = torch.cat((activations[name], input[0].detach()), 0)
    return hookFoo

def transform_to_features(activations, batch_size):
    with torch.no_grad():
        feats = torch.Tensor([]).to(DEVICE)
        for key in activations.keys():
            val = activations[key]
            cur_feat = val[-batch_size:]
            if len(val.shape) == 4:
                cur_feat = val[-batch_size:].mean(dim=[2,3])
            feats = torch.cat((feats, cur_feat), dim=1)
            activations[key] = None
        torch.cuda.empty_cache()
    return feats

def train_model(train_loader, model, criterion, optimizer, model_vae, activations, epochs=90, loss_list=[], device='cuda'):
    if activations == 'encoder':
      activations = activations_encoder
    elif activations == 'decoder':
      activations = activations_decoder

    model.train()
    loss_list = []
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                model_vae(images)
            # print(activations_decoder['layer1'].shape)
            if activations != 'mix':
              features_enc = transform_to_features(activations, labels.shape[0])
            else:
              features_enc = transform_to_features(activations_encoder, labels.shape[0])
              features_dec = transform_to_features(activations_decoder, labels.shape[0])
              features_enc = torch.cat((features_enc, features_dec), dim=1)


            features_enc = features_enc.to(device)
            outputs = model(features_enc)
            optimizer.zero_grad()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        loss_list.append(running_loss / len(train_loader))
        # print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
    return loss_list


def test_model(model, test_loader, model_vae, activations, device='cuda'):
    if activations == 'encoder':
      activations = activations_encoder
    elif activations == 'decoder':
      activations = activations_decoder

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                model_vae(images)
            if activations != 'mix':
              features_enc = transform_to_features(activations, labels.shape[0])
            else:
              features_enc = transform_to_features(activations_encoder, labels.shape[0])
              features_dec = transform_to_features(activations_decoder, labels.shape[0])
              features_enc = torch.cat((features_enc, features_dec), dim=1)

            features_enc.to(device)

            outputs = model(features_enc)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy}%")
    return accuracy

def main():
    device = DEVICE

    latent_dim = 6

    transform = transforms.ToTensor()

    mnist_data = datasets.MNIST(root='./todata', train=True,
                                download=True, transform = transform)

    train_indices, test_indices = split_dataset(mnist_data, num_train_per_class=SIZE_PER_CLASS, num_test_per_class=SIZE_PER_CLASS)

    train_subset = Subset(mnist_data, train_indices)
    test_subset = Subset(mnist_data, test_indices)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=32)

    conv_vae = vae_mnist.ConvVAE(latent_dim).to(device)
    conv_vae.load_state_dict(torch.load(PTH, map_location=device))

    input_size = 236
    num_classes = 10

    model = LinearNet(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    conv_vae.to(device)

    train_model(train_loader, model, criterion, optimizer, conv_vae, activations='encoder', epochs=90, loss_list=[], device=device);
    test_model(model, test_loader, conv_vae, activations='encoder', device=device)


if __name__ == '__main__':
    main()