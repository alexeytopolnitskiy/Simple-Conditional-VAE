import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, Dataset
import tqdm
from models import CVAE
import utils

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_epochs = 15
batch_size = 20
input_size = 784  # 28*28
hidden_size = 250
latent_size = 80
num_of_classes = 10
lr = 0.001

train_dataset = datasets.MNIST(
    './data',
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor()]
    )
)

test_dataset = datasets.MNIST(
    './data',
    train=False,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor()]
    )
)


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

model = CVAE(input_size, hidden_size, latent_size, num_of_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in tqdm.tqdm(range(n_epochs)):
  
  model.train()
  train_loss = 0
  
  for x, y in train_dataloader:
    
    x = x.view(-1, input_size).to(device)
    y = utils.y_to_onehot(y, batch_size, num_of_classes).to(device)
    
    optimizer.zero_grad()
    x_mu, x_logvar, z, z_mu, z_logvar = model(x, y)
    loss = model.loss_calc(x, x_mu, z_mu, z_logvar)
    loss.backward()
    train_loss += loss.item()
    optimizer.step()
    
  model.eval()
  test_loss = 0
  
  with torch.no_grad():
    
    for x, y in test_dataloader:
    
      x = x.view(-1, input_size).to(device)
      y = utils.y_to_onehot(y, batch_size, num_of_classes).to(device)

      x_mu, x_logvar, z, z_mu, z_logvar = model(x, y)
      loss = model.loss_calc(x, x_mu, z_mu, z_logvar)
      test_loss += loss.item()
      
  print('Epoch is {}. Train loss = {}. Test loss = {}'.format(epoch, train_loss/len(train_dataset), test_loss/len(test_dataset))) 
