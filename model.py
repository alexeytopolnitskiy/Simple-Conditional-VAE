# Conditional VAE model

class CVAE(nn.Module):
  
  def __init__(self, input_size, hidden_size, latent_size, num_of_classes):
    
    super(CVAE, self).__init__()
    
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.latent_size = latent_size
    self.num_of_classes = num_of_classes    
    
    self.enc_fc = nn.Linear(self.input_size + self.num_of_classes, hidden_size)
    self.enc_mu = nn.Linear(self.hidden_size, self.latent_size)
    self.enc_logvar = nn.Linear(self.hidden_size, self.latent_size)
    
    self.dec_fc = nn.Linear(self.latent_size + self.num_of_classes, self.hidden_size)
    self.dec_mu = nn.Linear(self.hidden_size, self.input_size)
    self.dec_logvar = nn.Linear(self.hidden_size, self.input_size)
    
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()    
    
  def Encoder(self, x):
    
    x = self.relu(self.enc_fc(x))
    enc_mu = self.enc_mu(x)
    enc_logvar = self.enc_logvar(x)
    
    return enc_mu, enc_logvar
  
  def Decoder(self, z):
    
    z = self.relu(self.dec_fc(z))
    dec_mu = self.sigmoid(self.dec_mu(z))
    dec_logvar = self.sigmoid(self.dec_logvar(z))
    
    return dec_mu, dec_logvar
  
  def forward(self, x, y):
    
    x_and_y = torch.cat((x, y), dim=1)
    enc_mu, enc_logvar = self.Encoder(x_and_y)
    z = self.reparameterize(enc_mu, enc_logvar)
    z_and_y = torch.cat((z, y), dim=1)
    dec_mu, dec_logvar = self.Decoder(z_and_y)
    
    return dec_mu, dec_logvar, z, enc_mu, enc_logvar
  
  def reconstruct_x(self, x, y):
    
    new_x, _, _, _, _ = self.forward(x, y)
    
    return new_x
  
  def generate_x(self, y):
    
    z = torch.randn(1, self.latent_size).to(device)
    z_and_y = torch.cat((z, y), dim=1)
    generated_x, _ = self.Decoder(z_and_y)
    
    return generated_x
    
  def kl_calc(self, z_mu, z_logvar):
    
    return -0.5 * torch.sum(1 + z_logvar - z_mu**2 - z_logvar.exp())
  
  def loss_calc(self, x, reconstructed_x, z_mu, z_logvar):
    
    kl = self.kl_calc(z_mu, z_logvar)
    re = F.binary_cross_entropy(reconstructed_x, x, size_average=False)
    
    return re + kl 
  
  @staticmethod
  def reparameterize(mu, logvar):
    
    std = logvar.mul(0.5).exp_()
    eps = torch.FloatTensor(std.size()).normal_().to(mu.device)
    
    return eps.mul(std).add_(mu)   