import torch
import matplotlib.pyplot as plt

def y_to_onehot(y, batch_size, n):
      
    y = y.view(-1, 1)
    y_onehot = torch.FloatTensor(batch_size, n)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
  
    return y_onehot

def plot(n_images, h, l, num_of_classes):

    fig, axes = plt.subplots(h, l, figsize=(l*3, h*3))

    for i in range(h):
      
        for j in range(l):
        
            y = torch.randint(0, num_of_classes, (1, 1)).to(dtype=torch.long)
            y_hot = y_to_onehot(y, 1, num_of_classes).to(device)
            new_image = model.generate_x(y_hot)
            new_image = new_image.view(28, 28).data
            axes[i][j].imshow(new_image.cpu(), cmap='gray')
            axes[i][j].set_xticks(())
            axes[i][j].set_xticks(())
            axes[i][j].set_title('Generated {}'.format(y.item()))

    plt.show()
    plt.savefig('10_random_di.png')
