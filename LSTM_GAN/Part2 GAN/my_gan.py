import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

from torch.autograd import Variable
import matplotlib.pyplot as plt
import pylab

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Construct generator. You should experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity
        self.fc1 = nn.Linear(args.latent_dim, 128)
        self.lk_relu1 = nn.LeakyReLU(0.2)
        
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.lk_relu2 = nn.LeakyReLU(0.2)
        
        self.fc3 = nn.Linear(256, 512)
        self.bn3 = nn.BatchNorm1d(num_features=512)
        self.lk_relu3 = nn.LeakyReLU(0.2)
        
        self.fc4 = nn.Linear(512, 1024)
        self.bn4 = nn.BatchNorm1d(num_features=1024)
        self.lk_relu4 = nn.LeakyReLU(0.2)
        
        self.fc5 = nn.Linear(1024, 784)
        self.non_linear = nn.Tanh()
        

    def forward(self, z):
        # Generate images from z
        x = self.lk_relu1(self.fc1(z))
        x = self.lk_relu2( self.bn2(self.fc2(x)) )
        x = self.lk_relu3( self.bn3(self.fc3(x)) )
        x = self.lk_relu4( self.bn4(self.fc4(x)) )
        x = self.non_linear(self.fc5(x))
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You should experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity
        self.fc1 = nn.Linear(784, 512)
        self.lk_relu1 = nn.LeakyReLU(0.2)
        
        self.fc2 = nn.Linear(512, 256)
        self.lk_relu2 = nn.LeakyReLU(0.2)
        
        self.fc3 = nn.Linear(256, 1)
        self.non_linear = nn.Sigmoid()
        
        
    def forward(self, img):
        # return discriminator score for img
        x = self.lk_relu1(self.fc1(img))
        x = self.lk_relu2(self.fc2(x))
        x = self.non_linear(self.fc3(x))
        return x


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def reset_grad(optimizer_G, optimizer_D):
    optimizer_G.zero_grad()
    optimizer_D.zero_grad()

def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    
    sample_dir = './images/samples'
    save_dir = './images/save'
    
    # Device setting
    discriminator = discriminator
    generator = generator
    
    # Binary cross entropy loss and optimizer
    criterion = nn.BCELoss()
    
    # Statistics to be saved
    d_losses = np.zeros(args.n_epochs)
    g_losses = np.zeros(args.n_epochs)
    real_scores = np.zeros(args.n_epochs)
    fake_scores = np.zeros(args.n_epochs)
    
    # Create the labels which are later used as input for the BCE loss
    real_labels = torch.ones(args.batch_size, 1)
    real_labels = Variable(real_labels)
    fake_labels = torch.zeros(args.batch_size, 1)
    fake_labels = Variable(fake_labels)
    
    total_step = len(dataloader)
    
    # Generate samples for testing
    num_test_samples = 2
    test_noise = Variable(torch.randn(num_test_samples, args.latent_dim))
    
    
    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Train Discriminator
            if i+1 > 800:
                break
            imgs = imgs.view(args.batch_size, -1)
            imgs = Variable(imgs)
            outputs = discriminator(imgs)
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs
            
            z = torch.randn(args.batch_size, args.latent_dim)
            z = Variable(z)
            fake_images = generator(z)
            outputs = discriminator(fake_images)
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs
            
            # Backprop and optimize
            d_loss = d_loss_real + d_loss_fake
            reset_grad(optimizer_G, optimizer_D)
            d_loss.backward()
            optimizer_D.step()
            # ---------------
            

            # Train Generator
            z = torch.randn(args.batch_size, args.latent_dim)
            z = Variable(z)
            fake_images = generator(z)
            outputs = discriminator(fake_images)
            
            # Backprop and optimize
            g_loss = criterion(outputs, real_labels)
            reset_grad(optimizer_G, optimizer_D)
            g_loss.backward()
            optimizer_G.step()
            # -------------------
            
            # =================================================================== #
            #                          Update Statistics                          #
            # =================================================================== #
            d_loss = d_loss.item()
            g_loss = g_loss.item()

            d_losses[epoch] = d_losses[epoch]*(i/(i+1.)) + d_loss*(1./(i+1.))
            g_losses[epoch] = g_losses[epoch]*(i/(i+1.)) + g_loss*(1./(i+1.))
            real_scores[epoch] = real_scores[epoch]*(i/(i+1.)) + real_score.mean().item()*(1./(i+1.))
            fake_scores[epoch] = fake_scores[epoch]*(i/(i+1.)) + fake_score.mean().item()*(1./(i+1.))

            if (i+1) % 200 == 0:
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                      .format(epoch, args.n_epochs, i+1, total_step, d_loss, g_loss, 
                              real_score.mean().item(), fake_score.mean().item()))
            
            
            # Save Images
            # -----------
            gen_imgs = fake_images
            gen_imgs = gen_imgs.view(gen_imgs.size(0), 1, 28, 28)
            
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                save_image(denorm(gen_imgs[:25].data),
                           'images/{}.png'.format(batches_done),
                           nrow=5, normalize=True)
            if (batches_done+1) % 100 == 0:
               
                test_images = generator(test_noise)
                test_images = test_images.view(test_images.size(0), 1, 28, 28)
                save_image(denorm(test_images.data), 'images/test/{}.png'.format(batches_done))
        
        #######
        # Save real images
        if (epoch+1) == 1:
            imgs = imgs.view(imgs.size(0), 1, 28, 28)
            save_image(denorm(imgs.data), os.path.join(sample_dir, 'real_images.png'))

        # Save sampled images
        fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
        save_image(denorm(fake_images.data), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))

        # Save and plot Statistics
        np.save(os.path.join(save_dir, 'd_losses.npy'), d_losses)
        np.save(os.path.join(save_dir, 'g_losses.npy'), g_losses)
        np.save(os.path.join(save_dir, 'fake_scores.npy'), fake_scores)
        np.save(os.path.join(save_dir, 'real_scores.npy'), real_scores)

        plt.figure()
        pylab.xlim(0, args.n_epochs + 1)
        plt.plot(range(1, args.n_epochs + 1), d_losses, label='d loss')
        plt.plot(range(1, args.n_epochs + 1), g_losses, label='g loss')    
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'loss.pdf'))
        plt.close()

        plt.figure()
        pylab.xlim(0, args.n_epochs + 1)
        pylab.ylim(0, 1)
        plt.plot(range(1, args.n_epochs + 1), fake_scores, label='fake score')
        plt.plot(range(1, args.n_epochs + 1), real_scores, label='real score')    
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'accuracy.pdf'))
        plt.close()
        
        # Save model at checkpoints
        if (epoch+1) % 50 == 0:
            torch.save(generator.state_dict(), os.path.join(save_dir, 'G--{}.ckpt'.format(epoch+1)))
            torch.save(discriminator.state_dict(), os.path.join(save_dir, 'D--{}.ckpt'.format(epoch+1)))
        #######
    
    # Save the model checkpoints 
    torch.save(generator.state_dict(), 'G.ckpt')
    torch.save(discriminator.state_dict(), 'D.ckpt')

def interpolation(generator):
    test_num = 500
    save_interval = 10
    for i in range(test_num):
        z = torch.randn(args.batch_size, args.latent_dim)
        z = Variable(z)
        fake_images = generator(z)
        
        if i % save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                gen_imgs = fake_images
                gen_imgs = gen_imgs.view(gen_imgs.size(0), 1, 28, 28)
                save_image(denorm(gen_imgs[:1].data),
                           'images/interpolation/{}.png'.format(i) ) 
            
            
def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True)

        
    # Initialize models and optimizers
    generator = Generator()
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)
    
    interpolation(generator)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    # torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_known_args()[0]

    main()

