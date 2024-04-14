import torch

from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import datasets, transforms


class Discriminator(nn.Module):
    def __init__(self, image_len: int):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(image_len, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)


class Generator(nn.Module):
    """
    Generator class. Accepts a tensor of size 100 as input and return a tensor of 724.
    """
    def __init__(self, z_dim, image_dim):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(z_dim, 256),          # z_dim -> 256
            nn.LeakyReLU(),
            nn.Linear(256, 256),            # 256 -> 256
            nn.LeakyReLU(),
            nn.Linear(256, image_dim),      # 256 -> 728
            nn.Tanh()
        )

    def forward(self, x):
        return self.generator(x)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    learning_rate = 3e-4
    z_dim = 64
    image_dim = 28 * 28 * 1
    batch_size = 32
    num_epoch = 500

    discriminator = Discriminator(image_dim).to(device)
    generator = Generator(z_dim, image_dim).to(device)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    fixed_noise = torch.randn((batch_size, z_dim)).to(device)

    dataset = datasets.MNIST(root='/dataset', transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optim_disc = optim.Adam(discriminator.parameters(), lr=learning_rate)
    optim_gen = optim.Adam(generator.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    fake_writer = SummaryWriter(f'logs/fake')
    real_writer = SummaryWriter(f'logs/real')
    step = 0

    for epoch in range(num_epoch):
        for batch_id, (real, _) in enumerate(dataloader):
            real = real.view(-1, 784).to(device)
            batch_size = real.shape[0]

            # Train discriminator: max log(D(real)) + log(1 - D(G(z))
            noise = torch.randn(batch_size, z_dim).to(device)
            fake = generator(noise)

            disc_real = discriminator(real).view(-1)
            disc_real_loss = criterion(disc_real, torch.ones_like(disc_real))

            disc_fake = discriminator(fake).view(-1)
            disc_fake_loss = criterion(disc_fake, torch.zeros_like(disc_fake))

            disc_loss = (disc_real_loss + disc_fake_loss) / 2
            discriminator.zero_grad()
            disc_loss.backward(retain_graph=True)
            optim_disc.step()

            # Train generator: min log(1 - D(G(z))) <--> max log(D(F(z))
            out = discriminator(fake).view(-1)
            gen_loss = criterion(out, torch.ones_like(out))
            generator.zero_grad()
            gen_loss.backward()
            optim_gen.step()

            if batch_id == 0:
                print(
                    f'Epoch: {epoch}/{num_epoch}\n'
                    f'Generator loss: {gen_loss:.4f} '
                    f'Discriminator loss: {disc_loss:.4f}'
                )

                with torch.no_grad():
                    fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
                    data = real.reshape(-1, 1, 28, 28)
                    image_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                    image_grid_real = torchvision.utils.make_grid(data, normalize=True)
                    fake_writer.add_image('fake', image_grid_fake, global_step=step)
                    real_writer.add_image('real', image_grid_real, global_step=step)
                    step += 1


if __name__ == '__main__':
    main()
