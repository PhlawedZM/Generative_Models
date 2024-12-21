import os.path
import random
import tqdm
import torch
import torchsummary
import torchvision.utils
import PIL.Image as Image
import datasets
import torch.optim as optimizer
import torch.nn as net
import torch.nn.functional as fun
import torchvision.transforms as transforms
import torch.utils.data as data
from tqdm import tqdm
from torchsummary import summary


class Generator(net.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Process Image and Get Detail
        self.encoder = net.Sequential(
            net.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            net.LeakyReLU(negative_slope=0.2),
            net.BatchNorm2d(64),

            net.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            net.LeakyReLU(negative_slope=0.2),
            net.BatchNorm2d(128),

            net.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            net.LeakyReLU(negative_slope=0.2),
            net.BatchNorm2d(256),

            net.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            net.LeakyReLU(negative_slope=0.2),
            net.BatchNorm2d(512),

            net.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            net.LeakyReLU(negative_slope=0.2),
            net.BatchNorm2d(1024),
        )

        # Generate Image from Passed Detail
        self.decoder = net.Sequential(
            net.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            net.LeakyReLU(negative_slope=0.2),
            net.BatchNorm2d(512),

            net.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            net.LeakyReLU(negative_slope=0.2),
            net.BatchNorm2d(256),

            net.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            net.LeakyReLU(negative_slope=0.2),
            net.BatchNorm2d(128),

            net.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            net.LeakyReLU(negative_slope=0.2),
            net.BatchNorm2d(64),

            net.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            net.Sigmoid()  # Use Tahn if you want values -1,1
        )

    def forward(self, x):
        x1 = self.encoder[0:3](x)
        x2 = self.encoder[3:6](x1)
        x3 = self.encoder[6:9](x2)
        x4 = self.encoder[9:12](x3)
        x5 = self.encoder[12:15](x4)

        # Connection to have feature learned at certain res to be transferred
        # Since we are dynamic we will have slightly different tensors hence interpolation
        # Depending on model we might want nearest (sharper images, slightly less accurate especially at lower res)
        # Also important, x+=var does not create a new tensor in memory which hurts performance of the model with skip connections.
        x = self.decoder[0:3](x5)
        x = fun.interpolate(x, size=x4.shape[2:], mode='bilinear', align_corners=False)
        x = x + x4

        x = self.decoder[3:6](x)
        x = fun.interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x = x + x3

        x = self.decoder[6:9](x)
        x = fun.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = x + x2

        x = self.decoder[9:12](x)
        x = fun.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = x + x1

        x = self.decoder[12:15](x)
        return x


class Discriminator(net.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Process Passed Image
        self.encoder = net.Sequential(
            net.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            net.LeakyReLU(negative_slope=0.2),  # Do not use in-place.
            net.BatchNorm2d(64),

            net.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            net.LeakyReLU(negative_slope=0.2),
            net.BatchNorm2d(128),

            net.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            net.LeakyReLU(negative_slope=0.2),
            net.BatchNorm2d(256),

            net.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            net.LeakyReLU(negative_slope=0.2),
            net.BatchNorm2d(512),

            net.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            net.LeakyReLU(negative_slope=0.2),
            net.BatchNorm2d(1024),
        )

        self.adaptive_pool = net.AdaptiveAvgPool2d((1, 1))  # Use this for dynamic tensors (cited 2015)

        self.fc = net.Sequential(
            net.Linear(1024, 1),
            net.Sigmoid()
        )

        self.flatten = net.Flatten(1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def test():
    model = Generator().cuda()
    summary(model, (3, 2000, 2000))


def train():
    generator = Generator().cuda()
    discriminator = Discriminator().cuda()

    if os.path.exists('generator.pth'):
        generator.load_state_dict(torch.load('generator.pth'))

    if os.path.exists('discriminator.pth'):
        discriminator.load_state_dict(torch.load('discriminator.pth'))

    generator.cuda()
    discriminator.cuda()

    # Use this if you want to use values between -1,1
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((250, 250), antialias=True)])

    dataset = datasets.CGANImageFolder(input_folder="train/input", stylized_folder="train/stylized",
                                       transform=transform)
    dataloader = data.DataLoader(dataset=dataset, collate_fn=datasets.padded_collate_fn, batch_size=1, num_workers=14,
                                 shuffle=True, pin_memory=True)


    criterion = net.BCELoss()

    optimizer_g = optimizer.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optimizer.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    epochs = 2000

    for epoch in range(epochs):

        bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=150)
        count = 0
        loss_d = torch.tensor(0).cuda()

        for batch in bar:
            input_images, stylized_images = zip(*batch)
            input = torch.stack(input_images).cuda()
            stylized = torch.stack(stylized_images).cuda()

            real_label = torch.ones(stylized.size(0), 1).cuda() + torch.tensor(random.uniform(-0.2, 0)).cuda()
            fake_label = torch.zeros(input.size(0), 1).cuda() + torch.tensor(random.uniform(0, 0.2)).cuda()

            generated = generator(input)

            if epoch % 2 == 0:
                for _ in range(5):
                    output_stylized = discriminator(stylized)
                    loss_d1 = criterion(output_stylized, real_label)

                    output_generated = discriminator(generated)
                    loss_d2 = criterion(output_generated, fake_label)

                    loss_d = (loss_d1 + loss_d2) / 2

                    optimizer_d.zero_grad()
                    loss_d.backward(retain_graph=True)
                    optimizer_d.step()

            output = discriminator(generated)
            loss_g = criterion(output, real_label)
            if epoch % 2 != 0:
                optimizer_g.zero_grad()
                loss_g.backward()
                optimizer_g.step()

            bar.set_postfix(Generator_Loss=loss_g.item(), Discriminator_Loss=loss_d.item())

            if epoch % 10 == 0:
                torch.save(generator.state_dict(), 'generator.pth')
                torch.save(discriminator.state_dict(), 'discriminator.pth')

                output = generated.detach().cpu()
                image = output.squeeze(0)
                image = image * 255  # images are between 0 and 1, we need them as 0, 255
                image = image.permute(1, 2, 0).numpy()  # Swap RGB, H & W
                image = image.astype('uint8')
                pil = Image.fromarray(image)  # Normalize if needed
                pil.save(f'train/generated/generated_{epoch}_{count}.jpg')
                count += 1


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    torch.autograd.set_detect_anomaly(True)
    train()
    # test()
