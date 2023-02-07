import torch
import torchvision.datasets as datasets  # Standard datasets
from tqdm import tqdm
from torch import nn, optim
from model import VariationalAutoEncoder
from model import Generator
from model import Discriminator
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 50
BATCH_SIZE = 32
ERR_TERM = 1e-10
LR = 3e-4  # Karpathy constant

# Dataset Loading
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
fixed_noise = torch.randn((BATCH_SIZE, Z_DIM)).to(DEVICE)
#model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
disc = Discriminator(INPUT_DIM).to(DEVICE)
gen = Generator(INPUT_DIM).to(DEVICE)
#optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
opt_disc = optim.Adam(disc.parameters(), lr=LR)
opt_gen = optim.Adam(gen.parameters(), lr=LR)
#loss_fn = nn.BCELoss(reduction="sum")
criterion = nn.GaussianNLLLoss(reduction="sum")
step = 0


# Start Training
for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader))
    for batch_idx, (real, _) in loop:
        real = real.view(-1, 784).to(DEVICE)
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, Z_DIM).to(DEVICE)
        fake = gen(noise)
        disc_real_mu, disc_real_sigma = disc(real)#.view(-1)
        #kl_div_real
        lossD_real = -torch.sum(1 + torch.log(disc_real_sigma.pow(2)+ERR_TERM) - disc_real_mu.pow(2) - disc_real_sigma.pow(2))
        #lossD_real = kl_div_real #criterion(disc_real, torch.ones_like(disc_real))
        disc_fake_mu, disc_fake_sigma = disc(fake)#.view(-1)
        #kl_div_fake
        lossD_fake = -torch.sum(1 + torch.log(disc_fake_sigma.pow(2)+ERR_TERM) - disc_fake_mu.pow(2) - disc_fake_sigma.pow(2))
        #lossD_fake = kl_div_fake #criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real - (lossD_fake))/ 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ### Train Generator:
        output_mu, output_sigma = disc(fake)#.view(-1)
        #kl_div_G = -torch.sum(1 + torch.log(output_sigma.pow(2)) - output_mu.pow(2) - output_sigma.pow(2))#criterion(output_mu, noise, output_sigma)
        #epsilon_out = torch.randn_like(output_sigma)
        #z_out = output_mu + output_sigma * epsilon_out
        lossG = criterion(output_mu, noise, output_sigma)
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        #loop.set_postfix(loss=loss.item())
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(train_loader)} \
                      Loss D: {lossD:.4f}, comprised of real loss: {lossD_real:.4f} and fake loss {lossD_fake:.4f}, and loss G: {lossG:.4f}"#, with kldiv loss {kl_div_G:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                #img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                #img_grid_real = torchvision.utils.make_grid(data, normalize=True)

               # writer_fake.add_image(
               #     "Mnist Fake Images", img_grid_fake, global_step=step
               # )
               # writer_real.add_image(
               #     "Mnist Real Images", img_grid_real, global_step=step
               # )
                step += 1
    #for i, (x, _) in loop:
     #   # Forward pass
     #   x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
     #   x_reconstructed, mu, sigma = model(x)

        # Compute loss
       # reconstruction_loss = loss_fn(x_reconstructed, x)
       # kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

        # Backprop
       # loss = reconstruction_loss + kl_div
       # optimizer.zero_grad()
       # loss.backward()
       # optimizer.step()
       # loop.set_postfix(loss=loss.item())


#model = model.to("cpu")
disc = disc.to("cpu")
gen = gen.to("cpu")
def inference(digit, num_examples=1):
    """
    Generates (num_examples) of a particular digit.
    Specifically we extract an example of each digit,
    then after we have the mu, sigma representation for
    each digit we can sample from that.

    After we sample we can run the decoder part of the VAE
    and generate examples.
    """
    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break

    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = disc.encode(images[d].view(1, 784))
        encodings_digit.append((mu, sigma))

    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = gen.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"generated_{digit}_ex{example}.png")

for idx in range(10):
    inference(idx, num_examples=5)





