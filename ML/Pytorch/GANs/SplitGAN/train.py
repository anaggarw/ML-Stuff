"""
Training for CycleGAN

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-05: Initial coding
* 2022-12-21: Small revision of code, checked that it works with latest PyTorch version
"""

import torch
from dataset import HorseZebraDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Encoder, Decoder


def train_fn(
    disc_H, disc_Z, enc, gen_Z, gen_H, loader, opt_disc, opt_gen, opt_enc, l1, mse, d_scaler, g_scaler, e_scaler
):
    H_reals = 0
    H_fakes = 0
    Z_reals = 0
    Z_fakes = 0
    ERR_TERM = 9.0e-12
    loop = tqdm(loader, leave=True)

    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            mu_z, sigma_z = enc(zebra)
            epsilon_z = torch.randn_like(sigma_z)
            z_z = mu_z + sigma_z * epsilon_z
            mu_h, sigma_h = enc(horse)
            epsilon_h = torch.randn_like(sigma_h)
            z_h = mu_h + sigma_h * epsilon_h


            fake_horse = gen_H(z_z)
            reconstructed_horse = gen_H(z_h)
            fake_zebra = gen_Z(z_h)
            reconstructed_zebra = gen_Z(z_z)



            D_H_real = disc_H(horse)
            D_H_fake = disc_H(fake_horse.detach())
            D_H_fake_recon_H = disc_H(reconstructed_horse.detach())
            D_H_fake_recon_Z = disc_H(reconstructed_zebra.detach())
            D_H_fake_Z1 = disc_H(zebra)
            D_H_fake_Z2 = disc_H(fake_zebra.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = 0.5*mse(D_H_fake_Z1, torch.zeros_like(D_H_fake_Z1)) + 0.5*(0.5*mse(D_H_fake, torch.zeros_like(D_H_fake)) + 0.5*(0.5*mse(D_H_fake_recon_H, torch.zeros_like(D_H_fake_recon_H))+ 0.5*(0.5*mse(D_H_fake_recon_Z, torch.zeros_like(D_H_fake_recon_Z)) + 0.5*mse(D_H_fake_Z2, torch.zeros_like(D_H_fake_Z2)))))
            D_H_loss = D_H_real_loss + D_H_fake_loss

           
            
            D_Z_real = disc_Z(zebra)
            D_Z_fake = disc_Z(fake_zebra.detach())
            D_Z_fake_recon_H = disc_Z(reconstructed_horse.detach())
            D_Z_fake_recon_Z = disc_Z(reconstructed_zebra.detach())
            D_Z_fake_H1 = disc_Z(horse)
            D_Z_fake_H2 = disc_Z(fake_horse.detach())
            Z_reals += D_Z_real.mean().item()
            Z_fakes += D_Z_fake.mean().item()
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = 0.5*mse(D_Z_fake_H1, torch.zeros_like(D_Z_fake_H1)) + 0.5*(0.5*mse(D_Z_fake, torch.zeros_like(D_Z_fake)) + 0.5*(0.5*mse(D_Z_fake_recon_Z, torch.zeros_like(D_Z_fake_recon_Z))+ 0.5*(0.5*mse(D_Z_fake_recon_H, torch.zeros_like(D_Z_fake_recon_H)) + 0.5*mse(D_Z_fake_H2, torch.zeros_like(D_Z_fake_H2)))))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put it togethor
            D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_horse)
            D_H_fake_Z = disc_H(fake_zebra)
            D_H_fake_recon_H = disc_H(reconstructed_horse)
            D_H_fake_recon_Z = disc_H(reconstructed_zebra)
            D_Z_fake = disc_Z(fake_zebra)
            D_Z_fake_H = disc_Z(fake_horse)
            D_Z_fake_recon_H = disc_Z(reconstructed_horse)
            D_Z_fake_recon_Z = disc_Z(reconstructed_zebra)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))+mse(D_H_fake_recon_H, torch.ones_like(D_H_fake_recon_H)) - (mse(D_Z_fake_H, torch.ones_like(D_Z_fake_H))+mse(D_Z_fake_recon_H, torch.ones_like(D_Z_fake_recon_H)))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))+mse(D_Z_fake_recon_Z, torch.ones_like(D_Z_fake_recon_Z)) - (mse(D_H_fake_Z, torch.ones_like(D_H_fake_Z))+mse(D_H_fake_recon_Z, torch.ones_like(D_H_fake_recon_Z)))


            # identity loss (remove these for efficiency if you set lambda_identity=0)
            #identity_zebra = gen_Z(zebra)
            #identity_horse = gen_H(horse)
            identity_zebra_loss = l1(zebra, reconstructed_zebra)#-l1(zebra, fake_horse)
            identity_horse_loss = l1(horse, reconstructed_horse)#-l1(horse, fake_zebra)

            # add all togethor
            G_loss = (
                0
                +loss_G_Z
                + loss_G_H
                #+ cycle_zebra_loss * config.LAMBDA_CYCLE
                #+ cycle_horse_loss * config.LAMBDA_CYCLE
                + identity_horse_loss * config.LAMBDA_IDENTITY
                + identity_zebra_loss * config.LAMBDA_IDENTITY
            )
        kl_div_H_loss = -torch.mean(1 + torch.log(sigma_h.pow(2)+ERR_TERM) - mu_h.pow(2) - sigma_h.pow(2))
        kl_div_Z_loss = -torch.mean(1 + torch.log(sigma_z.pow(2)+ERR_TERM) - mu_z.pow(2) - sigma_z.pow(2))
        kl_div_loss = kl_div_H_loss + kl_div_Z_loss
        kl_div = kl_div_loss.mean().item()
        opt_gen.zero_grad()
        opt_enc.zero_grad()
        g_scaler.scale(G_loss).backward(retain_graph=True)
        e_scaler.scale(kl_div_loss).backward()
        g_scaler.step(opt_gen)
        e_scaler.step(opt_enc)
        g_scaler.update()
        e_scaler.update()

        if idx % 25 == 0:
            save_image(fake_horse * 0.5 + 0.5, f"saved_images/horse_{idx}.png")
            save_image(fake_zebra * 0.5 + 0.5, f"saved_images/zebra_{idx}.png")
            save_image(reconstructed_horse * 0.5 + 0.5, f"saved_images/recon_horse_{idx}.png")
            save_image(reconstructed_zebra * 0.5 + 0.5, f"saved_images/recon_zebra_{idx}.png")
            save_image(horse * 0.5 + 0.5, f"saved_images/real_horse_{idx}.png")
            save_image(zebra * 0.5 + 0.5, f"saved_images/real_zebra_{idx}.png")

        loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1), Z_real=Z_reals / (idx + 1), Z_fake=Z_fakes / (idx + 1), KL_div = kl_div)


def main():
    print(config.DEVICE)
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    #gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    #gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    enc = Encoder(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Decoder(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_Z = Decoder(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_enc = optim.Adam(
        list(enc.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    
    opt_gen = optim.Adam(
        list(gen_H.parameters()) + list(gen_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_ENC,
            enc,
            opt_enc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_H,
            gen_H,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z,
            gen_Z,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H,
            disc_H,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z,
            disc_Z,
            opt_disc,
            config.LEARNING_RATE,
        )

    dataset = HorseZebraDataset(
        root_horse=config.TRAIN_DIR + "/horses",
        root_zebra=config.TRAIN_DIR + "/zebras",
        transform=config.transforms,
    )
    val_dataset = HorseZebraDataset(
        root_horse="cyclegan_test/horse1",
        root_zebra="cyclegan_test/zebra1",
        transform=config.transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    e_scaler = torch.cuda.amp.GradScaler()
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_H,
            disc_Z,
            enc,
            gen_Z,
            gen_H,
            loader,
            opt_disc,
            opt_gen,
            opt_enc,
            L1,
            mse,
            d_scaler,
            g_scaler,
            e_scaler,
        )

        if config.SAVE_MODEL:
            save_checkpoint(enc, opt_enc, filename=config.CHECKPOINT_ENC)
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)


if __name__ == "__main__":
    main()
