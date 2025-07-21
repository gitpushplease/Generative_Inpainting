from utils.losses import (
    wgan_generator_loss,
    wgan_discriminator_loss,
    compute_gradient_penalty
)
import torch.nn as nn


l1_loss = nn.L1Loss()
alpha = 100  
lambda_gp = 10  
n_critic = 5     

for epoch in range(epochs):
    for i, (input_tensor, real_img) in enumerate(dataloader):
        input_tensor = input_tensor.to(device)
        real_img = real_img.to(device)

       
        for _ in range(n_critic):
            fake_img = G(input_tensor)
            d_real = D(real_img)
            d_fake = D(fake_img.detach())

            loss_d = wgan_discriminator_loss(d_real, d_fake)
            gp = compute_gradient_penalty(D, real_img, fake_img.detach(), device)
            loss_d_total = loss_d + lambda_gp * gp

            optimizer_D.zero_grad()
            loss_d_total.backward()
            optimizer_D.step()

        
        fake_img = G(input_tensor)
        d_fake = D(fake_img)

        loss_g_adv = wgan_generator_loss(d_fake)
        loss_g_l1 = l1_loss(fake_img, real_img)
        loss_g = loss_g_adv + alpha * loss_g_l1

        optimizer_G.zero_grad()
        loss_g.backward()
        optimizer_G.step()

        if i % 100 == 0:
            print(f"Epoch {epoch}, Step {i} | D Loss: {loss_d_total.item():.4f} | G Loss: {loss_g.item():.4f}")
            save_tensor_image(fake_img[:4], f"outputs/epoch{epoch}_step{i}.png")
