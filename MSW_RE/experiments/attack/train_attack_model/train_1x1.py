import os
import argparse
import torch.nn
import numpy as np
from torch import nn
from unet import UNet
from torch.optim import Adam
from dataloader import MSGDS
from prettytable import PrettyTable
from unet import ClassicalDiscriminator
from skimage.util import view_as_windows
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split


def train(args):
    args_dict = vars(args)
    table = PrettyTable(["Argument", "Value"])
    for arg, value in args_dict.items():
        table.add_row([arg, value])
    print(table)

    train_name = os.path.basename(args.dataset_path)
    log_path = os.path.join(args.logs_path, str(args.alpha))
    os.makedirs(log_path, exist_ok=True)
    print(log_path)
    writer = SummaryWriter(log_path)

    # Create models
    generator = UNet(1, 1).to(args.device)
    discriminator = ClassicalDiscriminator().to(args.device)

    # Load pretrained if needed
    if args.continue_train:
        model_path = f"{args.checkpoint_path}/generator_{args.alpha}.pth"
        d_model_path = f"{args.checkpoint_path}/discriminator_{args.alpha}.pth"
        if os.path.exists(model_path):
            generator.load_state_dict(torch.load(model_path))
            print("Generator loaded.")
        if os.path.exists(d_model_path):
            discriminator.load_state_dict(torch.load(d_model_path))
            print("Discriminator loaded.")

    # Datasets
    dataset = MSGDS(args.dataset_path, args.im_size)
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=32)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, drop_last=True, num_workers=32)

    # Optimizers
    optimizer_G = Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.9))
    optimizer_D = Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.9))

    # Loss functions
    if args.loss_mode == "mse":
        criterion_G = nn.MSELoss()
    else:
        criterion_G = nn.BCELoss()
    criterion_D = nn.BCELoss()

    bce_val = nn.BCELoss()

    generator.train()
    discriminator.train()

    global_epoch = 5000
    himming_distance = 1.
    while global_epoch < args.num_epoch:

        for msg_batch, digital_msg in train_loader:
            msg_batch = msg_batch.to(args.device).float()
            digital_msg = digital_msg.to(args.device).float()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            pre_digital_msg = generator(msg_batch).detach()  # stop gradient

            real_pred = discriminator(digital_msg)
            fake_pred = discriminator(pre_digital_msg)

            real_label = torch.ones_like(real_pred)
            fake_label = torch.zeros_like(fake_pred)

            loss_real = criterion_D(real_pred, real_label)
            loss_fake = criterion_D(fake_pred, fake_label)
            d_loss = (loss_real + loss_fake) / 2

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            pre_digital_msg = generator(msg_batch)
            fake_pred = discriminator(pre_digital_msg)

            g_loss_mse = criterion_G(pre_digital_msg, digital_msg)
            g_loss_bce = bce_val(pre_digital_msg, digital_msg)
            g_loss_adv = criterion_D(fake_pred, torch.ones_like(fake_pred))

            # 加权组合 loss
            total_g_loss = args.lambda_bce * g_loss_mse + args.lambda_adv * g_loss_adv

            optimizer_G.zero_grad()
            total_g_loss.backward()
            optimizer_G.step()

            # Logging
            print(
                f"Epoch: {global_epoch} | hamming: {himming_distance} | BCE Loss: {g_loss_bce.item()} | D Loss: {d_loss.item():.4f} | G MSE: {g_loss_mse.item():.4f} | G Adv: {g_loss_adv.item():.4f}")

            logs_save(writer, genuine_msg=msg_batch, digital_msg=digital_msg,
                      pre_msg=torch.round(pre_digital_msg), d_loss=d_loss,
                      g_loss_mse=g_loss_mse, g_loss_adv=g_loss_adv, bce_loss=g_loss_bce,
                      global_step=global_epoch)

        if global_epoch % args.save_epoch == 0 or global_epoch == 1:
            os.makedirs(f"{args.checkpoint_path}", exist_ok=True)
            model_save_path = f"{args.checkpoint_path}/generator_{args.alpha}.pth"
            discriminator_save_path = f"{args.checkpoint_path}/discriminator_{args.alpha}.pth"
            torch.save(generator.state_dict(), model_save_path)
            torch.save(discriminator.state_dict(), discriminator_save_path)

        if (global_epoch + 1) % args.val_epoch == 0:
            generator.eval()
            with torch.no_grad():
                val_batch_msg, val_digital_msg = next(iter(val_loader))
                val_batch_msg = val_batch_msg.to(args.device)
                val_digital_msg = val_digital_msg.to(args.device)
                pre_val_msg = generator(val_batch_msg)
                himming_distance = torch.mean(torch.abs(val_digital_msg - torch.round(pre_val_msg)))
                print(f"Validation loss at epoch {global_epoch}: himming_distance: {himming_distance}")
            generator.train()

        global_epoch += 1


def postProcessingSimbolWise(image, symbol_size=5, thr=0.5):
    """An input image binarization with guaranteed integrity of each symbol

    Args:
        image: code to process
        symbol_size: size of the symbols' blocks in an input image
        thr: binarization threshold

    Returns:
        np.ndarray: binarized code"""

    symbols = view_as_windows(image, window_shape=symbol_size, step=symbol_size)
    symbols = symbols.reshape(symbols.shape[0], symbols.shape[1], -1)
    symbols = np.mean(symbols, axis=2)
    symbols[symbols < thr] = 0
    symbols[symbols != 0] = 1
    return np.repeat(np.repeat(symbols, symbol_size, axis=0), symbol_size, axis=1)


def logs_save(writer, genuine_msg, digital_msg, pre_msg, gp_loss=0, d_loss=0, g_loss_adv=0, bce_loss=0, g_loss_mse=0,
              global_step=0):
    writer.add_scalar('loss/gp_loss', gp_loss, global_step)
    writer.add_scalar('loss/d_loss', d_loss, global_step)
    writer.add_scalar('loss/bce_loss', bce_loss, global_step)
    writer.add_scalar('loss/g_loss_adv', g_loss_adv, global_step)
    writer.add_scalar('loss/g_loss_bce', g_loss_mse, global_step)
    N = genuine_msg.shape[0]
    index = np.random.randint(0, N)
    if global_step % 20 == 0:
        writer.add_image('encoded/genuine_msg', genuine_msg[index], global_step)
        writer.add_image('encoded/digital_msg', digital_msg[index], global_step)
        writer.add_image('encoded/predict_msg', pre_msg[index], global_step)
        writer.add_image('encoded/residual_msg', torch.abs(pre_msg[index] - digital_msg[index]), global_step)


def train_print(alpha, gpu_id, continue_train):
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str,
                        default=torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument('--dataset_path', type=str,
                        default=f'/data/chenjiale/datasets/cdp/alpha_{alpha}/'.replace('.', '_'))
    parser.add_argument('--alpha', nargs='+', type=float, default=alpha)
    parser.add_argument('--im_size', type=int, default=256)
    parser.add_argument('--loss_mode', type=str, default="mse")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument('--save_epoch', type=int, default=128)
    parser.add_argument('--val_epoch', type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/")
    parser.add_argument('--num_epoch', type=int, default=6000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--logs_path', type=str, default=r"logs")
    parser.add_argument('--continue_train', type=bool, default=continue_train)
    parser.add_argument('--lambda_bce', type=float, default=1.0)
    parser.add_argument('--lambda_adv', type=float, default=0.001)

    args = parser.parse_args()
    train(args)
