import os

import torch
import argparse
import kornia as K
from torch import optim
from prettytable import PrettyTable
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, random_split
from printscan.inn.dataloader import AttackDataset
import torch.nn.functional as F

from printscan.inn.models.encoder_decoder import INL


def train(args):
    """

    :param args:
    :return:
    """
    # 将args对象转换为字典
    args_dict = vars(args)
    table = PrettyTable(["Argument", "Value"])
    for arg, value in args_dict.items():
        table.add_row([arg, value])
    print(table)

    # logs
    log_path = args.logs_path
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)

    os.makedirs(f"{args.checkpoint_path}/", exist_ok=True)
    dataset = AttackDataset(args.dataset_path, args.dpi, args.ppi, args.img_size)
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=True, drop_last=True)

    # load model
    # model = INNChannel().to(args.device)
    model = INL().to(args.device)
    if args.continue_train:
        model_path = f"{args.checkpoint_path}/attack_model.pth"
        try:
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=args.device))
                print("ps_model pretrained model parameters loaded.")
            else:
                print("ps_model pretrained model parameters not found.")
        except FileNotFoundError:
            print("No pretrained model parameters found.")

    # optima
    optimization_model = optim.AdamW(model.parameters(), lr=args.lr)
    # train
    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        for _, (d_t_block_out, scan_block_out) in enumerate(train_loader):
            d_t_block_out = d_t_block_out.to(args.device).float()
            scan_block_out = scan_block_out.to(args.device).float()
            pre_scan_block_out = model(d_t_block_out, reverse=False)
            loss_forward = F.mse_loss(pre_scan_block_out, scan_block_out)

            pre_d_t_block_out = model(scan_block_out, reverse=True)
            # loss_backward = F.mse_loss(pre_d_t_block_out, d_t_block_out)

            # loss = loss_forward + loss_backward
            loss = loss_forward

            optimization_model.zero_grad()
            loss.backward()
            optimization_model.step()

            # print(f"loss: {loss.item()}, forward loss: {loss_forward.item()}, backward loss: {loss_backward.item()}")
            # logs_save(writer, loss=loss, loss_forward=loss_forward, loss_backward=loss_backward, pre_scan=pre_scan_block_out, pre_d_t=pre_d_t_block_out, global_step=global_step)

            print(f"epoch:{epoch + 1}, loss: {loss.item()}, forward loss: {loss_forward.item()}")
            logs_save(writer, loss=loss, loss_forward=loss_forward, pre_scan=pre_scan_block_out, real_d_t=d_t_block_out, real_scan=scan_block_out, pre_d_t=pre_d_t_block_out, global_step=global_step)
            global_step += 1

        if epoch % args.epochs == 0:
            attack_model_path = f"{args.checkpoint_path}/attack_model.pth"
            torch.save(model.state_dict(), attack_model_path)

        if (epoch + 1) % args.val_epochs:
            model.eval()
            with torch.no_grad():
                val_d_t_block_out, val_scan_block_out = next(iter(val_loader))
                val_d_t_block_out = val_d_t_block_out.to(args.device).float()
                val_scan_block_out = val_scan_block_out.to(args.device).float()

                scan_pre = model(val_d_t_block_out, reverse=False)
                pre_scan_psnr = K.metrics.psnr(val_scan_block_out, scan_pre, 1.).mean()
                pre_template_psnr = K.metrics.psnr(val_d_t_block_out, model(val_scan_block_out, reverse=True), 1.).mean()
                print(f"pre_Scan validate PSNR: {pre_scan_psnr.item()}, pre_template validate PSNR: {pre_template_psnr.item()}")
            model.train()


def logs_save(writer, loss=0, loss_forward=0, loss_backward=0, real_d_t=None, real_scan=None, pre_scan=None, pre_d_t=None, global_step=0):
    writer.add_scalar('loss/loss', loss, global_step)
    writer.add_scalar('loss/loss_forward', loss_forward, global_step)
    writer.add_scalar('loss/loss_backward', loss_backward, global_step)
    if (global_step + 1) % 20 == 0:
        writer.add_image('img/real_d_t', real_d_t[0], global_step)
        writer.add_image('img/pre_d_t', torch.clip(pre_d_t[0], 0, 1.), global_step)
        writer.add_image('img/real_scan', real_scan[0], global_step)
        writer.add_image('img/pre_scan', torch.clip(pre_scan[0], 0, 1), global_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), help="is cuda")
    parser.add_argument("--dataset_path", type=str, default="/media/dongli911/Documents/Datasets/MSW/ALL_DATA/")
    parser.add_argument("--dpi", type=int, default=600, help="print resolution")
    parser.add_argument("--ppi", type=int, default=1200, help="scan resolution")
    parser.add_argument("--logs_path", type=str, default="logs")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint_path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12800)
    parser.add_argument("--val_epochs", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--save_epoch", type=int, default=8)
    parser.add_argument("--continue_train", type=bool, default=True)
    args = parser.parse_args()

    train(args)
