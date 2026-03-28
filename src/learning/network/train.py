import os
import torch
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import pickle
import json


torch.manual_seed(2026)


def val_epoch(model, val_loader, criterion, args):
    epoch_loss = 0
    with tqdm(val_loader, unit="batch") as tepoch:
        for batch in tepoch:
            x = batch["representation" ] # B, C, T, H, W
            y = batch["target"] #B, T - 1, 12
            tepoch.set_description(f"Validating ")
            if torch.cuda.is_available():
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)

            # predict transformation
            estimated_transf = model(x.float()) # il modello returna [B, (T-1)*12]
            estimated_transf = estimated_transf.view(x.shape[0], args["clip_len"] - 1, 12) #reshapiamo

            # compute loss
            loss = compute_loss(estimated_transf, y, criterion, args)

            epoch_loss += loss.item()
            tepoch.set_postfix(val_loss=loss.item())

    return epoch_loss / len(val_loader)


def train_epoch(model, train_loader, criterion, optimizer, epoch, tensorboard_writer, args):
    epoch_loss = 0
    iter = (epoch - 1) * len(train_loader) + 1

    with tqdm(train_loader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            
            x = batch["representation"] # B, C, T, H, W
            y = batch["target"] #B, T - 1, 12
            
            if torch.cuda.is_available():
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)

            # predict pose
            estimated_transf = model(x.float())
            estimated_transf = estimated_transf.view(x.shape[0], args["clip_len"] - 1, 12)

            # compute loss
            loss = compute_loss(estimated_transf, y, criterion, args)

            # compute gradient and do optimizer step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())

            # log tensorboard
            tensorboard_writer.add_scalar('training_loss', loss.item(), iter)

            iter += 1
    return epoch_loss / len(train_loader)  
  

def train(model, train_loader, val_loader, criterion, optimizer, tensorboard_writer, args):
    checkpoint_path = args["checkpoint_path"]
    epochs = args["epoch"]
    init = args["epoch_init"]
    best_val = args["best_val"]
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(init, epochs +1):
        # training for one epoch
        model.train()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch, tensorboard_writer, args)

        # validate model
        if val_loader:
            with torch.no_grad():
                model.eval()
                val_loss = val_epoch(model, val_loader, criterion, args)

            print(f"Epoch: {epoch} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} \n")

            # save best mode
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                "best_val": best_val,
            }
            if val_loss < best_val:
                print(f"Saving new best model -- loss decreased from {best_val:.6f} to {val_loss:.6f} \n")
                best_val = val_loss
                state["best_val"] = best_val
                torch.save(state, os.path.join(checkpoint_path, "checkpoint_best.pth"))

            # log validation loss in TensorBoard
            tensorboard_writer.add_scalar("val_loss", val_loss, epoch)

        # save checkpoint every 20 epochs
        if not epoch%20:
            torch.save(state, os.path.join(checkpoint_path, "checkpoint_e{}.pth".format(epoch))) 
        # save last checkpoint
        torch.save(state, os.path.join(checkpoint_path, "checkpoint_last.pth"))  

        # log loss in TensorBoard
        tensorboard_writer.add_scalar("train_loss", train_loss, epoch)
    return


def get_optimizer(params, args):
    method = args["optimizer"]

    # initialize the optimizer
    if method == "Adam":
        optimizer = optim.Adam(params, lr=args["lr"])
    elif method == "SGD":
        optimizer = optim.SGD(params, lr=args["lr"],
                              momentum=args["momentum"],
                              weight_decay=args["weight_decay"])
    elif method == "RAdam":
        optimizer = optim.RAdam(params, lr=args["lr"])
    elif method == "Adagrad":
        optimizer = optim.Adagrad(params, lr=args["lr"],
                                  weight_decay=args["weight_decay"])

    # load checkpoint
    if args["checkpoint"] is not None:
        checkpoint = torch.load(os.path.join(args["checkpoint_path"], args["checkpoint"]))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return optimizer


def compute_loss(y_hat, y, criterion, args):
    if args["weighted_loss"] == None:
        loss = criterion(y_hat, y.float()) #Do the MSE on everything at the same time (flattened)
    else:
        y = torch.reshape(y, (y.shape[0], args["clip_len"]-1, 12))
        rot_idx = torch.tensor([0, 1, 2, 4, 5, 6, 8, 9, 10], device=y.device)
        transl_idx = torch.tensor([3, 7, 11], device=y.device)

        gt_rot = y[..., rot_idx].flatten()   # shape [B, T-1, 9]

        gt_transl = y[..., transl_idx].flatten()

        # predicted transformation
        y_hat = torch.reshape(y_hat, (y_hat.shape[0], args["clip_len"]-1, 12))
        estimated_rot = y_hat[..., rot_idx].flatten()
        estimated_transl = y_hat[..., transl_idx].flatten()

        # compute custom loss
        k = args["weighted_loss"]
        loss_angles = k * criterion(estimated_rot, gt_rot.float())
        loss_translation = criterion(estimated_transl, gt_transl.float())
        loss =  loss_angles + loss_translation   
    return loss



    