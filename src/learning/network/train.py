import os
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F


torch.manual_seed(2026)


def val_epoch(model, val_loader, criterion, args):
    epoch_loss = 0
    epoch_tr_loss = 0
    epoch_rot_loss = 0

    with tqdm(val_loader, unit="batch") as tepoch:
        for batch in tepoch:
            x = batch["representation" ] # B, C, T, H, W
            y = batch["target"] #B, T - 1, 6
            tepoch.set_description(f"Validating ")
            if torch.cuda.is_available():
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)

            # predict transformation
            estimated_transf = model(x.float()) # model returns [B, (T-1)*9]
            estimated_transf = estimated_transf.view(x.shape[0], args["clip_len"] - 1, 9) #safe reshaping

            # compute loss
            tr_loss, rot_loss = compute_loss(estimated_transf, y, criterion, args)
            loss = tr_loss + rot_loss

            #log three losses
            epoch_loss += loss.item()
            epoch_tr_loss += tr_loss.item()
            epoch_rot_loss += rot_loss.item()

            tepoch.set_postfix(
                loss=f"{loss.item():.4f}",
                tr=f"{tr_loss.item():.4f}",
                rot=f"{rot_loss.item():.4f}",
            )
            
        avg_total = epoch_loss / len(val_loader)
        avg_tr = epoch_tr_loss / len(val_loader)
        avg_rot = epoch_rot_loss / len(val_loader)

    return avg_total, avg_tr, avg_rot


def train_epoch(model, train_loader, criterion, optimizer, epoch, tensorboard_writer, args):
    epoch_loss = 0
    epoch_tr_loss = 0
    epoch_rot_loss = 0
    iter = (epoch - 1) * len(train_loader) + 1

    with tqdm(train_loader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            
            x = batch["representation"] # B, C, T, H, W
            y = batch["target"] #B, T - 1, 6
            
            if torch.cuda.is_available():
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)

            # predict transformation
            estimated_transf = model(x.float())
            estimated_transf = estimated_transf.view(x.shape[0], args["clip_len"] - 1, 9)

            # compute loss
            tr_loss, rot_loss = compute_loss(estimated_transf, y, criterion, args)
            loss = tr_loss + rot_loss

            # compute gradient and do optimizer step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            #log three losses
            epoch_loss += loss.item()
            epoch_tr_loss += tr_loss.item()
            epoch_rot_loss += rot_loss.item()

            tepoch.set_postfix(
                loss=f"{loss.item():.4f}",
                tr=f"{tr_loss.item():.4f}",
                rot=f"{rot_loss.item():.4f}",
            )
            # log tensorboard
            tensorboard_writer.add_scalar('training_loss', loss.item(), iter)
            iter += 1
            
        
        avg_total = epoch_loss / len(train_loader)
        avg_tr = epoch_tr_loss / len(train_loader)
        avg_rot = epoch_rot_loss / len(train_loader)
   
    return avg_total, avg_tr, avg_rot
  

def train(model, train_loader, val_loader, criterion, optimizer, tensorboard_writer, args, stats):
    checkpoint_path = args["checkpoint_path"]
    epochs = args["epoch"]
    init = args["epoch_init"]
    best_val = args["best_val"]
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(init, epochs +1):
        # training for one epoch
        model.train()
        train_loss, train_tr_loss, train_rot_loss = train_epoch(model, train_loader, criterion, optimizer, epoch, tensorboard_writer, args)

        # validate model
        if val_loader:
            with torch.no_grad():
                model.eval()
                val_loss, val_tr_loss, val_rot_loss = val_epoch(model, val_loader, criterion, args)

            tqdm.write(
                f"Epoch {epoch} | "
                f"train total={train_loss:.4f} tr={train_tr_loss:.4f} rot={train_rot_loss:.4f} | "
                f"val total={val_loss:.4f} tr={val_tr_loss:.4f} rot={val_rot_loss:.4f}"
            )

            # save best mode
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                "best_val": best_val,
                "target_mean": stats["mean"],
                "target_std": stats["std"],
            }
            if val_loss < best_val:
                tqdm.write(
                    f"Saving new best model: val_loss {best_val:.6f} -> {val_loss:.6f}"
                )
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
    y = y.reshape(y.shape[0], args["clip_len"] - 1, 6).float()
    y_hat = y_hat.reshape(y_hat.shape[0], args["clip_len"] - 1, 9)

    gt_transl = y[..., :3]
    gt_rotvec = y[..., 3:]

    estimated_transl = y_hat[..., :3]
    estimated_rotvec = y_hat[..., 3:6]
    estimated_cov = y_hat[..., 6:]

    loss_translation = criterion(estimated_transl, gt_transl)
    loss_rot = criterion(estimated_rotvec, gt_rotvec)

    k = 1.0 if args["weighted_loss"] is None else args["weighted_loss"]
    return loss_translation , k * loss_rot
    
