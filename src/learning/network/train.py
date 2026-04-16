import os
import torch
import torch.optim as optim
from tqdm import tqdm


torch.manual_seed(2026)


def val_epoch(model, val_loader, criterion, args):
    epoch_loss = 0

    with tqdm(val_loader, unit="batch") as tepoch:
        for batch in tepoch:
            x = batch["representation" ] # B, C, T, H, W
            y = batch["target"] # B, T - 1, 3
            tepoch.set_description(f"Validating ")
            if torch.cuda.is_available():
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)

            # predict transformation
            estimated_transf = model(x.float()) # model returns [B, (T-1)*3]
            estimated_transf = estimated_transf.view(x.shape[0], args["clip_len"] - 1, 3) # safe reshaping

            # compute loss
            loss = compute_loss(estimated_transf, y, criterion, args)

            # log loss
            epoch_loss += loss.item()

            tepoch.set_postfix(
                loss=f"{loss.item():.4f}",
            )
            
        avg_total = epoch_loss / len(val_loader)

    return avg_total


def train_epoch(model, train_loader, criterion, optimizer, epoch, tensorboard_writer, args):
    epoch_loss = 0
    iter = (epoch - 1) * len(train_loader) + 1

    with tqdm(train_loader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            
            x = batch["representation"] # B, C, T, H, W
            y = batch["target"] # B, T - 1, 3
            
            if torch.cuda.is_available():
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)

            # predict transformation
            estimated_transf = model(x.float())
            estimated_transf = estimated_transf.view(x.shape[0], args["clip_len"] - 1, 3)

            # compute loss
            loss = compute_loss(estimated_transf, y, criterion, args)

            # compute gradient and do optimizer step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # log loss
            epoch_loss += loss.item()

            tepoch.set_postfix(
                loss=f"{loss.item():.4f}",
            )
            # log tensorboard
            tensorboard_writer.add_scalar('training_loss', loss.item(), iter)
            iter += 1
            
        
        avg_total = epoch_loss / len(train_loader)
   
    return avg_total
  

def train(model, train_loader, val_loader, criterion, optimizer, tensorboard_writer, args, stats):
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

            tqdm.write(
                f"Epoch {epoch} | "
                f"train loss={train_loss:.4f} | "
                f"val loss={val_loss:.4f}"
            )

            # if parallelized, you need to extract the raw model
            raw_model = model.module if hasattr(model, "module") else model
            
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
        optimizer = optim.Adam(params, lr=args["lr"], weight_decay=args["weight_decay"])
    elif method == "AdamW":
        optimizer = optim.AdamW(params, lr=args["lr"], weight_decay=args["weight_decay"])
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
        checkpoint = torch.load(os.path.join(args["checkpoint_path"], args["checkpoint"]), weights_only=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return optimizer


def compute_loss(y_hat, y, criterion, args):
    y = y.reshape(y.shape[0], args["clip_len"] - 1, 3).float()
    y_hat = y_hat.reshape(y_hat.shape[0], args["clip_len"] - 1, 3)
    return criterion(y_hat, y)
    
