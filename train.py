import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

from PIL import Image
from glob import glob
from time import time
import argparse
import logging
import os
import timm
from tqdm import tqdm
import wandb_utils

def read_row(line_number):
    """
    Get data name and corresponding num_classes.
    """
    with open('utils/name_class.txt', "r") as file:
        for current_line, line in enumerate(file, start=1):
            if current_line == line_number:
                name, classes = line.strip().split('\t')
                return name, int(classes)
    return None, None  


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def create_logger(logging_dir, args):
    """
    Create a logger that writes to a log file and stdout.
    """
    # if dist.get_rank() == 0 or args.single_gpu:  # real logger
    if args.single_gpu:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def main(args):
    """
    Trains a new SiT model.
    """
    name, num_classes = read_row(args.index)

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
 
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    local_batch_size = args.global_batch_size 
    print(f"Starting single GPU training on {device}.")

    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")  # e.g., SiT-XL/2 --> SiT-XL-2 (for naming folders)
    experiment_name = f"{name}-{experiment_index:03d}-{model_string_name}" 
    experiment_dir = f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir, args)
    logger.info(f"Experiment directory created at {experiment_dir}")

    entity = 'liduo1202'
    project = 'Time2Image'
    if args.wandb:
        wandb_utils.initialize(args, entity, experiment_name, project)
 

    # Create model:
    model = timm.create_model(args.model, pretrained=False, num_classes=num_classes)
    logger.info(f"Dataset {name} with {num_classes} classes")


    model = model.to(device)

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")


    # Setup data:
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(), 
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    train_data_dir = os.path.join(args.data_path,name,'train')
    test_data_dir = os.path.join(args.data_path,name,'test')

    trainset = ImageFolder(train_data_dir, transform=transform_train)
    testset = ImageFolder(test_data_dir, transform=transform_test)

    train_sampler = RandomSampler(trainset) 

    test_sampler = SequentialSampler(testset)

    train_loader = DataLoader(
        trainset,
        batch_size=local_batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(
        testset,
        batch_size=local_batch_size,
        sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info(f"Dataset contains {len(trainset):,} images ({train_data_dir})")
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    # opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=0)
    
    total_steps = args.epochs * len(train_loader)

    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
    opt,
    num_warmup_steps=0.1 * total_steps,
    num_training_steps=total_steps
)
    # Prepare models for training:
   

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    criterion = torch.nn.CrossEntropyLoss()

    # Labels to condition the model with (feel free to change):

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if not args.single_gpu:
            train_sampler.set_epoch(epoch)  # Only call set_epoch when using DistributedSampler
        epoch_iterator = tqdm(train_loader)
        model.train() 
        correct = 0
        for i, batch  in enumerate(epoch_iterator):
            epoch_iterator.set_description(f"Epoch {epoch+1} ")
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            scheduler.step()
            opt.zero_grad()

            # Log loss values:
            log_steps += 1
            train_steps += 1
            # if train_steps % args.log_every == 0:
                # Measure training speed:
            pred = logits.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(y.view_as(pred)).sum().item()
            train_accuracy = 100. * correct / len(train_loader.dataset)
            # logger.info(f"Train set: Average loss: {loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} ({train_accuracy:.0f}%)")
            epoch_iterator.set_postfix(acc='%.6f'%train_accuracy, loss='%.3f' %loss)

            if args.wandb:
                wandb_utils.log(
                    { "train loss": loss },
                    step=train_steps
                )
            # Reset monitoring variables:
            running_loss = 0
            log_steps = 0
            start_time = time()

            # Save checkpoint:
            # if train_steps % args.ckpt_every == 0 and train_steps > 0:
            #     checkpoint = {
            #         "model": model.module.state_dict(),
            #     }
            #     checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
            #     torch.save(checkpoint, checkpoint_path)
            #     logger.info(f"Saved checkpoint to {checkpoint_path}")
 

    
        logger.info(f"Starting inference on test dataset for epoch {epoch + 1}...")
        model.eval()  # Set the model to inference mode
        test_loss = 0
        correct = 0
        i = 0
        with torch.no_grad():  # Disable gradient calculation for inference
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                test_loss += criterion(logits, y).item()  # Sum up batch loss
                pred = logits.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
                correct += pred.eq(y.view_as(pred)).sum().item()
                i += 1
        # Reduce test loss and correct count over all processes
        test_loss = torch.tensor(test_loss, device=device) / i
        correct = torch.tensor(correct, device=device)
        test_accuracy = 100. * correct / len(test_loader.dataset)
        logger.info(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_accuracy:.0f}%)")
        if args.wandb:
            wandb_utils.log(
                {"test loss": test_loss, "test accuracy": test_accuracy},
                step=(epoch + 1) * len(train_loader)
            )
        

    model.eval()  
    logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train SiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default='data-img_size224-patch_size16')
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str,  choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14",'ResNet'], default="vit_base_patch16_224")
    parser.add_argument("--image-size", type=int, choices=[224, 384, 512], default=224)
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--single-gpu", default=True, help="Run the training on a single GPU.")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    main(args)