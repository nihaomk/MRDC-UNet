import os
import re
import time
import datetime
from glob import glob
import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from net import MRDCUNet,unet
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
import transforms as T
from sklearn.model_selection import train_test_split
torch.autograd.set_detect_anomaly(True)

class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)
        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 565
    crop_size = 480

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(base_size, mean=mean, std=std)


def create_model(num_classes):
    model = MRDCUNet(in_channels=3, num_classes=num_classes, base_c=32)
    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_classes = args.num_classes + 1

    mean = (0.691, 0.447, 0.305)
    std = (0.098, 0.110,  0.096)

    results_file = "result/results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#自定义添加
    img_ids = glob(os.path.join(args.data_path, 'images', '*' + args.img_extension))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.15, random_state=41)
    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(args.data_path, 'images'),
        mask_dir=os.path.join(args.data_path, 'masks'),
        transforms=get_transform(train=True, mean=mean, std=std))
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(args.data_path, 'images'),
        mask_dir=os.path.join(args.data_path, 'masks'),
        transforms=get_transform(train=False, mean=mean, std=std))
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True)

    model = create_model(num_classes=num_classes)

    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.ASGD(params_to_optimize, lr=0.2, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)


    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    best_dice = 0.
    start_time = time.time()

    writer = SummaryWriter('MRDC_LOG', flush_secs=1)
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")
        # write into txt
        with open(results_file, "a") as f:
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice:.3f}\n"
            f.write(train_info + val_info + "\n\n")
        numbers = [float(num) for num in re.findall(r'\d+\.\d+|\d+', val_info)]
        writer.add_scalar('correct/global',numbers[0],epoch)
        writer.add_scalar('correct/signal',numbers[1],epoch)
        writer.add_scalar('Iou/iot',numbers[3],epoch)
        writer.add_scalar('Iou/mean iou',numbers[5],epoch)
        writer.add_scalar('train/loss',mean_loss,epoch)
        writer.add_scalar('other/lr',lr,epoch)

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}

        if epoch % 100 == 0:
            torch.save(save_file, "save_weights/model_{}.pth".format(epoch))

        if best_dice < dice:
            best_dice = dice
            torch.save(save_file, "save_weights/model_best.pth")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch MRDC-UNet training")

    #data
    parser.add_argument("--data-path", default="./data/", help="Dataset root")
    parser.add_argument('--img_extension',default='.png',help='image name extension')
    parser.add_argument('--mask_extension', default='.png', help='mask name extension')
    parser.add_argument('--num_classes', default=1, type=int,help='number of classes')

    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=600, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.02, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
