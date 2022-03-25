import os
import torch
import torch.nn as nn # for constructing a neural net class
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from datetime import datetime
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from autoencoder import ResNet_autoencoder as Autoencoder
from autoencoder import Bottleneck, DeconvBottleneck
from utils import *
from dataset import Dataset
import torchvision
import argparse
import torch.distributed as dist


# First we need to define constants:
# constants
NUM_EPOCHS = 1000
LEARNING_RATE = 5e-4
BATCH_SIZE = 1
IMG_SIZE = (512, 512)
gpu = 0

# latent dimensionalities for the experiments
dim_powers = list(range(10, 18))
global latent_dims
latent_dims = [2 ** x for x in dim_powers]

# image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMG_SIZE)
])


def train(model, train_loader, test_loader, 
          num_epochs : int = 5, 
          batch_size : int = 64, 
          learning_rate : float = 5e-4,
          log_dir : str = '/scratch/german/runs/',
          checkpoints_dir :str = '/scratch/german/checkpoints/renal_autoencoder_resnet50_v2',
         gpu : int = 0):
    torch.manual_seed(42)
    criterion = nn.MSELoss()
    # criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5)  # <--

    old_val_loss = np.Inf
    # outputs = []
    log_name = datetime.now().strftime('%y-%h_%d-%H-%M') + "_autoencoder_renal_resnet50"
    tb_writer = SummaryWriter(log_dir=log_dir + log_name)
    model.train()

    for epoch in range(num_epochs):
        comp_loss = 0
        for img, name in train_loader:
            print("TRAIN .. Img shape is ... ", img.shape)
            if gpu >= 0:
                img = img.cuda(gpu)
            _, recon = model(img)
            # print("IMG type: ", type(img))
            # print("RECON type: ", type(recon))
            # recon = recon.type(torch.uint8)
            # img = img.to(torch.int64)
            # recon = recon.to(torch.int64)
            # img = torch.tensor(img, dtype=torch.long)
            loss = criterion(recon, img)
            comp_loss += loss.item() * len(img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        comp_loss /= len(train_loader.dataset)

        if epoch % 5 == 0:
            cross_corr, sum_square_dif, val_loss = validate(model, epoch, tb_writer,
                                                            test_loader, batch_size=batch_size)
            print("Validation cross correlation: {:.4f}".format(float(cross_corr)))
            print('Epoch:{}, Loss:{:.4f}'.format(epoch + 1, float(loss)))

            if val_loss < old_val_loss:
                print("Lowered loss {0} < {1}, saving model weights ...".format(float(val_loss), float(old_val_loss)))
                torch.save(model.state_dict(), os.path.join(checkpoints_dir,
                                                            "cp_%i.pth" % epoch))
                old_val_loss = val_loss

            tb_writer.add_scalar('Loss/train', comp_loss, epoch)
            tb_writer.add_scalar('Loss/validation', val_loss, epoch)
            tb_writer.add_scalar('Cross-correlation/validation', cross_corr, epoch)
            tb_writer.add_scalar('Sum square difference/validation', sum_square_dif, epoch)


def validate(model, epoch,
             tb_writer, test_loader, batch_size=4):
    model.eval()
    torch.manual_seed(42)

    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    # criterion_a = nn.MSELoss()
    # criterion_b = nn.L1Loss()
    loss = 0
    # if gpu >= 0:
    #    data, target = data.cuda(gpu), target.cuda(gpu)

    for img, name in test_loader:
        # img, _ = data
        print("VALIDATION .. Img shape is ... ", img.shape)
        if gpu >= 0:
            img = img.cuda(gpu)
            
        _, recon = model(img)
        recon_uint = (recon * 255).type(torch.uint8)
        print("Batch validation size {0}".format(batch_size))
        print("Original image shape {0}".format(img.shape))
        print("Reconstructed image shape {0}".format(recon_uint.shape))
        save_decoded_image(recon_uint, img, epoch, name, batch_size=batch_size)
        cross_corr = cross_correlation(recon, img)
        sum_square_dif = sum_square_diff(recon, img)
        loss_ = criterion(recon, img)
        loss += loss_.item() * len(img)

        tb_writer.add_image("image_validation_set", make_grid(img,
                                                              nrow=batch_size), epoch)
        tb_writer.add_image("image_validation_reconstructions", make_grid(recon_uint,
                                                                          nrow=batch_size), epoch)

    loss /= len(test_loader.dataset)

    return cross_corr, sum_square_dif, loss


def train_parallel(gpu, args,
                  img_path_train : str = "/home/gsergei/imgs_linus/train",
                  img_path_test : str = "/home/gsergei/imgs_linus/test"):
    dist.init_process_group(
        backend='nccl',
        world_size=args.gpus,
        rank=gpu
    )

    torch.manual_seed(42)
    torch.cuda.set_device(gpu)
    # model = Autoencoder()
    model = Autoencoder(Bottleneck, DeconvBottleneck, [3, 4, 6, 3], 1).cuda(gpu)
    # model = Autoencoder(BasicBlock, DeconvBasicBlock, [2, 2, 2, 2], 10).cuda(gpu)

    # pretrained_dict = torch.load("resnet50-0676ba61.pth")
    # model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k,
    #                   v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    
    if args.load < 0:
        fl = os.path.join(args.dir_checkpoint, find_last_checkpoint(args.dir_checkpoint))
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
        model.load_state_dict(torch.load(fl, map_location=map_location))
        print('(%i) Model loaded from %s' % (gpu, fl), flush=True)
    if args.load > 0:
        fl = args.dir_checkpoint + 'cp_%i.pth' % args.load
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
        model.load_state_dict(torch.load(fl, map_location=map_location))
        print('(%i) Model loaded from %s' % (gpu, fl), flush=True)

    dataset_train = Dataset(img_path_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, num_replicas=args.gpus, rank=gpu)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, 
                                               shuffle=False, num_workers=0, pin_memory=True, sampler=train_sampler)

    dataset_test = Dataset(img_path_test)
    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, num_replicas=args.gpus, rank=gpu)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, 
                                             shuffle=False, num_workers=0, pin_memory=True,
                                             sampler=test_sampler, drop_last=True)

    try:
        train(model,
              train_loader,
              test_loader,
              gpu=gpu,
              checkpoints_dir=args.checkpoints_dir,
              # epoch1=args.load,
              batch_size=args.batch_size,
              num_epochs=args.epochs,
              learning_rate=args.lr)
              # weights=args.weights)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

# The old paths to original dataset (only 2 genotypes)
# img_path_train = "/home/gsergei/imgs_linus/train"
# img_path_test = "/home/gsergei/imgs_linus/test"

# New path to extended image dataset:
# img_path_train = "/home/gsergei/imgs_linus_ext/train"
# img_path_test = "/home/gsergei/imgs_linus_ext/test"
img_path_train = "/home/gsergei/imgs_linus/train"
img_path_test = "/home/gsergei/imgs_linus/test"

trainset = Dataset(img_path_train)
testset = Dataset(img_path_test)

trainloader = torch.utils.data.DataLoader(trainset)
testloader = torch.utils.data.DataLoader(testset)


# model = Autoencoder()
# model = model.cuda(gpu)
# make_dir()
# outputs = train(model, trainset, testset,
#                num_epochs=NUM_EPOCHS,
#                batch_size=BATCH_SIZE,
#               learning_rate=LEARNING_RATE)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-d', '--dims', metavar='D', type=int, default=32,
    #                    help='Dimensionality of representations', dest='n_dims')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batch_size')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=int, default=0,
                        help='Load model from a checkpoint')
    parser.add_argument('-g', '--gpus', dest='gpus', type=int, default=4,
                        help='Number of GPUs')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-w', '--withweights', dest='weights', action='store_true',
                        help='Weigh semantic classes')
    parser.add_argument('-L', '--log', dest='log', type=str, default=None, help='Name of a log file')
    parser.add_argument('-D', '--dir-checkpoint', dest='checkpoints_dir', type=str, default='../../checkpoints', help='Directory with checkpoints')
    # parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
    #                    help='Percent of the data that is used as validation (0-100)')
    return parser.parse_args()


if __name__ == '__main__':
    #logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    weights = None
    dir_checkpoint = args.checkpoints_dir
    # ("/inst_%i%s/" % (args.n_dims, ["","_w"][args.weights]))
    if args.weights:
        args.weights = [0.5, 1.0, 0.75]
    else:
        args.weights = None

    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)
        print('Created checkpoint directory %s' % dir_checkpoint, flush=True)
    args.dir_checkpoint = dir_checkpoint

    os.environ['MASTER_ADDR'] = 'localhost' #'127.0.0.1'  #
    os.environ['MASTER_PORT'] = '8890'  #
    mp.spawn(train_parallel, nprocs=args.gpus, args=(args,))
    
    # plot_tpr_ppv(Log.last_tsv(start_name='training_'))
    # plot_mult_tpr_ppv(Log.last_tsv(start_name='testing_'))
