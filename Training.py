import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import random

from torch.optim import Adam
from torch.utils.data import DataLoader
from read_dataset import ChestXrayDataSet
from Triplet_loss import TripletMarginLoss
from PK_sampler import PKSampler
from model import TCDHmodule
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


def train_epoch(model, dataloader, optimizer, criterion1, criterion2, device,alpha):
    # model: TCDHmodel
    # opterizer: Adam
    # criterion1: Reconstruction construction
    # criterion2: Triplet construction
    model.train()
    trainloss = 0.0

    for i, data in enumerate(dataloader):
        samples, targets = data[0].to(device), data[1].to(device)
        decoder, embeddings,_, hash = model(samples)
        loss_R = criterion1(decoder, samples)
        loss_T = criterion2(hash, targets)+criterion2(embeddings, targets)
        loss = alpha * loss_R+loss_T
        # Backward pass
        optimizer.zero_grad()
        loss.backward(torch.ones_like(loss))
        optimizer.step()
        trainloss += loss.item()
        Aloss = 100 * trainloss /len(dataloader.dataset)
    return Aloss


def evaluate(model, dataloader, device):
    model.eval()
    model.to(device)
    full_batch_output = torch.cuda.FloatTensor()
    full_batch_label = torch.cuda.LongTensor()

    with torch.no_grad():
        for i in dataloader:
            inputs, targets = i[0].to(device), i[1].to(device)
            _, _, outputs, _ = model(inputs)
            full_batch_output = torch.cat((full_batch_output, outputs.data), 0)
            full_batch_label = torch.cat((full_batch_label, targets.data), 0)

        test_binary = (torch.sign(full_batch_output)*0.5)+0.5
        test_label = full_batch_label

    test_binary = test_binary.cpu().numpy()
    tst_binary = np.asarray(test_binary, np.int32)
    tst_label = test_label.cpu().numpy()


    query_times = test_binary.shape[0]
    len = tst_binary.shape[0]
    AP = np.zeros(query_times)
    precision_radius = np.zeros(query_times)
    Ns = np.arange(1, len + 1)
    sum_tp = np.zeros(len)
    sum_r = np.zeros(len)

    _dists,_labels =[], []
    for i in range(query_times):
        query_label = tst_label[i]
        query_binary = test_binary[i, :]

        query_result = np.count_nonzero(query_binary != tst_binary, axis=1)  # Hamming distance
        sort_indices = np.argsort(query_result) #sort

        buffer_yes = np.equal(query_label, tst_label[sort_indices]).astype(int) #label equal
        Recall = np.cumsum(buffer_yes)/np.sum(buffer_yes)
        P = np.cumsum(buffer_yes) / Ns
        # P@H≤2
        precision_radius[i] = P[np.where(np.sort(query_result) > 2)[0][0]]
        AP[i] = np.sum(P * buffer_yes) / sum(buffer_yes)
        # Precision and Recall
        sum_tp = sum_tp + np.cumsum(buffer_yes)
        sum_r = sum_r+Recall

    _dists = torch.Tensor(_dists)
    _labels = torch.Tensor(_labels)

    kappas = [1,5,10,20, 40, 60, 80, 100, 120, 140, 160, 180, 200]  # kappas
    print('precision within Hamming radius 2:', np.mean(precision_radius))
    precision_at_k = sum_tp / Ns / query_times
    recall_at_k = sum_r/query_times
    print('recall=', recall_at_k[kappas])
    print('precision at k:', precision_at_k[kappas])
    map = np.mean(AP)
    print('mAP:', map)


def save(model, epoch, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = '_' + args.model
    file_name += '_seed_'+str(args.seed)+'_epoch_'+str(epoch)+'_ckpt.pth'
    save_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), save_path)


def main(args):
    start_time = time.time()
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    p = args.labels_per_batch
    k = args.samples_per_label
    batch_size = p * k

    model = TCDHmodule (num_classes = 1000, hash_bit =48)
    model.to(device)

    criterion2 = TripletMarginLoss(margin=args.margin)
    optimizer2 = Adam(model.parameters(), lr=args.lr)
    criterion1 = nn.MSELoss()

    # targets is a list where the i_th element corresponds to the label of i_th dataset element.
    # This is required for PKSampler to randomly sample from exactly p classes.

    train_dataset = ChestXrayDataSet(data_dir=os.path.join(args.dataset_dir, 'test'),
                                     image_list_file=args.test_image_list)
    targets = train_dataset.labels
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=PKSampler(targets, p, k),
                              num_workers=args.workers)

    test_dataset = ChestXrayDataSet(data_dir=os.path.join(args.dataset_dir, 'test'),
                                    image_list_file=args.test_image_list)

    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size,
                             shuffle=False,
                             num_workers=args.workers)


    for epoch in range(1, args.epochs+1):
        print('Training...')
        train_loss = train_epoch(model, train_loader, optimizer2, criterion1, criterion2, device, alpha=args.alpha)
        print('\n EPOCH {}/{} \t train loss1 {:.3f}'.format(epoch, args.epochs, train_loss))

    print('Evaluating...')
    evaluate(model, test_loader, device)
    print('\n Saving...')
    save(model, epoch, args.save_dir)
    end_time = time.time()
    run_time = end_time - start_time
    print('>>Time: {:.4f}'.format(run_time))


### Parameter
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Embedding Learning')
    parser.add_argument('--model', default='TCDH',
                        help='')
    parser.add_argument('--dataset', default='covid',
                        help='Dataset to use (covid)')
    parser.add_argument('--dataset-dir', default=r'D:\archive',
                        help='Test dataset directory path')
    parser.add_argument('--train-image-list', default='./train_split.txt',
                        help='Train image list')
    parser.add_argument('--test-image-list', default='./test_COVIDx4.txt',
                        help='Test image list')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='Number of data loading workers')
    parser.add_argument('-p', '--labels-per-batch', default=2, type=int,
                        help='Number of unique labels/classes per batch')
    parser.add_argument('-k', '--samples-per-label', default=16, type=int,
                        help='Number of samples per label in a batch')
    parser.add_argument('--eval-batch-size', default=32,type=int)
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='Number of training epochs to run')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Triplet loss margin')
    parser.add_argument('--lr', default=0.0001,  #
                        type=float, help='Learning rate')
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='Construction of reconstruction loss')
    parser.add_argument('--seed', type=int, default=2,
                        help='Random seed to use')
    parser.add_argument('--save-dir', default='./checkpoints',
                        help='Model save directory')
    parser.add_argument('--resume', default='',
                        help='Resume from checkpoint')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)


