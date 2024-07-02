import os
import torch
import torch.nn as nn
import torchvision.models as models
from dataset import dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from simclr import SimCLR

def parse_option():
    parser = argparse.ArgumentParser('Vision Models for Classification')
    # Training setting
    parser.add_argument('--batch_size', type=int, default=128, 
                    help='batch_size')
    parser.add_argument('--epoch', type=int, default=200,
                    help='number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                    help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    # dataset & model
    parser.add_argument('--root', type=str, default='./dataset/',
                    help='dataset path')
    parser.add_argument('--seed', type=int, default=850011,
                    help='seed for initializing training')
    parser.add_argument('--save_dir', type=str, default='./save',
                    help='path to save models')
    parser.add_argument('--num', type=int, default=40000,
                    help='the number of training samples')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
    
    args = parser.parse_args()
    return args


def main():

    args = parse_option()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    

    
    os.makedirs(args.save_dir, exist_ok=True)
    save = os.path.join(args.save_dir, f'traindata_{args.num}')
    os.makedirs(save, exist_ok=True)
    
    batch_size = 128
    device = args.device if torch.cuda.is_available() else 'cpu'

    log_dir = './train_logs'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(log_dir ,datetime.now().strftime("%m%d-%H%M")+f'_traindata_{args.num}'))

    train_data = dataset(root=os.path.join(args.root,'train.txt'), num=args.num)
    val_data = dataset(root=os.path.join(args.root,'val.txt'), num=10000)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size//2, shuffle=False, num_workers=16)
    
    model = SimCLR()
    model.to(device)
    epoch = args.epoch
    T = args.temperature
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = CosineAnnealingLR(optimizer, T_max=epoch, eta_min=1e-8)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=1e-8,last_epoch=-1) # update in every step instead of epoch
    criterion = nn.CrossEntropyLoss().to(device)

    
    best_acc = 0

    train_labels = torch.arange(batch_size).to(device)
    for ep in range(1,epoch+1):
        print(f'Training for epoch {ep}')
        # import pdb;pdb.set_trace()
        # writer.add_scalar('lr',scheduler.get_last_lr()[0],ep)
        running_loss = 0.

        for batch in tqdm(train_loader,total=len(train_loader)):
            x1, x2 = batch
            # import pdb;pdb.set_trace()
            x1 = x1.to(device) # (b,c,h,w)
            x2 = x2.to(device)
            
            optimizer.zero_grad()
            
            y1 = model(x1) # (b,512,1,1)
            y2 = model(x2)

            b, f, _, _ = y1.shape

            y1 = y1.view(b,f)
            y2 = y2.view(b,f)

            y1 = torch.nn.functional.normalize(y1,dim=1)
            y2 = torch.nn.functional.normalize(y2,dim=1)

            similarity_matrix = torch.matmul(y1, y2.T)

            info_nce_loss = criterion(similarity_matrix/T, train_labels)

            # import pdb;pdb.set_trace()
            info_nce_loss.backward()
            optimizer.step()
            running_loss += info_nce_loss.item()

            # warmup for the first 10 epochs
            if ep > 10:
                scheduler.step()


        print(f'Train loss: {running_loss/len(train_loader)}')
        # scheduler.step()
        
        # print(f'Validating for epoch {ep}')
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, total=len(val_loader)):
                x1, x2 = batch
                x1 = x1.to(device) # (b,c,h,w)
                x2 = x2.to(device)
        
                optimizer.zero_grad()
                
                y1 = model(x1) # (b,512,1,1)
                y2 = model(x2)

                b = y1.shape[0]
                y1 = y1.view(b,f)
                y2 = y2.view(b,f)

                y1 = torch.nn.functional.normalize(y1,dim=1)
                y2 = torch.nn.functional.normalize(y2,dim=1)

                similarity_matrix = torch.matmul(y1, y2.T)

                labels = torch.arange(len(y1)).to(device)
                info_nce_loss = criterion(similarity_matrix/T, labels).item()
            
                val_loss += info_nce_loss

                predicted = torch.argmax(similarity_matrix, 1)
                # labels = torch.argmax(y, 1)
                total += predicted.size(0)
                # labels = torch.range()
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
        print(f'Val loss: {val_loss}')
        print(f'Val Top-1 Accuracy:{accuracy}')
        
        writer.add_scalar('Loss/train',running_loss/len(train_loader), ep)
        writer.add_scalar('Loss/validation',val_loss, ep)
        writer.add_scalar('Top-1 Accuracy', accuracy, ep)

        if accuracy > best_acc:
            torch.save(model.state_dict(), os.path.join(save,f'ep_{ep}_best_{accuracy}.pth'))
            best_acc = accuracy
    
    torch.save(model.state_dict(), os.path.join(save, 'last.pth'))
    print(f'best acc: {best_acc}')
        
if __name__=="__main__":
    main()    

    