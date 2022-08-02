import time
import torch
import argparse
from pathlib import Path
from pyhocon import ConfigFactory

from model.data import Dataset, DataLoader
from model.model import Model


class Trainer:

    def __init__(self, conf, gpu):
        # load configuration
        self.config = ConfigFactory.parse_file('./coref.conf')[conf]
        use_cuda = gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        # load dataset with training data
        path = self.config['train_data_path']
        self.dataset = Dataset(self.config, path, training=True)
        self.dataloader = DataLoader(self.dataset, shuffle=True)

    def train(self, name, amp=False):
        # Print infos to console
        print(f'### Start Training ###')
        print(f'running on: {self.device}')
        print(f'running for: {self.config["epochs"]} epochs')
        print(f'number of batches: {len(self.dataloader)}')
        print(f'saving ckpts to: {name}\n')

        # initialize model and move to gpu if available
        model = Model(self.config, self.device)
        model.to(self.device)
        model.train()

        # define loss and optimizer
        lr, step_size, gamma = self.config['lr'], self.config['decay_step'], self.config['decay_rate']
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        # gradient scaler for fp16 training
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

        # create folder if not already existing
        self.path = Path(f'./data/ckpt/{name}')
        self.path.mkdir(exist_ok=True)
        # load latest checkpoint from path
        epoch = self.load_ckpt(model, optimizer, scheduler, scaler)

        # run indefinitely until keyboard interrupt
        for e in range(epoch, self.config['epochs']):
            init_epoch_time = time.time()
            for i, batch in enumerate(self.dataloader):
                optimizer.zero_grad()

                # forward and backward pass
                with torch.cuda.amp.autocast(enabled=amp):
                    scores, labels, _, _, _, _ = model(*batch)
                    loss = self.compute_loss(scores, labels)
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                if (i+1) % 100 == 0:
                    print(f'Batch {i+1:04d} of {len(self.dataloader)}')

            self.save_ckpt(e, model, optimizer, scheduler, scaler)
            epoch_time = time.time() - init_epoch_time
            print(f'Epoch {e:03d} took: {epoch_time}s\n')

    def save_ckpt(self, epoch, model, optimizer, scheduler, scaler):
        path = self.path.joinpath(f'ckpt_epoch-{epoch:03d}.pt.tar')
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict()
        }, path)

    def load_ckpt(self, model, optimizer, scheduler, scaler):
        # check if any checkpoint accessible
        ckpts = list(self.path.glob('ckpt_epoch-*.pt.tar'))
        if not ckpts:
            print(f'\nNo checkpoint found: Start training from scratch\n')
            return 0

        # get latest checkpoint
        latest_ckpt = max(ckpts, key=lambda p: p.stat().st_ctime)
        print(f'\nCheckpoint found: Load {latest_ckpt}\n')
        # load latest checkpoint and return next epoch
        latest_ckpt = torch.load(latest_ckpt)
        model.load_state_dict(latest_ckpt['model'])
        optimizer.load_state_dict(latest_ckpt['optimizer'])
        scheduler.load_state_dict(latest_ckpt['scheduler'])
        scaler.load_state_dict(latest_ckpt['scaler'])
        return latest_ckpt['epoch'] + 1

    @staticmethod
    def compute_loss(scores, labels):
        # apply mask to get only scores of gold antecedents
        gold_scores = scores + torch.log(labels.float())
        # marginalize gold scores
        gold_scores = torch.logsumexp(gold_scores, [1])
        scores = torch.logsumexp(scores, [1])
        return torch.sum(scores - gold_scores)

    @staticmethod
    def collate(batch):
        return batch[0]


if __name__ == '__main__':
    # parse command line arguments
    folder = time.strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser(description='Train e2e coreference resolution model with BERT.')
    parser.add_argument('-c', metavar='CONF', default='base', help='configuration (see coref.conf)')
    parser.add_argument('-f', metavar='FOLDER', default=folder, help='snapshot folder (data/ckpt/<FOLDER>)')
    parser.add_argument('--cpu', action='store_true', help='train on CPU even when GPU is available')
    parser.add_argument('--amp', action='store_true', help='use amp optimization')
    args = parser.parse_args()
    # run training
    Trainer(args.c, not args.cpu).train(args.f, args.amp)
