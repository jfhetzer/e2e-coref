import time
import torch
import argparse
from pathlib import Path
from pyhocon import ConfigFactory, HOCONConverter
from transformers import AdamW, get_linear_schedule_with_warmup

from model.data import Dataset, DataLoader
from model.model import Model, ModelDiscriminator


class Trainer:

    def __init__(self, conf, gpu, split):
        self.config = ConfigFactory.parse_file('./coref.conf')[conf]
        # configure devices (cpu or up to two sequential gpus)
        use_cuda = gpu and torch.cuda.is_available()
        self.device1 = torch.device('cuda:0' if use_cuda else 'cpu')
        self.device2 = torch.device('cuda:1' if use_cuda else 'cpu')
        self.device2 = self.device2 if split else self.device1
        # load dataset with training data
        self.dataset = Dataset(self.config, training=True)
        self.dataloader = DataLoader(self.dataset, shuffle=True)
        self.config['train_data_file'] = self.config['adv_data_file']
        self.adv_dataset = Dataset(self.config, training=True)
        self.adv_dataloader = DataLoader(self.adv_dataset, shuffle=True)

    def train(self, name, amp=False, checkpointing=False):
        # Print infos to console
        print(f'### Start Training ###')
        print(f'running on: {self.device1} {self.device2}')
        print(f'running for: {self.config["epochs"]} epochs')
        print(f'number of batches: {len(self.dataloader)}')
        print(f'saving ckpts to: {name}\n')

        # print full config
        print(HOCONConverter.convert(self.config, 'hocon'))

        # initialize model and move to gpu if available
        model = Model(self.config, self.device1, self.device2, checkpointing)
        model.bert_model.to(self.device1)
        model.task_model.to(self.device2)
        model.train()

        # initialize discriminator and move to gpu if available
        discriminator = ModelDiscriminator(self.config, model.bert_model)
        discriminator.to(self.device1)
        discriminator.train()

        # define loss and optimizer
        lr_bert, lr_task = self.config['lr_bert'], self.config['lr_task']
        lr_gen, lr_dis = self.config['lr_gen'], self.config['lr_dis']

        # exclude bias and layer-norm from weight decay
        bert_params_wd = []
        bert_params_no_wd = []
        for name_, param in model.bert_model.named_parameters():
            group = bert_params_no_wd if 'bias' in name_ or 'LayerNorm' in name_ else bert_params_wd
            group.append(param)

        # bert fine-tuning
        train_steps = len(self.dataloader) * self.config["epochs"]
        warmup_steps = int(train_steps * 0.1)
        optimizer_bert = AdamW([{'params': bert_params_wd, 'weight_decay': 0.01},
                                {'params': bert_params_no_wd, 'weight_decay': 0.0}], lr=lr_bert, correct_bias=False)
        scheduler_bert = get_linear_schedule_with_warmup(optimizer_bert, warmup_steps, train_steps)

        # task specific optimizer
        optimizer_task = torch.optim.Adam(model.task_model.parameters(), lr=lr_task)
        scheduler_task = get_linear_schedule_with_warmup(optimizer_task, warmup_steps, train_steps)

        # generator optimizer
        optimizer_gen = AdamW([{'params': bert_params_wd, 'weight_decay': 0.01},
                               {'params': bert_params_no_wd, 'weight_decay': 0.0}], lr=lr_gen, correct_bias=False)
        scheduler_gen = get_linear_schedule_with_warmup(optimizer_gen, warmup_steps, train_steps)

        # discriminator optimizer
        optimizer_dis = torch.optim.Adam(discriminator.discriminator.parameters(), lr=lr_dis)
        scheduler_dis = get_linear_schedule_with_warmup(optimizer_dis, warmup_steps, train_steps)

        # gradient scaler for fp16 training
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

        # create folder if not already existing
        self.path = Path(f'./data/ckpt/{name}')
        self.path.mkdir(exist_ok=True)
        # load latest checkpoint from path
        epoch = self.load_ckpt(model, optimizer_bert, optimizer_task, optimizer_dis, optimizer_gen, scheduler_bert,
                               scheduler_task, scheduler_dis, scheduler_gen, scaler)

        iter_gen_src = iter(self.dataloader)
        iter_dis_src = iter(self.dataloader)
        iter_gen_tgt = iter(self.adv_dataloader)
        iter_dis_tgt = iter(self.adv_dataloader)

        # run indefinitely until keyboard interrupt
        for e in range(epoch, self.config['epochs']):
            init_epoch_time = time.time()
            for i, batch in enumerate(self.dataloader):
                # ### TRAIN TASK ###
                optimizer_bert.zero_grad()
                optimizer_task.zero_grad()

                # forward and backward pass
                for param in model.parameters():
                    param.grad = None
                for param in discriminator.parameters():
                    param.grad = None
                with torch.cuda.amp.autocast(enabled=amp):
                    scores, labels, _, _, _, _ = model(*batch)
                    loss = self.compute_loss(scores, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer_bert)
                scaler.unscale_(optimizer_task)

                # update weights and lr
                torch.nn.utils.clip_grad_norm_(model.bert_model.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(model.task_model.parameters(), 1)
                scaler.step(optimizer_bert)
                scaler.step(optimizer_task)

                # ### TRAIN DISCRIMINATOR ###
                for param in model.parameters():
                    param.grad = None
                for param in discriminator.parameters():
                    param.grad = None
                for param in discriminator.bert_model.parameters():
                    param.requires_grad = False
                for param in discriminator.discriminator.parameters():
                    param.requires_grad = True
                batch_src, iter_dis_src = Trainer.next(iter_dis_src, self.dataloader)
                batch_tgt, iter_dis_tgt = Trainer.next(iter_dis_tgt, self.dataloader)

                optimizer_dis.zero_grad()
                with torch.cuda.amp.autocast(enabled=amp):
                    p_lang_src = discriminator(batch_src[0])
                    p_lang_tgt = discriminator(batch_tgt[0])
                    loss = self.compute_dis_loss(p_lang_src, p_lang_tgt)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer_dis)

                torch.nn.utils.clip_grad_norm_(discriminator.discriminator.parameters(), 1)
                scaler.step(optimizer_dis)

                # ### TRAIN GENERATOR ###
                for param in model.parameters():
                    param.grad = None
                for param in discriminator.parameters():
                    param.grad = None
                for param in discriminator.bert_model.parameters():
                    param.requires_grad = True
                for param in discriminator.discriminator.parameters():
                    param.requires_grad = False
                batch_src, iter_gen_src = Trainer.next(iter_gen_src, self.dataloader)
                batch_tgt, iter_gen_tgt = Trainer.next(iter_gen_tgt, self.dataloader)

                optimizer_gen.zero_grad()
                with torch.cuda.amp.autocast(enabled=amp):
                    p_lang_src = discriminator(batch_src[0])
                    p_lang_tgt = discriminator(batch_tgt[0])
                    loss = self.compute_gen_loss(p_lang_src, p_lang_tgt)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer_gen)

                torch.nn.utils.clip_grad_norm_(discriminator.bert_model.parameters(), 1)
                scaler.step(optimizer_gen)

                scaler.update()
                scheduler_bert.step()
                scheduler_task.step()
                scheduler_dis.step()
                scheduler_gen.step()

                if (i+1) % 100 == 0:
                    print(f'Batch {i+1:04d} of {len(self.dataloader)}')

            self.save_ckpt(e, model, optimizer_bert, optimizer_task, optimizer_dis, optimizer_gen, scheduler_bert,
                           scheduler_task, scheduler_dis, scheduler_gen, scaler)
            epoch_time = time.time() - init_epoch_time
            epoch_time = time.strftime('%H:%M:%S', time.gmtime(epoch_time))
            print(f'Epoch {e:03d} took: {epoch_time}\n')

    @staticmethod
    def next(iterator, dataloader):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(dataloader)
            return next(iterator), iterator

    def save_ckpt(self, epoch, model, optimizer_bert, optimizer_task, optimizer_dis, optimizer_gen, scheduler_bert,
                  scheduler_task, scheduler_dis, scheduler_gen, scaler):
        path = self.path.joinpath(f'ckpt_epoch-{epoch:03d}.pt.tar')
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer_bert': optimizer_bert.state_dict(),
            'optimizer_task': optimizer_task.state_dict(),
            'optimizer_dis': optimizer_dis.state_dict(),
            'optimizer_gen': optimizer_gen.state_dict(),
            'scheduler_bert': scheduler_bert.state_dict(),
            'scheduler_task': scheduler_task.state_dict(),
            'scheduler_dis': scheduler_dis.state_dict(),
            'scheduler_gen': scheduler_gen.state_dict(),
            'scaler': scaler.state_dict()
        }, path)

    def load_ckpt(self, model, optimizer_bert, optimizer_task, optimizer_dis, optimizer_gen, scheduler_bert,
                  scheduler_task, scheduler_dis, scheduler_gen, scaler):
        # check if any checkpoint accessible
        ckpts = list(self.path.glob('*'))
        if not ckpts:
            return 0

        # load latest checkpoint and return next epoch
        latest_ckpt = max(ckpts, key=lambda p: p.stat().st_ctime)
        latest_ckpt = torch.load(latest_ckpt, map_location=self.device1)
        model.load_state_dict(latest_ckpt['model'])
        optimizer_bert.load_state_dict(latest_ckpt['optimizer_bert'])
        optimizer_task.load_state_dict(latest_ckpt['optimizer_task'])
        optimizer_dis.load_state_dict(latest_ckpt['optimizer_dis'])
        optimizer_gen.load_state_dict(latest_ckpt['optimizer_gen'])
        scheduler_bert.load_state_dict(latest_ckpt['scheduler_bert'])
        scheduler_task.load_state_dict(latest_ckpt['scheduler_task'])
        scheduler_dis.load_state_dict(latest_ckpt['scheduler_dis'])
        scheduler_gen.load_state_dict(latest_ckpt['scheduler_gen'])
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
    def compute_dis_loss(p_source, p_target):
        return -torch.log(p_source) - torch.log(1 - p_target)

    @staticmethod
    def compute_gen_loss(p_source, p_target):
        return -torch.log(1 - p_source) - torch.log(p_target)

    @staticmethod
    def collate(batch):
        return batch[0]


if __name__ == '__main__':
    # parse command line arguments
    folder = time.strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser(description='Train e2e coreference resolution model with BERT.')
    parser.add_argument('-c', metavar='CONF', default='bert-multilingual-base', help='configuration (see coref.conf)')
    parser.add_argument('-f', metavar='FOLDER', default=folder, help='snapshot folder (data/ckpt/<FOLDER>)')
    parser.add_argument('--cpu', action='store_true', help='train on CPU even when GPU is available')
    parser.add_argument('--amp', action='store_true', help='use amp optimization')
    parser.add_argument('--check', action='store_true', help='use gradient checkpointing')
    parser.add_argument('--split', action='store_true', help='split the model across two GPUs')
    args = parser.parse_args()
    # run training
    Trainer(args.c, not args.cpu, args.split).train(args.f, args.amp, args.check)
