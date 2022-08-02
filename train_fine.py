import time
import torch
import argparse
from pathlib import Path
from pyhocon import ConfigFactory, HOCONConverter
from transformers import AdamW, get_linear_schedule_with_warmup

from evaluate import Evaluator
from model.data import Dataset, DataLoader
from model.model import Model


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
        self.evaluator = Evaluator(conf)

    def train(self, name, lr_bert, lr_task, epochs, amp=False, checkpointing=False, dump_path=None):
        # Print infos to console
        print(f'### Start Training ###')
        print(f'running on: {self.device1} {self.device2}')
        print(f'running for: {epochs} epochs')
        print(f'number of batches: {len(self.dataloader)}')
        print(f'saving ckpts to: {name}\n')

        # print full config
        print(HOCONConverter.convert(self.config, 'hocon'))

        # initialize model and move to gpu if available
        model = Model(self.config, self.device1, self.device2, checkpointing)
        model.bert_model.to(self.device1)
        model.task_model.to(self.device2)
        model.train()

        # exclude bias and layer-norm from weight decay
        bert_params_wd = []
        bert_params_no_wd = []
        for name_, param in model.bert_model.named_parameters():
            group = bert_params_no_wd if 'bias' in name_ or 'LayerNorm' in name_ else bert_params_wd
            group.append(param)

        # bert fine-tuning
        train_steps = len(self.dataloader) * epochs
        warmup_steps = int(train_steps * 0.1)
        optimizer_bert = AdamW([{'params': bert_params_wd, 'weight_decay': 0.01},
                                {'params': bert_params_no_wd, 'weight_decay': 0.0}], lr=lr_bert, correct_bias=False)
        scheduler_bert = get_linear_schedule_with_warmup(optimizer_bert, warmup_steps, train_steps)

        # task specific optimizer
        optimizer_task = torch.optim.Adam(model.task_model.parameters(), lr=lr_task)
        scheduler_task = get_linear_schedule_with_warmup(optimizer_task, warmup_steps, train_steps)

        # gradient scaler for fp16 training
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

        # create folder if not already existing
        self.path = Path(f'./data/ckpt/{name}')
        self.path.mkdir(exist_ok=True)
        # load latest checkpoint from path
        self.load_ckpt(model)

        # run indefinitely until keyboard interrupt
        for e in range(epochs):
            init_epoch_time = time.time()
            for i, batch in enumerate(self.dataloader):
                optimizer_bert.zero_grad()
                optimizer_task.zero_grad()

                # forward and backward pass
                with torch.cuda.amp.autocast(enabled=amp):
                    scores, labels, _, _, _, _ = model(*batch)
                    loss = self.compute_loss(scores, labels)
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer_bert)
                scaler.unscale_(optimizer_task)

                # update weights and lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                scaler.step(optimizer_bert)
                scaler.step(optimizer_task)
                scaler.update()
                scheduler_bert.step()
                scheduler_task.step()
                if (i+1) % 100 == 0:
                    print(f'Batch {i+1:04d} of {len(self.dataloader)}')

            if dump_path:
                self.eval(model, name, e, lr_bert, lr_task, dump_path, amp)
            else:
                self.save_ckpt(e, model)
            epoch_time = time.time() - init_epoch_time
            epoch_time = time.strftime('%H:%M:%S', time.gmtime(epoch_time))
            print(f'Epoch {e:03d} took: {epoch_time}\n')

    def eval(self, model, name, epoch, lr_bert, lr_task, dump_path, amp):
        model.eval()
        dump_path = Path(dump_path) / f'{name}_{lr_bert}_{lr_task}_{epoch:02d}.pickle'
        print(f'##### EVAL {name} | {lr_bert} | {lr_task} | {epoch:02d} #####')
        self.evaluator.eval_model(model, dump_path, amp)
        model.train()

    def save_ckpt(self, epoch, model):
        # this snapshot is not ment to resume fine-tuning but only for evaluation
        path = self.path.joinpath(f'ckpt_epoch-{epoch:03d}.pt.tar')
        torch.save({'model': model.state_dict()}, path)

    def load_ckpt(self, model):
        ckpts = list(self.path.glob('ckpt_epoch-*.pt.tar'))
        latest_ckpt = max(ckpts, key=lambda p: p.stat().st_ctime)
        model.load_state_dict(torch.load(latest_ckpt)['model'])

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
    parser.add_argument('-c', metavar='CONF', default='bert-base', help='configuration (see coref.conf)')
    parser.add_argument('-f', metavar='FOLDER', default=folder, help='snapshot folder (data/ckpt/<FOLDER>)')
    parser.add_argument('-d', metavar='DUMP', default=None, help='folder to dump predictions into')
    parser.add_argument('-eps', metavar='EPOCHS', type=int, default=10, help='number of epochs')
    parser.add_argument('-lr_b', metavar='LR_BERT', type=float, default=0.0001, help='number of GPUs to which the model is distributed')
    parser.add_argument('-lr_t', metavar='LR_TASK', type=float, default=0.0001, help='number of GPUs to which the model is distributed')
    parser.add_argument('--cpu', action='store_true', help='train on CPU even when GPU is available')
    parser.add_argument('--amp', action='store_true', help='use amp optimization')
    parser.add_argument('--check', action='store_true', help='use gradient checkpointing')
    parser.add_argument('--split', action='store_true', help='split the model across two GPUs')
    args = parser.parse_args()
    # run training
    Trainer(args.c, not args.cpu, args.split).train(args.f, args.lr_b, args.lr_t, args.eps, args.amp, args.check, args.d)
