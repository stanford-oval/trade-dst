import torch
import numpy as np
import random
import logging
import shutil
import os
import warnings
import sys
from tqdm import tqdm

from torch.optim import lr_scheduler
from transformers.optimization import AdamW, WarmupLinearSchedule

from models.TRADE import TRADE
from utils.config import args

from tensorboardX import SummaryWriter

'''
python myTrain.py -dec= -bsz= -hdd= -dr= -lr=
'''

warnings.simplefilter("ignore", UserWarning)

def run():

    if os.path.exists(args['log_dir']):
        if args['delete_ok']:
            shutil.rmtree(args['log_dir'])
        else:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args['log_dir']))
    os.makedirs(args['log_dir'], exist_ok=False)

    # create logger
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    logger = logging.getLogger(__name__)
    log_file = os.path.join(args['log_dir'], 'log.txt')
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # init tensorboard writer
    tensorboard_writer = SummaryWriter(os.path.join(args['log_dir'], args['dataset']))

    seed = args['seed']

    if args['encoder'] == 'BERT' and args['bert_model'] is None:
        raise ValueError('bert_model should be specified when using BERT encoder.')

    early_stop = args['earlyStop']

    if args['dataset']=='multiwoz':
        from utils.utils_multiWOZ_DST import prepare_data_seq
        early_stop = None
    else:
        logger.info("You need to provide the --dataset information")
        exit(1)

    # Configure models and load data
    avg_best, cnt, acc = 0.0, 0, 0.0
    train, dev, test, test_special, lang, SLOTS_LIST, gating_dict, domain_dict, max_word = \
        prepare_data_seq(True, args['task'], False, batch_size=int(args['batch']), train_sampler='random', test_sampler='random')

    if args['local_rank'] == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args['local_rank'])
        device = torch.device("cuda", args['local_rank'])
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    logger.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(args['local_rank'] != -1)))

    if args['gradient_accumulation_steps'] < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args['gradient_accumulation_steps']))
    num_train_steps = int(len(train) / args['batch'] / args['gradient_accumulation_steps'] * args['max_epochs'])

    if args['decoder'] == 'TRADE':
        model = TRADE(
        hidden_size=int(args['hidden']),
        lang=lang,
        path=args['path'],
        task=args['task'],
        lr=float(args['learn']),
        dropout=float(args['drop']),
        slots=SLOTS_LIST,
        gating_dict=gating_dict,
        domain_dict=domain_dict,
        t_total=num_train_steps,
        nb_train_vocab=max_word,
        device=device,
        logger=logger
        )
    else:
        raise ValueError("Model {} specified does not exist".format(args['decoder']))

    model.to(device)
    if args['local_rank'] != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    core = model.module if hasattr(model, 'module') else model

    for epoch in range(args['max_epochs']):

        if epoch >= args['epoch_threshold']:
            if args['trim']:
                turns_keep = epoch - args['epoch_threshold'] + 1
            else:
                turns_keep = 100000
            train, dev, test, test_special, lang, SLOTS_LIST, gating_dict, domain_dict, max_word = \
                prepare_data_seq(True, args['task'], sequicity=False, batch_size=int(args['batch']), train_sampler='dialogue', test_sampler='dialogue', turns_keep=turns_keep)

        logger.info("Epoch:{}".format(epoch))
        # Run the train function
        if args['is_kube']:
            pbar = enumerate(train)
        else:
            pbar = tqdm(enumerate(train), total=len(train))
        for i, data in pbar:
            batch = {}

            # wrap all numerical values as tensors for multi-gpu training
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                elif isinstance(v, list):
                    if k in ['ID', 'turn_belief', 'context_plain', 'turn_uttr_plain']:
                        batch[k] = v
                    else:
                        batch[k] = torch.tensor(v).to(device)
                else:
                    # print('v is: {} and this ignoring {}'.format(v, k))
                    pass

            loss = model(batch, int(args['clip']), SLOTS_LIST[1], reset=(i==0), n_gpu=n_gpu, epoch=epoch)

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']

            loss.backward()
            tensorboard_writer.add_scalar('train/batch_loss', loss, i + 1)

            if (i + 1) % args['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(core.parameters(), args['clip'])

                for name, param in core.named_parameters():
                    if param.grad is not None:
                        try:
                            tensorboard_writer.add_histogram('train/gradient_norms_{}'.format(name), param.grad, i + 1)
                        except:
                            logger.error('unexpected value for param {} with grad value of {}'.format(name, param.grad))

                core.optimizer.step()
                if isinstance(core.scheduler, WarmupLinearSchedule):
                    core.scheduler.step()

        logger.info(core.print_loss()) #TODO

        tensorboard_writer.add_scalar('train/loss_avg', core.loss / core.print_every, epoch)
        tensorboard_writer.add_scalar('train/loss_ptr', core.loss_ptr / core.print_every, epoch)
        tensorboard_writer.add_scalar('train/loss_gate', core.loss_gate / core.print_every, epoch)

        if (epoch+1) % int(args['evalp']) == 0:

            acc = core.evaluate(dev, avg_best, SLOTS_LIST[2], device, early_stop)
            if isinstance(core.scheduler, lr_scheduler.ReduceLROnPlateau):
                core.scheduler.step(acc)

            if acc >= avg_best:
                avg_best = acc
                cnt = 0
                best_model = core
            else:
                cnt += 1

            if cnt == args["patience"] or (acc==1.0 and early_stop==None):
                logger.info("Ran out of patient, early stop...")
                break

    tensorboard_writer.close()

if __name__ == '__main__':
    run()