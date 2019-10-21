import torch
import numpy as np
import random
import logging
from tqdm import tqdm
import shutil
import os
import warnings

from torch.optim import lr_scheduler
from transformers.optimization import AdamW, WarmupLinearSchedule

from models.TRADE import TRADE
from utils.config import args

'''
python myTrain.py -dec= -bsz= -hdd= -dr= -lr=
'''

warnings.simplefilter("ignore", UserWarning)

def run():

    seed = args['seed']

    if args['encoder'] == 'BERT' and args['bert_model'] is None:
        raise ValueError('bert_model should be specified when using BERT encoder.')

    early_stop = args['earlyStop']

    if args['dataset']=='multiwoz':
        from utils.utils_multiWOZ_DST import prepare_data_seq
        early_stop = None
    else:
        print("You need to provide the --dataset information")
        exit(1)

    # Configure models and load data
    avg_best, cnt, acc = 0.0, 0, 0.0
    train, dev, test, test_special, lang, SLOTS_LIST, gating_dict, max_word = prepare_data_seq(True, args['task'], False, batch_size=int(args['batch']))

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
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

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
        t_total=num_train_steps,
        nb_train_vocab=max_word,
        device=device
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
        print("Epoch:{}".format(epoch))
        # Run the train function
        pbar = tqdm(enumerate(train), total=len(train))
        for i, data in pbar:
            batch = {}
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                else:
                    batch[k] = v
            loss = model(batch, int(args['clip']), SLOTS_LIST[1], reset=(i==0), n_gpu=n_gpu)

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']

            loss.backward()

            if (i + 1) % args['gradient_accumulation_steps'] == 0:
                # model.optimize(args['clip'])
                torch.nn.utils.clip_grad_norm_(core.parameters(), args['clip'])
                core.optimizer.step()
                if isinstance(core.scheduler, WarmupLinearSchedule):
                    core.scheduler.step()

            # pbar.set_description(core.print_loss()) #TODO

        if((epoch+1) % int(args['evalp']) == 0):

            acc = model.evaluate(dev, avg_best, SLOTS_LIST[2], early_stop)
            if isinstance(core.scheduler, lr_scheduler.ReduceLROnPlateau):
                core.step(acc)

            if(acc >= avg_best):
                avg_best = acc
                cnt = 0
                best_model = core
            else:
                cnt += 1

            if(cnt == args["patience"] or (acc==1.0 and early_stop==None)):
                print("Ran out of patient, early stop...")
                break

if __name__ == '__main__':
    run()