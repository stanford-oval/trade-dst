import torch
import numpy as np
import random
import logging
from tqdm import tqdm
import shutil
import os
import warnings

from models.TRADE import TRADE
from utils.config import args, USE_CUDA

'''
python myTrain.py -dec= -bsz= -hdd= -dr= -lr=
'''

warnings.simplefilter("ignore", UserWarning)

def run():

    seed = args['seed']

    if args['encoder'] == 'BERT' and args['bert_model'] is None:
        raise ValueError('bert_model should be specified when using BERT encoder.')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if USE_CUDA:
        torch.cuda.manual_seed_all(seed)

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
        nb_train_vocab=max_word)
    else:
        raise ValueError("Model {} specified does not exist".format(args['decoder']))

    # print("[Info] Slots include ", SLOTS_LIST)
    # print("[Info] Unpointable Slots include ", gating_dict)

    for epoch in range(200):
        print("Epoch:{}".format(epoch))
        # Run the train function
        pbar = tqdm(enumerate(train),total=len(train))
        for i, data in pbar:
            model.train_batch(data, int(args['clip']), SLOTS_LIST[1], reset=(i==0))
            model.optimize(args['clip'])
            pbar.set_description(model.print_loss())
            # print(data)
            # exit(1)

        if((epoch+1) % int(args['evalp']) == 0):

            acc = model.evaluate(dev, avg_best, SLOTS_LIST[2], early_stop)
            model.scheduler.step(acc)

            if(acc >= avg_best):
                avg_best = acc
                cnt=0
                best_model = model
            else:
                cnt+=1

            if(cnt == args["patience"] or (acc==1.0 and early_stop==None)):
                print("Ran out of patient, early stop...")
                break

if __name__ == '__main__':
    run()