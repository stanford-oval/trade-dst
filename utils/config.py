import argparse

PAD_token = 1
SOS_token = 3
EOS_token = 2
UNK_token = 0

MAX_LENGTH = 10

def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ['true', '1']:
        return True
    elif v.lower() in ['false', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Unable to parse the argument')

parser = argparse.ArgumentParser(description='TRADE Multi-Domain DST')

# Training Setting
parser.add_argument('-ds', '--dataset', help='dataset', required=False, default="multiwoz")
parser.add_argument('-t', '--task', help='Task Number', required=False, default="dst")
parser.add_argument('-path', '--path', help='path of the file to load', required=False)
parser.add_argument('-sample', '--sample', help='Number of Samples', required=False, default=None)
parser.add_argument('-patience', '--patience', help='', required=False, default=6, type=int)
parser.add_argument('-es', '--earlyStop', help='Early Stop Criteria, BLEU or ENTF1', required=False, default='BLEU')
parser.add_argument('-all_vocab', '--all_vocab', help='', required=False, default=1, type=int)
parser.add_argument('-imbsamp', '--imbalance_sampler', help='', required=False, default=0, type=int)
parser.add_argument('-noshuff', '--no_shuffle_sampler', help='', required=False, default=0, type=int)
parser.add_argument('-data_ratio', '--data_ratio', help='', required=False, default=100, type=float)
parser.add_argument('-um', '--unk_mask', help='mask out input token to UNK', type=int, required=False, default=1)
parser.add_argument('-bsz', '--batch', help='Batch_size', required=False, type=int)

# Testing Setting
parser.add_argument('-rundev', '--run_dev_testing', help='', required=False, default=0, type=int)
parser.add_argument('-viz', '--vizualization', help='vizualization', type=int, required=False, default=0)
parser.add_argument('-gs', '--genSample', help='Generate Sample', type=int, required=False, default=0)
parser.add_argument('-evalp', '--evalp', help='evaluation period', required=False, default=1)
parser.add_argument('-an', '--addName', help='An add name for the save folder', required=False, default='')
parser.add_argument('-eb', '--eval_batch', help='Evaluation Batch_size', required=False, type=int, default=0)

# Model architecture
parser.add_argument('-gate', '--use_gate', help='', required=False, default=1, type=int)
parser.add_argument('--gate_weight', help='gate loss weight', required=False, default=1, type=float)
parser.add_argument('--gate_mask', help='gate loss weight', required=False, default=0, type=int)
parser.add_argument('-domain', '--use_domain', help='', required=False, default=0, type=int)
parser.add_argument('--domain_weight', help='domain loss weight', required=False, default=1, type=float)
parser.add_argument('--domain_mask', help='domain loss weight', required=False, default=0, type=int)
parser.add_argument('-tlw', '--trainable_loss_weights', help='gate and domain weights can be trainable', required=False, default=0, type=int)
parser.add_argument('-le', '--load_embedding', help='', required=False, default=0, type=int)
parser.add_argument('-femb', '--fix_embedding', help='', required=False, default=0, type=int)
parser.add_argument('-paral', '--parallel_decode', help='', required=False, default=1, type=int)
parser.add_argument('--cell_type', help='cell type to use for RNN models', required=False, default='GRU', choices=['LSTM', 'GRU'])
parser.add_argument('--pretrain_domain_embeddings', help='', required=False, default=False, action='store_true')
parser.add_argument('--merge_embed', help='merging strategy to combine slot and domain embeddings', required=False, default='sum', choices=['sum', 'mean', 'concat'])

# for TPRNN
parser.add_argument("--nSymbols", default=50, type=int, help="# of symbols")
parser.add_argument("--nRoles", default=35, type=int, help="# of roles")
parser.add_argument("--dSymbols", default=30, type=int, help="embedding size of symbols")
parser.add_argument("--dRoles", default=30, type=int, help="embedding size of roles")
parser.add_argument("--temperature", default=1.0, type=float, help="softmax temperature for aF and aR")
parser.add_argument("--scale_val", type=float, default=1.0, help='initial value of scale factor')
parser.add_argument("--train_scale", type=str2bool, default=False, help='whether scale factor should be trainable')

# Model Hyper-Parameters
parser.add_argument('-dec', '--decoder', help='decoder model', required=False)
parser.add_argument('-hdd', '--hidden', help='Hidden size', required=False, type=int, default=400)
parser.add_argument('-lr', '--learn', help='Learning Rate', required=False, type=float)
parser.add_argument('-dr', '--drop', help='Drop Out', required=False, type=float)
parser.add_argument('-lm', '--limit', help='Word Limit', required=False, default=-10000)
parser.add_argument('-clip', '--clip', help='gradient clipping', required=False, default=10, type=int)
parser.add_argument('-tfr', '--teacher_forcing_ratio', help='teacher_forcing_ratio', type=float, required=False, default=0.5)
parser.add_argument('-gas', '--gradient_accumulation_steps', help='Number of updates to accumulate before performing an optimization step',
                    type=int, default=1)
parser.add_argument('--max_epochs', help='maximum number of epochs', required=False, default=200, type=int)
parser.add_argument('-wp', "--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for")

# Unseen Domain Setting
parser.add_argument('-l_ewc', '--lambda_ewc', help='regularization term for EWC loss', type=float, required=False,
                    default=0.01)
parser.add_argument('-fisher_sample', '--fisher_sample', help='number of sample used to approximate fisher mat',
                    type=int, required=False, default=0)
parser.add_argument("--all_model", action="store_true")
parser.add_argument("--domain_as_task", action="store_true")
parser.add_argument('--run_except_4d', help='', required=False, default=1, type=int)
parser.add_argument("--strict_domain", action="store_true")
parser.add_argument('-exceptd', '--except_domain', help='', required=False, default="", type=str)
parser.add_argument('--except_domain_dev', help='like -exceptd, but only for dev set', required=False, default="", type=str)
parser.add_argument('-onlyd', '--only_domain', help='', required=False, default="", type=str)

# extra parameters
parser.add_argument('--seed', help='seed for random operations', required=False, default="123", type=int)
parser.add_argument('--log_dir', help='Save logs here', required=False, default="./log", type=str)
parser.add_argument('--data_dir', help='Load data from here', required=False, default="./data", type=str)
parser.add_argument("--delete_ok", type=str2bool, default=False, help='whether to delete the result directory if it already exists')

parser.add_argument("--is_kube", type=str2bool, default=True, help="turn on specific requirements for kubernetes (e.g. having progress bar, ...)")

parser.add_argument("--num_turns", type=int, default=-1, help='number of previous turns to encode at each turn')
parser.add_argument("--use_state_enc", type=int, default=0, help='')
parser.add_argument("--epoch_threshold", type=int, default=100000, help='')
parser.add_argument('-gtr', '--gold_turn_ratio', help='gold_turn_ratio', type=float, required=False, default=0.5)

# bert parameters
parser.add_argument("--bert_model", default=None, type=str, help="Bert pre-trained model selected")
parser.add_argument("--do_lower_case", type=str2bool, default=False, help="Set this flag if you are using an uncased model.")
parser.add_argument("--num_bert_layers", type=int, default=12, help='num_bert_layers to use in our model')
parser.add_argument("--encoder", type=str, default='RNN', choices=['RNN', 'BERT', 'TPRNN'], help='type of encoder to use for context')
parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

parser.add_argument('-mcl', "--max_context_length", type=int, default=-1, help="maximum length of context should not be larger than 512 when using BERT as encoder")

args = vars(parser.parse_args())

if args["load_embedding"]:
    args["hidden"] = 400
    print("[Warning] Using hidden size = 400 for pretrained word embedding (300 + 100)...")
if args["fix_embedding"]:
    args["addName"] += "FixEmb"
if args["except_domain"] != "":
    args["addName"] += "Except" + args["except_domain"]
if args["only_domain"] != "":
    args["addName"] += "Only" + args["only_domain"]

args['batch'] = int(args['batch'] / args['gradient_accumulation_steps'])

if args['encoder'] == 'BERT' and 'uncased' in args['bert_model'] and not args['do_lower_case']:
    print('do_lower_case should be True if uncased bert models are used')
    print('changing do_lower_case from False to True')
    args['do_lower_case'] = True

if args['encoder'] == 'BERT':
    args['max_context_length'] = 512 - 30


print(str(args))
