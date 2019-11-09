from models.TRADE import TRADE
from utils.config import args
import warnings

'''
python3 myTest.py -ds= -path= -bsz=
'''

warnings.simplefilter("ignore", UserWarning)

def run():

    directory = args['path'].split("/")
    print(directory)
    HDD = directory[2].split('HDD')[1].split('BSZ')[0]
    decoder = directory[1].split('-')[0]
    BSZ = int(args['batch']) if args['batch'] else int(directory[2].split('BSZ')[1].split('DR')[0])
    args["decoder"] = decoder
    args["HDD"] = HDD
    print("HDD", HDD, "decoder", decoder, "BSZ", BSZ)

    if args['dataset']=='multiwoz':
        from utils.utils_multiWOZ_DST import prepare_data_seq
    else:
        print("You need to provide the --dataset information")

    train, dev, test, test_special, lang, SLOTS_LIST, gating_dict, max_word = prepare_data_seq(False, args['task'], False, batch_size=BSZ)

    import pdb; pdb.set_trace()

    if args['decoder'] == 'TRADE':
        model = TRADE(
            int(HDD),
            lang=lang,
            path=args['path'],
            task=args["task"],
            lr=0,
            dropout=0,
            slots=SLOTS_LIST,
            gating_dict=gating_dict,
            t_total=-1,
            device='cpu',
            nb_train_vocab=max_word)
    else:
        raise ValueError("Model {} specified does not exist".format(args['decoder']))

    if args["run_dev_testing"]:
        print("Development Set ...")
        acc_dev = model.evaluate(dev, 1e7, SLOTS_LIST[2], device='cpu', save_string="dev")

    if args['except_domain']!="" and args["run_except_4d"]:
        print("Test Set on 4 domains...")
        acc_test_4d = model.evaluate(test_special, 1e7, SLOTS_LIST[2], device='cpu', save_string="test_4dom")

    # TODO: add test -onlyd instead of doing it in evaluate-job.sh

    print("Test Set ...")
    acc_test = model.evaluate(test, 1e7, SLOTS_LIST[3], device='cpu', save_string="test")

if __name__ == '__main__':
    run()


