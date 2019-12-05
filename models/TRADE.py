from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
import random
import json
import torch
import torch.nn as nn
import os
import numpy as np
import logging

from utils.masked_cross_entropy import masked_cross_entropy_for_value
from utils.config import args, PAD_token, SOS_token, EOS_token, UNK_token
from models.modules import TPRencoder_LSTM
from tqdm import tqdm

from transformers.modeling_bert import BertModel
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers.optimization import AdamW, WarmupLinearSchedule

class TRADE(nn.Module):
    def __init__(self, hidden_size, lang, path, task, lr, dropout, slots, gating_dict, domain_dict, t_total, device, logger, nb_train_vocab=0):
        super(TRADE, self).__init__()
        self.name = "TRADE"
        self.task = task
        self.hidden_size = hidden_size    
        self.lang = lang[0]
        self.mem_lang = lang[1] 
        self.lr = lr 
        self.dropout = dropout
        self.slots = slots[0]
        self.slot_temp = slots[2]
        self.gating_dict = gating_dict
        self.domain_dict = domain_dict
        self.device = device
        self.nb_gate = len(gating_dict)
        self.gate_weight = nn.Parameter(torch.tensor(args['gate_weight'], device=device), requires_grad=bool(args['trainable_loss_weights']))
        self.nb_domain = len(domain_dict)
        self.domain_weight = nn.Parameter(torch.tensor(args['domain_weight'], device=device), requires_grad=bool(args['trainable_loss_weights']))
        self.cross_entorpy = nn.CrossEntropyLoss()
        self.cell_type = args['cell_type']
        self.logger = logger

        self.inverse_unpoint_slot = dict([(v, k) for k, v in self.gating_dict.items()])


        if args['encoder'] == 'RNN':
            self.encoder = EncoderRNN(self.lang.n_words, hidden_size, self.dropout, self.device, self.cell_type)

            self.decoder = Generator(self.lang, self.encoder.embedding, self.lang.n_words, hidden_size, self.dropout, self.slots, self.nb_gate, self.nb_domain, self.device, self.cell_type)
        elif args['encoder'] == 'TPRNN':
            self.encoder = EncoderTPRNN(self.lang.n_words, hidden_size, self.dropout, self.device, self.cell_type,
                                        args['nSymbols'], args['nRoles'], args['dSymbols'], args['dRoles'],
                                        args['temperature'], args['scale_val'], args['train_scale'])
            self.decoder = Generator(self.lang, self.encoder.embedding, self.lang.n_words, hidden_size, self.dropout, self.slots, self.nb_gate, self.nb_domain, self.device, self.cell_type)
        else:
            self.encoder = BERTEncoder(hidden_size, self.dropout, self.device)
            self.decoder = Generator(self.lang, None, self.lang.n_words, hidden_size, self.dropout, self.slots, self.nb_gate, self.nb_domain, self.device, self.cell_type)

        self.state_encoder = EncoderRNN(self.lang.n_words, hidden_size, self.dropout, self.device, self.cell_type)

        if path:
            self.logger.info("MODEL {} LOADED".format(str(path)))
            trained_encoder = torch.load(str(path)+'/enc.th', map_location=self.device)
            trained_state_encoder = torch.load(str(path)+'/state_enc.th', map_location=self.device)
            trained_decoder = torch.load(str(path)+'/dec.th', map_location=self.device)

            self.encoder.load_state_dict(trained_encoder.state_dict())
            self.state_encoder.load_state_dict(trained_state_encoder.state_dict())
            self.decoder.load_state_dict(trained_decoder.state_dict())


        # Initialize optimizers and criterion
        if args['encoder'] == 'RNN':
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=1, min_lr=0.0001, verbose=True)
        else:
            if args['local_rank'] != -1:
                t_total = t_total // torch.distributed.get_world_size()

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=False)
            self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=args['warmup_proportion'] * t_total, t_total=t_total)

        self.reset()

    def print_loss(self):    
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr = self.loss_ptr / self.print_every
        print_loss_gate = self.loss_gate / self.print_every
        print_loss_class = self.loss_class / self.print_every
        # print_loss_domain = self.loss_domain / self.print_every
        self.print_every += 1     
        return 'L:{:.2f},LP:{:.2f},LG:{:.2f}'.format(print_loss_avg,print_loss_ptr,print_loss_gate)
    
    def save_model(self, dec_type):
        directory = 'save/TRADE-'+args["addName"]+args['dataset']+str(self.task)+'/'+'HDD'+str(self.hidden_size)+'BSZ'+str(args['batch'])+'DR'+str(self.dropout)+str(dec_type)                 
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory + '/enc.th')
        torch.save(self.state_encoder, directory + '/state_enc.th')
        torch.save(self.decoder, directory + '/dec.th')
    
    def reset(self):
        self.loss, self.print_every, self.loss_ptr, self.loss_gate, self.loss_domain, self.loss_class = 0, 1, 0, 0, 0, 0

    def forward(self, data, clip, slot_temp, reset=0, n_gpu=0, epoch=0):
        if reset: self.reset()
        # Zero gradients of both optimizers
        self.optimizer.zero_grad()
        
        # Encode and Decode
        use_teacher_forcing = random.random() < args["teacher_forcing_ratio"]
        all_point_outputs, gates, domains, words_point_out, words_class_out =\
            self.encode_and_decode(data, use_teacher_forcing, slot_temp, epoch, True)

        # ignore_gate_idx = [v for k, v in self.gating_dict.items() if k in ['dontcare', 'none']]
        # ignore_domain_idx = [v for k, v in self.domain_dict.items() if k in ['absent']]

        gates_mask = None
        domains_mask = None
        if args['gate_mask']:
            gates_mask = torch.argmax(gates.transpose(0, 1).contiguous(), dim=-1)
        if args['domain_mask']:
            domains_mask = torch.argmax(domains.transpose(0, 1).contiguous(), dim=-1)

        loss_ptr = masked_cross_entropy_for_value(
            all_point_outputs.transpose(0, 1).contiguous(),
            data["generate_y"].contiguous(),
            data["y_lengths"],
            gates_mask,
            domains_mask)
        loss_gate = self.cross_entorpy(gates.transpose(0, 1).contiguous().view(-1, gates.size(-1)), data["gating_label"].contiguous().view(-1))
        loss_domain = self.cross_entorpy(domains.transpose(0, 1).contiguous().view(-1, domains.size(-1)), data["domain_label"].contiguous().view(-1))

        loss = loss_ptr
        if args["use_gate"]:
            loss += self.gate_weight * loss_gate
        if args["use_domain"]:
            loss += self.domain_weight * loss_domain

        self.loss_grad = loss
        self.loss_ptr_to_bp = loss_ptr
        
        # Update parameters with optimizers
        self.loss += loss.item()
        self.loss_ptr += loss_ptr.item()
        self.loss_gate += loss_gate.item()
        self.loss_domain += loss_domain.item()

        return self.loss_grad

    def optimize_GEM(self, clip):
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.optimizer.step()
        if isinstance(self.scheduler, WarmupLinearSchedule):
            self.scheduler.step()

    def encode_and_decode(self, data, use_teacher_forcing, slot_temp, epoch, training):
        if args['encoder'] == 'RNN' or args['encoder'] == 'TPRNN':
            # Build unknown mask for memory to encourage generalization
            if args['unk_mask'] and self.decoder.training:
                story_size = data['context'].size()
                rand_mask = np.ones(story_size)
                bi_mask = np.random.binomial([np.ones((story_size[0], story_size[1]))], 1-self.dropout)[0]
                rand_mask = rand_mask * bi_mask
                rand_mask = torch.Tensor(rand_mask).to(self.device)
                story = data['context'] * rand_mask.long()
            else:
                story = data['context']

            story = story.to(self.device)
            encoded_outputs, encoded_hidden = self.encoder(story, data['context_len'])

        # Encode dialog history
        # story  32 396
        # data['context_len'] 32
        elif args['encoder'] == 'BERT':
            # import pdb; pdb.set_trace()
            story = data['context']
            # story_plain = data['context_plain']

            all_input_ids = data['all_input_ids']
            all_input_mask = data['all_input_mask']
            all_segment_ids = data['all_segment_ids']
            all_sub_word_masks = data['all_sub_word_masks']

            encoded_outputs, encoded_hidden = self.encoder(all_input_ids, all_input_mask, all_segment_ids, all_sub_word_masks)
            encoded_hidden = encoded_hidden.unsqueeze(0)
        batch_size = data['prev_generate_y'].shape[0]
        if epoch < args['epoch_threshold'] and training:
            prev_generate_y = data['prev_generate_y'].reshape(batch_size, -1).to(self.device)
        else:
            gold_turn_ratio = random.random() < args["gold_turn_ratio"] if training else 0
            if gold_turn_ratio:
                prev_generate_y = data['prev_generate_y'].reshape(batch_size, -1).to(self.device)
            else:
                prev_generate_y = []

                for b in range(batch_size):
                    cur_encoded_outputs = encoded_outputs[[b], ...]
                    cur_encoded_hidden = encoded_hidden[:, [b], ...]
                    max_res_len = data['generate_y'].size(2) if self.encoder.training else 10
                    cur_point_outputs, cur_gate_outputs, cur_domains_output, cur_words_point_out, cur_words_class_out =\
                                     self.decoder(1,  cur_encoded_hidden, cur_encoded_outputs, data['context_len'][[b]],
                                     story[[b], ...],
                                     max_res_len, data['generate_y'][[b], ...],  use_teacher_forcing, slot_temp)
                    predicted_slots = self.generate_slots(cur_point_outputs, cur_gate_outputs, cur_domains_output, cur_words_point_out, cur_words_class_out, slot_temp)
                    prev_generate_y.append(predicted_slots)

                prev_y, prev_y_lengths = self.merge_multi_response(prev_generate_y)
                prev_generate_y = prev_y.reshape(batch_size, -1).to(self.device)

        state_encoded_outputs, state_encoded_hidden = self.state_encoder(prev_generate_y, None)
        if args['use_state_enc']:
            final_outputs = torch.cat([encoded_outputs, state_encoded_outputs], dim=1)
            final_hidden = encoded_hidden
            new_story = torch.cat([story, prev_generate_y], dim=1)
        else:
            final_outputs = encoded_outputs
            final_hidden = encoded_hidden
            new_story = story

        # Get the words that can be copied from the memory
        # import pdb; pdb.set_trace()
        batch_size = len(data['context_len'])
        self.copy_list = data['context_plain']
        max_res_len = data['generate_y'].size(2) if self.encoder.training else 10

        all_point_outputs, all_gate_outputs, all_domains_output, words_point_out, words_class_out = self.decoder(batch_size, \
            final_hidden, final_outputs, data['context_len'], new_story, max_res_len, data['generate_y'], \
            use_teacher_forcing, slot_temp)

        return all_point_outputs, all_gate_outputs, all_domains_output, words_point_out, words_class_out

    def merge_multi_response(self, sequences):
        '''
        merge from batch * nb_slot * slot_len to batch * nb_slot * max_slot_len
        '''
        lengths = []
        for bsz_seq in sequences:
            length = [len(v) for v in bsz_seq]
            lengths.append(length)
        max_len = max([max(l) for l in lengths])
        padded_seqs = []
        for bsz_seq in sequences:
            pad_seq = []
            for v in bsz_seq:
                v = v + [PAD_token] * (max_len-len(v))
                pad_seq.append(v)
            padded_seqs.append(pad_seq)
        padded_seqs = torch.tensor(padded_seqs)
        lengths = torch.tensor(lengths)
        return padded_seqs, lengths

    def generate_slots(self, cur_point_outputs, cur_gate_outputs, cur_domains_output, cur_words_point_out, cur_words_class_out, slot_temp):

        predict_belief_bsz_ptr = []
        gate = torch.argmax(cur_gate_outputs.transpose(0, 1)[0], dim=1)
        domain = torch.argmax(cur_domains_output.transpose(0, 1)[0], dim=1)
        # import pdb; pdb.set_trace()

        # pointer-generator results
        use_gate = args["use_gate"]
        use_domain = args["use_domain"]
        if use_gate or use_domain:
            for si, (sg, sd) in enumerate(zip(gate, domain)):
                if (sg==self.gating_dict["none"] and use_gate) or (sd==self.domain_dict["absent"] and use_domain):
                    continue
                elif (sg==self.gating_dict["ptr"] and use_gate) and ((not use_domain) or (sd==self.domain_dict["present"] and use_domain)):
                    pred = np.transpose(cur_words_point_out[si])[0]
                    st = []
                    for e in pred:
                        if e== 'EOS': break
                        else: st.append(e)
                    st = " ".join(st)
                    if st == "none":
                        continue
                    else:
                        predict_belief_bsz_ptr.append(slot_temp[si]+"-"+str(st))
                elif (not use_domain) or (sd==self.domain_dict["present"] and use_domain):
                    predict_belief_bsz_ptr.append(slot_temp[si]+"-"+self.inverse_unpoint_slot[sg.item()])
                else:
                    continue
        else:
            for si, _ in enumerate(gate):
                pred = np.transpose(cur_words_point_out[si])[0]
                st = []
                for e in pred:
                    if e == 'EOS': break
                    else: st.append(e)
                st = " ".join(st)
                if st == "none":
                    continue
                else:
                    predict_belief_bsz_ptr.append(slot_temp[si]+"-"+str(st))

        predict_belief_bsz_ptr_final = ["none"]*len(slot_temp)
        for v in predict_belief_bsz_ptr:
            domain, slot_name, value = v.split('-', maxsplit=2)
            index = slot_temp.index(domain + '-' + slot_name)
            predict_belief_bsz_ptr_final[index] = value

        final_val = []
        for value in predict_belief_bsz_ptr_final:
            v = [self.lang.word2index[word] if word in self.lang.word2index else UNK_token for word in value.split()] + [EOS_token]
            final_val.append(v)

        return final_val

    def evaluate(self, dev, matric_best, slot_temp, device, save_dir="", save_string = "", early_stop=None):
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)
        self.logger.info("STARTING EVALUATION")
        all_prediction = {}

        if args['is_kube']:
            pbar = enumerate(dev)
        else:
            pbar = tqdm(enumerate(dev), total=len(dev))
        for j, data_dev in pbar: 
            # Encode and Decode
            eval_data = {}
            # wrap all numerical values as tensors for multi-gpu training
            for k, v in data_dev.items():
                if isinstance(v, torch.Tensor):
                    eval_data[k] = v.to(device)
                elif isinstance(v, list):
                    if k in ['ID', 'turn_belief', 'context_plain', 'turn_uttr_plain']:
                        eval_data[k] = v
                    else:
                        eval_data[k] = torch.tensor(v).to(device)
                else:
                    # print('v is: {} and this ignoring {}'.format(v, k))
                    pass
            batch_size = len(data_dev['context_len'])
            with torch.no_grad():
                _, gates, domains, words, class_words = self.encode_and_decode(eval_data, False, slot_temp, epoch=0, training=False)

            for bi in range(batch_size):
                if data_dev["ID"][bi] not in all_prediction.keys():
                    all_prediction[data_dev["ID"][bi]] = {}
                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]] = {"turn_belief": data_dev["turn_belief"][bi]}
                predict_belief_bsz_ptr, predict_belief_bsz_class = [], []
                gate = torch.argmax(gates.transpose(0, 1)[bi], dim=1)
                domain = torch.argmax(domains.transpose(0, 1)[bi], dim=1)
                # import pdb; pdb.set_trace()

                # pointer-generator results
                use_gate = args["use_gate"]
                use_domain = args["use_domain"]
                if use_gate or use_domain:
                    for si, (sg, sd) in enumerate(zip(gate, domain)):
                        if (sg==self.gating_dict["none"] and use_gate) or (sd==self.domain_dict["absent"] and use_domain):
                            continue
                        elif (sg==self.gating_dict["ptr"] and use_gate) and ((not use_domain) or (sd==self.domain_dict["present"] and use_domain)):
                            pred = np.transpose(words[si])[bi]
                            st = []
                            for e in pred:
                                if e== 'EOS': break
                                else: st.append(e)
                            st = " ".join(st)
                            if st == "none":
                                continue
                            else:
                                predict_belief_bsz_ptr.append(slot_temp[si]+"-"+str(st))
                        elif (not use_domain) or (sd==self.domain_dict["present"] and use_domain):
                            predict_belief_bsz_ptr.append(slot_temp[si]+"-"+self.inverse_unpoint_slot[sg.item()])
                        else:
                            continue
                else:
                    for si, _ in enumerate(gate):
                        pred = np.transpose(words[si])[bi]
                        st = []
                        for e in pred:
                            if e == 'EOS': break
                            else: st.append(e)
                        st = " ".join(st)
                        if st == "none":
                            continue
                        else:
                            predict_belief_bsz_ptr.append(slot_temp[si]+"-"+str(st))

                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]]["pred_bs_ptr"] = predict_belief_bsz_ptr

                #if set(data_dev["turn_belief"][bi]) != set(predict_belief_bsz_ptr) and args["genSample"]:
                #    print("True", set(data_dev["turn_belief"][bi]) )
                #    print("Pred", set(predict_belief_bsz_ptr), "\n")

        if args["genSample"]:
            if save_dir is not "" and not os.path.exists(save_dir):
                os.mkdir(save_dir)
            json.dump(all_prediction, open(os.path.join(save_dir, "prediction_{}_{}.json".format(self.name, save_string)), 'w'), indent=4)
            self.logger.info("saved generated samples: {}".format(os.path.join(save_dir, "prediction_{}_{}.json".format(self.name, save_string))))

        joint_acc_score_ptr, F1_score_ptr, turn_acc_score_ptr = self.evaluate_metrics(all_prediction, "pred_bs_ptr", slot_temp)

        evaluation_metrics = {"Joint Acc":joint_acc_score_ptr, "Turn Acc":turn_acc_score_ptr, "Joint F1":F1_score_ptr}
        self.logger.info(evaluation_metrics)
        print(evaluation_metrics)

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)

        joint_acc_score = joint_acc_score_ptr # (joint_acc_score_ptr + joint_acc_score_class)/2
        F1_score = F1_score_ptr

        if (early_stop == 'F1'):
            if (F1_score >= matric_best):
                self.save_model('ENTF1-{:.4f}'.format(F1_score))
                self.logger.info("MODEL SAVED")
            return F1_score
        else:
            if (joint_acc_score >= matric_best):
                self.save_model('ACC-{:.4f}'.format(joint_acc_score))
                self.logger.info("MODEL SAVED")
            return joint_acc_score

    def evaluate_metrics(self, all_prediction, from_which, slot_temp):
        total, turn_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0
        for d, v in all_prediction.items():
            for t in range(len(v)):
                cv = v[t]
                if set(cv["turn_belief"]) == set(cv[from_which]):
                    joint_acc += 1
                total += 1

                # Compute prediction slot accuracy
                temp_acc = self.compute_acc(set(cv["turn_belief"]), set(cv[from_which]), slot_temp)
                turn_acc += temp_acc

                # Compute prediction joint F1 score
                temp_f1, temp_r, temp_p, count = self.compute_prf(set(cv["turn_belief"]), set(cv[from_which]))
                F1_pred += temp_f1
                F1_count += count

        joint_acc_score = joint_acc / float(total) if total!=0 else 0
        turn_acc_score = turn_acc / float(total) if total!=0 else 0
        F1_score = F1_pred / float(F1_count) if F1_count!=0 else 0
        return joint_acc_score, F1_score, turn_acc_score

    def compute_acc(self, gold, pred, slot_temp):
        miss_gold = 0
        miss_slot = []
        for g in gold:
            if g not in pred:
                miss_gold += 1
                miss_slot.append(g.rsplit("-", 1)[0])
        wrong_pred = 0
        for p in pred:
            if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
                wrong_pred += 1
        ACC_TOTAL = len(slot_temp)
        ACC = len(slot_temp) - miss_gold - wrong_pred
        ACC = ACC / float(ACC_TOTAL)
        return ACC

    def compute_prf(self, gold, pred):
        TP, FP, FN = 0, 0, 0
        if len(gold)!= 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in pred:
                if p not in gold:
                    FP += 1
            precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
            recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
        else:
            if len(pred)==0:
                precision, recall, F1, count = 1, 1, 1, 1
            else:
                precision, recall, F1, count = 0, 0, 0, 1
        return F1, recall, precision, count


class BERTEncoder(nn.Module):
    def __init__(self, hidden_size, dropout, device):
        super(BERTEncoder, self).__init__()

        self.device = device
        # Load config and pre-trained model
        pre_trained_model = BertModel.from_pretrained(args['bert_model'], cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1))
        bert_config = pre_trained_model.config

        # modify config if you want
        bert_config.num_hidden_layers = args['num_bert_layers']

        self.bert = BertModel(bert_config)

        # load desired layers from pre-trained model
        self.bert.load_state_dict(pre_trained_model.state_dict(), strict=False)

        self.proj = nn.Linear(bert_config.hidden_size, hidden_size)

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, all_input_ids, all_input_mask, all_segment_ids, all_sub_word_masks):

        sequence_output, pooled_output = self.bert(all_input_ids, attention_mask=all_input_mask, token_type_ids=all_segment_ids)

        output = self.proj(sequence_output)
        hidden = self.proj(pooled_output)

        return output, hidden


class EncoderTPRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout, device, cell_type, nSymbols, nRoles, dSymbols, dRoles, temperature, scale_val, train_scale, n_layers=1):
        super(EncoderTPRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.device = device
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_token)
        self.embedding.weight.data.normal_(0, 0.1)
        self.cell_type = cell_type
        self.nSymbols = nSymbols
        self.nRoles = nRoles
        self.dSymbols = dSymbols
        self.dRoles = dRoles
        self.temperature = temperature
        self.scale_val = scale_val
        self.train_scale = train_scale
        self.proj = nn.Linear(self.dSymbols * self.dRoles * 2, hidden_size)

        encoder_args= {'in_dim': hidden_size, 'hidden_size': hidden_size, 'n_layers': n_layers, 'cell_type': cell_type, 'dropout': dropout,
                       'bidirectional': True, 'batch_first': True, 'nSymbols': self.nSymbols, 'nRoles': self.nRoles,
                       'dSymbols': self.dSymbols, 'dRoles': self.dRoles, 'temperature': self.temperature, 'scale_val': self.scale_val,
                       'train_scale': self.train_scale}
        self.rnn = TPRencoder_LSTM(encoder_args)

        if args["load_embedding"]:
            with open(os.path.join("data/", 'emb{}.json'.format(vocab_size))) as f:
                E = json.load(f)
            new = self.embedding.weight.data.new
            self.embedding.weight.data.copy_(new(E))
            self.embedding.weight.requires_grad = True

        if args["fix_embedding"]:
            self.embedding.weight.requires_grad = False

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        embedded = self.dropout_layer(embedded)

        output, (last_output, aFs, aRs), R_loss = self.rnn.call(embedded)

        outputs = self.proj(output)
        hidden = self.proj(last_output)

        return outputs, hidden.unsqueeze(0)


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout, device, cell_type, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size  
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.device = device
        self.cell_type = cell_type
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_token)
        self.embedding.weight.data.normal_(0, 0.1)
        if self.cell_type == 'GRU':
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True, batch_first=True)
        elif self.cell_type == 'LSTM':
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.rnn.flatten_parameters()
        # self.domain_W = nn.Linear(hidden_size, nb_domain)

        if args["load_embedding"]:
            with open(os.path.join("data/", 'emb{}.json'.format(vocab_size))) as f:
                E = json.load(f)
            new = self.embedding.weight.data.new
            self.embedding.weight.data.copy_(new(E))
            self.embedding.weight.requires_grad = True

        if args["fix_embedding"]:
            self.embedding.weight.requires_grad = False

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        if self.cell_type == 'GRU':
            return torch.zeros(2, bsz, self.hidden_size).to(self.device)
        if self.cell_type == 'LSTM':
            return (torch.zeros(2, bsz, self.hidden_size).to(self.device), torch.zeros(2, bsz, self.hidden_size).to(self.device))

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        embedded = self.dropout_layer(embedded)
        total_length = embedded.size(1)
        # embedded  32, 344, 400
        hidden = self.get_state(input_seqs.size(0))
        # import pdb; pdb.set_trace()
        #hidden 2, 32, 400
        if input_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(embedded, hidden)
        if input_lengths is not None:
           outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, total_length=total_length)
        # outputs  32, 344, 800
        # They sum hidden and output states from different directions but WHY?! #TODO
        hidden = hidden[0] + hidden[1]
        # hidden 32 400
        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]
        # outputs  32, 344, 400
        return outputs, hidden.unsqueeze(0)


class Generator(nn.Module):
    def __init__(self, lang, shared_emb, vocab_size, hidden_size, dropout, slots, nb_gate, nb_domain, device, cell_type):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.lang = lang
        if shared_emb:
            self.embedding = shared_emb
        else:
            self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_token)
            self.embedding.weight.data.normal_(0, 0.1)
            if args["load_embedding"]:
                with open(os.path.join("data/", 'emb{}.json'.format(vocab_size))) as f:
                    E = json.load(f)
                new = self.embedding.weight.data.new
                self.embedding.weight.data.copy_(new(E))
                self.embedding.weight.requires_grad = True

            if args["fix_embedding"]:
                self.embedding.weight.requires_grad = False

        self.cell_type = cell_type
        self.dropout_layer = nn.Dropout(dropout)
        if self.cell_type == 'GRU':
            self.rnn = nn.GRU(hidden_size, hidden_size, dropout=dropout)
        elif self.cell_type == 'LSTM':
            self.rnn = nn.LSTM(hidden_size, hidden_size, dropout=dropout)
        self.nb_gate = nb_gate
        self.nb_domain = nb_domain
        self.hidden_size = hidden_size
        self.W_ratio = nn.Linear(3*hidden_size, 1)
        self.W_slot_embed = nn.Linear(2*hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.slots = slots
        self.device = device

        self.W_gate = nn.Linear(hidden_size, self.nb_gate)
        self.W_domain = nn.Linear(hidden_size, self.nb_domain)

        # Create independent slot embeddings
        if args['pretrain_domain_embeddings']:
            self.domain_w2i = {}

            domains = list(set(slot.split('-')[0] for slot in self.slots))
            domains.sort()
            for domain in domains:
                self.domain_w2i[domain] = len(self.domain_w2i)

            if args["load_embedding"]:
                with open(os.path.join("data/", 'emb{}.json'.format(vocab_size))) as f:
                    E = json.load(f)

                self.domain_emb = []
                for domain in domains:
                    domain_idx = self.domain_w2i[domain]
                    domain_emb = E[self.lang.word2index[domain]]
                    self.domain_emb.append(torch.tensor([domain_emb], device=self.device, requires_grad=False))
                self.domain_emb = torch.cat(self.domain_emb)
            else:
                self.domain_emb = torch.zeros((len(domains), hidden_size), requires_grad=False)

            self.logger.info('Using pretrained domain embedding: {}'.format(self.domain_emb.size()))

        self.slot_w2i = {}
        for slot in self.slots:
            if not args['pretrain_domain_embeddings']:
                if slot.split("-")[0] not in self.slot_w2i.keys():
                    self.slot_w2i[slot.split("-")[0]] = len(self.slot_w2i)
            if slot.split("-")[1] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split("-")[1]] = len(self.slot_w2i)
        self.Slot_emb = nn.Embedding(len(self.slot_w2i), hidden_size)
        self.Slot_emb.weight.data.normal_(0, 0.1)

    def forward(self, batch_size, encoded_hidden, encoded_outputs, encoded_lens, story, max_res_len, target_batches, use_teacher_forcing, slot_temp):
        all_point_outputs = torch.zeros(len(slot_temp), batch_size, max_res_len, self.vocab_size, device=self.device)
        all_gate_outputs = torch.zeros(len(slot_temp), batch_size, self.nb_gate, device=self.device)
        all_domain_outputs = torch.zeros(len(slot_temp), batch_size, self.nb_domain, device=self.device)

        # Get the slot embedding 
        slot_emb_dict = {}
        for i, slot in enumerate(slot_temp):
            # Domain embbeding
            if args['pretrain_domain_embeddings']:
                assert slot.split("-")[0] in self.domain_w2i.keys()
                domain_w2idx = [self.domain_w2i[slot.split("-")[0]]]
                domain_w2idx = torch.tensor(domain_w2idx)
                domain_w2idx = domain_w2idx.to(self.device)
                domain_emb = self.domain_emb[domain_w2idx]
            else:
                assert slot.split("-")[0] in self.slot_w2i.keys()
                domain_w2idx = [self.slot_w2i[slot.split("-")[0]]]
                domain_w2idx = torch.tensor(domain_w2idx)
                domain_w2idx = domain_w2idx.to(self.device)
                domain_emb = self.Slot_emb(domain_w2idx)
            # Slot embbeding
            assert slot.split("-")[1] in self.slot_w2i.keys()
            slot_w2idx = [self.slot_w2i[slot.split("-")[1]]]
            slot_w2idx = torch.tensor(slot_w2idx)
            slot_w2idx = slot_w2idx.to(self.device)
            slot_emb = self.Slot_emb(slot_w2idx)

            # Combine two embeddings as one query
            if args['merge_embed'] == 'sum':
                combined_emb = domain_emb + slot_emb
            elif args['merge_embed'] == 'mean':
                combined_emb = (domain_emb + slot_emb) / 2
            elif args['merge_embed'] == 'concat':
                combined_emb = self.W_slot_embed(torch.cat([domain_emb, slot_emb], dim=-1))
            slot_emb_exp = combined_emb.expand_as(encoded_hidden)
            if i == 0:
                slot_emb_arr = slot_emb_exp.clone()
            else:
                slot_emb_arr = torch.cat((slot_emb_arr, slot_emb_exp), dim=0)

        if args["parallel_decode"]:
            # Compute pointer-generator output, putting all (domain, slot) in one batch
            decoder_input = self.dropout_layer(slot_emb_arr).view(-1, self.hidden_size) # (batch*|slot|) * emb
            hidden = encoded_hidden.repeat(1, len(slot_temp), 1) # 1 * (batch*|slot|) * emb
            words_point_out = [[] for i in range(len(slot_temp))]
            words_class_out = []
            
            for wi in range(max_res_len):
                dec_state, hidden = self.rnn(decoder_input.expand_as(hidden), hidden)

                enc_out = encoded_outputs.repeat(len(slot_temp), 1, 1)
                enc_len = encoded_lens * len(slot_temp)
                context_vec, logits, prob = self.attend(enc_out, hidden.squeeze(0), enc_len)

                if wi == 0: 
                    all_gate_outputs = torch.reshape(self.W_gate(context_vec), all_gate_outputs.size())
                    all_domain_outputs = torch.reshape(self.W_domain(context_vec), all_domain_outputs.size())

                p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0))
                p_gen_vec = torch.cat([dec_state.squeeze(0), context_vec, decoder_input], -1)
                vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec))
                p_context_ptr = torch.zeros(p_vocab.size())
                p_context_ptr = p_context_ptr.to(self.device)

                prob = prob.to(self.device)
                p_context_ptr.scatter_add_(1, story.repeat(len(slot_temp), 1), prob)

                final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr + \
                                vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab
                pred_word = torch.argmax(final_p_vocab, dim=1)
                words = [self.lang.index2word[w_idx.item()] for w_idx in pred_word]
                
                for si in range(len(slot_temp)):
                    words_point_out[si].append(words[si*batch_size:(si+1)*batch_size])
                
                all_point_outputs[:, :, wi, :] = torch.reshape(final_p_vocab, (len(slot_temp), batch_size, self.vocab_size))
                
                if use_teacher_forcing:
                    decoder_input = self.embedding(torch.flatten(target_batches[:, :, wi].transpose(1,0)))
                else:
                    decoder_input = self.embedding(pred_word)   
                
                decoder_input = decoder_input.to(self.device)
        else:
            # Compute pointer-generator output, decoding each (domain, slot) one-by-one
            words_point_out = []
            counter = 0
            for slot in slot_temp:
                hidden = encoded_hidden
                words = []
                slot_emb = slot_emb_dict[slot]
                decoder_input = self.dropout_layer(slot_emb).expand(batch_size, self.hidden_size)
                for wi in range(max_res_len):
                    dec_state, hidden = self.rnn(decoder_input.expand_as(hidden), hidden)
                    context_vec, logits, prob = self.attend(encoded_outputs, hidden.squeeze(0), encoded_lens)
                    if wi == 0: 
                        all_gate_outputs[counter] = self.W_gate(context_vec)
                        all_domain_outputs[counter] = self.W_domain(context_vec)
                    p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0))
                    p_gen_vec = torch.cat([dec_state.squeeze(0), context_vec, decoder_input], -1)
                    vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec))
                    p_context_ptr = torch.zeros(p_vocab.size())
                    p_context_ptr = p_context_ptr.to(self.device)
                    p_context_ptr.scatter_add_(1, story, prob)
                    final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr + \
                                    vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab
                    pred_word = torch.argmax(final_p_vocab, dim=1)
                    words.append([self.lang.index2word[w_idx.item()] for w_idx in pred_word])
                    all_point_outputs[counter, :, wi, :] = final_p_vocab
                    if use_teacher_forcing:
                        decoder_input = self.embedding(target_batches[:, counter, wi]) # Chosen word is next input
                    else:
                        decoder_input = self.embedding(pred_word)   
                    decoder_input = decoder_input.to(self.device)
                counter += 1
                words_point_out.append(words)
        
        return all_point_outputs, all_gate_outputs, all_domain_outputs, words_point_out, []

    def attend(self, seq, cond, lens):
        """
        attend over the sequences `seq` using the condition `cond`.
        """

        scores_ = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
        max_len = max(lens)
        # import pdb; pdb.set_trace()
        for i, l in enumerate(lens):
            if l < max_len:
                scores_.data[i, l:] = -np.inf
        scores = F.softmax(scores_, dim=1)
        context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
        return context, scores_, scores

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1,0))
        scores = F.softmax(scores_, dim=1)
        return scores
