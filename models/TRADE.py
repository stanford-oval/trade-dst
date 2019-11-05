from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
import random
import json
import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm

from utils.masked_cross_entropy import masked_cross_entropy_for_value
from utils.config import args, PAD_token

from transformers.modeling_bert import BertModel
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers.optimization import AdamW, WarmupLinearSchedule

class TRADE(nn.Module):
    def __init__(self, hidden_size, lang, path, task, lr, dropout, slots, gating_dict, t_total, device, nb_train_vocab=0):
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
        self.device = device
        self.nb_gate = len(gating_dict)
        self.cross_entorpy = nn.CrossEntropyLoss()

        if args['encoder'] == 'RNN':
            self.encoder = EncoderRNN(self.lang.n_words, hidden_size, self.dropout, self.device)
            self.decoder = Generator(self.lang, self.encoder.embedding, self.lang.n_words, hidden_size, self.dropout, self.slots, self.nb_gate, self.device)
        else:
            self.encoder = BERTEncoder(hidden_size, self.dropout, self.device)
            self.decoder = Generator(self.lang, None, self.lang.n_words, hidden_size, self.dropout, self.slots, self.nb_gate, self.device)

        if path:
            print("MODEL {} LOADED".format(str(path)))
            trained_encoder = torch.load(str(path)+'/enc.th', map_location=self.device)
            trained_decoder = torch.load(str(path)+'/dec.th', map_location=self.device)

            self.encoder.load_state_dict(trained_encoder.state_dict())
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
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=args['learn'], correct_bias=False)
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
        torch.save(self.decoder, directory + '/dec.th')
    
    def reset(self):
        self.loss, self.print_every, self.loss_ptr, self.loss_gate, self.loss_class = 0, 1, 0, 0, 0

    def forward(self, data, clip, slot_temp, reset=0, n_gpu=0):
        if reset: self.reset()
        # Zero gradients of both optimizers
        self.optimizer.zero_grad()
        
        # Encode and Decode
        use_teacher_forcing = random.random() < args["teacher_forcing_ratio"]
        all_point_outputs, gates, words_point_out, words_class_out = self.encode_and_decode(data, use_teacher_forcing, slot_temp)
        # all_point_outputs  30 32 7 18311
        # gates  30 32 3
        # words_point_out UNK...
        # words_class_out []

        loss_ptr = masked_cross_entropy_for_value(
            all_point_outputs.transpose(0, 1).contiguous(),
            data["generate_y"].contiguous(), #[:,:len(self.point_slots)].contiguous(),
            data["y_lengths"]) #[:,:len(self.point_slots)])
        loss_gate = self.cross_entorpy(gates.transpose(0, 1).contiguous().view(-1, gates.size(-1)), data["gating_label"].contiguous().view(-1))

        if args["use_gate"]:
            loss = loss_ptr + loss_gate
        else:
            loss = loss_ptr

        self.loss_grad = loss
        self.loss_ptr_to_bp = loss_ptr
        
        # Update parameters with optimizers
        self.loss += loss.data
        self.loss_ptr += loss_ptr.item()
        self.loss_gate += loss_gate.item()

        return self.loss_grad


    def optimize_GEM(self, clip):
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.optimizer.step()
        if isinstance(self.scheduler, WarmupLinearSchedule):
            self.scheduler.step()

    def encode_and_decode(self, data, use_teacher_forcing, slot_temp):
        if args['encoder'] == 'RNN':
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
            # encoded_outputs, encoded_hidden = self.encoder(story.transpose(0, 1), data['context_len'])
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

        # Get the words that can be copied from the memory
        # import pdb; pdb.set_trace()
        batch_size = len(data['context_len'])
        self.copy_list = data['context_plain']
        max_res_len = data['generate_y'].size(2) if self.encoder.training else 10

        all_point_outputs, all_gate_outputs, words_point_out, words_class_out = self.decoder.forward(batch_size, \
            encoded_hidden, encoded_outputs, data['context_len'], story, max_res_len, data['generate_y'], \
            use_teacher_forcing, slot_temp)

        return all_point_outputs, all_gate_outputs, words_point_out, words_class_out
    def evaluate(self, dev, matric_best, slot_temp, device, save_dir="", save_string = "", early_stop=None):
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)  
        print("STARTING EVALUATION")
        all_prediction = {}
        inverse_unpoint_slot = dict([(v, k) for k, v in self.gating_dict.items()])
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
                _, gates, words, class_words = self.encode_and_decode(eval_data, False, slot_temp)

            for bi in range(batch_size):
                if data_dev["ID"][bi] not in all_prediction.keys():
                    all_prediction[data_dev["ID"][bi]] = {}
                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]] = {"turn_belief":data_dev["turn_belief"][bi]}
                predict_belief_bsz_ptr, predict_belief_bsz_class = [], []
                gate = torch.argmax(gates.transpose(0, 1)[bi], dim=1)

                # pointer-generator results
                if args["use_gate"]:
                    for si, sg in enumerate(gate):
                        if sg==self.gating_dict["none"]:
                            continue
                        elif sg==self.gating_dict["ptr"]:
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
                        else:
                            predict_belief_bsz_ptr.append(slot_temp[si]+"-"+inverse_unpoint_slot[sg.item()])
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

                if set(data_dev["turn_belief"][bi]) != set(predict_belief_bsz_ptr) and args["genSample"]:
                    print("True", set(data_dev["turn_belief"][bi]) )
                    print("Pred", set(predict_belief_bsz_ptr), "\n")  

        if args["genSample"]:
            if save_dir is not "" and not os.path.exists(save_dir):
                os.mkdir(save_dir)
            json.dump(all_prediction, open(os.path.join(save_dir, "prediction_{}_{}.json".format(self.name, save_string)), 'w'), indent=4)
            print("saved generated samples", os.path.join(save_dir, "prediction_{}_{}.json".format(self.name, save_string)))

        joint_acc_score_ptr, F1_score_ptr, turn_acc_score_ptr = self.evaluate_metrics(all_prediction, "pred_bs_ptr", slot_temp)

        evaluation_metrics = {"Joint Acc":joint_acc_score_ptr, "Turn Acc":turn_acc_score_ptr, "Joint F1":F1_score_ptr}
        print(evaluation_metrics)

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)

        joint_acc_score = joint_acc_score_ptr # (joint_acc_score_ptr + joint_acc_score_class)/2
        F1_score = F1_score_ptr

        if (early_stop == 'F1'):
            if (F1_score >= matric_best):
                self.save_model('ENTF1-{:.4f}'.format(F1_score))
                print("MODEL SAVED")  
            return F1_score
        else:
            if (joint_acc_score >= matric_best):
                self.save_model('ACC-{:.4f}'.format(joint_acc_score))
                print("MODEL SAVED")
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

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout, device, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size  
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.device = device
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_token)
        self.embedding.weight.data.normal_(0, 0.1)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True, batch_first=True)
        # self.domain_W = nn.Linear(hidden_size, nb_domain)

        if args["load_embedding"]:
            with open(os.path.join("data/", 'emb{}.json'.format(vocab_size))) as f:
                E = json.load(f)
            new = self.embedding.weight.data.new
            self.embedding.weight.data.copy_(new(E))
            self.embedding.weight.requires_grad = True
            print("Encoder embedding requires_grad", self.embedding.weight.requires_grad)

        if args["fix_embedding"]:
            self.embedding.weight.requires_grad = False

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        return torch.zeros(2, bsz, self.hidden_size).to(self.device)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        embedded = self.dropout_layer(embedded)
        total_length = embedded.size(1)
        # embedded  344, 32, 400
        hidden = self.get_state(input_seqs.size(0))
        # import pdb; pdb.set_trace()
        #hidden 2, 32, 400
        if input_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        outputs, hidden = self.gru(embedded, hidden)
        if input_lengths is not None:
           outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, total_length=total_length)
        # outputs  344, 32, 800
        # They sum hidden and output states from different directions but WHY?! #TODO
        hidden = hidden[0] + hidden[1]
        # hidden 32 400
        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]
        # outputs  344, 32, 400
        return outputs, hidden.unsqueeze(0)


class Generator(nn.Module):
    def __init__(self, lang, shared_emb, vocab_size, hidden_size, dropout, slots, nb_gate, device):
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
                print("Encoder embedding requires_grad", self.embedding.weight.requires_grad)

            if args["fix_embedding"]:
                self.embedding.weight.requires_grad = False

        self.dropout_layer = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=dropout)
        self.nb_gate = nb_gate
        self.hidden_size = hidden_size
        self.W_ratio = nn.Linear(3*hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.slots = slots
        self.device = device

        self.W_gate = nn.Linear(hidden_size, nb_gate)

        # Create independent slot embeddings
        if args['pretrain_domain_embeddings']:
            self.domain_w2i = {}

            with open(os.path.join("data/", 'emb{}.json'.format(vocab_size))) as f:
                E = json.load(f)

            domains = list(set(slot.split('-')[0] for slot in self.slots))
            domains.sort()

            self.domain_emb = []
            for domain in domains:
                domain_idx = len(self.domain_w2i)
                self.domain_w2i[domain] = domain_idx
                domain_emb = E[self.lang.word2index[domain]]
                self.domain_emb.append(torch.tensor([domain_emb], device=self.device, requires_grad=False))

            self.domain_emb = torch.cat(self.domain_emb)
            print('Using pretrained domain embedding', self.domain_emb.size())

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
        all_point_outputs = all_point_outputs.to(self.device)
        all_gate_outputs = all_gate_outputs.to(self.device)
        
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
            combined_emb = domain_emb + slot_emb
            slot_emb_dict[slot] = combined_emb
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
                dec_state, hidden = self.gru(decoder_input.expand_as(hidden), hidden)

                enc_out = encoded_outputs.repeat(len(slot_temp), 1, 1)
                enc_len = encoded_lens * len(slot_temp)
                context_vec, logits, prob = self.attend(enc_out, hidden.squeeze(0), enc_len)

                if wi == 0: 
                    all_gate_outputs = torch.reshape(self.W_gate(context_vec), all_gate_outputs.size())

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
                    dec_state, hidden = self.gru(decoder_input.expand_as(hidden), hidden)
                    context_vec, logits, prob = self.attend(encoded_outputs, hidden.squeeze(0), encoded_lens)
                    if wi == 0: 
                        all_gate_outputs[counter] = self.W_gate(context_vec)
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
        
        return all_point_outputs, all_gate_outputs, words_point_out, []

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


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
