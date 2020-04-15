# Model.py

import torch
import torch.nn as nn
from copy import deepcopy
from transformer_classifier.Model_Transformer.train_utils import GloveEmbeddings_CTGR_rand, Word2vecEmbeddings_CTGR_oov, Word2vecEmbeddings_CTGR_rand, Embeddings_segment, FinetunedEmbeddings_CTGR_rand, FinetunedEmbeddings_CTGR_unk,FinetunedEmbeddings,Embeddings, PositionalEncoding, SegmentEmbedding, PadEncoding
from transformer_classifier.Model_Transformer.attention import MultiHeadedAttention
from transformer_classifier.Model_Transformer.encoder import EncoderLayer, Encoder
from transformer_classifier.Model_Transformer.feed_forward import PositionwiseFeedForward
from transformer_classifier.Model_Transformer.utils import *
import pickle
# torch.set_printoptions(edgeitems=40,linewidth=80)
torch.set_printoptions(profile="full")

class Transformer(nn.Module):
    def __init__(self, config, src_vocab, torch_text, model_setting):
        super(Transformer, self).__init__()
        self.config = config
        
        h, N, dropout = self.config.h, self.config.N, self.config.dropout
        d_model, d_ff = self.config.d_model, self.config.d_ff
        
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        self.encoder = Encoder(EncoderLayer(config.d_model, deepcopy(attn), deepcopy(ff), dropout), N)
        if model_setting == 'standard':
            self.src_embed = nn.Sequential(Embeddings(config.d_model, src_vocab),
                                           deepcopy(position))  # Embeddings followed by PE
        elif model_setting == 'word2vec':
            self.src_embed = nn.Sequential(Word2vecEmbeddings_CTGR_rand(config.d_model, src_vocab, torch_text),
                                           deepcopy(position))  # Embeddings followed by PE
        elif model_setting == 'dual':
            self.src_embed = nn.Sequential(FinetunedEmbeddings(config.d_model, src_vocab, torch_text),
                                           deepcopy(position))  # Embeddings followed by PE
        elif model_setting == 'word2vec_oov':
            self.src_embed = nn.Sequential(Word2vecEmbeddings_CTGR_oov(config.d_model, src_vocab, torch_text),
                                           deepcopy(position))  # Embeddings followed by PE
        elif model_setting == 'dual+kb':
            self.src_embed = nn.Sequential(FinetunedEmbeddings_CTGR_rand(config.d_model, src_vocab, torch_text),
                                           deepcopy(position))  # Embeddings followed by PE
        elif model_setting == 'dual+kb+unk':
            self.src_embed = nn.Sequential(FinetunedEmbeddings_CTGR_unk(config.d_model, src_vocab, torch_text),
                                           deepcopy(position))  # Embeddings followed by PE
        else:
            print('model setting error')
            exit()

        # Fully-Connected Layer
        self.fc = nn.Linear(
            self.config.d_model,
            self.config.output_size
        )
        
        # Softmax non-linearity
        self.softmax = nn.Softmax()

    def forward(self, x):
        embedded_sents = self.src_embed(x.permute(1,0)) # shape = (batch_size, sen_len, d_model)
        encoded_sents = self.encoder(embedded_sents)
        
        # Convert input to (batch_size, d_model) for linear layer
        final_feature_map = encoded_sents[:,-1,:]
        final_out = self.fc(final_feature_map)
        return self.softmax(final_out)
    
    def add_optimizer(self, optimizer):
        self.optimizer = optimizer
        
    def add_loss_op(self, loss_op):
        self.loss_op = loss_op
    
    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2
                
    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []
        
        # Reduce learning rate as number of epochs increase
        if (epoch == int(self.config.max_epochs/3)) or (epoch == int(2*self.config.max_epochs/3)):
            self.reduce_lr()

        for i, batch in enumerate(train_iterator):

            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                x = batch.text.cuda()
                y = (batch.label - 1).type(torch.cuda.LongTensor)
            else:
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()
    
            if i % 100 == 0:
                print("Iter: {}".format(i+1))
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                losses = []
                
                # Evalute Accuracy on validation set
                val_accuracy = evaluate_model(self, val_iterator)
                print("\tVal Accuracy: {:.4f}".format(val_accuracy))
                self.train()
                
        return train_losses, val_accuracies


class Transformer_segment(nn.Module):
    def __init__(self, config, src_vocab, torch_text, model_setting):
        super(Transformer_segment, self).__init__()
        self.config = config
        self.torch_text = torch_text
        if model_setting == 'word2vec' or 'word2vec' or 'standard':
            with open('/hdd/user4/InfoSc/out_extra/cat2vec_converted_word2vec.bin', 'rb') as f:
                cat2vec_converted = pickle.load(f)
        elif model_setting == 'dual' or model_setting == 'dual+kb':
            with open('/hdd/user4/InfoSc/out_extra/cat2vec_converted_dual.bin', 'rb') as f:
                cat2vec_converted = pickle.load(f)
        elif model_setting == 'glove' :
            with open('/hdd/user4/InfoSc/out_extra/glove_category.bin', 'rb') as f:
                cat2vec_converted = pickle.load(f)
        else:
            print('model setting error')


        self.catlist = cat2vec_converted.keys()
        self.itos = self.torch_text.vocab.itos
        self.seg_lookup = [1 if k in self.catlist else 0 for k in self.itos]


        h, N, dropout = self.config.h, self.config.N, self.config.dropout
        d_model, d_ff = self.config.d_model, self.config.d_ff

        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.segment = SegmentEmbedding(d_model, dropout)


        self.encoder = Encoder(EncoderLayer(config.d_model, deepcopy(attn), deepcopy(ff), dropout), N)
        if model_setting == 'standard':
            self.src_embed = nn.Sequential(Embeddings_segment(config.d_model, src_vocab),
                                           deepcopy(position))  # Embeddings followed by PE
        elif model_setting == 'word2vec':
            self.src_embed = nn.Sequential(Word2vecEmbeddings_CTGR_rand(config.d_model, src_vocab, torch_text),
                                           deepcopy(position))  # Embeddings followed by PE
        elif model_setting == 'word2vec_oov':
            self.src_embed = nn.Sequential(Word2vecEmbeddings_CTGR_oov(config.d_model, src_vocab, torch_text),
                                           deepcopy(position))  # Embeddings followed by PE
        elif model_setting == 'dual':
            self.src_embed = nn.Sequential(FinetunedEmbeddings(config.d_model, src_vocab, torch_text),
                                           deepcopy(position))  # Embeddings followed by PE
        elif model_setting == 'dual+kb':
            self.src_embed = nn.Sequential(FinetunedEmbeddings_CTGR_rand(config.d_model, src_vocab, torch_text),
                                           deepcopy(position))  # Embeddings followed by PE
        elif model_setting == 'dual+kb+unk':
            self.src_embed = nn.Sequential(FinetunedEmbeddings_CTGR_unk(config.d_model, src_vocab, torch_text),
                                           deepcopy(position))  # Embeddings followed by PE
        elif model_setting == 'glove':
            self.src_embed = nn.Sequential(GloveEmbeddings_CTGR_rand(config.d_model, src_vocab, torch_text),
                                           deepcopy(position))  # Embeddings followed by PE
        else:
            print('model setting error')
            exit()

        # Fully-Connected Layer
        self.fc = nn.Linear(
            self.config.d_model,
            self.config.output_size
        )

        # Softmax non-linearity
        self.softmax = nn.Softmax()

    def forward(self, x, t):
        embedded_sents = self.src_embed(x.permute(1, 0))  # shape = (batch_size, sen_len, d_model)
        embedded_sents += self.segment.forward(t.permute(1, 0))
        encoded_sents = self.encoder(embedded_sents)
        # Convert input to (batch_size, d_model) for linear layer
        final_feature_map = encoded_sents[:, -1, :]
        final_out = self.fc(final_feature_map)
        return self.softmax(final_out)

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2

    def run_epoch_lr_scheduler(self, train_iterator, val_iterator, epoch, lr_scheduler):
        train_losses = []
        val_accuracies = []
        losses = []

        # Reduce learning rate as number of epochs increase
        # if (epoch == int(self.config.max_epochs / 3)) or (epoch == int(2 * self.config.max_epochs / 3)):
        #     self.reduce_lr()

        for i, batch in enumerate(train_iterator):

            self.optimizer.zero_grad()
            lr_scheduler.step(i)
            if torch.cuda.is_available():
                x = batch.text.cuda()
                t = torch.tensor(np.take(self.seg_lookup, batch.text.numpy())).cuda()
                y = (batch.label - 1).type(torch.cuda.LongTensor)
            else:
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            y_pred = self.__call__(x, t)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()


            if i % 100 == 0:
                print("Iter: {}".format(i + 1))
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                losses = []

                # Evalute Accuracy on validation set
                val_accuracy = evaluate_model_seg(self, val_iterator, self.seg_lookup)
                val_accuracies.append(val_accuracy)
                print("\tVal Accuracy: {:.4f}".format(val_accuracy))
                self.train()
        return train_losses, val_accuracies

    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []

        # Reduce learning rate as number of epochs increase
        if (epoch == int(self.config.max_epochs / 3)) or (epoch == int(2 * self.config.max_epochs / 3)):
            self.reduce_lr()

        for i, batch in enumerate(train_iterator):

            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                x = batch.text.cuda()
                t = torch.tensor(np.take(self.seg_lookup, batch.text.numpy())).cuda()
                y = (batch.label - 1).type(torch.cuda.LongTensor)
            else:
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            y_pred = self.__call__(x, t)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()

            if i % 100 == 0:
                print("Iter: {}".format(i + 1))
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                losses = []

                # Evalute Accuracy on validation set
                val_accuracy = evaluate_model_seg(self, val_iterator, self.seg_lookup)
                val_accuracies.append(val_accuracy)
                print("\tVal Accuracy: {:.4f}".format(val_accuracy))
                self.train()
        return train_losses, val_accuracies

class Transformer_segment_maxpool(nn.Module):
    def __init__(self, config, src_vocab, torch_text, model_setting):
        super(Transformer_segment_maxpool, self).__init__()
        self.config = config
        self.torch_text = torch_text
        if model_setting == 'word2vec':
            with open('/hdd/user4/InfoSc/out_extra/cat2vec_converted_word2vec.bin', 'rb') as f:
                cat2vec_converted = pickle.load(f)
        elif model_setting == 'dual':
            with open('/hdd/user4/InfoSc/out_extra/cat2vec_converted_dual.bin', 'rb') as f:
                cat2vec_converted = pickle.load(f)
        self.catlist = cat2vec_converted.keys()
        self.itos = self.torch_text.vocab.itos
        self.seg_lookup = [1 if k in self.catlist else 0 for k in self.itos]


        h, N, dropout = self.config.h, self.config.N, self.config.dropout
        d_model, d_ff = self.config.d_model, self.config.d_ff

        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.segment = SegmentEmbedding(d_model, dropout)


        self.encoder = Encoder(EncoderLayer(config.d_model, deepcopy(attn), deepcopy(ff), dropout), N)
        if model_setting == 'standard':
            self.src_embed = nn.Sequential(Embeddings_segment(config.d_model, src_vocab),
                                           deepcopy(position))  # Embeddings followed by PE
        elif model_setting == 'word2vec':
            self.src_embed = nn.Sequential(Word2vecEmbeddings_CTGR_rand(config.d_model, src_vocab, torch_text),
                                           deepcopy(position))  # Embeddings followed by PE
        elif model_setting == 'dual':
            self.src_embed = nn.Sequential(FinetunedEmbeddings(config.d_model, src_vocab, torch_text),
                                           deepcopy(position))  # Embeddings followed by PE
        elif model_setting == 'dual+kb':
            self.src_embed = nn.Sequential(FinetunedEmbeddings_CTGR_rand(config.d_model, src_vocab, torch_text),
                                           deepcopy(position))  # Embeddings followed by PE
        elif model_setting == 'dual+kb+unk':
            self.src_embed = nn.Sequential(FinetunedEmbeddings_CTGR_unk(config.d_model, src_vocab, torch_text),
                                           deepcopy(position))  # Embeddings followed by PE
        else:
            print('model setting error')
            exit()

        # Fully-Connected Layer
        self.fc = nn.Linear(
            self.config.d_model,
            self.config.output_size
        )

        # Softmax non-linearity
        self.softmax = nn.Softmax()

    def forward(self, x, t):
        embedded_sents = self.src_embed(x.permute(1, 0))  # shape = (batch_size, sen_len, d_model)
        embedded_sents += self.segment.forward(t.permute(1, 0))
        encoded_sents = self.encoder(embedded_sents)
        # Convert input to (batch_size, d_model) for linear layer
        input = torch.tensor(encoded_sents)
        input1 = input.transpose(1, 2)
        input2 = input1.cpu().detach().numpy()
        input3 = np.amax(input2, axis=2)
        input4 = torch.tensor(input3)
        final_feature_map = input4
        final_out = self.fc(final_feature_map.cuda())
        return self.softmax(final_out)

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2

    def run_epoch_lr_scheduler(self, train_iterator, val_iterator, epoch, lr_scheduler):
        train_losses = []
        val_accuracies = []
        losses = []

        # Reduce learning rate as number of epochs increase
        # if (epoch == int(self.config.max_epochs / 3)) or (epoch == int(2 * self.config.max_epochs / 3)):
        #     self.reduce_lr()

        for i, batch in enumerate(train_iterator):

            self.optimizer.zero_grad()
            lr_scheduler.step(i)
            if torch.cuda.is_available():
                x = batch.text.cuda()
                t = torch.tensor(np.take(self.seg_lookup, batch.text.numpy())).cuda()
                y = (batch.label - 1).type(torch.cuda.LongTensor)
            else:
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            y_pred = self.__call__(x, t)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()


            if i % 100 == 0:
                print("Iter: {}".format(i + 1))
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                losses = []

                # Evalute Accuracy on validation set
                val_accuracy = evaluate_model_seg(self, val_iterator, self.seg_lookup)
                val_accuracies.append(val_accuracy)
                print("\tVal Accuracy: {:.4f}".format(val_accuracy))
                self.train()
        return train_losses, val_accuracies

    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []

        # Reduce learning rate as number of epochs increase
        if (epoch == int(self.config.max_epochs / 3)) or (epoch == int(2 * self.config.max_epochs / 3)):
            self.reduce_lr()

        for i, batch in enumerate(train_iterator):

            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                x = batch.text.cuda()
                t = torch.tensor(np.take(self.seg_lookup, batch.text.numpy())).cuda()
                y = (batch.label - 1).type(torch.cuda.LongTensor)
            else:
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            y_pred = self.__call__(x, t)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()

            if i % 100 == 0:
                print("Iter: {}".format(i + 1))
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                losses = []

                # Evalute Accuracy on validation set
                val_accuracy = evaluate_model_seg(self, val_iterator, self.seg_lookup)
                val_accuracies.append(val_accuracy)
                print("\tVal Accuracy: {:.4f}".format(val_accuracy))
                self.train()
        return train_losses, val_accuracies

class Transformer_segment_mask(nn.Module):
    def __init__(self, config, src_vocab, torch_text, model_setting):
        super(Transformer_segment_mask, self).__init__()
        self.config = config
        self.torch_text = torch_text
        if model_setting == 'word2vec' or model_setting == 'word2vec_oov' or model_setting == 'standard':
            with open('/hdd/user4/InfoSc/out_extra/cat2vec_converted_word2vec.bin', 'rb') as f:
                cat2vec_converted = pickle.load(f)
        elif model_setting == 'dual':
            with open('/hdd/user4/InfoSc/out_extra/cat2vec_converted_dual.bin', 'rb') as f:
                cat2vec_converted = pickle.load(f)
        self.catlist = list(cat2vec_converted.keys())
        self.catlist.append('<sep>')
        self.itos = self.torch_text.vocab.itos
        self.seg_lookup = [1 if k in self.catlist else 0 for k in self.itos]
        self.pad_lookup = [0 if k == '<pad>' else 1 for k in self.itos]

        h, N, dropout = self.config.h, self.config.N, self.config.dropout
        d_model, d_ff = self.config.d_model, self.config.d_ff

        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.segment = SegmentEmbedding(d_model, dropout)
        self.pad_encoding = PadEncoding(d_model, dropout)


        self.encoder = Encoder(EncoderLayer(config.d_model, deepcopy(attn), deepcopy(ff), dropout), N)
        if model_setting == 'standard':
            self.src_embed = nn.Sequential(Embeddings_segment(config.d_model, src_vocab),
                                           deepcopy(position))  # Embeddings followed by PE
        elif model_setting == 'word2vec':
            self.src_embed = nn.Sequential(Word2vecEmbeddings_CTGR_rand(config.d_model, src_vocab, torch_text),
                                           deepcopy(position))  # Embeddings followed by PE
        elif model_setting == 'dual':
            self.src_embed = nn.Sequential(FinetunedEmbeddings(config.d_model, src_vocab, torch_text),
                                           deepcopy(position))  # Embeddings followed by PE
        elif model_setting == 'dual+kb':
            self.src_embed = nn.Sequential(FinetunedEmbeddings_CTGR_rand(config.d_model, src_vocab, torch_text),
                                           deepcopy(position))  # Embeddings followed by PE
        elif model_setting == 'dual+kb+unk':
            self.src_embed = nn.Sequential(FinetunedEmbeddings_CTGR_unk(config.d_model, src_vocab, torch_text),
                                           deepcopy(position))  # Embeddings followed by PE
        elif model_setting == 'word2vec_oov':
            self.src_embed = nn.Sequential(Word2vecEmbeddings_CTGR_oov(config.d_model, src_vocab, torch_text),
                                           deepcopy(position))  # Embeddings followed by PE
        else:
            print('model setting error')
            exit()

        # Fully-Connected Layer
        self.fc = nn.Linear(
            self.config.d_model,
            self.config.output_size
        )

        # Softmax non-linearity
        self.softmax = nn.Softmax()

    def forward(self, x, t, pad):
        pad_expand = self.pad_encoding(pad.permute(1,0))
        embedded_sents = self.src_embed(x.permute(1, 0))  # shape = (batch_size, sen_len, d_model)
        embedded_sents += self.segment.forward(t.permute(1, 0))
        encoded_sents = self.encoder(embedded_sents, pad_expand)
        # Convert input to (batch_size, d_model) for linear layer
        # print(encoded_sents.size())
        # final_feature_map = encoded_sents[:, -1, :]
        final_feature_map = encoded_sents[:, 0, :]
        final_out = self.fc(final_feature_map)
        return self.softmax(final_out)

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2

    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []

        # Reduce learning rate as number of epochs increase
        if (epoch == int(self.config.max_epochs / 3)) or (epoch == int(2 * self.config.max_epochs / 3)):
            self.reduce_lr()

        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                x = batch.text.cuda()
                t = torch.tensor(np.take(self.seg_lookup, batch.text.numpy())).cuda()
                pad = torch.tensor(np.take(self.pad_lookup, batch.text.numpy())).cuda()
                y = (batch.label - 1).type(torch.cuda.LongTensor)
            else:
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            y_pred = self.__call__(x, t, pad)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()

            if i % 100 == 0:
                print("Iter: {}".format(i + 1))
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                losses = []

                # Evalute Accuracy on validation set
                val_accuracy = evaluate_model_seg_mask(self, val_iterator, self.seg_lookup, self.pad_lookup)
                print("\tVal Accuracy: {:.4f}".format(val_accuracy))
                self.train()

        return train_losses, val_accuracies