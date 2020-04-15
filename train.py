# train.py




from transformer_classifier.Model_Transformer.utils import *
from transformer_classifier.Model_Transformer.model import *
from transformer_classifier.Model_Transformer.config import Config
import sys
import torch.optim as optim
from torch import nn
import torch
import os
import time
import datetime

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"




if __name__=='__main__':
    config = Config()
    train_file = '/hdd/user4/InfoSc/transformer_classifier/data/ag_title_all_wo.train' #TREC = TREC_ctgr.train, AG = ag_news_ctgr.train
    if len(sys.argv) > 2:
        train_file = sys.argv[1]
    test_file = '/hdd/user4/InfoSc/transformer_classifier/data/ag_title_all_wo.test'
    if len(sys.argv) > 3:
        test_file = sys.argv[2]
    
    dataset = Dataset(config)
    dataset.load_data(train_file, test_file)



    # Create Model with specified optimizer and loss function
    ##############################################################
    model_setting = 'standard' #standard, dual, dual+kb
    model = Transformer(config, len(dataset.vocab), dataset, model_setting)
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    NLLLoss = nn.NLLLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(NLLLoss)
    ##############################################################
    
    train_losses = []
    val_accuracies = []
    start = time.time()
    for i in range(config.max_epochs):
        print("Epoch: {}".format(i))
        train_loss, val_accuracy = model.run_epoch(dataset.train_iterator, dataset.val_iterator, i)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

    train_acc = evaluate_model(model, dataset.train_iterator)
    val_acc = evaluate_model(model, dataset.val_iterator)
    test_acc = evaluate_model(model, dataset.test_iterator)

    print('Final Training Accuracy: {:.4f}'.format(train_acc))
    print('Final Validation Accuracy: {:.4f}'.format(val_acc))
    print('Final Test Accuracy: {:.4f}'.format(test_acc))
    time_consumed = int(time.time()-start)
    print('time:',str(datetime.timedelta(seconds=time_consumed)))