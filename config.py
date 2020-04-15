# config.py

class Config(object):
    N = 6 #6 in Transformer Paper
    d_model = 300 #512 in Transformer Paper
    d_ff = 512 #2048 in Transformer Paper
    h = 6
    dropout = 0.0
    output_size = 4
    lr = 0.00005
    max_epochs = 300
    batch_size = 32
    max_sen_len = 128

