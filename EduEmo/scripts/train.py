"""
Usage:
    main.py [options]

Options:
    -h --help                         show this screen
    --loss-type=<str>                 Which loss to use cross-ent|corr|joint. [default: joint]
    --max-length=<int>                text length [default: 128]
    --output-dropout=<float>          prob of dropout applied to the output layer [default: 0.1]
    --seed=<int>                      fixed random seed number [default: 42]
    --train-batch-size=<int>          batch size [default: 32]
    --eval-batch-size=<int>           batch size [default: 32]
    --max-epoch=<int>                 max epoch [default: 20]
    --ffn-lr=<float>                  ffn learning rate [default: 0.001]
    --bert-lr=<float>                 bert learning rate [default: 2e-5]
    --transformer-lr=<float>          transformer learning rate [default: 2e-5]
    --lang=<str>                      language choice [default: English]
    --dev-path=<str>                  file path of the dev set [default: ../data/SemEval2018-Task1-all-data/English/E-c/2018-E-c-En-dev.txt]
    --train-path=<str>                file path of the train set [default: ../data/SemEval2018-Task1-all-data/English/E-c/2018-E-c-En-train.txt]
    --preprocess-save-path=<str>      file path of the preprocess files [default: ../data/save_data/]
    --alpha-loss=<float>              weight used to balance the loss [default: 0.2]
    --transformer-n_layers=<int>      number of transformer layers [default: 6]
    --aux_adv_training=<str>          auxiliary_adversarial training [default: aux_fgm] 
"""

from learner import Trainer
from model import EduEmo
from data_loader import LoadDataClass
from torch.utils.data import DataLoader
import torch
from docopt import docopt
import datetime
import json
import numpy as np

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"RE_DIGITS":1,"UNKNOWN":2,"PADDING":0}
        self.word2count = {"RE_DIGITS":1,"UNKNOWN":1,"PADDING":1}
        self.index2word = {0: "PADDING", 1: "RE_DIGITS", 2: "UNKNOWN"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.strip('\n').strip('\r').split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def collate_fn(batch):
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    lengths = [item[2] for item in batch]
    label_idxs = [item[3] for item in batch]
    edu_lists = [item[4] for item in batch]

    inputs = torch.tensor([item.detach().numpy() for item in inputs])
    targets = torch.tensor(targets)
    lengths = torch.tensor(lengths)
    label_idxs = torch.tensor([item.detach().numpy() for item in label_idxs])

    return [inputs,targets,lengths,label_idxs,edu_lists]

args = docopt(__doc__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if str(device) == 'cuda:0':
    print("Currently using GPU: {}".format(device))
    np.random.seed(int(args['--seed']))
    torch.cuda.manual_seed_all(int(args['--seed']))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    print("Currently using CPU")
#####################################################################
# Save hyper-parameter values ---> config.json
# Save model weights ---> filename.pt using current time
#####################################################################
now = datetime.datetime.now()
filename = now.strftime("%Y-%m-%d-%H:%M:%S")
fw = open('./configs/' + filename + '.json', 'a')
model_path = filename + '.pt'
args['--checkpoint-path'] = model_path
json.dump(args, fw, sort_keys=True, indent=2)
#####################################################################
# Define Dataloaders
#####################################################################
train_dataset = LoadDataClass(args, args['--train-path'],args['--preprocess-save-path']+'train')
train_data_loader = DataLoader(train_dataset,
                               batch_size=int(args['--train-batch-size']),
                               shuffle=True,
                               collate_fn=collate_fn
                               )
print('The number of training batches: ', len(train_data_loader))
dev_dataset = LoadDataClass(args, args['--dev-path'], args['--preprocess-save-path']+'dev')
dev_data_loader = DataLoader(dev_dataset,
                             batch_size=int(args['--eval-batch-size']),
                             shuffle=False,
                             collate_fn=collate_fn
                             )
print('The number of validation batches: ', len(dev_data_loader))
#############################################################################
# Define Model & Training Pipeline
#############################################################################
model = EduEmo(output_dropout=float(args['--output-dropout']),
                lang=args['--lang'],
                joint_loss=args['--loss-type'],
                alpha=float(args['--alpha-loss']),
                n_layers = int(args['--transformer-n_layers']))
#############################################################################
# Start Training
#############################################################################
learn = Trainer(model, train_data_loader, dev_data_loader, aux_adv_train=args['--aux_adv_training'],filename=filename)
learn.fit(
    num_epochs=int(args['--max-epoch']),
    args=args,
    device=device
)
