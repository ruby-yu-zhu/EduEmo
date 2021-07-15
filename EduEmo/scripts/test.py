"""
Usage:
    main.py [options]

Options:
    -h --help                         show this screen
    --model-path=<str>                path of the trained model [default: 2021-07-15-20:51:31_checkpoint.pt]
    --max-length=<int>                text length [default: 128]
    --seed=<int>                      seed [default: 0]
    --test-batch-size=<int>           batch size [default: 32]
    --lang=<str>                      language choice [default: English]
    --test-path=<str>                 file path of the test set [default: ../data/SemEval2018-Task1-all-data/English/E-c/2018-E-c-En-test-gold.txt]
    --preprocess-save-path=<str>      file path of the preprocess files [default: ../data/save_data/]
"""
from learner import EvaluateOnTest
from model import EduEmo
from data_loader import LoadDataClass
from torch.utils.data import DataLoader
import torch
from docopt import docopt
import numpy as np

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
# Define Dataloaders
#####################################################################
test_dataset = LoadDataClass(args, args['--test-path'],args['--preprocess-save-path']+'test')
test_data_loader = DataLoader(test_dataset,
                              batch_size=int(args['--test-batch-size']),
                              shuffle=False,
                              collate_fn=collate_fn)
print('The number of Test batches: ', len(test_data_loader))
#############################################################################
# Run the model on a Test set
#############################################################################
model = EduEmo(lang=args['--lang'])
learn = EvaluateOnTest(model, test_data_loader, model_path='./models/' + args['--model-path'])
learn.predict(device=device)


