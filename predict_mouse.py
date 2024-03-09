import os
import  numpy as np
import torch
from model import Transformer, VirusCNN
from    torch import optim
from    torch import nn
from    torch.nn import functional as F
import random
from tqdm import tqdm
# from VirusCNN_siamese import VirusCNN

from sklearn.metrics import roc_auc_score, average_precision_score
import argparse
# import loralib as lora



#####################################################################
##########################  Input Params  ###########################
#####################################################################

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--save', type=str, default = '_shuffle')
parser.add_argument('--threshold', type=float, default = 0.6)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--embed_size', type=int, default = 512)
parser.add_argument('--batch_size', type=int, default = 1024)
parser.add_argument('--forward_expansion', type=int, default = 2)
parser.add_argument('--out_size', type=int, default = 256)
parser.add_argument('--heads', type=int, default = 8)
parser.add_argument('--dorp_out', type=float, default=0.1)
parser.add_argument('--nce', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--reverse', type=int, default=0)
parsers = parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(1001)

#########load data##########

train_x, train_y, test_x, test_y = [torch.tensor(np.load(f'test_data/m_data/{i}.npy')) for i in ['train_x', 'train_y', 'test_x', 'test_y']]




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
b_z = parsers.batch_size

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_x.to(device), train_y.to(device)),
    batch_size=b_z, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(test_x.to(device), test_y.to(device)), 
    batch_size=b_z, 
    shuffle=False
)


src_pad_idx = 0
num_pcs = 4
src_vocab_size = num_pcs+1
num_layers = parsers.num_layers
out_size = parsers.out_size
embed_size=parsers.embed_size
forward_expansion=parsers.forward_expansion
heads=parsers.heads
dorp_out = parsers.dorp_out



model = Transformer(
                            out_size   = parsers.out_size,
                            src_vocab_size = 5, 
                            src_pad_idx = 0,
                            embed_size = parsers.embed_size, 
                            num_layers = parsers.num_layers,
                            forward_expansion = parsers.forward_expansion,
                            heads = parsers.heads,
                            device=device, 
                            max_length=48, 
                            dropout = parsers.dorp_out
                )

model = model.to(device)

CNN = VirusCNN()
CNN = CNN.to(device)
params = list(CNN.parameters()) 
optimizer = optim.Adam(params, lr=0.001)#args.learning_rate
criterion = nn.BCEWithLogitsLoss()


# 定义测试函数
def test(model, test_loader, CNN):
    model.eval()
    CNN.eval()
    y_true = []
    y_pred = []
    running_loss = 0.0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader, 0):
            sigmoid = torch.nn.Sigmoid()


            c_inputs = torch.eye(4).long()[(inputs - 1).long()]
            c_inputs = c_inputs.float()
            c_inputs = c_inputs.unsqueeze(1).to(device)
            
            

            c_outputs = CNN(c_inputs)
            outputs = model(inputs)
            


            outputs = torch.max(c_outputs, outputs)
            outputs = sigmoid(outputs)


            y_true.extend(labels.tolist())

            y_pred.extend(outputs.squeeze().tolist())
    auc = roc_auc_score(y_true, y_pred)
    aupr = average_precision_score(y_true, y_pred)
    return auc, aupr


train_losses = []
test_losses = []
accuracies = []
precisions = []
recalls = []
f1_scores = []
aucs = []
auprs = []
best_loss = float('inf')


model_dict = torch.load(f'params/model_mouse.pkl', map_location='cpu')
CNN_dict = torch.load(f'params/CNN_mouse.pkl', map_location='cpu')

model.load_state_dict(model_dict)
model = model.to(device)
CNN.load_state_dict(CNN_dict)
CNN = CNN.to(device)

auc, aupr = test(model, test_loader, CNN)
print('AUC: {:.4f}, AUPR: {:.4f}'.format(auc, aupr))
