import os
import numpy as np
import torch
from model import Transformer, VirusCNN
from init import matrix_from_fasta, setup_seed, filter_reads
import argparse
import math

#####################################################################
##########################  Input Params  ###########################
#####################################################################

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--in_folder', default='', required=True, help='Folder path containing FASTA files')
parser.add_argument('--out_folder', default='', required=True)
parser.add_argument('--threshold', type=float, default = 0.6, required=False)
parser.add_argument('--num_layers', type=int, default=3, required=False)
parser.add_argument('--embed_size', type=int, default=512, required=False)
parser.add_argument('--batch_size', type=int, default=1024, required=False)
parser.add_argument('--forward_expansion', type=int, default=2, required=False)
parser.add_argument('--out_size', type=int, default=256, required=False)
parser.add_argument('--heads', type=int, default=8, required=False)
parser.add_argument('--dorp_out', type=float, default=0.1, required=False)
parser.add_argument('--output_file', default='out/predictions1.txt', help='File to save the predictions')
parsers = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
setup_seed(1001)

#########        load data              ##########
def process_fasta_files(folder_path):
    predictions = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.fasta'):
            print(filename)
            file_path = os.path.join(folder_path, filename)
            
            # Extract sequence name
            file_name = os.path.splitext(filename)[0]
            
            test_x = torch.tensor(matrix_from_fasta(file_path))
            b_z = parsers.batch_size
            test_loader = torch.utils.data.DataLoader(
                test_x.clone().detach().to(device),
                # torch.tensor(test_x).to(device),
                batch_size=b_z,
                shuffle=False
            )

            #########        load model              ##########
            model = Transformer(
                            out_size=parsers.out_size,
                            src_vocab_size=5, 
                            src_pad_idx=0,
                            embed_size=parsers.embed_size, 
                            num_layers=parsers.num_layers,
                            forward_expansion=parsers.forward_expansion,
                            heads=parsers.heads,
                            device=device, 
                            max_length=48, 
                            dropout=parsers.dorp_out
                )
            
            CNN = VirusCNN()
            model = model.to(device)
            CNN = CNN.to(device)

            model = torch.nn.DataParallel(model)
            model_dict = torch.load(f'params/transformer_params.pkl', map_location='cpu')
            CNN_dict = torch.load(f'params/CNN_params.pkl', map_location='cpu')
            model.load_state_dict(model_dict)
            CNN.load_state_dict(CNN_dict)
            
            def batch_mean(inputs, outputs):
                bat = math.ceil(len(inputs[0]) / 48)
                batches = [[] for _ in range(bat)]

                for i in range(bat):
                    batches[i] = outputs[i::bat]

                result = torch.mean(torch.stack(batches), dim=0)
                
                return result

            def predict_reads(model, CNN, test_loader):
                model.eval()
                CNN.eval()
                y_pred = []
                with torch.no_grad():
                    for number, inputs in enumerate(test_loader, 0):
                        sigmoid = torch.nn.Sigmoid()
                        if len(inputs[0]) > 48:

                            ret = []
                            for i in range(len(inputs)):


                                sequence = inputs[i]
                                segment_length = 48
                                for start_idx in range(0, len(sequence), segment_length):
                                    fragment = sequence[start_idx:(start_idx + segment_length)]
                                    
                                    if len(fragment) < segment_length:
                                        fragment = sequence[-48:]
                                    ret.append(fragment)
                            ret = torch.stack(ret)

                            #one-hot编码
                            c_inputs = torch.eye(4).long()[(ret - 1).long()]
                            c_inputs = c_inputs.float()
                            c_inputs = c_inputs.unsqueeze(1).to(device)
                                                    
                            c_outputs = CNN(c_inputs)
                            c_outputs = batch_mean(inputs, c_outputs)

                            outputs = model(ret)
                            outputs = batch_mean(inputs, outputs)

                        else:
                            c_inputs = torch.eye(4).long()[(inputs - 1).long()]
                            c_inputs = c_inputs.float()
                            c_inputs = c_inputs.unsqueeze(1).to(device)

                            outputs = model(inputs)
                            c_outputs = CNN(c_inputs)

                        outputs = torch.max(c_outputs, outputs)

                        outputs = sigmoid(outputs)
                        if outputs.shape[0] == 1:
                            y_pred = y_pred
                        else:
                            y_pred.extend(outputs.squeeze().tolist())
                    y_pred = torch.tensor(y_pred)
                    y_pred1 = torch.where(y_pred > parsers.threshold)[0]
                    
                return y_pred, y_pred1

            reads_score, read_indices, = predict_reads(model, CNN, test_loader)
            filter_reads(file_path, read_indices, parsers.out_folder)
            
            # Store predictions along with file name
            predictions.append((file_name, reads_score.tolist()))

    # Write predictions to a text file
    with open(parsers.output_file, 'w') as f:
        for seq_name, pred in predictions:
            f.write(f'{seq_name}\t{pred}\n')

process_fasta_files(parsers.in_folder)
