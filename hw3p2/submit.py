import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from os.path import join
from sklearn.metrics import accuracy_score
import gc
import zipfile
import pandas as pd
from tqdm import tqdm
import os
import datetime
import phonemes
# imports for decoding and distance calculation
import ctcdecode
import Levenshtein
from ctcdecode import CTCBeamDecoder
import csv
import time
import warnings
from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)
# !jupyter nbextension enable --py widgetsnbextension


# PHONEME_MAP is the list that maps the phoneme to a single character. 
# The dataset contains a list of phonemes but you need to map them to their corresponding characters to calculate the Levenshtein Distance
# You final submission should not have the phonemes but the mapped string
# No TODOs in this cell
num_classes = 41
PHONEME_MAP = [
    " ",
    ".", #SIL
    "a", #AA
    "A", #AE
    "h", #AH
    "o", #AO
    "w", #AW
    "y", #AY
    "b", #B
    "c", #CH
    "d", #D
    "D", #DH
    "e", #EH
    "r", #ER
    "E", #EY
    "f", #F
    "g", #G
    "H", #H
    "i", #IH 
    "I", #IY
    "j", #JH
    "k", #K
    "l", #L
    "m", #M
    "n", #N
    "N", #NG
    "O", #OW
    "Y", #OY
    "p", #P 
    "R", #R
    "s", #S
    "S", #SH
    "t", #T
    "T", #TH
    "u", #UH
    "U", #UW
    "v", #V
    "W", #W
    "?", #Y
    "z", #Z
    "Z" #ZH
]
phe_dict = {}
tensor_dict = {}
PHONEMES = phonemes.PHONEMES
for idx, i in enumerate(PHONEMES):
    phe_dict[i] = PHONEME_MAP[idx]
    tensor_dict[PHONEME_MAP[idx]] = idx

def maplst(lst):
    res =[]
    for i in lst:
        res.append(phe_dict[i])
    res = np.array(res)
    # res = res.astype(np.float)
    return np.array(res)
def maptotensor(lst):
    res =[]
    for i in lst:
        res.append(tensor_dict[i])
    res = np.array(res)
    # res = res.astype(np.float)
    return np.array(res)

# This cell is where your actual TODOs start
# You will need to implement the Dataset class by your own. You may also implement it similar to HW1P2 (dont require context)
# The steps for implementation given below are how we have implemented it.
# However, you are welcomed to do it your own way if it is more comfortable or efficient. 

class LibriSamples(torch.utils.data.Dataset):

    def __init__(self, data_path, partition= "train"): # You can use partition to specify train or dev

        self.X_dir = os.path.join(data_path,partition,"mfcc/")# TODO: get mfcc directory path
        self.Y_dir = os.path.join(data_path,partition,"transcript/")# TODO: get transcript path

        self.X_files = os.listdir(self.X_dir)# TODO: list files in the mfcc directory
        self.Y_files = os.listdir(self.Y_dir)# TODO: list files in the transcript directory

        # TODO: store PHONEMES from phonemes.py inside the class. phonemes.py will be downloaded from kaggle.
        # You may wish to store PHONEMES as a class attribute or a global variable as well.
        self.PHONEMES = phonemes.PHONEMES

        assert(len(self.X_files) == len(self.Y_files))

        pass

    def __len__(self):
        return len(self.X_files)

    def __getitem__(self, ind):
        X_path = self.X_dir + self.X_files[ind]
        Y_path = self.Y_dir + self.Y_files[ind]
        X = torch.Tensor(np.load(X_path))# TODO: Load the mfcc npy file at the specified index ind in the directory
        # Y = maplst(np.load(Y_path)[1:-1])# TODO: Load the corresponding transcripts
        Y = np.load(Y_path)[1:-1]
        # print(Y)
        # print(Y.shape)
        # Y2 = PHONEMES.index(i) for i in np.load(Y_path)[1:-1]
        # print(f"Y {Y}")
        # print(f"Y2 {Y2}")
        # Remember, the transcripts are a sequence of phonemes. Eg. np.array(['<sos>', 'B', 'IH', 'K', 'SH', 'AA', '<eos>'])
        # You need to convert these into a sequence of Long tensors
        # Tip: You may need to use self.PHONEMES
        # Remember, PHONEMES or PHONEME_MAP do not have '<sos>' or '<eos>' but the transcripts have them. 
        # You need to remove '<sos>' and '<eos>' from the trancripts. 
        # Inefficient way is to use a for loop for this. Efficient way is to think that '<sos>' occurs at the start and '<eos>' occurs at the end.
        Yy = torch.LongTensor([PHONEMES.index(i) for i in Y])
        # Yy = torch.Tensor(maptotensor(Y)).type(torch.LongTensor)# TODO: Convert sequence of  phonemes into sequence of Long tensors
        # print(f"X {X.shape}")
        # print(f"Y {Y.shape}")
        return X, Yy
    
    def collate_fn(self,batch):

        batch_x = [x for x,y in batch]
        batch_y = [y for x,y in batch]
        # print(batch_x[0].shape)

        new_lst = []
        for idx, i in enumerate(batch_x):
            new_lst.append(batch_x[idx])
        batch_x_pad = pad_sequence(new_lst)
        # batch_x_pad = pad_sequence([i for i in batch_x], batch_first=False)# TODO: pad the sequence with pad_sequence (already imported)
        lengths_x = [len(i) for i in batch_x]# TODO: Get original lengths of the sequence before padding
        lengths_x_pad = [len(i) for i in batch_x_pad]
        # print(f"batch_x {batch_x}")
        

        # print(f"test_pad.len {test_pad.shape}")
        # print(f"batch_x.len {len(batch_x)}")
        # print(f"lengths_x {lengths_x}")
        # print(f"lengths_x_pad {lengths_x_pad}")

        new_lst = []
        for idx, i in enumerate(batch_y):
            new_lst.append(batch_y[idx])
        batch_y_pad = pad_sequence(new_lst)

        # batch_y_pad = pad_sequence(batch_y) # TODO: pad the sequence with pad_sequence (already imported)
        lengths_y = [len(i) for i in batch_y] # TODO: Get original lengths of the sequence before padding

        return batch_x_pad, batch_y_pad, torch.tensor(lengths_x), torch.tensor(lengths_y)
from torch.utils.data.dataset import Subset

# You can either try to combine test data in the previous class or write a new Dataset class for test data
class LibriSamplesTest(torch.utils.data.Dataset):

    def __init__(self, data_path, test_order): # test_order is the csv similar to what you used in hw1
        self.data_path = data_path
        test_csv_pth = os.path.join(data_path,'test',test_order)
        subset = list(pd.read_csv(test_csv_pth).file)
        # subset = self.parse_csv(test_csv_pth)
        test_order_list = subset# TODO: open test_order.csv as a list
        self.X_names = [i for i in subset]# TODO: Load the npy files from test_order.csv and append into a list
        # You can load the files here or save the paths here and load inside __getitem__ like the previous class
    @staticmethod
    def parse_csv(filepath):
        subset = []
        with open(filepath) as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                subset.append(row[0])
        return subset[0:]
    def __len__(self):
        return len(self.X_names)
    
    def __getitem__(self, ind):
        # TODOs: Need to return only X because this is the test dataset
        X_path = os.path.join(self.data_path,'test','mfcc',self.X_names[ind])
        X = torch.Tensor(np.load(X_path))
        return X
    
    def collate_fn(self, batch):
        batch_x = [x for x in batch]
        new_lst = []
        for idx, i in enumerate(batch_x):
            new_lst.append(batch_x[idx])
        batch_x_pad = pad_sequence(new_lst)
        # batch_x_pad = pad_sequence([i for i in batch_x], batch_first=False)# TODO: pad the sequence with pad_sequence (already imported)
        lengths_x = [len(i) for i in batch_x]# TODO: Get original lengths of the sequence before padding
        # lengths_x_pad = [len(i) for i in batch_x_pad]
        # batch_x = [x for x in batch]
        # batch_x_pad = pad_sequence(batch_x)# TODO: pad the sequence with pad_sequence (already imported)
        # lengths_x = [len(i) for i in batch_x]# TODO: Get original lengths of the sequence before padding

        return batch_x_pad, torch.tensor(lengths_x)

# # Optional
# # Test code for checking shapes and return arguments of the train and val loaders


# for data in train_loader:
#     x, y, lx, ly = data # if you face an error saying "Cannot unpack", then you are not passing the collate_fn argument
#     # print(f"ly {ly}")
#     # print(f"lx {lx}")
#     print(x.shape, y.shape, lx.shape, ly.shape)
#     break

class Network(nn.Module):

    def __init__(self): # You can add any extra arguments as you wish

        super(Network, self).__init__()

        # Embedding layer converts the raw input into features which may (or may not) help the LSTM to learn better 
        # For the very low cut-off you dont require an embedding layer. You can pass the input directly to the  LSTM
        # self.embedding = 
        
        self.lstm = nn.LSTM(input_size=13,hidden_size= 256, num_layers= 1)# TODO: # Create a single layer, uni-directional LSTM with hidden_size = 256
        # Use nn.LSTM() Make sure that you give in the proper arguments as given in https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

        self.classification = nn.Linear(256,num_classes)# TODO: Create a single classification layer using nn.Linear()

    def forward(self, x, lx): # TODO: You need to pass atleast 1 more parameter apart from self and x

        # x is returned from the dataloader. So it is assumed to be padded with the help of the collate_fn
        packed_input = pack_padded_sequence(x,lx,enforce_sorted=False)# TODO: Pack the input with pack_padded_sequence. Look at the parameters it requires

        out1, (out2, out3) = self.lstm(packed_input)# TODO: Pass packed input to self.lstm
        # As you may see from the LSTM docs, LSTM returns 3 vectors. Which one do you need to pass to the next function?
        out, lengths  = pad_packed_sequence(out1)# TODO: Need to 'unpack' the LSTM output using pad_packed_sequence

        out = self.classification(out)# TODO: Pass unpacked LSTM output to the classification layer
        # out = # Optional: Do log softmax on the output. Which dimension?
        # print(out[0,0,:])
        out = torch.nn.functional.log_softmax(out,dim=2)
        # print(out[0,0,:])
        # print(sum(out[0,0,:]))
        return out, lengths # TODO: Need to return 2 variables

model = Network().to(device)
print(model)


def calculate_levenshtein(h, y, lh, ly, decoder, PHONEME_MAP):

    # h - ouput from the model. Probability distributions at each time step 
    # y - target output sequence - sequence of Long tensors
    # lh, ly - Lengths of output and target
    # decoder - decoder object which was initialized in the previous cell
    # PHONEME_MAP - maps output to a character to find the Levenshtein distance

    h = h.permute(1, 0, 2)# TODO: You may need to transpose or permute h based on how you passed it to the criterion
    # Print out the shapes often to debug
    t1 =time.time()
    beam_results, _, _, out_lens = decoder.decode(h,seq_lens=lh)
    t2 = time.time()
    # print(f"time cost {t2-t1}")
    # TODO: call the decoder's decode method and get beam_results and out_len (Read the docs about the decode method's outputs)
    # Input to the decode method will be h and its lengths lh 
    # You need to pass lh for the 'seq_lens' parameter. This is not explicitly mentioned in the git repo of ctcdecode.

    batch_size = h.shape[0]# TODO
    # print(f"batch_szie {batch_size}")

    dist = 0

    # dist = 0
    # h = np.zeros((100,))  
    # y = y.cpu().detach().numpy().astype(int)
    for i in range(batch_size): # Loop through each element in the batch

    # for j in range(100)
        h_sliced = beam_results[i][0][:out_lens[i,0]]
    # print(h_sliced.shape)
        # TODO: Get the output as a sequence of numbers from beam_results
        # Remember that h is padded to the max sequence length and lh contains lengths of individual sequences
        # Same goes for beam_results and out_lens
        # You do not require the padded portion of beam_results - you need to slice it with out_lens 
        # If it is confusing, print out the shapes of all the variables and try to understand

        h_string = "".join([PHONEME_MAP[j] for j in h_sliced])# TODO: MAP the sequence of numbers to its corresponding characters with PHONEME_MAP and merge everything as a single string
        # print(f"ly.shape {ly.shape}")
        # print(f"y.shape {y.shape}")
        y_sliced = y[i][:ly[i]]
        y_string = "".join([PHONEME_MAP[j] for j in y_sliced])# TODO: MAP the sequence of numbers to its corresponding characters with PHONEME_MAP and merge everything as a single string
        # print(f"h_string {h_string} h_string.len {len(h_string)}")
        # print(f"y_string {y_string} y_string.len {len(y_string)}")
        per_dist = Levenshtein.distance(h_string, y_string)
        # print(f"{i} {per_dist} ")
        dist += per_dist

    dist/=batch_size
    return dist
    # print(f"dist {dist}")



lr = 2e-3
batch_size = 256
epochs = 40
num_classes = 41
wandb.config = {
  "learning_rate": lr,
  "epochs": epochs,
  "batch_size": batch_size
}

root = 'hw3p2_student_data/hw3p2_student_data' # TODO: Where your hw3p2_student_data folder is

train_data = LibriSamples(root, 'train')
val_data = LibriSamples(root, 'dev')
test_data = LibriSamplesTest(root, 'test_order.csv')

train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=False,collate_fn=train_data.collate_fn, num_workers=8)# TODO: Define the train loader. Remember to pass in a parameter (function) for the collate_fn argument 
val_loader = DataLoader(val_data,batch_size=batch_size,shuffle=False,collate_fn=val_data.collate_fn, num_workers=8)# TODO: Define the val loader. Remember to pass in a parameter (function) for the collate_fn argument 
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=False,collate_fn=test_data.collate_fn, num_workers=8)# TODO: Define the test loader. Remember to pass in a parameter (function) for the collate_fn argument 

print("Batch size: ", batch_size)
print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))

criterion = nn.CTCLoss()# TODO: What loss do you need for sequence to sequence models? 
# Do you need to transpose or permute the model output to find out the loss? Read its documentation
optimizer = torch.optim.Adam(model.parameters(),lr=lr)# TODO: Adam works well with LSTM (use lr = 2e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * epochs))

decoder = ctcdecode.CTCBeamDecoder(labels = PHONEME_MAP,
    model_path=None,
    alpha=0,
    beta=0,
    cutoff_top_n=40,
    cutoff_prob=1.0,
    beam_width=3,
    num_processes=8,
    blank_id=0,
    log_probs_input=True)# TODO: Intialize the CTC beam decoder
# Check out https://github.com/parlance/ctcdecode for the details on how to implement decoding
# Do you need to give log_probs_input = True or False?

def pred(h,lh,decoder,PHONEME_MAP):
    h = h.permute(1, 0, 2)
    beam_results, _, _, out_lens = decoder.decode(h,seq_lens=lh)
    h_string_lst = []
    batch_size = h.shape[0]
    for i in range(batch_size): 

        h_sliced = beam_results[i][0][:out_lens[i,0]]
        h_string = "".join([PHONEME_MAP[j] for j in h_sliced])
        h_string_lst.append(h_string)
    return h_string_lst

# TODO: Write your model evaluation code for the test dataset
# You can write your own code or use from the previous homewoks' stater notebooks
# You can't calculate loss here. Why?
def submit_test(model):
    model.eval()
    true_y_list = []
    pred_y_list = []
    with torch.no_grad():
        for i in range(1):
            # X = test_samples[i]

            # test_items = test_item(X, context=args['context'])
            # test_loader = torch.utils.data.DataLoader(test_items, batch_size=args['batch_size'], shuffle=False)

            for x, lx in test_loader:
                # data = data.float().to(device)
                x = x.cuda()
                # y = y.cuda()
                out, out_len = model(x,lx)
                pred_y = pred(out,out_len,decoder,PHONEME_MAP)
                # print(out.shape)
                # pred_y = torch.argmax(out, axis=2)
                # print(pred_y.shape)
                pred_y_list.extend(pred_y)
    # print(pred_y_list)
    # now_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    f = open(f"bad.csv", "w")
    f.write("id,predictions\n")
    for idx, i  in enumerate(pred_y_list):
        f.write(f"{idx},{i}\n")
    f.close()

model.load_state_dict(torch.load("7_1.0379955768585205_25_03_2022_22_30_06model.pth"))

submit_test(model)

! kaggle competitions submit -c 11-785-s22-hw3p2 -f bad.csv -m "Message"

    # with open('good.csv', 'w', newline='') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames = ['id','label'])
    #     # writer.writerow(['id','label'])
    #     writer.writeheader() 
    #     for idx, i  in enumerate(pred_y_list):
    #         writer.writerow([idx,i])
        
    
    
