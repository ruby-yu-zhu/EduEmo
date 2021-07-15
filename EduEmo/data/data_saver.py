from torch.utils.data import Dataset
from transformers import BertTokenizer
from tqdm import tqdm
import torch
import pandas as pd
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor
import json
import run_segbot as segbot


def twitter_preprocessor():
    preprocessor = TextPreProcessor(
        normalize=['url', 'email', 'phone', 'user'],
        annotate={"hashtag", "elongated", "allcaps", "repeated", 'emphasis', 'censored'},
        all_caps_tag="wrap",
        fix_text=False,
        segmenter="twitter_2018",
        corrector="twitter_2018",
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True).tokenize).pre_process_doc
    return preprocessor


class DataClass(Dataset):
    def __init__(self, args, filename, savepath):
        self.args = args
        self.filename = filename
        self.savepath = savepath
        self.max_length = int(args['--max-length'])
        self.data, self.labels = self.load_dataset()

        if args['--lang'] == 'English':
            self.bert_tokeniser = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        elif args['--lang'] == 'Chinese':
            self.bert_tokeniser = BertTokenizer.from_pretrained("bert-base-chinese")
        self.inputs, self.lengths, self.label_indices, self.edu_token_lists = self.process_data()

    def load_dataset(self):
        """
        :return: dataset after being preprocessed and tokenised
        """
        df = pd.read_csv(self.filename, sep='\t')
        x_train, y_train = df.Tweet.values, df.iloc[:, 2:].values
        return x_train, y_train

    def lists2txt(self, lists,file_path):
        output = open(file_path,'w+')
        for i in range(len(lists)):
	        for j in range(len(lists[i])):
		        output.write(str(lists[i][j]))
		        output.write(' ')   
	        output.write('\n')      
        output.close()

    def list2txt(self, lists,file_path):
        output = open(file_path,'w+')
        for i in range(len(lists)):
            output.write(str(lists[i]))
            output.write(' ')       
        output.close()

    def process_data(self):
        desc = "PreProcessing dataset {}...".format('')
        preprocessor = twitter_preprocessor()

        if self.args['--lang'] == 'English':
            segment_a = "anger anticipation disgust fear joy love optimism hopeless sadness surprise or trust?"
            label_names = ["anger", "anticipation", "disgust", "fear", "joy",
                           "love", "optimism", "hopeless", "sadness", "surprise", "trust"]
        elif self.args['--lang'] == 'Chinese':
            segment_a = "乐恨爱悲忧惊怒欲"
            label_names = ["乐", "恨", "爱", "悲", "忧", "惊", "怒", "欲"]

        inputs, lengths, label_indices, edu_token_lists = [], [], [], []
        for x in tqdm(self.data, desc=desc):
            x = ' '.join(preprocessor(x))
            output_seg = segbot.main_input_output(x)
            token_index = []
            tmp_token_index=14 #Chinese: 9
            for ss in output_seg:
                tokens=self.bert_tokeniser.tokenize(ss)
                tmp_token_index=tmp_token_index+len(tokens)
                if ss==output_seg[-1]:
                    tmp_token_index=tmp_token_index+1
                token_index.append(tmp_token_index)
            x = self.bert_tokeniser.encode_plus(segment_a,
                                                x,
                                                add_special_tokens=True,
                                                max_length=self.max_length,
                                                pad_to_max_length=True,
                                                truncation=True)
            input_id = x['input_ids']
            input_length = len([i for i in x['attention_mask'] if i == 1])
            inputs.append(input_id)
            lengths.append(input_length)
            edu_token_lists.append(token_index)
            #label indices
            label_idxs = [self.bert_tokeniser.convert_ids_to_tokens(input_id).index(label_names[idx])
                             for idx, _ in enumerate(label_names)]
            label_indices.append(label_idxs)
        
        self.lists2txt(inputs, self.savepath+'inputs.txt')
        self.list2txt(lengths, self.savepath+'data_length.txt')
        self.lists2txt(label_indices, self.savepath+'label_indices.txt')
        self.lists2txt(edu_token_lists, self.savepath+'edu_token_lists.txt')
        inputs = torch.tensor(inputs, dtype=torch.long)
        data_length = torch.tensor(lengths, dtype=torch.long)
        label_indices = torch.tensor(label_indices, dtype=torch.long)
        
        return inputs, data_length, label_indices, edu_token_lists

    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        label_idxs = self.label_indices[index]
        length = self.lengths[index]
        edu_token_list = self.edu_token_lists[index]
        return inputs, labels, length, label_idxs, edu_token_list

    def __len__(self):
        return len(self.inputs)
