from numpy.core.fromnumeric import nonzero
import torch
from tqdm import tqdm
from collections import Counter 
from sklearn.feature_extraction.text import TfidfVectorizer
from dataclasses import dataclass
from copy import copy

def build_image_lookup_table(keyword_ls):
    lookup_table = {}
    for idx,keywords in enumerate(keyword_ls):
        for word in keywords:
            if word in lookup_table:
                lookup_table[word].append(idx)
            else:
                lookup_table[word] = [idx,]
    return lookup_table

def build_keyword_net(keyword_ls):
    keyword_net = {}
    for keywords in keyword_ls:
        for idx in range(len(keywords)):
            for jdx in range(idx+1,len(keywords)):
                
                pair = tuple(sorted([keywords[idx],keywords[jdx]]))
                if pair in keyword_net:
                    keyword_net[pair]+=1
                else:
                    keyword_net[pair]=1
    #print(f"keyword cooccurrence: {len(keyword_net)}")
    return keyword_net

def get_txt(f):
    return [x.rstrip('\n') for x in open(f).readlines()]

def build_tfidf_vec(src_tokenizer,data_args):

    train_src = get_txt(data_args.train_src_dir)
    stop_word_ls = [x.strip('\n') for x in open(data_args.stop_word_dir).readlines()]

    ## get tf-idf
    tfidf_vec = TfidfVectorizer(tokenizer=src_tokenizer.split)
    corpus = []
    for s in train_src:
        s = s.split(" ")
        for w in s[:]:
            if w in stop_word_ls:s.remove(w)
        corpus.append(" ".join(s))
    tfidf_matrix = tfidf_vec.fit_transform(corpus)


    keyword_ls = []
    non_zero = tfidf_matrix.nonzero()
    pre = 0
    temp_ls = []
    tfidf_vec_id2w = {v:k for k,v in tfidf_vec.vocabulary_.items()}
    for i in range(len(non_zero[0])):
        if non_zero[0][i] == pre:
            temp_ls.append((tfidf_vec_id2w[non_zero[1][i]],tfidf_matrix[non_zero[0][i],non_zero[1][i]]))
        else:
            pre = non_zero[0][i]
            temp_ls.sort(key=lambda x:x[1],reverse=True)
            keyword_ls.append([x[0] for x in temp_ls])
            temp_ls.clear()
    keyword_ls = [x[:data_args.w] for x in keyword_ls]
    
    return tfidf_vec,keyword_ls

def get_top_m_keywords(s,keyword_net,tfidf_vec,m=5,w2ww=None):
    keywords = get_keywords(s,tfidf_vec)
    co_keyword_ls = []
    for w in keywords:
        if w in w2ww:
            pairs = w2ww[w]
            for pair in pairs:
                co_keyword = pair[0] if pair[0]!= w else pair[1]
                co_keyword_ls.extend([co_keyword]*keyword_net[pair])
    counter = Counter(co_keyword_ls)
    top_w_keywords = " ".join([x[0] for x in counter.most_common(m)])
    return top_w_keywords

def get_top_m_images(s,image_lookup_table,tfidf_vec,m=5):

    keywords = get_keywords(s,tfidf_vec)
    image_ls = []
    for w in keywords:
        if w in image_lookup_table:
            image_ls.extend(image_lookup_table[w])
    counter = Counter(image_ls)
    top_m_images = [x[0] for x in counter.most_common(m)]
    return top_m_images

def get_keywords(s,tfidf_vec,w=8):
    """
    input:
        s:string
    """
    matrix = tfidf_vec.transform([s])
    non_zero = matrix.nonzero()
    tfidf_id2w = tfidf_vec.get_feature_names_out()
    ret = []
    for i in range(len(non_zero[0])):
        ret.append((tfidf_id2w[non_zero[1][i]],matrix[non_zero[0][i],non_zero[1][i]]))
    # for idx,x in enumerate(a):
    #     if x !=0:
    #         ret.append((tfidf_id2w[idx],x))
    ret.sort(key=lambda x:x[1],reverse=True)
    ret = [x[0] for x in ret[:w]]
    return ret

def build_dataset(src,trg,src_tokenizer,trg_tokenizer,data_args,tfidf_vec,image_lookup_table=None,keyword_net=None):

    ret = []
    w2ww = {}
    if keyword_net is not None:
        for k in keyword_net.keys():
            w1 = k[0]
            w2 = k[1]
            if w1 in w2ww:w2ww[w1].append(k)
            else:w2ww[w1] = [k]
            if w2 in w2ww:w2ww[w2].append(k)
            else:w2ww[w2]=[k]

    # for s,t in tqdm(zip(src,trg),total=len(src)):
    for s,t in zip(src,trg):
        if image_lookup_table is not None:
            image = get_top_m_images(s,image_lookup_table,tfidf_vec,data_args.m)
            t = {
                "src":src_tokenizer.encode(s),
                'trg':trg_tokenizer.encode(t),
                'images':image,
            }
            ret.append(t)
        
        elif keyword_net is not None:

            keywords = get_top_m_keywords(s,keyword_net,tfidf_vec,data_args.m,w2ww)
            t = {
                "src":src_tokenizer.encode(s),
                'trg':trg_tokenizer.encode(t),
                'keywords':src_tokenizer.encode(keywords),
            }
            ret.append(t)
        
        else:
            t = {
                "src":src_tokenizer.encode(s),
                'trg':trg_tokenizer.encode(t),}
            ret.append(t)

    return ret

def get_dataset(data_args):

    train_src = get_txt(data_args.train_src_dir)#[:1000]
    train_trg = get_txt(data_args.train_trg_dir)
    
    ## get src vocab
    l = []
    for x in [x.split(" ") for x in train_src]:l.extend(x)
    counter = Counter(l)
    src_id2w = [x for x in counter if counter[x]>=data_args.min_freq]
    src_tokenizer = Tokenizer(src_id2w)
    #print(f"Src vocab:{len(src_id2w)}")

    ## get trg vocab
    l = []
    for x in [x.split(" ") for x in train_trg]:l.extend(x)
    counter = Counter(l)
    trg_id2w = [x for x in counter if counter[x]>=data_args.min_freq]
    trg_tokenizer = Tokenizer(trg_id2w)
    #print(f"Trg vocab:{len(trg_id2w)}")

    dev_src = get_txt(data_args.dev_src_dir)
    dev_trg = get_txt(data_args.dev_trg_dir)

    # if data_args.trg_lang == ''
    # test_src = get_txt(data_args.test_src_dir)
    # test_trg = get_txt(data_args.test_trg_dir)
    
    tfidf_vec,keyword_ls = build_tfidf_vec(src_tokenizer,data_args)
    image_lookup_table,keyword_net = None,None
    if data_args.use_image:
        
        image_lookup_table = build_image_lookup_table(keyword_ls)

        train_dataset = Dataset(build_dataset(train_src,train_trg,src_tokenizer,trg_tokenizer,data_args,tfidf_vec,image_lookup_table=image_lookup_table))
        dev_dataset = Dataset(build_dataset(dev_src,dev_trg,src_tokenizer,trg_tokenizer,data_args,tfidf_vec,image_lookup_table=image_lookup_table))
        # test_dataset = Dataset(build_dataset(test_src,test_trg,src_tokenizer,trg_tokenizer,data_args,tfidf_vec,image_lookup_table=image_lookup_table))

    
    elif data_args.use_keyword:
        
        keyword_net = build_keyword_net(keyword_ls)
        train_dataset = Dataset(build_dataset(train_src,train_trg,src_tokenizer,trg_tokenizer,data_args,tfidf_vec,keyword_net=keyword_net))
        dev_dataset = Dataset(build_dataset(dev_src,dev_trg,src_tokenizer,trg_tokenizer,data_args,tfidf_vec,keyword_net=keyword_net))
        # test_dataset = Dataset(build_dataset(test_src,test_trg,src_tokenizer,trg_tokenizer,data_args,tfidf_vec,keyword_net=keyword_net))
    
    else:

        train_dataset = Dataset(build_dataset(train_src,train_trg,src_tokenizer,trg_tokenizer,data_args,tfidf_vec))
        dev_dataset = Dataset(build_dataset(dev_src,dev_trg,src_tokenizer,trg_tokenizer,data_args,tfidf_vec))
        # test_dataset = Dataset(build_dataset(test_src,test_trg,src_tokenizer,trg_tokenizer,data_args,tfidf_vec))

    return train_dataset,dev_dataset,src_tokenizer,trg_tokenizer,tfidf_vec,image_lookup_table,keyword_net

@dataclass
class DataCollator:
    
    pad_token_id:int = 1
    device:int = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __call__(self,features):
        """
        features: list of dict
        this funtion is for padding
        """
        batch_size = len(features)

        src = [x['src'] for x in features]
        trg = [x['trg'] for x in features]

        labels = [x[1:] for x in trg]
        decoder_input_ids = [x[:-1] for x in trg]

        # pad src
        max_src_len = max(len(x) for x in src)
        attention_mask = [[1]*len(x) + [0]*(max_src_len-len(x)) for x in src]
        src = [x+[self.pad_token_id]*(max_src_len-len(x)) for x in src]

        # pad labels and decoder_input_ids        
        max_trg_len = max(len(x) for x in labels)
        labels = [x+[-100]*(max_trg_len-len(x)) for x in labels]
        decoder_input_ids = [x+[self.pad_token_id]*(max_trg_len-len(x)) for x in decoder_input_ids]

        ## pad image or keywords
        use_extra = 'images' in features[0].keys() or 'keywords' in features[0].keys()
        if use_extra:
            if "images" in features[0].keys():
                extra = [x['images'] for x in features]
            elif 'keywords' in features[0].keys():
                extra = [x['keywords'] for x in features]

            max_extra_len = max(len(x) for x in extra)
            # extra_attention_mask = [[1]*len(x) + [0]*(max_extra_len-len(x)) for x in extra]
            extra = [x+[self.pad_token_id]*(max_extra_len-len(x)) for x in extra]
            ret =  {
                'input_ids':torch.tensor(src), # b,l
                'attention_mask':torch.tensor(attention_mask),# b,l
                'decoder_input_ids':torch.tensor(decoder_input_ids), # b,l
                'labels':torch.tensor(labels), # b,l
                'extra':torch.tensor(extra), #b,l
                # "extra_attention_mask":torch.tensor(extra_attention_mask) #b,l
                }
        else:
            ret =  {
                'input_ids':torch.tensor(src), # b,l
                'attention_mask':torch.tensor(attention_mask),# b,l
                'decoder_input_ids':torch.tensor(decoder_input_ids), # b,l
                'labels':torch.tensor(labels), # b,l
                }
        
        for k in ret:
            ret[k] = ret[k].to(self.device)
        return ret

class Tokenizer:
    def __init__(self,id2w):
        
        ## add bos,eos,unk,pad
        id2w.insert(0,'<sep>') #4
        id2w.insert(0,'<bos>') #3
        id2w.insert(0,'<eos>') #2
        id2w.insert(0,'<pad>') #1
        id2w.insert(0,'<unk>') #0
        
        self.special_token_ls = ['<bos>','<unk>','<eos>','<pad>','<sep>']
        self.id2w = id2w
        self.w2id = {x:idx for idx,x in enumerate(id2w)}
        
        self.pad_token_id = self.w2id['<pad>']
        self.bos_token_id = self.w2id['<bos>']
        self.eos_token_id = self.w2id['<eos>']
        self.unk_token_id = self.w2id['<unk>']

    def __len__(self):
        return len(self.id2w)

    def encode(self,s):
        s_ls = s.split(" ")
        ret = []
        ret.append(self.bos_token_id)
        for w in s_ls:
            ret.append(self.w2id.get(w,self.unk_token_id))
        ret.append(self.eos_token_id)
        return ret

    def split(self,s):
        s_ls = s.split(" ")
        ret = []
        for w in s_ls:
            if w in self.w2id.keys():ret.append(w)
        return ret

    def encode_batch(self,list_of_s):
        ret = []
        for s in list_of_s:
            ret.append(self.encode(s))
        return ret

    def decode(self,s):
        ret = []
        for idx in s:
            token = self.id2w[idx]
            if token not in self.special_token_ls:
                ret.append(token)
        return " ".join(ret)

    def decode_batch(self,list_of_s,skip_special_tokens=True):
        ret = []
        for s in list_of_s:
            ret.append(self.decode(s))
        return ret
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self,dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,index):
        return self.dataset[index]
