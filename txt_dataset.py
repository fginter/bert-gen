import sys
import torch
import transformers
import itertools
import random
import os

###############
# Related to the dataset we work with which is split to a number of
# text files, whose name codes the source (news, discussions, etc)

def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = itertools.cycle(itertools.islice(nexts, num_active))

def balance_filenames(fnames):
    """
    Sorts filenames by the first 3 chars of their basename in an interleaving manner
    this allows balance between different data sources, if they are coded in filenames
    """
    prefixes={}
    for fname in fnames:
        prefixes.setdefault(os.path.basename(fname)[:3],[]).append(fname)
    min_len=min(len(lst) for lst in prefixes.values())
    final_lists=[]
    for prefix,lst in prefixes.items():
        lst=lst[:min_len]
        random.shuffle(lst)
        final_lists.append(lst)
    filenames=list(roundrobin(*final_lists))
    return filenames


def docs_from_plaintext_files(file_names,max_doc=50):
    """
    Consumes inp which is one sentence per line, empty line between documents. 
    Yields the documents as lists of sentences.

    max_doc: how many documents to read from the input? None: everything
    """
    for f_name in file_names:
        with open(f_name,"rt") as inp:
            yielded=0
            current_doc=[]
            for line in inp:
                line=line.strip()
                if not line and current_doc:
                    yield current_doc
                    yielded+=1
                    if max_doc and yielded>=max_doc:
                        break
                    current_doc=[]
                    continue
                current_doc.append(line)
            else:
                if current_doc:
                    yield current_doc


def shuffle_stream(s,buff=3000):
    """
    Yield data from s in a random order - maintains and shuffles a buffer
    so setting buff=3000 means we read 3000 items, shuffle, yield - and then repeat
    for another 3000. Etc.

    s: anything you can iterate over
    """
    data=(x for x in s) #make this into generator so the islice does the right thing, consuming
    while True:
        buffer=list(itertools.islice(data,0,buff))
        if not buffer:
            break
        random.shuffle(buffer)
        yield from buffer    

  
def document_examples(documents,tokenizer,min_trigger=40,max_trigger=50,max_length=60):
    """
    yields examples from one document at a time, each being a "minibatch" of training examples produced from one text
    span, these can be later stitched into the actual training minibatches by a separate process

    documents: iterable over documents
    min_trigger: minimum number of tokens which act as a trigger
    max_trigger: maximum number of tokens which act as a trigger
    max_len: maximum length of generated examples (including the trigger)
    """
    for document in documents:
        tokenized_sentences=[tokenizer.tokenize(sentence) for sentence in document]
        ids=[tokenizer.convert_tokens_to_ids(tokenized_sentence) for tokenized_sentence in tokenized_sentences]
        ids=[torch.tensor(token_ids,dtype=torch.long) for token_ids in ids] #list of tensors, each being a sequence of token ids in a sentence
        ids=torch.cat(ids)
        document_len=len(ids)
        current_index=0 #this is where we currently are, and will start generating
        while True:
            trigger_len=random.randint(min_trigger,max_trigger)
            remains_in_document=document_len-current_index
            if remains_in_document<=trigger_len: #not enough text left in this document
                break
            end_index=min(current_index+max_length,document_len)
            trigger=ids[current_index:current_index+trigger_len]
            pred_target=ids[current_index+trigger_len:end_index]
            yield sentence_example(trigger,pred_target,tokenizer)
            current_index=current_index+max_length #jump to the end of the current example, so we have no overlap
        
        
def examples(fnames,tokenizer,min_trigger=10,max_trigger=60,max_length=80,max_doc_per_file=3000,shuffle_buff=30000):
    balanced_fnames=balance_filenames(fnames)
    print("BALANCED NAMES:",len(balanced_fnames))
    docs_balanced=docs_from_plaintext_files(balanced_fnames,max_doc=max_doc_per_file)
    docs_balanced_shuffled=shuffle_stream(docs_balanced,buff=shuffle_buff)
    for e in document_examples(docs_balanced_shuffled,tokenizer,min_trigger=min_trigger,max_trigger=max_trigger,max_length=max_length):
        yield e


####################
##

def sentence_example(trigger_ids,sent_ids,tokenizer):
    """
    OOO
    """
    CLS,SEP,MASK,PAD,UNUSED0=tokenizer.convert_tokens_to_ids(["[CLS]","[SEP]","[MASK]","[PAD]","[unused0]"])
    # len=7
    #   0   1 2 3 4 5 6
    # [cls] a b c d e f

    #how many positions can I predict
    to_predict=sent_ids.shape[0]
    inputs_gold=sent_ids.unsqueeze(0).expand((to_predict,-1))
    inputs=inputs_gold.clone() #copies the sentence "to_predict" times
    #extra_col=torch.full((to_predict,1),PAD,dtype=torch.long) #we'll need an extra c
    #inputs_xtra_col=torch.cat((inputs,extra_col),-1) #...I'll refer to this later
    inputs=torch.tril(inputs) #only keep the values below the diagonal, this is what the transformer can see
    diag=torch.arange(to_predict) #diagonal indices
    inputs[diag,diag]=MASK #main diagonal is masked, this is what the transformer must not see
    #inputs[diag,diag+1]=MASK #and the diagonal above that is filled with another MASK token, telling this is where the seqs end
    all_pad=torch.tensor([[PAD]],dtype=torch.long).expand((to_predict,to_predict)) #
    inputs+=torch.triu(all_pad,1) #upper triangle from second diagonal up is just padding
    # 999 is mask, 666 is SEP and 888 is PAD, [4,5,6,7] was the sequence
    #tensor([[999, 888, 888, 888],
    #        [  4, 999, 888, 888],
    #        [  4,   5, 999, 888],
    #        [  4,   5,   6, 999]])

    triggers=trigger_ids.unsqueeze(0).expand((to_predict,-1)).clone() #copies the trigger "to_predict" times
    #everything starts with CLS and SEP
    #triggers=torch.cat((torch.full((to_predict,1),CLS,dtype=torch.long),torch.full((to_predict,1),SEP,dtype=torch.long),triggers),-1)
    triggers=torch.cat((torch.full((to_predict,1),CLS,dtype=torch.long),triggers),-1) #try without sep
        
    #what we need now is the masks for the attention
    inp_mask=torch.tril(torch.ones(inputs.shape,dtype=torch.long))

    #let's try right-aligned position embeddings
    max_len=128
    triggers.shape[1]
    x=torch.arange(max_len-triggers.shape[-1]-to_predict,max_len+1).unsqueeze(0).expand((to_predict,-1))
    off=torch.arange(to_predict-1,-1,-1).unsqueeze(-1)
    position_indices=torch.tril(x+off,triggers.shape[1]+1)
    #Tensor([[122, 123, 124, 125, 126, 127, 128,   0,   0,   0,   0,   0,   0],
    #       [121, 122, 123, 124, 125, 126, 127, 128,   0,   0,   0,   0,   0],
    #       [120, 121, 122, 123, 124, 125, 126, 127, 128,   0,   0,   0,   0],
    #       [119, 120, 121, 122, 123, 124, 125, 126, 127, 128,   0,   0,   0],
    #       [118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,   0,   0],
    #       [117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,   0],
    #       [116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128]])

    
    #tensor([[1., 1., 0., 0., 0.],
    #    [1., 1., 1., 0., 0.],
    #    [1., 1., 1., 1., 0.],
    #    [1., 1., 1., 1., 1.]])

    #prepend ones for the trigger part of the sentence
    attention_mask=torch.cat((torch.ones(triggers.shape,dtype=torch.long),inp_mask),-1) #and of course thr trigger we can see
    
    
    # #and finally, we need the gold output with all tokens masked as -1 except the ones we are supposed to be predicting
    # #this is needed for the loss later on, so we do not train on tokens the model can trivially see
    diagonal=(inputs_gold+1).triu().tril()-1  #the +1 -1 thing just makes sure that masking is -1 not zero
    #tensor([[ 4, -1, -1, -1, -1],
    #        [-1,  5, -1, -1, -1],
    #        [-1, -1,  6, -1, -1],
    #        [-1, -1, -1,  7, -1]])
    
    gold=torch.cat((torch.full(triggers.shape,-1,dtype=torch.long),diagonal),-1) #...and all the inputs are masked at -1


    triggers_and_inputs=torch.cat((triggers,inputs),-1)
    #and now we have all we need for this sentence
    return triggers_and_inputs, attention_mask, position_indices, gold


#####################
###

def blocks2batch(blocks,padding_value):
    #blocks are Batch x Len (examples from one sentence)
    #they need to be stacked vertically
    import torch
    max_len=max(b.shape[-1] for b in blocks) #which is the longest? - that is how much we need to pad
    return torch.cat(tuple(torch.nn.functional.pad(b,(0,max_len-b.shape[-1]),"constant",padding_value) for b in blocks))
    # batch1=torch.ones((5,15))
    # batch2=torch.ones((6,17))
    # batch3=torch.ones((11,5))
    # batches=[batch1,batch2,batch3]
    # max_len=max(b.shape[-1] for b in batches)
    # torch.cat(tuple(torch.nn.functional.pad(b,(0,max_len-b.shape[-1]),"constant",0) for b in batches))
    # tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
    #         [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
    #         [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
    #         [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
    #         [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
    #         [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
    #         [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
    #         [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
    #         [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
    #         [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
    #         [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
    #         [1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #         [1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #         [1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #         [1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #         [1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #         [1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #         [1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #         [1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #         [1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #         [1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #         [1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])


def batch(examples,padding_element,max_elements=25000):
    batch_examples,batch_masks,batch_position_indices,batch_golds=[],[],[],[]
    batch_sizes=[]
    batch_length=0
    for s_ex in examples:
        if not s_ex: #in case this is None or something such, should not happen!
            continue
        data_in,attention_mask,pos_indices,gold_out=s_ex
        batch_examples.append(data_in)
        batch_masks.append(attention_mask)
        batch_position_indices.append(pos_indices)
        batch_golds.append(gold_out)
        batch_sizes.append(gold_out.shape[1])
        batch_length+=gold_out.shape[0]
        
        total_elements=max(batch_sizes)*batch_length #how big a matrix do we get after padding?
        if total_elements>=max_elements:
            #we can yield a padded batch
            #print(batch_examples)
            padded_batch_in=blocks2batch(batch_examples,padding_element)
            padded_batch_masks=blocks2batch(batch_masks,0)
            padded_batch_position_indices=blocks2batch(batch_position_indices,0)
            padded_batch_golds=blocks2batch(batch_golds,-1)
            batch_examples,batch_masks,batch_position_indices,batch_golds=[],[],[],[]
            batch_sizes=[]
            batch_length=0
            yield padded_batch_in, padded_batch_masks, padded_batch_position_indices, padded_batch_golds

if __name__=="__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--files", nargs="+", help=".txt files used to train")
    args = parser.parse_args()

    transformers.BERT_PRETRAINED_MODEL_ARCHIVE_MAP["bert-base-finnish-cased"]="http://dl.turkunlp.org/finbert/torch-transformers/bert-base-finnish-cased/pytorch_model.bin"
    transformers.BERT_PRETRAINED_CONFIG_ARCHIVE_MAP["bert-base-finnish-cased"]="http://dl.turkunlp.org/finbert/torch-transformers/bert-base-finnish-cased/config.json"
    transformers.tokenization_bert.PRETRAINED_VOCAB_FILES_MAP["vocab_file"]["bert-base-finnish-cased"]="http://dl.turkunlp.org/finbert/torch-transformers/bert-base-finnish-cased/vocab.txt"
    transformers.tokenization_bert.PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES["bert-base-finnish-cased"]=512
    transformers.tokenization_bert.PRETRAINED_INIT_CONFIGURATION["bert-base-finnish-cased"]={'do_lower_case': False}

    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-finnish-cased")
    CLS,SEP,MASK=tokenizer.convert_tokens_to_ids(["[CLS]","[SEP]","[MASK]"])

    trigger_ids=tokenizer.convert_tokens_to_ids(tokenizer.tokenize("olen väärässä kuten aina"))
    sent_ids=tokenizer.convert_tokens_to_ids(tokenizer.tokenize("joten lähden kotiin"))

    trigger_ids=torch.tensor(trigger_ids, dtype=torch.long)
    sent_ids=torch.tensor(sent_ids, dtype=torch.long)

    print("Trigger ids:", trigger_ids)
    print("Sent ids:",sent_ids)
    
    sent,mask,pos_indices,gold=sentence_example(trigger_ids,sent_ids,tokenizer)
    print("input",sent)
    print("attmask",mask)
    print("posidx",pos_indices)
    print("gold",gold)
    
    
    # exs=examples(args.files,tokenizer,min_trigger=10,max_trigger=60,max_length=80,max_doc_per_file=50,shuffle_buff=3000)
    # for ex_batch in batch(exs,padding_element=MASK):
    #     data_in,attention_mask,gold_out=ex_batch
    #     print(data_in.shape,attention_mask.shape,gold_out.shape)
