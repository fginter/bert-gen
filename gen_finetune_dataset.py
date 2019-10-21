import sys
import torch
import random

def document_batch(min_trigger=10,max_trigger=30,max_length=60):
    """
    min_trigger: minimum number of tokens which act as a trigger
    max_trigger: maximum number of tokens which act as a trigger
    max_len: maximum length of generated examples (including the trigger)
    """
    tokenized_sentences=[tokenizer.tokenize(sentence) for sentence in document]
    ids=[tokenizer.convert_tokens_to_ids(tokenized_sentence) for tokenized_sentence in tokenized_sentences]
    ids=[torch.tensor(token_ids) for token_ids in ids] #list of tensors, each being a sequence of token ids in a sentence
    ids=torch.cat(ids)
    current_index=0 #this is where we currently are, and will start generating
    while True:
        trigger_len=random.randint(min_trigger,max_trigger)
        
        
    

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

    

def sentence_example(trigger_ids,sent_ids,tokenizer):
    """
    """
    print("hi")
    CLS,SEP,MASK=tokenizer.convert_tokens_to_ids(["[CLS]","[SEP]","[MASK]"])
    # len=7
    #   0   1 2 3 4 5 6
    # [cls] a b c d e f

    #how many positions can I predict
    to_predict=sent_ids.shape[0]
    inputs=sent_ids.unsqueeze(0).expand((to_predict,-1)) #copies the sentence "to_predict" times
    triggers=trigger_ids.unsqueeze(0).expand((to_predict,-1)) #copies the trigger "to_predict" times
    #what we need now is the masks for the attention

    # min_lead_in=3
    # s_len=7
    # to_predict=s_len-min_lead_in-1
    # lower_triangle=torch.tril(torch.ones((to_predict,to_predict+1)))
    # mask=torch.cat((torch.ones((to_predict,min_lead_in)),lower_triangle),-1)
    # print(mask)
    # tensor([[1., 1., 1., 1., 0., 0., 0.],
    #         [1., 1., 1., 1., 1., 0., 0.],
    #         [1., 1., 1., 1., 1., 1., 0.]])

    lower_triangle=torch.tril(torch.ones((to_predict,to_predict+1)))
    attention_mask=torch.cat((torch.ones(triggers.shape),lower_triangle),-1)
    
    # #and finally, we need the gold output with all tokens masked as -1 except the ones we are supposed to be predicting
    # #this is needed for the loss later on, so we do not train on tokens the model can trivially see
    #
    # print("texts",new_texts)
    # filter=torch.cat((torch.zeros((5,4),dtype=torch.long),torch.eye(5,dtype=torch.long)),-1).cuda()
    # print("filter",filter)
    # gold=filter*(new_texts+1)-1
    # print("gold",gold)
    # texts tensor([[  102,  3668,   145,  3093,   374,   145, 22936,   142, 23967],
    #         [  102,  3668,   145,  3093,   374,   145, 22936,   142, 23967],
    #         [  102,  3668,   145,  3093,   374,   145, 22936,   142, 23967],
    #         [  102,  3668,   145,  3093,   374,   145, 22936,   142, 23967],
    #         [  102,  3668,   145,  3093,   374,   145, 22936,   142, 23967]],
    #        device='cuda:0')
    # filter tensor([[0, 0, 0, 0, 1, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 1, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 1, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 1, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 1]], device='cuda:0')
    # gold tensor([[   -1,    -1,    -1,    -1,   374,    -1,    -1,    -1,    -1],
    #         [   -1,    -1,    -1,    -1,    -1,   145,    -1,    -1,    -1],
    #         [   -1,    -1,    -1,    -1,    -1,    -1, 22936,    -1,    -1],
    #         [   -1,    -1,    -1,    -1,    -1,    -1,    -1,   142,    -1],
    #         [   -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1, 23967]],
    #        device='cuda:0')

    triggers_and_inputs=torch.cat((triggers,inputs),-1)
    filtr=torch.cat((torch.zeros(triggers.shape,dtype=torch.long),torch.eye(to_predict,dtype=torch.long)),-1)
    gold=filtr*(triggers_and_inputs+1)-1 #the +1 -1 thing just makes sure that masking is -1 and not 0
    #print(filtr.shape, inputs.shape, attention_mask.shape)
    #and now we have all we need for this sentence
    return inputs, attention_mask, gold 

def batch(sent_examples,tokenizer,max_elements=100):
    CLS,SEP,MASK=tokenizer.convert_tokens_to_ids(["[CLS]","[SEP]","[MASK]"])
    batch_examples,batch_masks,batch_golds=[],[],[]
    batch_sizes=[]
    batch_length=0
    for sentence_text in sent_examples:
        s_ex=sentence_example(sentence_text,tokenizer)
        if not s_ex: #too short to produce examples
            continue
        data_in,attention_mask,gold_out=s_ex
        batch_examples.append(data_in)
        batch_masks.append(attention_mask)
        batch_golds.append(gold_out)
        batch_sizes.append(gold_out.shape[1])
        batch_length+=gold_out.shape[0]
        
        total_elements=max(batch_sizes)*batch_length #how big a matrix do we get after padding?
        if total_elements>=max_elements:
            #we can yield a padded batch
            #print(batch_examples)
            padded_batch_in=blocks2batch(batch_examples,MASK)
            padded_batch_masks=blocks2batch(batch_masks,0)
            padded_batch_golds=blocks2batch(batch_golds,0)
            batch_examples,batch_masks,batch_golds=[],[],[]
            batch_sizes=[]
            batch_length=0
            yield padded_batch_in, padded_batch_masks, padded_batch_golds
    else:
        if batch_sizes:
            yield padded_batch_in, padded_batch_masks, padded_batch_golds

ID,FORM,LEMMA,UPOS,XPOS,FEATS,HEAD,DEPREL,DEPS,MISC=range(10)

def read_conllu(inp,max_sent=0,drop_tokens=True,drop_nulls=True):
    comments=[]
    sent=[]
    yielded=0
    for line in inp:
        line=line.strip()
        if line.startswith("#"):
            comments.append(line)
        elif not line:
            if sent:
                yield sent,comments
                yielded+=1
                if max_sent>0 and yielded==max_sent:
                    break
                sent,comments=[],[]
        else:
            cols=line.split("\t")
            if drop_tokens and "-" in cols[ID]:
                continue
            if drop_nulls and "." in cols[ID]:
                continue
            sent.append(cols)
    else:
        if sent:
            yield sent,comments

def get_text(comments):
    for c in comments:
        if c.startswith("# text = "):
            return c[len("# text = "):]
    return None
            
def sent_examples_from_conllu(inp):
    for sent,comments in read_conllu(inp):
        txt=get_text(comments)
        if not txt:
            continue
        txt=txt.strip()
        if not txt:
            continue
        yield txt
            

if __name__=="__main__":
    import torch
    import transformers
    transformers.BERT_PRETRAINED_MODEL_ARCHIVE_MAP["bert-base-finnish-cased"]="http://dl.turkunlp.org/finbert/torch-transformers/bert-base-finnish-cased/pytorch_model.bin"
    transformers.BERT_PRETRAINED_CONFIG_ARCHIVE_MAP["bert-base-finnish-cased"]="http://dl.turkunlp.org/finbert/torch-transformers/bert-base-finnish-cased/config.json"
    transformers.tokenization_bert.PRETRAINED_VOCAB_FILES_MAP["vocab_file"]["bert-base-finnish-cased"]="http://dl.turkunlp.org/finbert/torch-transformers/bert-base-finnish-cased/vocab.txt"
    transformers.tokenization_bert.PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES["bert-base-finnish-cased"]=512
    transformers.tokenization_bert.PRETRAINED_INIT_CONFIGURATION["bert-base-finnish-cased"]={'do_lower_case': False}


    
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-finnish-cased")
    print("done loading stuff")
    # sent_examples=sent_examples_from_conllu(sys.stdin)
    # for idx,x in enumerate(batch(sent_examples, tokenizer)):
    #     print(idx,end="\r")
    ones=torch.ones((7,),dtype=torch.Long)
    print(sentence_example(ones,ones+1,tokenizer))
