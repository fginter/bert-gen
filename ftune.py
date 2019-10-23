import time
import datetime
import sys
import itertools
import transformers
import torch
import traceback

transformers.BERT_PRETRAINED_MODEL_ARCHIVE_MAP["bert-base-finnish-cased"]="http://dl.turkunlp.org/finbert/torch-transformers/bert-base-finnish-cased/pytorch_model.bin"
transformers.BERT_PRETRAINED_CONFIG_ARCHIVE_MAP["bert-base-finnish-cased"]="http://dl.turkunlp.org/finbert/torch-transformers/bert-base-finnish-cased/config.json"
transformers.tokenization_bert.PRETRAINED_VOCAB_FILES_MAP["vocab_file"]["bert-base-finnish-cased"]="http://dl.turkunlp.org/finbert/torch-transformers/bert-base-finnish-cased/vocab.txt"
transformers.tokenization_bert.PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES["bert-base-finnish-cased"]=512
transformers.tokenization_bert.PRETRAINED_INIT_CONFIGURATION["bert-base-finnish-cased"]={'do_lower_case': False}
print("CUDA:",torch.cuda.is_available())

import gen_finetune_dataset as ds
import random
import os

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

def documents_from_filenames(fnames):
    for f_name in balance_filenames(fnames):
        with open(f_name,"rt") as f:
                print(datetime.datetime.now().isoformat(),"Open file",f_name,flush=True)
                doc_examples=ds.doc_examples_from_plaintext(f)
                yield from doc_examples


def shuffle_stream(s,buff=3000):
    data=(x for x in s) #make this into generator so the islice does the right thing, consuming
    while True:
        buffer=list(itertools.islice(data,0,buff))
        if not buffer:
            break
        random.shuffle(buffer)
        yield from buffer    

def batches_from_documents(docs,tokenizer):
    docs_shuffled=shuffle_stream(docs,buff=3000)
    batches=ds.batches_from_documents(docs_shuffled,tokenizer,max_length=80)
    yield from batches

if __name__=="__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--apex",default=False,action="store_true",help="Half-prec training w/ apex")
    parser.add_argument("--files", nargs="+", help=".txt files used to train")
    parser.add_argument("--out", help="out model filename")
    parser.add_argument("--log", help="logfile")
    args = parser.parse_args()

    if args.apex:
        import apex
    
    model=transformers.BertForMaskedLM.from_pretrained("bert-base-finnish-cased")
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-finnish-cased")
    model=model.cuda()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.025}
    ]
    t_total=5000000
    optimizer=transformers.optimization.AdamW(optimizer_grouped_parameters,lr=0.00001)
    scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=10000, t_total=t_total)

    if args.apex:
        opt_level = 'O1'
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level=opt_level)


    model.zero_grad()
    model.train()

    documents=documents_from_filenames(args.files)
    batches=batches_from_documents(documents,tokenizer)
    examples_seen=0
    batches_seen=0
    
    time_started=time.time()
    with open(args.log,"wt") as logfile:
        for idx,x in enumerate(batches):
            inp,mask,outp=x
            examples_seen+=inp.shape[0]
            inp=inp.cuda()
            mask=mask.cuda()
            outp=outp.cuda()
            optimizer.zero_grad()
            outputs=model(inp,attention_mask=mask,masked_lm_labels=outp)
            loss=outputs[0]
            if args.apex:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            scheduler.step()
            if idx and idx%10==0:
                elapsed=time.time()-time_started
                print(idx,loss.item(),datetime.datetime.now().isoformat(),examples_seen,idx/elapsed,"[batch/sec]",examples_seen/elapsed,"[ex/sec]",sep="\t",file=logfile,flush=True)
            if idx and idx%10000==0:
                model.save_pretrained(args.out)
            if idx>=t_total:
                break
