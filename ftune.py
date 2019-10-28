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

import txt_dataset
import random
import os




if __name__=="__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--apex",default=False,action="store_true",help="Half-prec training w/ apex")
    parser.add_argument("--files", nargs="+", help=".txt files used to train")
    parser.add_argument("--out", help="out model filename")
    parser.add_argument("--log", help="logfile")
    args = parser.parse_args()

    if args.apex:
        assert False, "not sure if this works, better avoid"
        import apex
    
    model=transformers.BertForMaskedLM.from_pretrained("bert-base-finnish-cased")
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-finnish-cased")
    model=model.cuda()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.025}
    ]
    t_total=200000
    optimizer=transformers.optimization.AdamW(optimizer_grouped_parameters,lr=0.001)
    scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=5000, t_total=t_total)

    if args.apex:
        opt_level = 'O1'
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level=opt_level)


    model.zero_grad()
    model.train()

    CLS,SEP,MASK,PAD=tokenizer.convert_tokens_to_ids(["[CLS]","[SEP]","[MASK]","[PAD]"])
    examples_seen=0
    batches_seen=0
    
    time_started=time.time()
    with open(args.log,"wt") as logfile:
        while batches_seen<t_total:
            print(datetime.datetime.now().isoformat(),"NEW EPOCH. Batches seen",batches_seen)
            exs=txt_dataset.examples(args.files,tokenizer,min_trigger=10,max_trigger=40,max_length=60,max_doc_per_file=15000,shuffle_buff=50000)
            batches=txt_dataset.batch(exs,padding_element=PAD,max_elements=10000)
            print("batchid","loss","time","examples","batchpersec","batchpersec","examplpersec","examplpersec",sep="\t",file=logfile)
            for idx,x in enumerate(batches):
                inp,mask,pos_i,outp=x
                examples_seen+=inp.shape[0]
                inp=inp.cuda()
                mask=mask.cuda()
                pos_i=pos_i.cuda()
                outp=outp.cuda()
                model.zero_grad()
                outputs=model(inp,position_ids=pos_i,attention_mask=mask,masked_lm_labels=outp)
                loss=outputs[0]
                if args.apex:
                    with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
                scheduler.step()
                if batches_seen and batches_seen%10==0:
                    elapsed=time.time()-time_started
                    print(batches_seen,loss.item(),datetime.datetime.now().isoformat(),examples_seen,batches_seen/elapsed,"[batch/sec]",examples_seen/elapsed,"[ex/sec]",sep="\t",file=logfile,flush=True)
                if batches_seen and batches_seen%10000==0:
                    model.save_pretrained(args.out)

                batches_seen+=1
                if batches_seen>=t_total:
                    break
