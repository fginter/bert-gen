import sys
import torch
import random




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
            

        
if __name__=="__main__":
    import torch
    import transformers
    transformers.BERT_PRETRAINED_MODEL_ARCHIVE_MAP["bert-base-finnish-cased"]="http://dl.turkunlp.org/finbert/torch-transformers/bert-base-finnish-cased/pytorch_model.bin"
    transformers.BERT_PRETRAINED_CONFIG_ARCHIVE_MAP["bert-base-finnish-cased"]="http://dl.turkunlp.org/finbert/torch-transformers/bert-base-finnish-cased/config.json"
    transformers.tokenization_bert.PRETRAINED_VOCAB_FILES_MAP["vocab_file"]["bert-base-finnish-cased"]="http://dl.turkunlp.org/finbert/torch-transformers/bert-base-finnish-cased/vocab.txt"
    transformers.tokenization_bert.PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES["bert-base-finnish-cased"]=512
    transformers.tokenization_bert.PRETRAINED_INIT_CONFIGURATION["bert-base-finnish-cased"]={'do_lower_case': False}

    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-finnish-cased")
    #print("done loading stuff")
    # sent_examples=sent_examples_from_conllu(sys.stdin)
    # for idx,x in enumerate(batch(sent_examples, tokenizer)):
    #     print(idx,end="\r")
    #ones=torch.ones((7,),dtype=torch.long)
    #print(sentence_example(ones+3,ones+4,tokenizer))
    docs=doc_examples_from_plaintext(sys.stdin)
    for doc in docs:
        for b in document_batch(doc,tokenizer):
            for x in b:
                print(x.shape)
