import transformers
import torch

transformers.BERT_PRETRAINED_MODEL_ARCHIVE_MAP["bert-base-finnish-cased"]="http://dl.turkunlp.org/finbert/torch-transformers/bert-base-finnish-cased/pytorch_model.bin"
transformers.BERT_PRETRAINED_CONFIG_ARCHIVE_MAP["bert-base-finnish-cased"]="http://dl.turkunlp.org/finbert/torch-transformers/bert-base-finnish-cased/config.json"
transformers.tokenization_bert.PRETRAINED_VOCAB_FILES_MAP["vocab_file"]["bert-base-finnish-cased"]="http://dl.turkunlp.org/finbert/torch-transformers/bert-base-finnish-cased/vocab.txt"
transformers.tokenization_bert.PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES["bert-base-finnish-cased"]=512
transformers.tokenization_bert.PRETRAINED_INIT_CONFIGURATION["bert-base-finnish-cased"]={'do_lower_case': False}

model=transformers.BertForMaskedLM.from_pretrained("model_ftune")
#model=transformers.BertForMaskedLM.from_pretrained("bert-base-finnish-cased")
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-finnish-cased")


model=model.cuda()

def gen(trigger,tokenizer,model,temperature=None,top_k=100):
    CLS,SEP,MASK=tokenizer.convert_tokens_to_ids(["[CLS]","[SEP]","[MASK]"])
    tokenized=tokenizer.tokenize(trigger)
    ids=[CLS]+tokenizer.convert_tokens_to_ids(tokenized)+[MASK]
    ids=torch.tensor(ids,dtype=torch.long)
    ids=ids.unsqueeze(0)
    length=ids.shape[1]
    mask=torch.ones((1,length),dtype=torch.long)
    mask[0,-1]=0
    with torch.no_grad():
        ids=ids.cuda()
        mask=mask.cuda()
        out=model(ids,attention_mask=mask)
    pred=generate_step(out[0],-1,temperature=temperature,top_k=top_k)
    pred_token=tokenizer.convert_ids_to_tokens(pred)
    return detokenize(tokenized+pred_token)

def detokenize(sent):
        """ Roughly detokenizes (mainly undoes wordpiece) """
        new_sent = []
        for i, tok in enumerate(sent):
            if tok.startswith("##"):
                new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
            else:
                new_sent.append(tok)
        return new_sent



def generate_step(out, gen_idx, temperature=None, top_k=0, sample=False, return_list=True):
        """ Generate a word from from out[gen_idx]
        
        args:
            - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
            - gen_idx (int): location for which to generate for
            - top_k (int): if >0, only sample from the top k most probable words
            - sample (Bool): if True, sample from full distribution. Overridden by top_k 
        """
        logits = out[:, gen_idx]
        if temperature is not None:
            logits = logits / temperature
        if top_k > 0:
            kth_vals, kth_idx = logits.topk(top_k, dim=-1)
            dist = torch.distributions.categorical.Categorical(logits=kth_vals)
            idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
        elif sample:
            dist = torch.distributions.categorical.Categorical(logits=logits)
            idx = dist.sample().squeeze(-1)
        else:
            idx = torch.argmax(logits, dim=-1)
        return idx.tolist() if return_list else idx



#trigger="Suomen tärkeimpiin teihin kuuluva Kehä III uhkaa sortua Vantaalla, ministeriö tyrmäsi rahoituspyynnön Vantaan Askistossa oleva tieosuus on vaarassa vaurioitua nopeasti ajokelvottomaksi. Urakan hinta-arvio on 26 miljoonaa euroa."
trigger="Yhtä huonokuntoiseksi runsaan 300 metrin pituisen tieosuuden arvioivat myös valtion Väylävirasto, Uudenmaan tulevan kaavan valmistelijat sekä seudun suuria liikennehankkeita kattava maankäyttösopimus"

for _ in range(15):
    generated=" ".join(gen(trigger,tokenizer,model,temperature=1.0,top_k=200))
    trigger=generated
print(generated)

