def sentence_example(sent,tokenizer,min_lead_in=3):
    """
    Todo - should we mask whole words, not just subwords...?

    min_lead_in: how many words of initial context should we at least keep?
    """
    CLS,SEP,MASK=tokenizer.convert_tokens_to_ids(["[CLS]","[SEP]","[MASK]"])
    tokenized=tokenizer.tokenize(sent)
    ids=[CLS]+tokenizer.convert_to_ids(tokenized)
    if len(ids)<=min_lead_in+1:
        #not enough for an example...
        return
    #   0   1 2 3 4 5 6
    # [cls] a b c d e f 
    for cut_idx in range(min_lead_in+1,len(ids)):
        yield ids[:cut_idx]+[MASK,SEP],cut_idx #cut_idx says the masked stuff is at position cut_idx

def batch(sent_examples,max_elements=1000):
    batch_examples,batch_masks=[],[]
    for sent_example,cut_idx in sent_examples:
        
        
