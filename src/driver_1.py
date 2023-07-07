import config
from POC import POC
import time
poc_obj=POC()
config_json=config.fetch_config()

def did_you_mean(query):
    try:
        sentence=poc_obj.fix_grammar(query)
        sentence,misspelled_words = poc_obj.mask_misspelled(sentence)
        j=1
        sentences=[sentence]
        temp=[]
        oldtemp=sentences
        for misspelled_word in misspelled_words:
            temp=[]
            for error in oldtemp:
                temp.extend(poc_obj.predict_mask(error,misspelled_word))
            oldtemp=temp
        sentences=oldtemp
    except Exception as e:
        config.log_error(e)
        sentence=query
        sentences=[sentence]
    return sentences

queries=["hams", "goosebrry pickkle","titkan watch","lokal trian","ranbow toyss stffed"]

corrections={}

#processing
for query in queries:
    corrections[query]={}
    corrections[query]['DID YOU MEAN']=poc_obj.did_you_mean(query)

# Print the result
print(f"Corrections: {corrections} ")
