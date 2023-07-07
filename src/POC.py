from fuzzywuzzy import fuzz
from spellchecker import SpellChecker
from gingerit.gingerit import GingerIt
import Levenshtein
import re
from pyaspeller import YandexSpeller
from transformers import BertTokenizer
import heapq
# from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import config
import nltk
# import fasttext
# import fasttext.util
#STS
from sentence_transformers import SentenceTransformer, util

class POC:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        #self.model = BertForMaskedLM.from_pretrained("bert-large-cased")
        self.config_json= config.fetch_config()
        #self.model = fasttext.load_model(self.config_json['driver']['model'])
        self.model= SentenceTransformer('all-MiniLM-L6-v2')
        self.n=self.config_json['driver']['topn']
        self.threshold=self.config_json['driver']['threshold']
        with open(self.config_json["preprocess"]["output"], 'r') as txt:
            self.dictionary = set(txt.read().split())

        with open(self.config_json["driver"]["dictionary"], 'r') as txt:
            self.dictionary2 = set(txt.read().split())
        self.dictionary.update(self.dictionary2)
        self.trie_root = TrieNode()
        for word in self.dictionary:
            insert_word(self.trie_root, word)
        self.speller = YandexSpeller()
        self.spell = SpellChecker()
        #self.STmodel = SentenceTransformer('bert-base-nli-mean-tokens')
        self.ginger_parser = GingerIt()
        print("MODEL READY")
        
    def did_you_mean(self,query):
        try:
            sentence=self.preprocess_query(query)
            query,misspelled_words = self.mask_misspelled(sentence)
            sentences=[query]
            temp=[]
            oldtemp=sentences
            for misspelled_word in misspelled_words:
                temp=[]
                for error in oldtemp:
                    temp.extend(self.predict_mask(error,misspelled_word))
                oldtemp=temp
            sentences=oldtemp
        except Exception as e:
            config.log_error(e)
            sentence=query
            sentences=[sentence]
        best_matches,relevance=self.find_best_matches(query,sentences, self.n)
        result={"best_matches":list(best_matches),"relevance":list(relevance)}
        return result

    def get_most_appropriate_word(self,phrase, options):
        masked_word = "["
        tokenized_phrase = nltk.word_tokenize(phrase)
        masked_index = tokenized_phrase.index(masked_word)  
        # Find the index of the first occurrence of "[MASKED]"
        tagged_phrase = nltk.pos_tag(tokenized_phrase)
        word, pos = tagged_phrase[masked_index]
        grammatically_appropriate_words = []

        for option in options:
            tagged_option = nltk.pos_tag([option])
            if tagged_option[0][1] == pos:
                grammatically_appropriate_words.append(option)
        return grammatically_appropriate_words
    
    def suggest_corrections(self, word):
        closest_words=search_closest_words(self.trie_root, word, self.n)
        filtered_words = []
        for close_word in closest_words:
            swap_distance = levenshtein_distance_with_swaps(word, close_word)
            if swap_distance <= self.threshold:
                filtered_words.append(close_word)
            if len(filtered_words)==0:
                filtered_words=[word]
        return filtered_words
      
    def preprocess_query(self,query):
        # Remove special characters and convert to lowercase
        processed_query = re.sub(r"[^\w\s\d]", " ", query.lower())
        processed_query = re.sub(r"\s+", " ", processed_query.strip())
        return processed_query

    def basic_spell(self,query):
        fixed = self.speller.spelled(query)
        return fixed

    def mask_misspelled(self,query):
        misspelled_words=[]
        # Split the processed query into individual words
        words = query.split()
        masked_query = []

        # Replace misspelled words with [MASK]
        for word in words:
            if word.lower()  not in self.spell:
                masked_query.append("[MASK]")
                misspelled_words.append(word)
            else:
                masked_query.append(word)

        return " ".join(masked_query),misspelled_words

    def cosine_similarity(self,vector1, vector2):
        v1_norm= np.linalg.norm(vector1)
        v2_norm =np.linalg.norm(vector2, axis=1)
        cosine_similarities = np.dot(vector1, vector2.T) / (v1_norm * v2_norm)
        return cosine_similarities[0]
    
    def find_best_matches(self,query, options, n=5):
        query_embedding = self.model.encode([query], convert_to_tensor=True)
        option_embeddings = self.model.encode(options, convert_to_tensor=True)

        similarities = util.pytorch_cos_sim(query_embedding, option_embeddings).flatten().numpy()
        # Compute the Levenshtein distances for all options
        distances = np.array([Levenshtein.distance(query, option) for option in options])

        # Adjust the similarities based on the Levenshtein distances
        similarities = similarities- 0.1*distances
        similarities = similarities.tolist()
        
        tags = self.config_json["driver"]["tags"]
        for tag in tags:
            tag_embedding = self.model.encode([tag], convert_to_tensor=True)
            tag_similarities = util.pytorch_cos_sim(tag_embedding, option_embeddings).flatten().tolist()
            similarities = [sim + 0.2 * tag_sim / len(tags) for sim, tag_sim in zip(similarities, tag_similarities)]
        # Rank the options
        ranked_options = [option for _, option in sorted(zip(similarities, options), reverse=True)]
        sorted_options, sorted_similarities=ranked_options,sorted(similarities, reverse=True)
        #check if length>n
        if n >= len(options):
            return sorted_options, sorted_similarities
        else:
            return sorted_options[:n], sorted_similarities[:n]
        
    def predict_mask(self,sentence,misspelled_word):

        masked_sentence = sentence
        # Predict candidate words using BERT
        input_ids = self.tokenizer.encode(masked_sentence, add_special_tokens=True, return_tensors="pt")
        # Get the masked token index
        try:
            masked_index = input_ids[0].tolist().index(self.tokenizer.mask_token_id)
        except Exception as e:
            config.log_error('[Get the masked token index]'+str(e))
            return sentence

        # Step 4: Calculate edit distance and select correction
        dict_words= self.suggest_corrections(misspelled_word)
        filtered_words=self.get_most_appropriate_word(sentence, dict_words)
        if len(filtered_words)==0:
            filtered_words = dict_words
        corrected_words,corrected_rel = self.find_best_matches(sentence, filtered_words,n=self.n)
        
        def insert_prediction(query, prediction, index):
            tokens = self.tokenizer.tokenize(query)
            if index <= len(tokens):
                tokens[index-1] = prediction
            else:
                tokens[-1] = prediction

            updated_query = self.tokenizer.convert_tokens_to_string(tokens)
            return updated_query
        
        updated_queries=[]
        for corrected_word in corrected_words:
            updated_query = insert_prediction(sentence, corrected_word, masked_index)
            updated_queries.append(updated_query)
        return updated_queries

    def fix_grammar(self,query):
        # Initialize GingerIt for grammar checking
        grammar_corrected_sentence = self.ginger_parser.parse(query)['result']
        Lev_dist=Levenshtein.distance(query,grammar_corrected_sentence)
        if Lev_dist>2:
            return query
        return grammar_corrected_sentence

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False

def insert_word(root, word):
    current_node = root

    for char in word:
        if char not in current_node.children:
            current_node.children[char] = TrieNode()
        current_node = current_node.children[char]
    current_node.is_word = True

def search_closest_words(root, search_term, k=5):
    closest_words = []
    min_heap = []

    def dfs(node, prefix):
        if node.is_word:
            edit_distance = levenshtein_distance_with_swaps(prefix, search_term)
            #score=fuzz.ratio(prefix,search_term)
            heapq.heappush(min_heap, (edit_distance, prefix))

        for char, child_node in node.children.items():
            dfs(child_node, prefix + char)

    dfs(root, '')

    while min_heap and len(closest_words) < k:
        _, word = heapq.heappop(min_heap)
        closest_words.append(word)

    return closest_words

def levenshtein_distance_with_swaps(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)
    return dp[m][n]