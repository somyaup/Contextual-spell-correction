import random
import pandas as pd
import Levenshtein as lev
from POC import POC as POC1
import config
import time
class ProductList:
    def __init__(self):
        self.df=self.__main__()
    # Function to introduce a random error in the phrase
    def introduce_error(self,phrase):
        error_types = ["insert", "delete", "replace"]
        error_type = random.choice(error_types)
        if error_type == "insert":
            index = random.randint(0, len(phrase))
            char = random.choice("abcdefghijklmnopqrstuvwxyz")
            phrase = phrase[:index] + char + phrase[index:]
        elif error_type == "delete":
            index = random.randint(0, len(phrase) - 1)
            phrase = phrase[:index] + phrase[index + 1:]
        elif error_type == "replace":
            index = random.randint(0, len(phrase) - 1)
            char = random.choice("abcdefghijklmnopqrstuvwxyz")
            phrase = phrase[:index] + char + phrase[index + 1:]
        return phrase
    def __main__(self):
        product_list = [
            "Men's sneakers",
            "LED TV",
            "Women's sweater",
            "Electronic devices",
            "Boys shoes",
            "Wall paint",
            "Home appliances",
            "Women's jeans",
            "Wireless headsets",
            "Computer accessories",
            "Fashionable clothes",
            "Men's wristwatch",
            "Traveling bag",
            "Women's tops",
            "Computer mouse",
            "Diamond earrings",
            "Baby clothes",
            "Kitchen appliances",
            "Ladies' handbag",
            "Wireless speakers",
            "Summer hat",
            "Smartphones",
            "Boys jumper",
            "Handheld vacuum",
            "Women's swimwear",
            "Outdoor furniture",
            "Girls' shoes",
            "Electric kettle",
            "Men's trousers",
            "Sportswear for women",
            "Kids toys",
            "Sunglasses",
            "Backpack",
            "Waterproof jacket",
            "Pizza delivery",
            "Formal dresses",
            "Bluetooth speakers",
            "Girls' skirts",
            "Wristwatch",
            "Indoor plants",
            "Unisex fragrance",
            "Party decoration",
            "Healthcare products",
            "Women's boots",
            "Wireless keyboard",
            "School supplies",
            "Chocolate cake",
            "Office chair",
            "Stainless steel utensils",
            "Kid's clothes",
            "Table lamp",
            "Makeup accessories",
            "Green Lantern"
        ]
        # Generate phrases with errors
        phrases_with_errors = [self.introduce_error(phrase) for phrase in product_list]

        # Create a dataframe
        df = pd.DataFrame({"Original": product_list, "Error": phrases_with_errors})
        return df
def find_closest_option(query, options):
    min_distance = float('inf')
    closest_option = None
    for option in options:
        distance = lev.distance(query, option)
        if distance < min_distance:
            min_distance = distance
            closest_option = option
    return closest_option
    
df= ProductList().df
# Start the timer
start_time = time.time()
poc_obj=POC1()
# End the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
# Print the result
print(f"LOAD trie time taken: {elapsed_time} seconds")
# Start the timer
start_time = time.time()
# Example usage
df['model_guess']=""
df['guesses']=-1
for index, row in df.iterrows():
    corrected_query = row['Original']
    query = row['Error']
    sentences=[]
    try:
        sentences=poc_obj.did_you_mean(query)['best_matches']
        sentence=find_closest_option(corrected_query.lower(),sentences)
        
    except Exception as e:
        config.log_error(e)
        sentence=query

    if  corrected_query.lower() == sentence.lower():
        df.loc[index,'model_guess']=corrected_query.lower()
        df.loc[index,'guesses']=0
        guess=0
    else:
        df.loc[index,'model_guess']=sentence
        guess=lev.distance(sentence.lower(),corrected_query.lower())
        df.loc[index,'guesses']=guess
    df.loc[index,'improvement']=lev.distance(query.lower(),corrected_query.lower())-guess
config_json=config.fetch_config()
df.to_csv(config_json["driver"]["metrics_data"])
# End the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the result
print(f"Time taken: {elapsed_time} seconds")
print(df.guesses.value_counts())
print(df.improvement.value_counts())