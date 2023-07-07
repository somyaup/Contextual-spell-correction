import json
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
import config

def extract_data(csv_file, output_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    #fill ratings 
    df['product_rating']= pd.to_numeric(df['product_rating'], errors='coerce')
    # Replace NaN values with 0
    df['product_rating'] = df['product_rating'].fillna(0)

    # Group the data by the 'Product' column and sum the 'Quantity Ordered' column
    product_quantity = df.groupby('product_name')['product_rating'].mean()

    # Sort the products based on the quantity in descending order
    sorted_products = product_quantity.sort_values(ascending=False)

    # Extract only the product names without any numerical text
    products = sorted_products.index.str.replace(r'\-', '').str.strip()
    
    # Extract only the product names without any numerical text
    products = sorted_products.index.str.replace(r'\d+', '').str.strip()

    # Convert the Index object to a Series
    products_series = pd.Series(products)

    # Write the product names to the output file
    products_series.to_csv(output_file, header=False, index=False)

    print(f"Data extracted and saved to {output_file}")

def replace_spaces_commas(input_file, output_file):
    with open(input_file, 'r') as file:
        text = file.read()

    modified_text = text.replace(' ', '\n').replace(',', '\n')
    
    
    #Remove non-alphabetic characters
    modified_text = re.sub(r'[^a-z\n]', '', modified_text, flags=re.IGNORECASE)

    # Remove duplicate rows
    modified_text = '\n'.join(list(set(modified_text.lower().split('\n'))))
    
    

    with open(output_file, 'w') as file:
        file.write(modified_text)

    print(f"Text processed and saved to {output_file}")

def remove_stopwords(file_path):
    # Load the stop words
    stop_words = set(stopwords.words('english'))

    # Read the file
    with open(file_path, 'r') as file:
        text = file.read()

    # Tokenize the text
    words = nltk.word_tokenize(text)

    # Remove stop words
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Join the filtered words back into a sentence
    filtered_text = '\n '.join(filtered_words)
    # Read the file
    with open(file_path, 'w') as file:
         file.write(filtered_text)
    return filtered_text

# Example usage
config_json= config.fetch_config()
input_csv= config_json["preprocess"]["input"]
df=pd.read_csv(input_csv)
df.product_rating.value_counts()
output_txt=config_json["preprocess"]["output"]
# extract_data(input_csv,output_txt )
# replace_spaces_commas(output_txt, output_txt)
remove_stopwords(output_txt)
fileterd=remove_stopwords(config_json['driver']['dictionary'])

