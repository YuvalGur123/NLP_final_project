from itertools import combinations
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from transformers import  BertTokenizer, BertForQuestionAnswering, \
    BartTokenizer, BartForConditionalGeneration
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from collections import Counter
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

#### The scopus api was giving alot of problems, so instead the program will run with a csv.
#### The csv contains all articles from the last 10 years with nlp in the title, keywords or abstract

cache_dir = 'D:/models'  # Configure a folder to save the pretrained models to. comment the line to save to default directory


#Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


df = pd.read_csv('scopus.csv')


# Function to preprocess the text
def preprocess_text(text):
    if pd.isna(text):
        return ''
    if not isinstance(text, str):
        text = str(text)
    tokens = word_tokenize(text)
    filtered_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if
                       token.isalpha() and token.lower() not in stop_words]
    return ' '.join(filtered_tokens)


df['processed_abstract'] = df['Abstract'].apply(preprocess_text)
df['processed_author_keywords'] = df['Author Keywords'].apply(preprocess_text)

# Save the updated DataFrame to a new CSV file
df.to_csv('processed_scopus.csv', index=False)

print("Preprocessing complete. The processed data is saved in 'processed_scopus.csv'.")


df = pd.read_csv('processed_scopus.csv')

df['combined_text'] = df['processed_abstract'] + ' ' + df['processed_author_keywords']

df['combined_text'] = df['combined_text'].fillna('')

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df['processed_abstract'])
feature_names = vectorizer.get_feature_names_out()

# Sum the TF-IDF scores for each word across all documents
sum_tfidf = X.sum(axis=0)
word_tfidf = dict(zip(feature_names, sum_tfidf.flat))

# Sort the words by TF-IDF score
sorted_words = sorted(word_tfidf.items(), key=lambda x: x[1], reverse=True)

# Prepare data for the bar chart
words, scores = zip(*sorted_words[:20])  # Top 20 words

# Create a bar chart
plt.figure(figsize=(10, 8))
plt.barh(words, scores, color='skyblue')
plt.xlabel('TF-IDF Score')
plt.title('Top 20 Words by TF-IDF Score')
plt.gca().invert_yaxis()
plt.show()

# Create a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(word_tfidf)


plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Words by TF-IDF Score')
plt.show()


# Word2Vec model

documents = df['processed_abstract'].apply(simple_preprocess)
model = Word2Vec(sentences=documents, vector_size=100, window=5, min_count=5, workers=4)


word_list = list(model.wv.index_to_key)

# Define the number of top words to retrieve
n = 20
top_n_words = word_list[:n]
word_freq = {word: model.wv.get_vecattr(word, 'count') for word in top_n_words}
df_word_freq = pd.DataFrame(list(word_freq.items()), columns=['Word', 'Frequency'])

# Sort the DataFrame by frequency in descending order
df_word_freq = df_word_freq.sort_values(by='Frequency', ascending=False)

# Plotting the top words and their frequencies
plt.figure(figsize=(10, 8))
plt.barh(df_word_freq['Word'], df_word_freq['Frequency'], color='skyblue')
plt.xlabel('Frequency')
plt.title('Top 20 Words in Word2Vec Model')
plt.gca().invert_yaxis()
plt.show()

word_vectors = model.wv
word_list = list(word_vectors.index_to_key)
word_array = np.array([word_vectors[word] for word in word_list])


# Standardize the word vectors
scaler = StandardScaler()
word_vectors_scaled = scaler.fit_transform(word_array)

# Define the autoencoder model
input_dim = word_vectors_scaled.shape[1]
encoding_dim = 50  # Size of the compressed representation

# Define the model architecture
input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoded)
encoder = tf.keras.models.Model(inputs=input_layer, outputs=encoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(word_vectors_scaled, word_vectors_scaled, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)

encoded_vectors = encoder.predict(word_vectors_scaled)
reconstructed_vectors = autoencoder.predict(word_vectors_scaled)

reconstruction_error = np.mean(np.square(word_vectors_scaled - reconstructed_vectors), axis=1)

word_importance = dict(zip(word_list, reconstruction_error))
sorted_importance = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)

df_word_importance = pd.DataFrame(list(word_importance.items()), columns=['Word', 'Reconstruction Error'])
df_word_importance = df_word_importance.sort_values(by='Reconstruction Error', ascending=False).head(20)

plt.figure(figsize=(10, 8))
plt.barh(df_word_importance['Word'], df_word_importance['Reconstruction Error'], color='skyblue')
plt.xlabel('Reconstruction Error')
plt.title('Top 20 Words by Reconstruction Error in AutoEncoder Model')
plt.gca().invert_yaxis()
plt.show()


# Spacy NER algorithm. this is a very heavy and long part of the code to run.
# Should only be ran once to get an impression of the code, then comment this block for any additional runs of the code

# #start to comment from here
#
# #Load the SpaCy model
# nlp = spacy.load('en_core_web_sm')
#
# #Load your data
# df = pd.read_csv('processed_scopus.csv')
#
#
# #Function to extract named entities from text
# def extract_entities(text):
#     doc = nlp(text)
#     return [(ent.text, ent.label_) for ent in doc.ents]
#
#
# # Apply NER to abstracts
# df['entities'] = df['Abstract'].apply(extract_entities)
#
# all_entities = [ent for sublist in df['entities'] for ent in sublist]
#
# # Count entities by type
# entity_types = [ent[1] for ent in all_entities]
# entity_counts = Counter(entity_types)
#
# output_file = 'output.txt'
#
# with open(output_file, 'w', encoding='utf-8') as f:
#     for index, row in df.iterrows():
#         f.write(f"Title: {row['Title']}\n")
#         f.write(f"Entities: {row['entities']}\n")
#         f.write("\n")
#
# print(f"Output saved to {output_file}")
#
# # Plot the results
# plt.figure(figsize=(10, 6))
# plt.bar(entity_counts.keys(), entity_counts.values(), color='skyblue')
# plt.xlabel('Entity Type')
# plt.ylabel('Count')
# plt.title('Distribution of Named Entity Types')
# plt.xticks(rotation=45)
# plt.show()
#
# # Display the results
# print(df[['Title', 'entities']].head())
#
# # stop the comment here


df2 = pd.read_csv('scopus.csv')

# Count the number of titles published each year
titles_per_year = df2.groupby('Year').size()

plt.figure(figsize=(12, 6))
titles_per_year.plot(kind='bar')
plt.xlabel('Year')
plt.ylabel('Number of Titles')
plt.title('Number of Titles Published Each Year')
plt.xticks(rotation=45)
plt.show()

# Count number of articles for each publisher
publisher_counts = df['Publisher'].value_counts()

# Get the top 10 publishers
top_10_publishers = publisher_counts.head(10)

# Plot the data
plt.figure(figsize=(10, 6))
sns.barplot(x=top_10_publishers.values, y=top_10_publishers.index, palette='viridis')

# Add titles and labels
plt.title('Top 10 Publishers')
plt.xlabel('Number of Publications')
plt.ylabel('Publisher')

# Display the plot
plt.show()


# Function to parse authors from a string, the authors are separated by a semicolon
def parse_authors(author_str):
    if pd.isna(author_str):
        return []  # Return an empty list if the author_str is NaN or not available
    if not isinstance(author_str, str):
        author_str = str(author_str)  # Convert non-string types to string
    return [author.strip() for author in author_str.split(';')]


df['author_list'] = df['Authors'].apply(parse_authors)
all_authors = [author for sublist in df['author_list'] for author in sublist]


author_counts = Counter(all_authors)

author_counts_df = pd.DataFrame(author_counts.items(), columns=['Author', 'Publication Count'])
author_counts_df = author_counts_df.sort_values(by='Publication Count', ascending=False)

# Get the top 10 authors
top_10_authors = author_counts_df.head(10)

plt.figure(figsize=(12, 8))
plt.barh(top_10_authors['Author'], top_10_authors['Publication Count'], color='skyblue')
plt.xlabel('Number of Publications')
plt.title('Top 10 Authors by Number of Publications')
plt.gca().invert_yaxis()
plt.show()


df['processed_author_keywords'] = df['processed_author_keywords'].fillna('')

# Initialize CountVectorizer to get the most common keywords
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X_keywords = vectorizer.fit_transform(df['processed_author_keywords'])
feature_names = vectorizer.get_feature_names_out()

# Sum the occurrences of each keyword across all documents
sum_keywords = X_keywords.sum(axis=0)
keyword_counts = dict(zip(feature_names, sum_keywords.flat))

# Sort the keywords by frequency
sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)

# Get the top 10 keywords
top_10_keywords = [keyword for keyword, count in sorted_keywords[:10]]
print("Top 10 Keywords:")
print(top_10_keywords)

df_keywords_per_year = df[['Year', 'processed_author_keywords']].copy()
df_keywords_per_year['processed_author_keywords'] = df_keywords_per_year['processed_author_keywords'].apply(
    lambda x: [word for word in x.split() if word in top_10_keywords]
)

keyword_yearly_counts = pd.DataFrame(index=top_10_keywords)

# Count articles per year for each of the top 10 keywords
for year in df_keywords_per_year['Year'].unique():
    year_df = df_keywords_per_year[df_keywords_per_year['Year'] == year]
    keyword_counts = year_df['processed_author_keywords'].explode().value_counts()
    keyword_yearly_counts[year] = keyword_counts.reindex(top_10_keywords, fill_value=0)

keyword_yearly_counts = keyword_yearly_counts.T


plt.figure(figsize=(14, 8))
ax = keyword_yearly_counts.plot(kind='bar', stacked=True, colormap='tab20')
plt.xlabel('Year')
plt.ylabel('Number of Articles')
plt.title('Number of Articles Published Each Year with Top 10 Keywords')
plt.xticks(rotation=45)
plt.legend(title='Keyword')
plt.tight_layout()
plt.show()


# Copy relevant columns
df_authors_per_year = df[['Year', 'author_list']].copy()

# Flatten the author list
df_authors_per_year = df_authors_per_year.explode('author_list')

# Count articles per author per year
author_yearly_counts = df_authors_per_year.groupby(['Year', 'author_list']).size().reset_index(name='Article Count')

# Filter authors who published at least 3 articles each year
filtered_authors = author_yearly_counts[author_yearly_counts['Article Count'] >= 3]

# Calculate the number of unique years in the dataset
unique_years = len(df['Year'].unique())

# Group by author and count the number of years they meet the condition
author_publication_counts = filtered_authors.groupby('author_list').size().reset_index(name='Years Met Condition')

# Filter authors who published at least 3 articles every year
authors_at_least_3_articles_each_year = author_publication_counts[
    author_publication_counts['Years Met Condition'] == unique_years
]

# Print results
print("Authors who published at least 3 articles each year:")
print(authors_at_least_3_articles_each_year)

# Plotting
plt.figure(figsize=(12, 8))

# Bar plot showing authors and the number of years they met the publication condition
plt.barh(authors_at_least_3_articles_each_year['author_list'], authors_at_least_3_articles_each_year['Years Met Condition'], color='skyblue')

plt.xlabel('Number of Years with at Least 3 Articles')
plt.ylabel('Authors')
plt.title('Authors Who Published at Least 3 Articles Each Year')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest values at the top

plt.show()

print("Authors who published at least 3 articles each year:")
print(authors_at_least_3_articles_each_year)


#find pairs of keywords with high correlation
def get_keyword_pairs(keywords):
    return list(combinations(sorted(keywords), 2))


# Create a DataFrame to hold keyword pairs
keyword_pairs_per_year = []

# Iterate through each row in the DataFrame
for _, row in df_keywords_per_year.iterrows():
    year = row['Year']
    keywords = row['processed_author_keywords']
    pairs = get_keyword_pairs(keywords)
    for pair in pairs:
        keyword_pairs_per_year.append({'Year': year, 'Pair': pair})

# Convert list to DataFrame
df_pairs_per_year = pd.DataFrame(keyword_pairs_per_year)

# Count occurrences of each keyword pair per year
pair_counts = df_pairs_per_year.groupby(['Year', 'Pair']).size().reset_index(name='Count')

# Find the top 5 pairs for each year
top_pairs_per_year = pair_counts.groupby('Year').apply(lambda x: x.nlargest(5, 'Count')).reset_index(drop=True)

# Plot the top pairs per year
plt.figure(figsize=(14, 8))

# Create a color map
years = top_pairs_per_year['Year'].unique()
cmap = plt.get_cmap('tab20', len(years))  # Use 'tab20' colormap for distinct colors

# Assign a color to each year
for i, year in enumerate(years):
    year_data = top_pairs_per_year[top_pairs_per_year['Year'] == year]
    plt.barh(year_data['Pair'].astype(str) + f' ({year})', year_data['Count'], color=cmap(i), label=f'Year {year}')

plt.xlabel('Count')
plt.ylabel('Keyword Pairs')
plt.title('Top 5 Keyword Pairs for Each Year')
plt.legend(title='Year')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest counts at the top
plt.xticks(rotation=45)
plt.show()

df['Title'] = df['Title'].fillna('')


# Function to check if a keyword is in the title
def is_keyword_in_title(keyword, title):
    return keyword in title


# Calculate the percentage of keywords in titles
def calculate_keyword_title_percentage(row):
    title = row['Title'].lower()
    keywords = row['processed_author_keywords'].split()
    if not keywords:
        return 0
    count_in_title = sum(is_keyword_in_title(keyword, title) for keyword in keywords)
    return (count_in_title / len(keywords)) * 100


df['keyword_in_title_percentage'] = df.apply(calculate_keyword_title_percentage, axis=1)

# Calculate average percentage of keywords in titles
average_percentage = df['keyword_in_title_percentage'].mean()
print(f"Average percentage of keywords present in the title: {average_percentage:.2f}%")


vectorizer = CountVectorizer(stop_words='english', max_features=50)
X = vectorizer.fit_transform(df['combined_text'])
feature_names = vectorizer.get_feature_names_out()

# Sum the occurrences of each word across all documents
sum_words = X.sum(axis=0)
word_counts = dict(zip(feature_names, sum_words.flat))

# Sort the words by frequency
sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

# Print the top 50 words
top_50_words = sorted_words[:50]
print("Top 50 Most Common Words:")
for word, count in top_50_words:
    print(f"{word}: {count}")

word_df = pd.DataFrame(X.toarray(), columns=feature_names)

# Calculate the correlation matrix
correlation_matrix = word_df.corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Top 50 Words')
plt.show()

# Identify strongly correlated words
threshold = 0.5  # Adjust threshold as needed
strongly_correlated = correlation_matrix[(correlation_matrix >= threshold) & (correlation_matrix != 1.0)]

# Set to track printed pairs
printed_pairs = set()

# Print strongly correlated word pairs
print("\nStrongly Correlated Words:")
for word1 in strongly_correlated.columns:
    for word2 in strongly_correlated.index:
        if not np.isnan(strongly_correlated.loc[word2, word1]):
            # Create a canonical pair representation
            pair = tuple(sorted((word1, word2)))
            if pair not in printed_pairs:
                printed_pairs.add(pair)
                print(f"{word1} and {word2} have correlation {strongly_correlated.loc[word2, word1]:.2f}")



# Summarization of abstracts
#### this part is limited to 100 abstracts. This number can be changed easily.
#### running the summarization part on all abstracts will take almost a week of run time.


df = pd.read_csv('scopus.csv')
df = df.head(100)   # remove this line to summarize all abstracts
# Load BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = BartForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Ensure model is in evaluation mode
model.eval()


# Function to summarize a list of abstracts using BART
def summarize_batch(abstracts):
    summaries = []
    for abstract in abstracts:
        if not abstract.strip():  # Skip empty abstracts
            summaries.append("")
            continue

        inputs = tokenizer(abstract, max_length=1024, return_tensors="pt", truncation=True, padding='max_length')
        input_ids = inputs["input_ids"].to(device)

        try:
            with torch.no_grad():
                summary_ids = model.generate(input_ids, max_length=150, min_length=30, length_penalty=2.0, num_beams=4,
                                             early_stopping=True)

            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        except Exception as e:
            print(f"Error generating summary: {e}")
            summaries.append("Error generating summary")
    return summaries


# Process abstracts in batches
batch_size = 10  # Adjust based on available memory
summaries = []
for start in range(0, len(df), batch_size):
    end = min(start + batch_size, len(df))
    batch_abstracts = df['Abstract'][start:end].tolist()
    batch_summaries = summarize_batch(batch_abstracts)
    summaries.extend(batch_summaries)

df['summary'] = summaries

# Save the DataFrame with summaries to a new CSV file
df.to_csv('summarized_scopus_bart.csv', index=False)

print("Summarization complete. The summarized data is saved in 'summarized_scopus_bart.csv'.")


# Getting the most common topics using a GPT model
# The model of choice was bert-large, gpt-2 was useless and any newer version of chat-gpt requires paid access


# Load pre-trained DistilBERT tokenizer and model
model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = BertForQuestionAnswering.from_pretrained(model_name, cache_dir=cache_dir)

MAX_LENGTH = 512  # BERT model's maximum token limit


def answer_question(question, context):
    # Tokenize input with truncation and padding
    inputs = tokenizer.encode_plus(
        question, context,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Find the start and end token indices for the answer
    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits) + 1

    # Convert token indices to answer string
    answer_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_index:end_index])
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    # Clean answer text, bert sometimes leave the internal tokens in the answer
    return answer.strip().replace('[CLS]', '').replace('[SEP]', '').strip()


def extract_topics_from_abstracts(abstracts, question="What are the main topics?"):
    all_topics = []

    for abstract in abstracts:
        if len(tokenizer.encode(abstract)) > MAX_LENGTH:
            # If the abstract is too long, chunk it
            chunks = [abstract[i:i + MAX_LENGTH - 2] for i in range(0, len(abstract), MAX_LENGTH - 2)]
            answers = []
            for chunk in chunks:
                answer = answer_question(question, chunk)
                if answer:
                    answers.append(answer)
            # Join all answers from chunks and process
            full_answer = ' '.join(answers)
            topics = full_answer
        else:
            topics = answer_question(question, abstract)

        if topics:
            all_topics.extend([topic.strip() for topic in topics.split(',')])

    # Count the frequency of each topic
    topic_counts = Counter(all_topics)

    return topic_counts.most_common()


# only get topics from the top 100 abstracts. the model takes several days to run on all of the abstracts
abstracts = df['Abstract'].head(100).tolist()

common_topics = extract_topics_from_abstracts(abstracts)
print("Most common topics:", common_topics)