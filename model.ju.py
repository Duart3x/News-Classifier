# # %%
# !pip install deep - translator
#

# %%
import networkx as nx
from collections import defaultdict
import json
from collections import Counter
from operator import neg
import matplotlib.pyplot as plt
from deep_translator import GoogleTranslator
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
# %%
nltk.download('vader_lexicon')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

# %%
# Step 2: Load the dataset
file_path = "results-publico-translated.csv"
columns = ["term", "url", "text", "title", "translated_text"]
df = pd.read_csv(file_path, header=1, names=columns)

# %%
# Check for missing values; address any missing data if necessary
has_missing = df.isnull().values.any()
if has_missing:
    print('Missing values found. Dropping rows with missing values.')
    df.dropna(inplace=True)
else:
    print('No missing values found.')
df['text'] = df['text'].str.strip().str.lower()
df.drop_duplicates(subset='text', keep='first', inplace=True)
print(len(df))

# %%
lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

stop_words.add('goes')
stop_words.add('newspaper')
stop_words.add('journalist')
stop_words.add('journalists')
stop_words.add('pt')
stop_words.add('rsf')
stop_words.add('aacs')
stop_words.add('still')
stop_words.add('year')
stop_words.add('today')
stop_words.add('day')
stop_words.add('to have')
stop_words.add('about')
stop_words.add('hurts')
stop_words.add('yesterday')
stop_words.add('where')
stop_words.add('one')
stop_words.add('two')
stop_words.add('three')
stop_words.add('four')
stop_words.add('five')
stop_words.add('six')
stop_words.add('seven')
stop_words.add('eight')
stop_words.add('nine')
stop_words.add('ten')
stop_words.add('because')
stop_words.add('years')
stop_words.add('said')

# Tokenize the text


def tokenize(text):
    tokens = word_tokenize(text, language='english')
    for token in tokens:
        lword = token.lower()
        if lword.isalpha() and (lword not in stop_words) and (len(lword) > 2):
            lemWord = lemmatizer.lemmatize(lword)
            yield lemWord


# Step 3: Tokenize the text
df['tokens'] = df['translated_text'].apply(lambda x: [t for t in tokenize(x)])
# join the tokens back into a single string
df['cleaned_text'] = df['tokens'].apply(lambda x: ' '.join(x))
df.head()

# %%
# Step 3: Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Step 4: Analyze Sentiment


def analyze_sentiment(text):
    scores = analyzer.polarity_scores(text)
    return scores['compound']


df['compound_score'] = df['cleaned_text'].apply(analyze_sentiment)

# Step 5: Map Compound Scores to Sentiment Labels


def map_sentiment(compound):
    return "positive" if compound > 0 else "negative"


df['sentiment'] = df['compound_score'].apply(map_sentiment)

# Step 6: Save results or preview
print(df[['text', 'compound_score', 'sentiment']].head())

# %%

# plot the sentiment distribution
fig = df['sentiment'].value_counts().plot(kind="pie", autopct='%1.1f%%')
fig.set_title('Sentiment Distribution')
fig.set_ylabel('')
fig.set_xlabel('')
fig.legend(labels=['Positive', 'Negative'])
plt.show()

# plot bar chart of sentiment distribution
fig = df['sentiment'].value_counts().plot(kind='bar')
fig.set_title('Sentiment Distribution')
fig.set_ylabel('Count')
fig.set_xlabel('Sentiment')
# draw number of positive and negative counts on the bars
for index, value in enumerate(df['sentiment'].value_counts()):
    plt.text(index, value, str(value))

plt.show()

# %%
# Get the most common Names of People in negative and positive texts

with open("./Legislaturas/X.json") as json_file:
    legislature_json = json.load(json_file)

legislature = legislature_json["Legislatura"]

l_init_date = legislature["DetalheLegislatura"]["dtini"]  # 2005-03-10
l_end_date = legislature["DetalheLegislatura"]["dtfim"]  # 2009-10-14

deputies = legislature["Deputados"]["pt_ar_wsgode_objectos_DadosDeputadoSearch"]
parties = legislature["GruposParlamentares"]["pt_gov_ar_objectos_GPOut"]

deputies = [{
    "depId": dep["depId"],
    "depNomeParlamentar": dep["depNomeParlamentar"],
    "depNomeCompleto": dep["depNomeCompleto"],
    "gpId": dep["depGP"]["pt_ar_wsgode_objectos_DadosSituacaoGP"]["gpId"] if type(dep["depGP"]["pt_ar_wsgode_objectos_DadosSituacaoGP"]) is not list else dep["depGP"]["pt_ar_wsgode_objectos_DadosSituacaoGP"][-1]["gpId"],
    "gpSigla": dep["depGP"]["pt_ar_wsgode_objectos_DadosSituacaoGP"]["gpSigla"] if type(dep["depGP"]["pt_ar_wsgode_objectos_DadosSituacaoGP"]) is not list else dep["depGP"]["pt_ar_wsgode_objectos_DadosSituacaoGP"][-1]["gpSigla"]
}
    for dep in deputies
]

dp_df = pd.DataFrame(
    deputies,
    columns=["depId", "depNomeParlamentar", "depNomeCompleto", "gpId", "gpSigla"],
)
party_df = pd.DataFrame(parties)

dp_df

# %%
with open("./Legislaturas/GOV.json") as json_file:
    gov_json = json.load(json_file)

gov = gov_json["membros"]

gov_df = pd.DataFrame(
    gov,
    columns=["nome", "cargos"],
)

gov_df

# %%
# Get one member of the government by the cargo


def get_member_by_cargo(cargo):
    for member in gov:
        if cargo.lower() in list(map(lambda x: x.lower(), member['cargos'])):
            return member['nome']
    return None

# %%


def get_people_names(text):
    people_names = []
    all_names = []
    # Do not tokenize
    deputies_names = dp_df['depNomeParlamentar'].values
    gov_names = gov_df['nome'].values

    # merge the two lists of names
    all_names.extend(deputies_names)
    all_names.extend(gov_names)

    # Use lower case to remove duplicates
    all_names = list(set(
        [name.lower() for name in all_names]
    ))
    for name in all_names:
        if name.lower() in text:
            people_names.append(name)

    # Check for the existence of positions referring to people
    cargos = gov_df['cargos'].values
    cargos = [cargo for sublist in cargos for cargo in sublist]

    for cargo in cargos:
        if cargo.lower() in text or cargo.replace('-', ' ').lower() in text:
            member = get_member_by_cargo(cargo)
            if member is not None:
                people_names.append(member)

    return people_names


df['people_names'] = df['text'].apply(get_people_names)
print(df[['text', 'people_names']].head())

# %%
# Get the most common names of people in positive and negative texts
positive_people_names = [name for names in df[df['sentiment']
                                              == 'positive']['people_names'] for name in names]
negative_people_names = [name for names in df[df['sentiment']
                                              == 'negative']['people_names'] for name in names]

positive_people_names_counter = Counter(positive_people_names)
negative_people_names_counter = Counter(negative_people_names)

print('Most common names of people in positive texts:',
      positive_people_names_counter.most_common(5))
print('Most common names of people in negative texts:',
      negative_people_names_counter.most_common(5))

# %%


def process_data(input_data):
    combined_counts = Counter()
    for name, count in input_data.items():
        normalized_name = name.lower()
        combined_counts[normalized_name] += count
    sorted_counts = sorted(combined_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_counts[:20]


# Convert combined_counts to lists for plotting
# Sort combined_counts by count in descending order
sorted_counts = process_data(positive_people_names_counter)
neg_sorted_counts = process_data(negative_people_names_counter)

# Convert sorted_counts to lists for plotting
names, counts = zip(*sorted_counts)
neg_names, neg_counts = zip(*neg_sorted_counts)

# Plotting the first dataset
plt.figure(figsize=(10, 10))
plt.bar(names, counts, color="skyblue")
plt.xlabel("Name")
plt.ylabel("Count")
plt.title("Positive: Name vs Count")
plt.xticks(rotation=75)
plt.tight_layout()
plt.show()

# Plotting the second dataset
plt.figure(figsize=(10, 10))
plt.bar(neg_names, neg_counts, color="lightgreen")
plt.xlabel("Name")
plt.ylabel("Count")
plt.title("Negative: Name vs Count")
plt.xticks(rotation=75)
plt.tight_layout()
plt.show()

# %%

# Create a graph
G = nx.Graph()

# Add edges from depNomeParlamentar to gpSigla (party)
for _, row in dp_df.iterrows():
    G.add_edge(row['depNomeParlamentar'], row['gpSigla'])

# Assign unique colors to parties
party_colors = {
    'PEV': 'green',
    'PSD': 'blue',
    'PS': 'red',
    'BE': 'darkred',
    'PCP': 'yellow',
    'CDS-PP': 'orange'
}

# Assign colors to nodes
node_colors = []
for node in G.nodes():
    if node in dp_df['gpSigla'].values:  # Party nodes
        node_colors.append(party_colors.get(node, 'gray'))
    else:  # Parliamentarian nodes
        # Color parliamentarians based on the party they are connected to
        party = dp_df[dp_df['depNomeParlamentar'] == node]['gpSigla'].values[0]
        node_colors.append(party_colors.get(party, 'gray'))

# Draw the graph
plt.figure(figsize=(30, 30))
pos = nx.spring_layout(G, k=0.7, scale=3)  # Spread nodes out

nx.draw(
    G, pos, with_labels=True,
    node_color=node_colors,  # Apply colors
    node_size=3000,
    font_size=10,
    font_weight="bold",
    edge_color="gray"
)

plt.title("Parliamentarians and Parties Network (Colored by Party)", fontsize=16)
plt.show()

# %%

# Calculate degree centrality
degree_centrality = nx.degree_centrality(G)

# Extract degree centrality for party nodes
party_degree_centrality = {node: degree_centrality[node] for node in dp_df['gpSigla'].unique()}

# Plot degree centrality
plt.figure(figsize=(10, 6))
plt.bar(
    party_degree_centrality.keys(),
    party_degree_centrality.values(),
    color=['orange', 'yellow', 'blue', 'red', 'darkred', 'green']
)
plt.title("Degree Centrality of Parties", fontsize=16)
plt.xlabel("Party", fontsize=14)
plt.ylabel("Degree Centrality", fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %%


def process_data_2(input_data):
    combined_counts = Counter()
    for name, count in input_data.items():
        normalized_name = name.lower()
        combined_counts[normalized_name] += count
    sorted_counts = sorted(combined_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_counts


normalized_positive_counts = process_data(positive_people_names_counter)
normalized_negative_counts = process_data(negative_people_names_counter)

name_to_party = {row['depNomeParlamentar'].lower(): row['gpSigla']
                 for index, row in dp_df.iterrows()}


# Initialize dictionaries to store the counts per party
positive_articles_by_party = defaultdict(int)
negative_articles_by_party = defaultdict(int)

# Count positive articles per party
for name, count in normalized_positive_counts:
    if name in name_to_party:  # Check if the name exists in the mapping
        party = name_to_party[name]
        positive_articles_by_party[party] += count

# Count negative articles per party
for name, count in normalized_negative_counts:
    if name in name_to_party:  # Check if the name exists in the mapping
        party = name_to_party[name]
        negative_articles_by_party[party] += count


# Convert the dictionaries to DataFrame
party_article_counts = pd.DataFrame({
    'positive_articles': positive_articles_by_party,
    'negative_articles': negative_articles_by_party
}).reset_index()

# Rename the columns for clarity
party_article_counts.columns = ['gpSigla', 'positive_articles', 'negative_articles']
sorted_party_article_counts = party_article_counts.sort_values(by=['positive_articles'], ascending=False)

# %%

# Plotting positive articles per party
plt.figure(figsize=(10, 6))
plt.bar(party_article_counts['gpSigla'], party_article_counts['positive_articles'], color='green')
plt.xlabel('Party (gpSigla)')
plt.ylabel('Number of Positive Articles')
plt.title('Number of Positive Articles per Party')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()

# Plotting negative articles per party
plt.figure(figsize=(10, 6))
plt.bar(party_article_counts['gpSigla'], party_article_counts['negative_articles'], color='red')
plt.xlabel('Party (gpSigla)')
plt.ylabel('Number of Negative Articles')
plt.title('Number of Negative Articles per Party')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()

