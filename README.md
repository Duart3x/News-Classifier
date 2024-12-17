# News Classifier
This project aims to classify news articles obtained from artigo.pt API into political
spectrums. 

The project is divided into three main parts: the data collection, data visualization and the model training.

## Data Collection ([collect.ipynb](collect.ipynb))
The data collection is done using the artigo.pt API and parlamento.pt json file 
with the information about the X Legislature of Portugal. 
The API is used to obtain news articles from different portuguese newspapers 
querying by the name of deputies and political parties.

The data retrieved from arquivo.pt is only the urls of the articles, 
so we need to scrape the articles to get the text.
The scraping is done using the newspaper3k library.

After that, the data is saved in a csv file with the following columns: term,url,text,title.

## Exploratory Data Analysis ([EDA.ipynb](EDA.ipynb))
The EDA is done using the data collected in the previous step. We plot some statistics about the data,
like the distribution of the length of the articles and we filter the articles by 
political category.

We also plot the most common words in the articles as the bigrams and trigrams. It is
also plotted the wordcloud of the most common words.

## AI Analysis ([model.ipynb](model.ipynb))
The AI analysis is done by sentiment analysis of the articles.
To do that, first we need to translate the articles to english using the googletrans library in the 
(translate.py)[translate.py] file.

After all the text is translated, we use Vader from the nltk library to classify the sentiment of the articles.
The sentiment is classified in two categories: positive and negative.
Then we search for the politicians names referenced in the articles and coorelate the sentiment of the article with the politician.
With this information we can visualize the sentiment of the newspapers towards the politicians and theirs parties.