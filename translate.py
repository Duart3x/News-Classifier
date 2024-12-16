from threading import Thread
from deep_translator import GoogleTranslator
import nltk
from pandas.core.api import DataFrame
import numpy as np
import pandas as pd

THREAD_NUMBER = 64


class ReturningThread(Thread):

    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


nltk.download("vader_lexicon")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

# Step 2: Load the dataset
file_path = "results-publico.csv"
columns = ["term", "url", "text", "title"]
df = pd.read_csv(file_path, header=1, names=columns)

# Check for missing values; address any missing data if necessary
has_missing = df.isnull().values.any()
if has_missing:
    print("Missing values found. Dropping rows with missing values.")
    df.dropna(inplace=True)
else:
    print("No missing values found.")
df["text"] = df["text"].str.strip().str.lower()
df.drop_duplicates(subset="text", keep="first", inplace=True)


def translate_text(translator, sentences, t_n, tries):
    print(f"#thread {t_n}: Translating {len(sentences)} sentences")
    # print(sentences)
    # return str(sentences)
    for i in range(tries):
        try:
            tr = " ".join(translator.translate_batch(sentences))
            print(tr)
            return tr
        except Exception as e:
            print(f"thread {t_n}, try {i}, error {e}")
    return None


def spli_in_5000_sentences(text):
    return [text[i:i + 4998] for i in range(0, len(text), 4998)]
    # return text.split(".")


# Step 4: Translate the text
df['sentences'] = df['text'].apply(spli_in_5000_sentences)
df['translated_text'] = ""
# df["translated_text"] = df["sentences"].apply(lambda x: translate_text(x, 0))
df.reset_index(drop=True, inplace=True)
print(df.head())


def thread(df: pd.DataFrame, t_n: int) -> DataFrame:
    translator = GoogleTranslator(source='pt', target='en')
    print("running from thread ", i)
    df["translated_text"] = df["sentences"].apply(lambda x: translate_text(translator, x, t_n, 3))
    return df


batches = np.array_split(df, THREAD_NUMBER)
for i in batches:
    print(i.__len__())

threads: list[Thread] = []

for i in range(0, THREAD_NUMBER):
    print("starting thread ", i)
    t = ReturningThread(target=thread, args=[batches[i], i])
    t.start()
    threads.append(t)

print(len(threads))

dfs = []

for t in threads:
    df_1 = t.join()
    dfs.append(df_1)

df_2 = pd.concat(dfs)
print(len(threads))

print(len(df_2))

# df_2['translated_text'] = df_2['translated_text'].apply(lambda x: ' '.join(str(x)))

# remove setences column
df_2 = df_2.drop(columns=['sentences'])

df_2.to_csv("results-publico-translated.csv", index=False)
