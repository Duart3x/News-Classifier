from threading import Lock, Thread

import nltk
import pandas as pd
from deep_translator import GoogleTranslator

THREAD_NUMBER = 48

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

translator = GoogleTranslator(source="pt", target="en")


def translate_text(sentences, i):
    print(f"#{i}: Translating {len(sentences)} sentences")
    return translator.translate_batch(sentences)


def spli_in_5000_sentences(text):
    return [text[i : i + 4998] for i in range(0, len(text), 4998)]


# Step 4: Translate the text
df["sentences"] = df["text"].apply(spli_in_5000_sentences)
df["translated_text"] = ""
df.reset_index(drop=True)

lock = Lock()


def thread(begin: int, end: int, lock: Lock) -> None:
    global df

    for i in range(begin, end + 1):
        try:
            translated = translate_text(df["sentences"][i], i)

            lock.acquire()
            df["translated_text"][i] = translated
            lock.release()
        except KeyError:
            print(i, "error")


step = len(df) // THREAD_NUMBER
missing = len(df) - step * THREAD_NUMBER
threads: list[Thread] = []

for i in range(0, THREAD_NUMBER):
    t = Thread(target=thread, args=[i * step, i + 1 * step - 1, lock])
    t.start()
    threads.append(t)

t = Thread(target=thread, args=[(THREAD_NUMBER) * step, len(df) - 1, lock])
t.start()
threads.append(t)

for t in threads:
    t.join()


print(len(df))
df.head()

df["translated_text"] = df["translated_text"].apply(lambda x: " ".join(x))
df.head()

# remove setences column
df = df.drop(columns=["sentences"])
df.head()

df.to_csv("results-publico-translated-test.csv", index=False)
