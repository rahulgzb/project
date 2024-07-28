import pandas as pd
from datasets import load_dataset
import re
import demoji

'''
do in it ..
- Analyze the structure and content of the dataset. Remove any irrelevant information. 
not using this since it already T5 already trained on large corpus of words which does not required these steps 
- Apply techniques such as tokenization, stop word removal, and stemming/lemmatization.

'''


def clean_text(text):
    # Remove URLs
    url_pattern = r'https?://\S+|www\.\S+'
    text = re.sub(url_pattern, '', text, flags=re.MULTILINE)

    # Remove emojis
    text =demoji.replace(text, '')

    #keeping english only since data contains hindi and mixed language
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Remove bullet points
    bullet_points = ['•', '●', '▪', '-', '*']
    for bullet in bullet_points:
        text = text.replace(bullet, ' ')
    ## relinked text 
    linked_text_pattern = r'\[linked_text:.*?\]'
    text = re.sub(linked_text_pattern, '', text, flags=re.MULTILINE)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def _create_splits(split_size,data):

    test= data.loc[data.target.isna()].dropna()
    train_val= data.loc[~data.target.isna()].dropna()

    train_val= train_val.sample(frac=1).reset_index(drop=True)
    len_train=int(len(train_val)*split_size)

    train= train_val[:len_train]
    val= train_val[len_train:]
    test.to_csv("../data/text.csv",index= False)
    val.to_csv("../data/val.csv",index=False)
    train.to_csv("../data/train.csv",index=False)
    print("all train val split files are created and loaded \n in the project/data directory")

 
def cleaner_function(text):
    if text:
        try:
           text=clean_text(text)
        except Exception as e:
            print(f"cannot be cleaned {text} due to {e}")
    return text

def data_cleaner(split_size=0.7):
    df=pd.read_csv("../data/full_data.csv")
    ## split size % split between val and train 
    print(f"shape of full data before _cleaning {df.shape}")
    df = df.drop_duplicates()
    df.columns= ['question', 'target']

    print("start cleaning for questions..")
    df.question=df.question.apply(cleaner_function)
    print("started cleaning for answers .. ")
    df.target=df.target.apply(cleaner_function)
    print('completed cleaning ..')
    ## 
    print("initiating train val text split ...")
    _create_splits(split_size,df)
    print(f"shape of train data before _cleaning {df.shape}")


   
def download_data():
    # data loading from hugging face 
    ds = load_dataset("toughdata/quora-question-answer-dataset")
    df = pd.DataFrame(ds["train"])
    print("data_downloaded..")
    df.to_csv("../data/full_data.csv",index=False)


if __name__=="__main__":
    data_cleaner()
    pass