
import streamlit as st

doc = st.text_area('Area for textual entry')

with st.beta_expander('Test text'):
    '''
    Next, we convert both the document as well as the candidate keywords/keyphrases to numerical data. We use BERT for this purpose as it has shown great results for both similarity- and paraphrasing tasks.
    There are many methods for generating the BERT embeddings, such as Flair, Hugginface Transformers, and now even spaCy with their 3.0 release! However, I prefer to use the sentence-transformers package as it allows me to quickly create high-quality embeddings that work quite well for sentence- and document-level embeddings.
    We install the package with pip install sentence-transformers. If you run into issues installing this package, then it might be helpful to install Pytorch first.
    Now, we are going to run the following code to transform our document and candidates into vectors:
    '''

text = """
Next, we convert both the document as well as the candidate keywords/keyphrases to numerical data. We use BERT for this purpose as it has shown great results for both similarity- and paraphrasing tasks.
There are many methods for generating the BERT embeddings, such as Flair, Hugginface Transformers, and now even spaCy with their 3.0 release! However, I prefer to use the sentence-transformers package as it allows me to quickly create high-quality embeddings that work quite well for sentence- and document-level embeddings.
We install the package with pip install sentence-transformers. If you run into issues installing this package, then it might be helpful to install Pytorch first.
Now, we are going to run the following code to transform our document and candidates into vectors:
"""
#doc
#st.beta_expander('Expander')


st.write('')
st.write('')

from sklearn.feature_extraction.text import CountVectorizer

n_gram_range = (3, 3)
stop_words = "english"

# Extract candidate words/phrases
count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([doc])
candidates = count.get_feature_names()

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
doc_embedding = model.encode([doc])
candidate_embeddings = model.encode(candidates)

from sklearn.metrics.pairwise import cosine_similarity

top_n = 10
distances = cosine_similarity(doc_embedding, candidate_embeddings)
keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

keywords








