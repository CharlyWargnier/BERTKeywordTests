
import streamlit as st

doc = st.text_area('Area for textual entry')

with st.beta_expander('Test text from post'):
    '''
        Supervised learning is the machine learning task of 
         learning a function that maps an input to an output based 
         on example input-output pairs.[1] It infers a function 
         from labeled training data consisting of a set of 
         training examples.[2] In supervised learning, each 
         example is a pair consisting of an input object 
         (typically a vector) and a desired output value (also 
         called the supervisory signal). A supervised learning 
         algorithm analyzes the training data and produces an 
         inferred function, which can be used for mapping new 
         examples. An optimal scenario will allow for the algorithm 
         to correctly determine the class labels for unseen 
         instances. This requires the learning algorithm to  
         generalize from the training data to unseen situations 
         in a 'reasonable' way (see inductive bias).

    '''

with st.beta_expander('Test text 01'):
    '''
    Next, we convert both the document as well as the candidate keywords/keyphrases to numerical data. We use BERT for this purpose as it has shown great results for both similarity- and paraphrasing tasks.
    There are many methods for generating the BERT embeddings, such as Flair, Hugginface Transformers, and now even spaCy with their 3.0 release! However, I prefer to use the sentence-transformers package as it allows me to quickly create high-quality embeddings that work quite well for sentence- and document-level embeddings.
    We install the package with pip install sentence-transformers. If you run into issues installing this package, then it might be helpful to install Pytorch first.
    Now, we are going to run the following code to transform our document and candidates into vectors:
    '''
with st.beta_expander('Test text 02'):
    '''

    1958–1975: Early life and the Jackson 5
    The single-story house has white walls, two windows, a central white door with a black door frame, and a black roof. In front of the house there is a walkway and multiple colored flowers and memorabilia.
    Jackson's childhood home in Gary, Indiana, pictured in March 2010 with floral tributes after his death
    Michael Joseph Jackson[7][8] was born in Gary, Indiana, near Chicago, on August 29, 1958.[9][10] He was the eighth of ten children in the Jackson family, a working-class African-American family living in a two-bedroom house on Jackson Street.[11][12] His mother, Katherine Esther Jackson (née Scruse), played clarinet and piano, had aspired to be a country-and-western performer, and worked part-time at Sears.[13] She was a Jehovah's Witness.[14] His father, Joseph Walter "Joe" Jackson, a former boxer, was a crane operator at U.S. Steel and played guitar with a local rhythm and blues band, the Falcons, to supplement the family's income.[15][16] Joe's great-grandfather, July "Jack" Gale, was a US Army scout; family lore held that he was also "a Native American medicine man".[17] Michael grew up with three sisters (Rebbie, La Toya, and Janet) and five brothers (Jackie, Tito, Jermaine, Marlon, and Randy).[15] A sixth brother, Marlon's twin Brandon, died shortly after birth.[18]
    Joe acknowledged that he regularly whipped Michael;[19] Michael said his father told him he had a "fat nose,"[20] and regularly physically and emotionally abused him during rehearsals. He recalled that Joe often sat in a chair with a belt in his hand as he and his siblings rehearsed, ready to physically punish any mistakes.[14][21] Katherine Jackson stated that although whipping is considered abuse in more modern times, it was a common way to discipline children when Michael was growing up.[22][23] Jackie, Tito, Jermaine and Marlon have said that their father was not abusive and that the whippings, which were harder on Michael because he was younger, kept them disciplined and out of trouble.[24] Jackson said his youth was lonely and isolated.[25]
    In 1964, Michael and Marlon joined the Jackson Brothers—a band formed by their father which included Jackie, Tito, and Jermaine—as backup musicians playing congas and tambourine.[26][27] Later that year, Michael began sharing lead vocals with Jermaine, and the group's name was changed to the Jackson 5.[28] The following year, the group won a talent show; Michael performed the dance to Robert Parker's 1965 song "Barefootin'" and singing lead to The Temptations' "My Girl".[29] From 1966 to 1968 they toured the Midwest; they frequently played at a string of black clubs known as the "Chitlin' Circuit" as the opening act for artists such as Sam & Dave, the O'Jays, Gladys Knight, and Etta James. The Jackson 5 also performed at clubs and cocktail lounges, where striptease shows were featured, and at local auditoriums and high school dances.[30][31] In August 1967, while touring the East Coast, they won a weekly amateur night concert at the Apollo Theater in Harlem.[32]
    Jackson (center) as a member of the Jackson 5 in 1972. The group were among the first African American performers to attain a crossover following.[33]
    The Jackson 5 recorded several songs for a Gary record label, Steeltown Records; their first single, "Big Boy", was released in 1968.[34] Bobby Taylor of Bobby Taylor & the Vancouvers brought the Jackson 5 to Motown after the group opened for Taylor at Chicago's Regal Theater in 1968. Taylor also produced some of their early recordings for the label, including a version of "Who's Lovin' You".[35] After signing with Motown, the Jackson family relocated from Gary to Los Angeles.[36] In 1969, executives at Motown decided Diana Ross should introduce the Jackson 5 to the public—partly to bolster her career in television—sending off what was considered Motown's last product of its "production line".[37] The Jackson 5 made their first television appearance in 1969 in the Miss Black America Pageant where they performed a cover of "It's Your Thing".[38] Rolling Stone later described the young Michael as "a prodigy" with "overwhelming musical gifts" who "quickly emerged as the main draw and lead singer".[39]
    In January 1970, "I Want You Back" became the first Jackson 5 song to reach number one on the US Billboard Hot 100; it stayed there for four weeks. Three more singles with Motown—"ABC", "The Love You Save", and "I'll Be There"—also topped the chart.[40] In May 1971, the Jackson family moved into a large house on a two-acre estate in Encino, California.[41] During this period, Michael developed from a child performer into a teen idol.[42] As he emerged as a solo performer in the early 1970s, he maintained ties to the Jackson 5. Between 1972 and 1975, Michael released four solo studio albums with Motown: Got to Be There (1972), Ben (1972), Music & Me (1973), and Forever, Michael (1975).[43] "Got to Be There" and "Ben", the title tracks from his first two solo albums, sold well as singles, as did a cover of Bobby Day's "Rockin' Robin".[44]
    The Jackson 5 were later described as "a cutting-edge example of black crossover artists."[45] They were frustrated by Motown's refusal to allow them creative input.[46] Jackson's performance of their top five single "Dancing Machine" on Soul Train popularized the robot dance.[47]

    '''

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

top_n = 5
distances = cosine_similarity(doc_embedding, candidate_embeddings)
keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

keywords

