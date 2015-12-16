# # Document retrieval from wikipedia data
import graphlab


# # Load some text data - from wikipedia, pages on people
people = graphlab.SFrame('people_wiki.gl/')
people.head()
len(people)


# # Explore the dataset and checkout the text it contains
# 
# ## Exploring the entry for president Obama
obama = people[people['name'] == 'Barack Obama']
obama
obama['text']


# ## Exploring the entry for actor George Clooney
clooney = people[people['name'] == 'George Clooney']
clooney['text']


# # Get the word counts for Obama article
obama['word_count'] = graphlab.text_analytics.count_words(obama['text'])
print obama['word_count']
obama_word_count_table = obama[['word_count']].stack('word_count', new_column_name = ['word','count'])
obama_word_count_table.head()
obama_word_count_table.sort('count',ascending=False)


# Most common words include uninformative words like "the", "in", "and",...

# # Compute TF-IDF for the corpus 
# 
# To give more weight to informative words, we weigh them by their TF-IDF scores.
people['word_count'] = graphlab.text_analytics.count_words(people['text'])
people.head()
tfidf = graphlab.text_analytics.tf_idf(people['word_count'])
tfidf
people['tfidf'] = tfidf


# ## Examine the TF-IDF for the Obama article
obama = people[people['name'] == 'Barack Obama']
obama[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)

clinton = people[people['name'] == 'Bill Clinton']
beckham = people[people['name'] == 'David Beckham']


# ## Is Obama closer to Clinton than to Beckham?
# 
# We will use cosine distance, which is given by
# 
# (1-cosine_similarity) 
# 
# and find that the article about president Obama is closer to the one about former president Clinton than that of footballer David Beckham.
graphlab.distances.cosine(obama['tfidf'][0],clinton['tfidf'][0])
graphlab.distances.cosine(obama['tfidf'][0],beckham['tfidf'][0])


# # Build a nearest neighbor model for document retrieval
# 
# We now create a nearest-neighbors model and apply it to document retrieval.  
knn_model = graphlab.nearest_neighbors.create(people,features=['tfidf'],label='name')
knn_model.query(obama)

# #Homework Starts Here
elton = people[people['name']=='Elton John']
elton_word_count_table = elton[['word_count']].stack('word_count', new_column_name = ['word','count'])
elton_word_count_table.sort('count', ascending = False)
elton[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)
type(elton['tfidf'])
type(elton[['tfidf']])

victoria = people[people['name']=='Victoria Beckham']
paul = people[people['name']=='Paul McCartney']
graphlab.distances.cosine(elton['tfidf'][0],victoria['tfidf'][0])
graphlab.distances.cosine(elton['tfidf'][0], paul['tfidf'][0])

word_count_model = graphlab.nearest_neighbors.create(people,features=['word_count'],label='name', distance = 'cosine')
tfidf_model = graphlab.nearest_neighbors.create(people,features=['tfidf'],label='name', distance = 'cosine')
word_count_model.query(elton)
tfidf_model.query(elton)
word_count_model.query(victoria)
tfidf_model.query(victoria)


