
# coding: utf-8

# #Predicting sentiment from product reviews
# 
# #Fire up GraphLab Create

# In[1]:

import graphlab


# #Read some product review data
# 
# Loading reviews for a set of baby products. 

# In[2]:

products = graphlab.SFrame('amazon_baby.gl/')


# #Let's explore this data together
# 
# Data includes the product name, the review text and the rating of the review. 

# In[3]:

products.head()


# #Build the word count vector for each review

# In[4]:

products['word_count'] = graphlab.text_analytics.count_words(products['review'])


# In[5]:

products.head()


# In[6]:

graphlab.canvas.set_target('ipynb')


# In[7]:

products['name'].show()


# #Examining the reviews for most-sold product:  'Vulli Sophie the Giraffe Teether'

# In[8]:

giraffe_reviews = products[products['name'] == 'Vulli Sophie the Giraffe Teether']


# In[9]:

len(giraffe_reviews)


# In[10]:

giraffe_reviews['rating'].show(view='Categorical')


# #Build a sentiment classifier

# In[11]:

products['rating'].show(view='Categorical')


# ##Define what's a positive and a negative sentiment
# 
# We will ignore all reviews with rating = 3, since they tend to have a neutral sentiment.  Reviews with a rating of 4 or higher will be considered positive, while the ones with rating of 2 or lower will have a negative sentiment.   

# In[12]:

#ignore all 3* reviews
products = products[products['rating'] != 3]


# In[13]:

#positive sentiment = 4* or 5* reviews
products['sentiment'] = products['rating'] >=4


# In[14]:

products.head()


# ##Let's train the sentiment classifier

# In[15]:

train_data,test_data = products.random_split(.8, seed=0)


# In[16]:

sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=['word_count'],
                                                     validation_set=test_data)


# #Evaluate the sentiment model

# In[17]:

sentiment_model.evaluate(test_data, metric='roc_curve')


# In[18]:

sentiment_model.show(view='Evaluation')


# #Applying the learned model to understand sentiment for Giraffe

# In[19]:

giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews, output_type='probability')


# In[20]:

giraffe_reviews.head()


# ##Sort the reviews based on the predicted sentiment and explore

# In[21]:

giraffe_reviews = giraffe_reviews.sort('predicted_sentiment', ascending=False)


# In[22]:

giraffe_reviews.head()


# ##Most positive reviews for the giraffe

# In[23]:

giraffe_reviews[0]['review']


# In[24]:

giraffe_reviews[1]['review']


# ##Show most negative reviews for giraffe

# In[25]:

giraffe_reviews[-1]['review']


# In[26]:

giraffe_reviews[-2]['review']


# In[27]:

selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']


# In[28]:

selected_words[0]


# In[29]:

def awesome_count(word_count_dictionary):
    if 'awesome' in word_count_dictionary:
        return word_count_dictionary['awesome']
    else:
        return 0


# In[30]:

products['awesome']=products['word_count'].apply(awesome_count)


# In[31]:

products.head()


# In[32]:

a = products[products['awesome']>0]


# In[33]:

len(a)


# In[34]:

a[0]['review']


# In[38]:

for word in selected_words:
    products[word]=products['word_count'].apply(
    lambda d: d.get(word,0))


# In[39]:

products.head()


# In[41]:

train_data, test_data = products.random_split(.8, seed = 0)
keyword_model = graphlab.logistic_classifier.create(train_data,
                                                   target = 'sentiment',
                                                   features = selected_words,
                                                   validation_set = test_data)


# In[42]:

keyword_model['coefficients']


# In[43]:

keyword_model.evaluate(test_data)


# In[44]:

sentiment_model.evaluate(test_data)


# In[74]:

keyword_model['coefficients'].sort('value', ascending = False)


# In[48]:

diaper_champ_reviews = products[products['name']=='Baby Trend Diaper Champ']


# In[49]:

len(diaper_champ_reviews)


# In[50]:

diaper_champ_reviews['predicted_sentiment'] = sentiment_model.predict(
diaper_champ_reviews, output_type = 'probability')


# In[51]:

diaper_champ_reviews.head()


# In[52]:

diaper_champ_reviews=diaper_champ_reviews.sort('predicted_sentiment', ascending = False)


# In[53]:

diaper_champ_reviews[0]['review']


# In[54]:

diaper_champ_reviews[-1]['review']


# In[55]:

diaper_champ_reviews[0]['predicted_sentiment']


# In[56]:

diaper_champ_reviews['key_predicted_sentiment'] = keyword_model.predict(
diaper_champ_reviews, output_type = 'probability')


# In[58]:

diaper_champ_reviews[0]['predicted_sentiment']


# In[60]:

diaper_champ_reviews[0]['key_predicted_sentiment']


# In[67]:

diaper_champ_reviews[0]['word_count']


# In[68]:

diaper_champ_reviews.head()


# In[71]:

for word in selected_words:
    print word, products[word].sum()


# In[76]:

products['sentiment'].sum()


# In[77]:

len(products['sentiment'])


# In[78]:

140259/float(166752)


# In[ ]:



