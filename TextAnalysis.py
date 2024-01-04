#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries and Dataset

# In[1]:


# data analysis
import pandas as pd
import numpy as np

# graphing
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

# natural language toolkit for sentiment analysis
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# progress bar
from tqdm import tqdm 

# word clouds
from wordcloud import WordCloud, STOPWORDS

import re


# We will be using two different datasets in this project - one to test the sentiment analysis library and show that it is working as intended, and one to perform actual sentiment analysis on social media posts to gain insight into the overall sentiment of Avengers Endgame on April 23rd, 2019, three days before its U.S. release.
# 
# The dataset we'll be using for testing the natural language toolkit is "Amazon Fine Food Reviews", published by Stanford Network Analysis Project to kaggle.com, located at https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews. This dataset contains information from amazon reviews, which have a text review and a score from one to five stars. This is an important feature for testing our sentiment analysis, as it gives us a user-reported numerical measurement to measure against our own scores, interpreted from the text. If these two scores are reasonably similar, then we can conclude that the nltk library is accurate enough for our purposes.

# In[2]:


# dataset is cut at 3000 rows for efficiency
df = pd.read_csv('Reviews.csv')
print(df.shape)


# In[3]:


df.head()


# # Preparing Test Data

# In[4]:


# plot a simple graph to show the distribution of star ratings in the dataset
reviewChart = df['Score'].value_counts().sort_index().plot(kind='bar', title='Number of Reviews by Stars', figsize = (10,5))
reviewChart.set_xlabel('Star Rating')
reviewChart.set_ylabel('Number of Reviews')
plt.show()


# In[5]:


# take a random review as an example
example = df['Text'][50]
print(example)


# In[6]:


# splits the review into individual strings, or "tokens", including punctuation
tokens = nltk.word_tokenize(example)
tokens[:10]


# In[7]:


# tags each word with a marker that marks what part of a sentence it is (noun, adjective, etc.)
tagged_tokens = nltk.pos_tag(tokens)
tagged_tokens[:10]


# In[8]:


# takes a list of tagged tokens and groups them into chunks of text
entities = nltk.chunk.ne_chunk(tagged_tokens)
entities.pprint()


# Once the words are appropriately tokenized, tagged, and chunked, we can use the sentiment intensity analyzer to determine how positive or negative each review is. To do this, we will need to repeat this process for the whole dataset, then analyze these tagged chunks.

# # Testing NLTK Sentiment Scoring
# We will use python's NLTK, or natural language toolkit, library in order to generate numerical scores for a review's overall sentiment towards an item. NLTK's Vader sentiment analyzer uses a bag of words approach, which means it approximates the positive or negative sentiment of each individual word then averages the score of all the words to find the overall scoring. Notably, this approach does not account for context between words, since it only analyzes words individually. However, it provides a decent starting point to analyze sentiment.

# In[9]:


sia = SentimentIntensityAnalyzer()


# In[10]:


sia.polarity_scores('I am so happy!')


# In[11]:


sia.polarity_scores('This is the worst thing ever.')


# In[12]:


# example = 'This oatmeal is not good. Its mushy, soft, I don't like it. Quaker Oats is the way to go.'
sia.polarity_scores(example)


# After running these examples, we can conclude that the scoring for the sentiment analysis is working properly. The first example was strongly positive, and resulted in a compound score of 0.65. The second example was strongly negative, and produced a compound score of -0.62. Therefore, we can see that the values are scored from -1 to 1, with negative scores being negative sentiments, 0 being neutral, and 1 being positive.
# 
# When we input our earlier example text from the dataset, which states "This oatmeal is not good. Its mushy, soft, I don't like it. Quaker Oats is the way to go", we find that we receive a compound score of -0.54, which is reasonably negative for such a bad review. Since it appears to be working well on these examples, we will now run the sentiment analysis on the entire dataset and save the values to a dataframe.

# In[13]:


# Run the polarity score on the entire dataset 
# note: this can take a while for large datasets, but the test dataset has been cut for speed
result = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    result[myid] = sia.polarity_scores(text)


# In[14]:


# import results into a dataframe and flip the dataframe
sentiment = pd.DataFrame(result).T
# rename index
sentiment = sentiment.reset_index().rename(columns={'index':'Id'})
# left join sentiment scores with original data
sentiment = sentiment.merge(df, how='left')


# In[15]:


sentiment.head()


# In[16]:


print(sentiment.shape)


# # Plot Results

# In[17]:


sns.set_style(rc = {'axes.facecolor': 'lightgray'})
compound_graph = sns.barplot(data=sentiment, x='Score', y='compound')
compound_graph.set_title('Overall Sentiment Score by Star Rating per Review')
compound_graph.set_xlabel('Star Rating')
compound_graph.set_ylabel('Sentiment Score')


# The compound score, which we have graphed here on the y-axis, is an aggregation of the positive and negative scores for a statement and provides a measurement of the overall sentiment. As we can see from this plot, the compound scores seem to generally follow the number of stars a review has given a product, which is a good sign. For further analysis, we can also check the positive, negative, and neutral scores to ensure they follow the star ratings as expected.

# In[18]:


sns.set_style(rc = {'axes.facecolor': 'lightgray'})
fig, axs = plt.subplots(1, 3, figsize=(12,5))
sns.barplot(data=sentiment, x='Score', y='pos', ax=axs[0])
sns.barplot(data=sentiment, x='Score', y='neu', ax=axs[1])
sns.barplot(data=sentiment, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive Sentiment')
axs[0].set_xlabel('Star Rating')
axs[0].set_ylabel('Sentiment (Positive)')

axs[1].set_title('Neutral Sentiment')
axs[1].set_xlabel('Star Rating')
axs[1].set_ylabel('Sentiment (Neutral)')

axs[2].set_title('Negative Sentiment')
axs[2].set_xlabel('Star Rating')
axs[2].set_ylabel('Sentiment (Negative)')
plt.tight_layout()
plt.show()


# The scores generally match the number of stars, as we would expect, and so we can conclude that this sentiment analysis is reasonably accurate, despite its shortcomings.
# 
# Now that we have fully tested the sentiment of the dataset as a whole, we could choose to filter it and check the sentiments for specific criteria, such as reviews for certain items. However, since this dataset already had review scores, that would provide little benefit over using the scores given by the review writers themselves. For a better use of this technology, we will run sentiment analysis on social media posts to gain an understanding of overall sentiment in a given community at a certain time.

# # Performing Sentiment Analysis on Twitter (X) Posts

# Using another dataset uploaded to kaggle.com titled "Twitter Dataset - #AvengersEndgame" by Kavita Lolayekar, we can use sentiment analysis to determine the reception of a movie on X, formerly known as Twitter.
# 
# We will begin by importing the dataset, running sentiment analysis on it, and storing the results in a dataframe.

# In[19]:


tweets = pd.read_csv('tweets.csv', encoding= 'unicode_escape')
tweets.head()


# In[20]:


len(tweets)


# In[21]:


# Run the polarity score on the entire dataset
# note: this can take a while
result2 = {}
for i, row in tqdm(tweets.iterrows(), total=len(tweets)):
    text = row['text']
    myid = row['index']
    result2[myid] = sia.polarity_scores(text)


# In[22]:


# import results into a dataframe and flip the dataframe
tweet_sentiment = pd.DataFrame(result2).T
# rename index
tweet_sentiment = tweet_sentiment.reset_index()

# left join sentiment scores with original data
tweet_sentiment = tweet_sentiment.merge(tweets, how='left')
tweet_sentiment.head()


# # Results of Sentiment Analysis

# In[23]:


sns.set_style(rc = {'axes.facecolor': 'lightgray'})
plt.figure(figsize=(8, 4))
sns.violinplot(data=tweet_sentiment, y='compound', color = 'purple')
plt.suptitle('Overall Sentiment on Avengers Endgame', fontsize=14, x=0.52, y=0.98)
plt.title('from Tweets Made on 2019-04-23', fontsize=10)
plt.ylabel('Sentiment')
plt.show()


# The overall reception seems to be mostly neutral, but leans more positive than negative. The large number of neutral tweets makes sense - although the movie had premiered in Los Angeles at the time, this data was actually collected three days before the movie's full release in the United States. Many posts about the movie are likely to be neutral discussion rather than strong opinions, since most people have not actually seen the movie yet. However, the volume of positive sentiments suggests an overall positive outlook for the movie, suggesting people are excited to see it.

# In[24]:


sns.set_style(rc = {'axes.facecolor': 'lightgray'})
fig, axs = plt.subplots(1, 3, figsize=(12,5))
sns.violinplot(data=tweet_sentiment, y='pos', ax=axs[0], color='green')
sns.violinplot(data=tweet_sentiment, y='neu', ax=axs[1], color='yellow')
sns.violinplot(data=tweet_sentiment, y='neg', ax=axs[2], color='red')
axs[0].set_title('Positive Sentiment')
axs[1].set_title('Neutral Sentiment')
axs[2].set_title('Negative Sentiment')
plt.tight_layout()
plt.show()


# We can further break down the sentiment scores into positive, neutral, and negative scores, to see how well represented each is. This is not an indication of overall sentiment, since the approach used scores each word individually to generate these positive and negative scores. Rather, it is an indication of how many positive or negative words were used in a particular twitter post. However, we still see a fairly similar pattern, where negative sentiment is rather low and neutral sentiment is high. There are some larger volumes of positive scores than negative, but nothing overwhelmingly positive. Again, this makes sense given the state of the movie at this time.

# In[25]:


sns.set_style(rc = {'axes.facecolor': 'lightgray'})
plt.figure(figsize=(8,4))
sns.scatterplot(data=tweet_sentiment, y ='retweetCount', x ='compound', palette='Spectral', hue='compound')
plt.suptitle('Number of Retweets by Overall Sentiment', fontsize=14, x=0.51, y=0.98)
plt.title('on Avengers Endgame from Tweets Made on 2019-04-23', fontsize=10)
plt.ylabel('Retweets')
plt.xlabel('Sentiment')
plt.show()


# The number of retweets based on sentiment is interesting, as we see that most posts with a high number of retweets are either neutral or moderately to highly positive. Only a few posts with somewhat negative sentiments gained much traction, whereas a large portion of positive tweets are at or above 5000 retweets. This suggests once again that the general sentiment in this community was excitement for the movie, and a positive impression of it. 

# This positive sentiment and excitement would prove to be the correct interpretation, as Avengers Endgame would result in a strong 2.79 billion dollar box office sales. However, if we wanted to further analyze what was being discussed about the movie on Twitter before its release, we could analyze the text of the posts themselves to see what words are most common.

# # Avengers Endgame Twitter Posts Word Cloud

# In order to create a word cloud using the text from the Twitter dataset, we will need to follow a similar method as we did to perform sentiment analysis. We will break the word up into tokens to begin with, but rather than tagging each token, we will clean the sentences to remove any unwanted words such as links or unicode tags, such as those for emojis, that will not display properly in the word cloud.

# In[26]:


# the string that will be used for the word cloud
comment_words = ''

for index, row in tweets.iterrows():
    text = row['text']
    tokens = text.split()

    # removes capital letters from the whole word
    tokens = [token.lower() for token in tokens]

    # filter out links and emoji codes
    tokens = [token for token in tokens if not re.search('https.+|<.+>', token)]

    # adds cleaned tokens to string
    comment_words += " ".join(tokens) + " "


# In[27]:


stopwords = ["rt", "re", "s", "u", "m","t"] + list(STOPWORDS)

wordcloud = WordCloud(width = 800, height = 800, colormap="PuOr_r", min_font_size = 2,
background_color ='black', max_words=50, stopwords = stopwords).generate(comment_words)
 
# plot the WordCloud image                       
plt.figure(figsize = (8, 8), facecolor = 'black')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()


# This word cloud is rather interesting, as we can quickly see the influence retweets have on the chosen words. "helloboon man", "everywhere", and "avengersendgame ads" seem to be very prominent, and that is because the phrase comes from a popular tweet containing the phrase "RT @HelloBoon: Man these #AvengersEndgame ads are everywhere https://t.co/Q0lNf5eJsX". The original tweet has over 70,000 retweets, which explains why it is so prevalent in the dataset. 
# 
# Besides the retweet content, we see a lot of discussion about Marvel superheroes, actors, and MCU, or the Marvel Cinematic Universe. We also note what appear to be ads for Funko Pops and BoxLunch, likely due to promotions or giveaways they ran relating to the movie, which often encourage people to retweet the post to enter the giveaway. The sentiment towards the movie seems to be generally positive here, with words like "ready", "win", and "favourite avenger", as well as discussion about famous actors very prevalent.
# 
# For thoroughness, we will also filter out only the original tweets (non-retweets), and view a word cloud of those as well to see if it differs significantly.

# In[28]:


non_retweets = tweets.query('isRetweet == False')
print(non_retweets.shape)


# In[29]:


# the string that will be used for the word cloud
noRt_words = ''

for index, row in non_retweets.iterrows():
    text = row['text']
    tokens = text.split()

    # removes capital letters from the whole word
    tokens = [token.lower() for token in tokens]

    # filter out links and emoji codes
    tokens = [token for token in tokens if not re.search('https.+|<.+>', token)]

    # adds cleaned tokens to string
    noRt_words += " ".join(tokens) + " "


# In[30]:


wordcloud2 = WordCloud(width = 800, height = 800, colormap="PuOr_r", min_font_size = 2, 
background_color ='black', max_words=50, stopwords = stopwords).generate(noRt_words)
 
# plot the WordCloud image                       
plt.figure(figsize = (8, 8), facecolor = 'black')
plt.imshow(wordcloud2, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()


# Having filtered out the retweets, we see that the word cloud is similar, but clearly different from the first. There are more words revolving around times, being ready to watch the movie, and tickets, implying the posters are purchasing theater tickets and excited to view the movie soon.
# 
# I found the inclusion of words like "tomorrow" especially interesting, since the movie was not released until April 26th in the U.S., and this data was collected on the 23rd. Looking into it further, I found that the movie was actually released on the 24th in Australia and the 25th in the U.K., implying that several of these tweets were excited Australian movie fans. 
# Aside from that, most of the words are as expected - revolving around the Avengers, movies, and Marvel.

# # Final Thoughts and Going Forward
# 

# Sentiment analysis is a powerful tool for analyzing qualitative data that can be difficult to categorize numerically, such as social media posts about a brand or product. While more powerful sentiment analysis tools exist, NLTK is a powerful library for performing a basic level of sentiment analysis and allows us to analyze large amounts of data that would otherwise be too difficult to sort through and categorize. Utilizing this data to visualize public sentiment, as well as tools like word clouds to visualize the most popular topics, could be a very useful marketing tool with the ever increasing growth of social media.
# 
# While I used old data to explore sentiment at a particular time in the past, this approach could easily be used for much more recent data, and can be collected relatively quickly with the prevalence of tools to scrape data from most social media sites. It would also be useful to filter the data for posts that contain certain words, such as a particular actor or character, and view sentiment for those individually.
