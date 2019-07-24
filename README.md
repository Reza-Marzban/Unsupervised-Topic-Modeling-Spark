# Unsupervised-Topic-Modeling-Spark
Unsupervised Topic Modeling of tweets using Spark

LDA and LSA Comparative analysis

Step 1: Data collection using Apache flume with the following keyword: #suicidal
Step 2: Data Cleaning on spark dataframes:
  For the data cleaning, the steps we carried out are removind the following tokens:
    stop words,
    numerical values,
    Symbols,
    unique words (df >=2), and 
    words with length < 4.
    
    
    
 Step 3: we performed LDA (Latent Dirichlet allocation) by using ml library in spark. and also LSA (Latent Semantic Analysis) by using Gensim library and found 5 topics with each algorithm over all cleaned and tokenized tweets.
 
  Step 4: We devised a way to compare their result and find the similarity between these topics of each algorithm. We created similarity score between each two topic based on simple weight multiplication of identical words.
  
  Step 5: We created a 5*5 matrix that showed the similarity score between each LDA and LSA topic, and then decide which two topics are equal and which are different.
  
  Step 6: We created a table that shows all found topics and their top 10 significant words. the topics that are similar are shown in a single row.

Results found: LDA and LSA topics may overlap, but each of them can find new unique topics, by applying both and finding similarity between their topics, we can extract more useful data from texts.
