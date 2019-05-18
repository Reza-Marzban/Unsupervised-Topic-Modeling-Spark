'''

 Reza Marzban

Mental Illness Topic Modelling
Twitter Keywors used: #suicidal
'''

from pyspark import SparkContext, SparkConf
from pyspark.sql import functions, SparkSession
from pyspark.ml.clustering import LDA
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from gensim.models import LsiModel
import numpy as np


def main():
    conf = SparkConf().setAppName("Program Number 1")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    # creates Spark Session
    spark = SparkSession.builder.appName("Program Number 1").getOrCreate()

    # tweets folder address on HDFS server -  ignore files with .tmp extensions (Flume active files).
    inputpath = "hdfs://hdfs input path"

    spark.conf.set("spark.sql.shuffle.partitions", 1)

    # get the raw tweets from HDFS
    raw_tweets = spark.read.format("json").option("inferScehma", "true").option("mode", "dropMalformed").load(inputpath)

    # get the tweet text from the raw data. text is transformed to lower case. Deletes re-tweets. and finally include an index for each tweet
    tweets = raw_tweets.select(functions.lower(functions.col("text"))).withColumnRenamed("lower(text)","text").distinct().withColumn("id", functions.monotonically_increasing_id())

    # Create a tokenizer that Filter away tokens with length < 4, and get rid of symbols like $,#,...
    tokenizer = RegexTokenizer().setPattern("[\\W_]+").setMinTokenLength(4).setInputCol("text").setOutputCol("tokens")

    # Tokenize tweets
    tokenized_tweets = tokenizer.transform(tweets)
    remover = StopWordsRemover().setInputCol("tokens").setOutputCol("cleaned")

    # remove stopwords
    cleaned_tweets = remover.transform(tokenized_tweets)

    # create a vector of words that at least appeared in two different tweets, and set maximum vocab size to 20000.
    vectorizer = CountVectorizer().setInputCol("cleaned").setOutputCol("features").setVocabSize(20000).setMinDF(2).fit(
        cleaned_tweets)
    wordVectors = vectorizer.transform(cleaned_tweets).select("id", "features")

    # LDA
    # create Latent Dirichlet Allocation model and run it on our data with 25 iteration and 5 topics
    lda = LDA(k=5, maxIter=25)
    # fit the model on data
    ldaModel = lda.fit(wordVectors)
    # create topics based on LDA
    lda_topics = ldaModel.describeTopics()
    # show LDA topics

    # ______________________________________________________________________________________________________________
    # LSA
    clean_tweets_list = []
    tweet_list = []
    # for creating the document term matrix for the LSIModel as input
    # this is needed as LSI needs tuples of (vocabulary_index, frequency) form
    for tweet_row in wordVectors.select('features').collect():
        tweet_list.clear()
        # reading the SparseVector of 'features' column (hence the 0 index) and zipping them to a list
        # idx = vocabulary_index, val=frequency of that word in that tweet
        for idx, val in zip(tweet_row[0].indices, tweet_row[0].values):
            # converting the frequency from float to integer
            tweet_list.append((idx, int(val)))
        clean_tweets_list.append(tweet_list[:])

    # calling the LSIModel and passing the number of topics as 5
    lsa_model = LsiModel(clean_tweets_list, num_topics=5)
    # show LSA topics

    # ______________________________________________________________________________________________________________
    # #Comparison

    # get the weights and indices of words from LDA topics in format of List[list[]]
    lda_wordIndices = [row['termIndices'] for row in lda_topics.collect()]
    lda_wordWeights = [row['termWeights'] for row in lda_topics.collect()]

    # get the weights and indices of words from LDA topics in format of numpy array with 5*wordCount shape.
    # each element is the weight of the corresponding word in that specific topic.
    lsa_weightsMatrix = lsa_model.get_topics()

    # function to calculate the similarity between an lsa topic and an lda topic.
    def topic_similarity_calculator(lsa_t, lda_t):
        (lda_index, lda_weight) = lda_t
        sum = 0
        for index, weight in zip(lda_index, lda_weight):
            sum = sum + (np.abs(lsa_t[index] * weight))
        return sum

    # run the similarity function on 25 possibilities (5 LSA * 5 LDA)
    similarity = []
    eachLSA = []
    for i in range(0, 5):
        eachLSA.clear()
        for j in range(0, 5):
            temp = topic_similarity_calculator(lsa_weightsMatrix[i], (lda_wordIndices[j], lda_wordWeights[j]))
            eachLSA.append(temp)
        similarity.append(eachLSA[:])

    # Print the similarity table
    # each row is a LDA topic and each column is an LSA topic.
    print(" ")
    print("Similarity table")
    def similarity_print(s):
        i = 1
        print("|--------------------------------------------------------|")
        print("|      |  LSA 1  |  LSA 2  |  LSA 3  |  LSA 4  |  LSA 5  |")
        print("|--------------------------------------------------------|")
        for one, two, three, four, five in zip(*similarity):
            print('|LDA {} | {:+1.4f} | {:+1.4f} | {:+1.4f} | {:+1.4f} | {:+1.4f} |'.format(i, one, two, three, four, five))
            print("|--------------------------------------------------------|")
            i = i + 1
	#creates the similarity matrix
    similarity_print(similarity)

    # ______________________________________________________________________________________________________________
    # Final result Table
    # Manually found the following Topics to be similar
    # (LSA1 - LDA1)
    # (LSA5 - LDA2)
    # rest are alone
    lsa_words_idx = []
    for idx, curr_topic in enumerate(lsa_weightsMatrix):
        lsa_words_idx.append(np.abs(curr_topic).argsort()[-10:][::-1])
    lsa_topics_bow = {}
    lda_topics_bow = {}
    lsa_bow_list = []
    lda_bow_list = []
    for curr_idx, (lda_topic, lsa_topic) in enumerate(zip(lda_wordIndices, lsa_words_idx)):
        lsa_bow_list.clear()
        lda_bow_list.clear()
        for idx in range(10):
            lsa_bow_list.append(vectorizer.vocabulary[lsa_topic[idx]])
            lda_bow_list.append(vectorizer.vocabulary[lda_topic[idx]])
        lsa_topics_bow[curr_idx] = lsa_bow_list[:]
        lda_topics_bow[curr_idx] = lda_bow_list[:]




    results = []
    names=[]
	# Creating word dictionary for LDA2 and LSA5
    lda2_lsa5 = lda_topics_bow[1][:]
    for word in (lsa_topics_bow[4]):
        if word not in lda2_lsa5:
            lda2_lsa5.append(word)
	# Creating word dictionary for LDA1 and LSA1
    lda1_lsa1 = lda_topics_bow[0][:]
    for word in (lsa_topics_bow[0]):
        if word not in lda1_lsa1:
            lda1_lsa1.append(word)
    results.append(lda1_lsa1)
    names.append("LDA1 - LSA1 ")
    results.append(lda2_lsa5)
    names.append("LDA2 - LSA5 ")
    results.append(lda_topics_bow[2])
    names.append("LDA3        ")
    results.append(lda_topics_bow[3])
    names.append("LDA4        ")
    results.append(lda_topics_bow[4])
    names.append("LDA5        ")
    results.append(lsa_topics_bow[1])
    names.append("LSA2        ")
    results.append(lsa_topics_bow[2])
    names.append("LSA3        ")
    results.append(lsa_topics_bow[3])
    names.append("LSA4        ")
    #printing the topics and related words
    print(" ")
    print("Topics Table")
    print("|------------------------------------------------------------------------------------------|")
    print("|    Topic     |  Significant Words                                                    |")
    print("|------------------------------------------------------------------------------------------|")
    for name, r in zip(names,results):
        print('| {} |  {} |'.format(name, r))
        print("|------------------------------------------------------------------------------------------|")
	
    print(" ")
    print(" ")


if __name__ == '__main__':
    main()
