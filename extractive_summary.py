import numpy as np
import pandas as pd
import re
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from nltk.corpus import wordnet
from nltk import pos_tag
from scipy import spatial
import networkx as nx
import zipfile
from datasets import load_dataset
from operator import itemgetter
from evaluate import load
import evaluate
rouge = evaluate.load('rouge')

#nltk.download('punkt')
#nltk.download('stopwords')
#train_dataset = load_dataset("ccdv/govreport-summarization", trust_remote_code = True)
#train = train_dataset['train']
test_dataset = load_dataset("ccdv/govreport-summarization", trust_remote_code = True, split = 'test[:100]')
test_data = test_dataset['report']
train = test_dataset

stop_words = stopwords.words('english')

# The number of sentences to keep in the extractive summary
N = 5

summary_list = []
# for i in tqdm(range(np.shape(train)[0])):
for i in tqdm(range(100)):
    train_report = train[i]['report']
    train_sentences = sent_tokenize(train_report)
    sentence_tokens = []
    for sentence in train_sentences:
        # Preprocessing data through removing all not-a-word or not-a-space characters
        new_sentence = re.sub(r'[^\w\s]', '', sentence.lower())
        # Generate word tokens for each sentence using word_tokenize
        # These tokens will be used to create word embeddings
        words_in_sentence = word_tokenize(new_sentence)
        words_in_sentence = [word for word in words_in_sentence if word not in stop_words]
        sentence_tokens.append(words_in_sentence)

    # Create a list of word embeddings for each word in the sentence using Word2Vec
    # size is set to 50 to create 50-dimensional word embeddings to capture more semantic information and relationships between words
    # min_count is set to 1 to include all words in the sentence
    word_to_vector = Word2Vec(sentences = sentence_tokens, vector_size = 1, min_count = 1, epochs = 1000)
    # Create a list of sentence embeddings for each sentence in the report
    sentence_embeddings = []
    # Find the maximum sentence length in the report to pad the sentence embeddings
    maximum_sentence_length = max([len(sentence_token) for sentence_token in sentence_tokens])
    # For each sentence in the report, calculate the mean of the word embeddings for each word in the sentence
    for sentence in sentence_tokens:
        sentence_embedding = [np.mean(word_to_vector.wv[word]) for word in sentence]
        # Pad the sentence embeddings to the maximum sentence length
        sentence_embedding = np.pad(sentence_embedding, (0, maximum_sentence_length - len(sentence_embedding)), 'constant')
        sentence_embeddings.append(sentence_embedding)
    
    # Use cosine similarity to calculate the similarity between each sentence in the report
    # Create a matrix to store the cosine similarity between each sentence
    similarity_matrix = np.zeros((len(sentence_tokens), len(sentence_tokens)))
    for j in range(len(sentence_tokens)):
        for k in range(len(sentence_tokens)):
            if j != k:
                # used 1 - cosine similarity to calculate the cosine distance
                similarity_matrix[j][k] = 1 - spatial.distance.cosine(sentence_embeddings[j], sentence_embeddings[k])
    print(np.shape(similarity_matrix))
    # Create a network to represent the similarity between each sentence in the report
    similarity_network = nx.from_numpy_array(similarity_matrix)
    text_rank_scores = nx.pagerank(similarity_network, max_iter = 10000)

    # keep track of the score and the corresponding sentence
    sentence_scores = {}
    for j in range(len(sentence_tokens)):
        sentence_scores[train_sentences[j]] = text_rank_scores[j]

    # Sort the sentences based on the text rank scores and keep the top N sentences
    top_n_sent = dict(sorted(sentence_scores.items(), key = itemgetter(1), reverse = True)[:N])
    summary = ''
    # Generate the extractive summary by concatenating the top N sentences in the same order appearing in the report
    for train_sentence in train_sentences:
        if train_sentence in top_n_sent:
            summary += train_sentence
    #print(summary)
    summary_list.append(summary)
candidates = summary_list
references = test_dataset['summary']
results = rouge.compute(predictions=candidates, references=references)
print(results)
#print("here is the shape: ", np.shape(train)[0])
#print("here is the dataset, ", train_dataset['train'][1]['summary']
#print("here is the dataset, ", train_dataset['train'][1]['summary'])
#train = zipfile.ZipFile('train.zip', 'r')


