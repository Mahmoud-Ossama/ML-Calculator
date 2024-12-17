import numpy as np
from collections import defaultdict
import re

class NaiveBayesTextClassifier:
    def __init__(self):
        self.vocabularies = defaultdict(dict)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.class_counts = defaultdict(int)
        self.classes = None

    def preprocess_text(self, text):
        # Convert to lowercase and split into words
        text = text.lower()
        # Remove special characters and split
        words = re.findall(r'\w+', text)
        return words

    def fit(self, texts, labels):
        self.classes = np.unique(labels)
        
        # Count words in each class
        for text, label in zip(texts, labels):
            words = self.preprocess_text(text)
            self.class_counts[label] += 1
            
            for word in words:
                self.word_counts[label][word] += 1
                self.vocabularies[label][word] = 1
        
        # Calculate total vocabulary
        self.total_vocab = len(set().union(*[set(v.keys()) for v in self.vocabularies.values()]))
        
    def predict(self, text):
        words = self.preprocess_text(text)
        scores = []
        
        for class_label in self.classes:
            # Calculate class probability
            class_prob = np.log(self.class_counts[class_label] / sum(self.class_counts.values()))
            
            # Calculate word probabilities
            word_probs = 0
            total_words = sum(self.word_counts[class_label].values())
            
            for word in words:
                # Add-one smoothing
                count = self.word_counts[class_label][word] + 1
                prob = count / (total_words + self.total_vocab)
                word_probs += np.log(prob)
            
            scores.append(class_prob + word_probs)
        
        return self.classes[np.argmax(scores)]

    def get_class_name(self, class_label):
        sentiment_map = {0: "Negative", 1: "Positive"}
        return sentiment_map.get(class_label, str(class_label))
