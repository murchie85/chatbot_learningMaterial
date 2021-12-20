# NLP 

Python has some of the most extensive open-source NLP libraries, including the `Natural Language Toolkit or NLTK.`  
https://www.nltk.org/

## Cleaning 

`Noise removal` — stripping text of formatting (e.g., HTML tags).

`Tokenization` — breaking text into individual words.  

`Normalization` — cleaning text data in any other way:  

`Stemming` is a blunt axe to chop off word prefixes and suffixes. “booing” and “booed” become “boo”, but “computer” may become “comput” and “are” would remain “are.”
  
`Lemmatization ` is a scalpel to bring words down to their root forms. For example, NLTK’s savvy lemmatizer knows “am” and “are” are related to “be.”
Other common tasks include lowercasing, stopwords removal, spelling correction, etc.


```python
# regex for removing punctuation!
import re
# nltk preprocessing magic
import nltk
nltk.download('punkt') # notebook only
nltk.download('wordnet') # notebook only
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# grabbing a part of speech function:
#from part_of_speech import get_part_of_speech

text = "So many squids are jumping out of suitcases these days that you can barely go anywhere without seeing one burst forth from a tightly packed valise. I went to the dentist the other day, and sure enough I saw an angry one jump out of my dentist's bag within minutes of arriving. She hardly even noticed."

cleaned = re.sub('\W+', ' ', text)
tokenized = word_tokenize(cleaned)

stemmer = PorterStemmer()
stemmed = [stemmer.stem(token) for token in tokenized]

## -- CHANGE these -- ##
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(x) for x in tokenized]

print("Stemmed text:")
print(stemmed)
print("\nLemmatized text:")
print(lemmatized)
```

    [nltk_data] Downloading package punkt to
    [nltk_data]     /Users/adammcmurchie/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package wordnet to
    [nltk_data]     /Users/adammcmurchie/nltk_data...
    [nltk_data]   Unzipping corpora/wordnet.zip.


    Stemmed text:
    ['So', 'mani', 'squid', 'are', 'jump', 'out', 'of', 'suitcas', 'these', 'day', 'that', 'you', 'can', 'bare', 'go', 'anywher', 'without', 'see', 'one', 'burst', 'forth', 'from', 'a', 'tightli', 'pack', 'valis', 'I', 'went', 'to', 'the', 'dentist', 'the', 'other', 'day', 'and', 'sure', 'enough', 'I', 'saw', 'an', 'angri', 'one', 'jump', 'out', 'of', 'my', 'dentist', 's', 'bag', 'within', 'minut', 'of', 'arriv', 'she', 'hardli', 'even', 'notic']
    
    Lemmatized text:
    ['So', 'many', 'squid', 'are', 'jumping', 'out', 'of', 'suitcase', 'these', 'day', 'that', 'you', 'can', 'barely', 'go', 'anywhere', 'without', 'seeing', 'one', 'burst', 'forth', 'from', 'a', 'tightly', 'packed', 'valise', 'I', 'went', 'to', 'the', 'dentist', 'the', 'other', 'day', 'and', 'sure', 'enough', 'I', 'saw', 'an', 'angry', 'one', 'jump', 'out', 'of', 'my', 'dentist', 's', 'bag', 'within', 'minute', 'of', 'arriving', 'She', 'hardly', 'even', 'noticed']


Why are the lemmatized verbs like "went" still conjugated? By default `lemmatize()` treats every word as a noun.

Give `lemmatize()` a second argument: `get_part_of_speech(token)` function added. This will tell our lemmatizer what part of speech the word is


```python
from nltk.corpus import wordnet

def get_part_of_speech(word):
  probable_part_of_speech = wordnet.synsets(word)
  pos_counts = Counter()
  pos_counts["n"] = len(  [ item for item in probable_part_of_speech if item.pos()=="n"]  )
  pos_counts["v"] = len(  [ item for item in probable_part_of_speech if item.pos()=="v"]  )
  pos_counts["a"] = len(  [ item for item in probable_part_of_speech if item.pos()=="a"]  )
  pos_counts["r"] = len(  [ item for item in probable_part_of_speech if item.pos()=="r"]  )
  
  most_likely_part_of_speech = pos_counts.most_common(1)[0][0]
  return most_likely_part_of_speech

# regex for removing punctuation!
import re
# nltk preprocessing magic
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

text = "So many squids are jumping out of suitcases these days that you can barely go anywhere without seeing one burst forth from a tightly packed valise. I went to the dentist the other day, and sure enough I saw an angry one jump out of my dentist's bag within minutes of arriving. She hardly even noticed."

cleaned = re.sub('\W+', ' ', text)
tokenized = word_tokenize(cleaned)

stemmer = PorterStemmer()
stemmed = [stemmer.stem(token) for token in tokenized]

## -- CHANGE these -- ##
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(x,get_part_of_speech(x)) for x in tokenized]

print("Stemmed text:")
print(stemmed)
print("\nLemmatized text:")
print(lemmatized)

```

    Stemmed text:
    ['So', 'mani', 'squid', 'are', 'jump', 'out', 'of', 'suitcas', 'these', 'day', 'that', 'you', 'can', 'bare', 'go', 'anywher', 'without', 'see', 'one', 'burst', 'forth', 'from', 'a', 'tightli', 'pack', 'valis', 'I', 'went', 'to', 'the', 'dentist', 'the', 'other', 'day', 'and', 'sure', 'enough', 'I', 'saw', 'an', 'angri', 'one', 'jump', 'out', 'of', 'my', 'dentist', 's', 'bag', 'within', 'minut', 'of', 'arriv', 'she', 'hardli', 'even', 'notic']
    
    Lemmatized text:
    ['So', 'many', 'squid', 'be', 'jump', 'out', 'of', 'suitcase', 'these', 'day', 'that', 'you', 'can', 'barely', 'go', 'anywhere', 'without', 'see', 'one', 'burst', 'forth', 'from', 'a', 'tightly', 'pack', 'valise', 'I', 'go', 'to', 'the', 'dentist', 'the', 'other', 'day', 'and', 'sure', 'enough', 'I', 'saw', 'an', 'angry', 'one', 'jump', 'out', 'of', 'my', 'dentist', 's', 'bag', 'within', 'minute', 'of', 'arrive', 'She', 'hardly', 'even', 'notice']



```python

```

    Stemmed text:
    ['So', 'mani', 'squid', 'are', 'jump', 'out', 'of', 'suitcas', 'these', 'day', 'that', 'you', 'can', 'bare', 'go', 'anywher', 'without', 'see', 'one', 'burst', 'forth', 'from', 'a', 'tightli', 'pack', 'valis', 'I', 'went', 'to', 'the', 'dentist', 'the', 'other', 'day', 'and', 'sure', 'enough', 'I', 'saw', 'an', 'angri', 'one', 'jump', 'out', 'of', 'my', 'dentist', 's', 'bag', 'within', 'minut', 'of', 'arriv', 'she', 'hardli', 'even', 'notic']
    
    Lemmatized text:
    ['So', 'many', 'squid', 'are', 'jumping', 'out', 'of', 'suitcase', 'these', 'day', 'that', 'you', 'can', 'barely', 'go', 'anywhere', 'without', 'seeing', 'one', 'burst', 'forth', 'from', 'a', 'tightly', 'packed', 'valise', 'I', 'went', 'to', 'the', 'dentist', 'the', 'other', 'day', 'and', 'sure', 'enough', 'I', 'saw', 'an', 'angry', 'one', 'jump', 'out', 'of', 'my', 'dentist', 's', 'bag', 'within', 'minute', 'of', 'arriving', 'She', 'hardly', 'even', 'noticed']


# parsing 

Parsing is an NLP process concerned with segmenting text based on syntax.  
NLTK has a few tricks up its sleeve to help you out:   

## Part-of-speech tagging (POS tagging) 

identifies parts of speech (verbs, nouns, adjectives, etc.). NLTK can do it faster (and maybe more accurately) than your grammar teacher.  


## Named entity recognition (NER) 
 
helps identify the proper nouns (e.g., “Natalia” or “Berlin”) in a text. This can be a clue as to the topic of the text and NLTK captures many for you.  

## Dependency grammar 
trees help you understand the relationship between the words in a sentence. It can be a tedious task for a human, so the Python library spaCy is at your service, even if it isn’t always perfect.   

## Regex parsing 
using Python’s ` re library`, allows for a bit more nuance. When coupled with `POS tagging`, you can identify specific phrase chunks. On its own, it can find you addresses, emails, and many other common patterns within large chunks of text.

# nltk tree


```python
"""
silly squid sentences parsed into dependency trees visually!
"""
    

import spacy
from nltk import Tree

dependency_parser = spacy.load('en')
squids_text = "So many squids are jumping out of suitcases these days. You can barely go anywhere without seeing one. I went to the dentist the other day. Sure enough, I saw an angry one jump out of my dentist's bag. She hardly even noticed."
parsed_squids = dependency_parser(squids_text)

# Assign my_sentence a new value:
my_sentence = "Your sentence goes here!"
my_parsed_sentence = dependency_parser(my_sentence)

def to_nltk_tree(node):
  if node.n_lefts + node.n_rights > 0:
    parsed_child_nodes = [to_nltk_tree(child) for child in node.children]
    return Tree(node.orth_, parsed_child_nodes)
  else:
    return node.orth_

for sent in parsed_squids.sents:
  to_nltk_tree(sent.root).pretty_print()
  
for sent in my_parsed_sentence.sents:
 to_nltk_tree(sent.root).pretty_print()
```

            jumping                
      _________|________________    
     |   |   squids    out      |  
     |   |     |        |       |   
     |   |    many      of     days
     |   |     |        |       |   
    are  .     So   suitcases these
    
              go                       
      ________|____________________     
     |   |    |       |      |  without
     |   |    |       |      |     |    
     |   |    |       |      |   seeing
     |   |    |       |      |     |    
    You can barely anywhere  .    one  
    
              went               
      _________|_________         
     |   |     to        |       
     |   |     |         |        
     |   |  dentist     day      
     |   |     |      ___|____    
     I   .    the   the     other
    
                       saw                           
      __________________|_________                    
     |   |   |    |              jump                
     |   |   |    |      _________|__________         
     |   |   |    |     |    |    |         out      
     |   |   |    |     |    |    |          |        
     |   |   |    |     |    |    |          of      
     |   |   |    |     |    |    |          |        
     |   |   |    |     |    |    |         bag      
     |   |   |    |     |    |    |          |        
     |   |   |  enough  |    |    |       dentist    
     |   |   |    |     |    |    |     _____|_____   
     ,   I   .   Sure   an angry one   my          's
    
        noticed         
      _____|__________   
    She  hardly even  . 
    
         goes         
      ____|______      
     |    |   sentence
     |    |      |     
    here  !     Your  
    



```python
"""
adding my sentence
"""
    

import spacy
from nltk import Tree

dependency_parser = spacy.load('en')
squids_text = "So many squids are jumping out of suitcases these days. You can barely go anywhere without seeing one. I went to the dentist the other day. Sure enough, I saw an angry one jump out of my dentist's bag. She hardly even noticed."
parsed_squids = dependency_parser(squids_text)

# Assign my_sentence a new value:
my_sentence = "Within the old decrepid vault that had been left for a millenia stirred an ancient evil, waiting to be awoken."
my_parsed_sentence = dependency_parser(my_sentence)

def to_nltk_tree(node):
  if node.n_lefts + node.n_rights > 0:
    parsed_child_nodes = [to_nltk_tree(child) for child in node.children]
    return Tree(node.orth_, parsed_child_nodes)
  else:
    return node.orth_

for sent in parsed_squids.sents:
  to_nltk_tree(sent.root).pretty_print()
  
for sent in my_parsed_sentence.sents:
 to_nltk_tree(sent.root).pretty_print()
```

            jumping                
      _________|________________    
     |   |   squids    out      |  
     |   |     |        |       |   
     |   |    many      of     days
     |   |     |        |       |   
    are  .     So   suitcases these
    
              go                       
      ________|____________________     
     |   |    |       |      |  without
     |   |    |       |      |     |    
     |   |    |       |      |   seeing
     |   |    |       |      |     |    
    You can barely anywhere  .    one  
    
              went               
      _________|_________         
     |   |     to        |       
     |   |     |         |        
     |   |  dentist     day      
     |   |     |      ___|____    
     I   .    the   the     other
    
                       saw                           
      __________________|_________                    
     |   |   |    |              jump                
     |   |   |    |      _________|__________         
     |   |   |    |     |    |    |         out      
     |   |   |    |     |    |    |          |        
     |   |   |    |     |    |    |          of      
     |   |   |    |     |    |    |          |        
     |   |   |    |     |    |    |         bag      
     |   |   |    |     |    |    |          |        
     |   |   |  enough  |    |    |       dentist    
     |   |   |    |     |    |    |     _____|_____   
     ,   I   .   Sure   an angry one   my          's
    
        noticed         
      _____|__________   
    She  hardly even  . 
    
                 Within                                                               
      _____________|________________                                                   
     |                            vault                                               
     |    __________________________|________                                          
     |   |   |     |                        left                                      
     |   |   |     |       __________________|______                                   
     |   |   |     |      |    |    |              for                                
     |   |   |     |      |    |    |               |                                  
     |   |   |     |      |    |    |            millenia                             
     |   |   |     |      |    |    |     __________|_____________                     
     |   |   |     |      |    |    |    |                     stirred                
     |   |   |     |      |    |    |    |    ____________________|___________         
     |   |   |     |      |    |    |    |   |             |               waiting    
     |   |   |     |      |    |    |    |   |             |                  |        
     |   |   |     |      |    |    |    |   |            evil              awoken    
     |   |   |     |      |    |    |    |   |       ______|______       _____|_____   
     .  the old decrepid that had  been  a   ,      an         ancient  to          be
    



```python

```
