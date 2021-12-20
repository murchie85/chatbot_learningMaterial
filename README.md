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
    [nltk_data]   Package wordnet is already up-to-date!


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




`Dependency grammar` trees help you understand the relationship between the words in a sentence. It can be a tedious task for a human, so the Python library spaCy is at your service, even if it isn’t always perfect.   

`Regex parsing`, using Python’s ` re library`, allows for a bit more nuance. When coupled with `POS tagging`, you can identify specific phrase chunks. On its own, it can find you addresses, emails, and many other common patterns within large chunks of text.

# Part-of-speech tagging (POS tagging) 

identifies parts of speech (verbs, nouns, adjectives, etc.). NLTK can do it faster (and maybe more accurately) than your grammar teacher.  


# Named entity recognition (NER)  
helps identify the proper nouns (e.g., “Natalia” or “Berlin”) in a text. This can be a clue as to the topic of the text and NLTK captures many for you.  


```python

```

# nltk tree


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
    


# bag of words 

The squids jumped out of the suitcases 

results in 

`{"the": 2, "squid": 1, "jump": 1, "out": 1, "of": 1, "suitcase": 1}`

“Why are your suitcases full of jumping squids?”
results in 

`{"why": 1, "be": 1, "your": 1, "suitcase": 1, "full": 1, "of": 1, "jump": 1, "squid": 1}`

bag-of-words can be an excellent way of looking at language when you want to make predictions concerning topic or sentiment of a text. `When grammar and word order are irrelevant, this is probably a good model to use.`


```python
# importing regex and nltk
import re, nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# importing Counter to get word counts for bag of words
from collections import Counter
# importing a passage from Through the Looking Glass
looking_glass_text = """
 However, the egg only got larger and larger, and more and more human: when she had come within a few yards of it, she saw that it had eyes and a nose and mouth; and when she had come close to it, she saw clearly that it was HUMPTY DUMPTY himself. It cant be anybody else! she said to herself. Im as certain of it, as if his name were written all over his face.

It might have been written a hundred times, easily, on that enormous face. Humpty Dumpty was sitting with his legs crossed, like a Turk, on the top of a high wallsuch a narrow one that Alice quite wondered how he could keep his balanceand, as his eyes were steadily fixed in the opposite direction, and he didnt take the least notice of her, she thought he must be a stuffed figure after all.

And how exactly like an egg he is! she said aloud, standing with her hands ready to catch him, for she was every moment expecting him to fall.

Its very provoking, Humpty Dumpty said after a long silence, looking away from Alice as he spoke, to be called an eggVery!

I said you looked like an egg, Sir, Alice gently explained. And some eggs are very pretty, you know she added, hoping to turn her remark into a sort of a compliment.

Some people, said Humpty Dumpty, looking away from her as usual, have no more sense than a baby!

Alice didnt know what to say to this: it wasnt at all like conversation, she thought, as he never said anything to her; in fact, his last remark was evidently addressed to a treeso she stood and softly repeated to herself:

     Humpty Dumpty sat on a wall:
     Humpty Dumpty had a great fall.
     All the Kings horses and all the Kings men
     Couldnt put Humpty Dumpty in his place again.

That last line is much too long for the poetry, she added, almost out loud, forgetting that Humpty Dumpty would hear her.

Dont stand there chattering to yourself like that, Humpty Dumpty said, looking at her for the first time, but tell me your name and your business.

My name is Alice, but

Its a stupid enough name! Humpty Dumpty interrupted impatiently. What does it mean?

Must a name mean something? Alice asked doubtfully.

Of course it must, Humpty Dumpty said with a short laugh: my name means the shape I amand a good handsome shape it is, too. With a name like yours, you might be any shape, almost.

Why do you sit out here all alone? said Alice, not wishing to begin an argument.

Why, because theres nobody with me! cried Humpty Dumpty. Did you think I didnt know the answer to that? Ask another.

Dont you think youd be safer down on the ground? Alice went on, not with any idea of making another riddle, but simply in her good-natured anxiety for the queer creature. That wall is so very narrow!

What tremendously easy riddles you ask! Humpty Dumpty growled out. Of course I dont think so! Why, if ever I did fall offwhich theres no chance ofbut if I did Here he pursed his lips and looked so solemn and grand that Alice could hardly help laughing. If I did fall, he went on, The King has promised mewith his very own mouthtoto

To send all his horses and all his men, Alice interrupted, rather unwisely.

Now I declare thats too bad! Humpty Dumpty cried, breaking into a sudden passion. Youve been listening at doorsand behind treesand down chimneysor you couldnt have known it!

I havent, indeed! Alice said very gently. Its in a book.

Ah, well! They may write such things in a book, Humpty Dumpty said in a calmer tone. Thats what you call a History of England, that is. Now, take a good look at me! Im one that has spoken to a King, I am: mayhap youll never see such another: and to show you Im not proud, you may shake hands with me! And he grinned almost from ear to ear, as he leant forwards (and as nearly as possible fell off the wall in doing so) and offered Alice his hand. She watched him a little anxiously as she took it. If he smiled much more, the ends of his mouth might meet behind, she thought: and then I dont know what would happen to his head! Im afraid it would come off!

Yes, all his horses and all his men, Humpty Dumpty went on. Theyd pick me up again in a minute, they would! However, this conversation is going on a little too fast: lets go back to the last remark but one.

Im afraid I cant quite remember it, Alice said very politely.

In that case we start fresh, said Humpty Dumpty, and its my turn to choose a subject (He talks about it just as if it was a game! thought Alice.) So heres a question for you. How old did you say you were?

Alice made a short calculation, and said Seven years and six months.

Wrong! Humpty Dumpty exclaimed triumphantly. You never said a word like it!

I though you meant How old are you? Alice explained.

If Id meant that, Id have said it, said Humpty Dumpty. 
"""

def get_part_of_speech(word):
  probable_part_of_speech = wordnet.synsets(word)
  pos_counts = Counter()
  pos_counts["n"] = len(  [ item for item in probable_part_of_speech if item.pos()=="n"]  )
  pos_counts["v"] = len(  [ item for item in probable_part_of_speech if item.pos()=="v"]  )
  pos_counts["a"] = len(  [ item for item in probable_part_of_speech if item.pos()=="a"]  )
  pos_counts["r"] = len(  [ item for item in probable_part_of_speech if item.pos()=="r"]  )
  
  most_likely_part_of_speech = pos_counts.most_common(1)[0][0]
  return most_likely_part_of_speech


# importing part-of-speech function for lemmatization
#from part_of_speech import get_part_of_speech
# got from earlier function 
nltk.download('stopwords') # notebook only
# Change text to another string:
text = looking_glass_text

cleaned = re.sub('\W+', ' ', text).lower()
tokenized = word_tokenize(cleaned)

# Filter out stopwords ( i / me , him...)
stop_words = stopwords.words('english')
filtered = [word for word in tokenized if word not in stop_words]

normalizer = WordNetLemmatizer()
normalized = [normalizer.lemmatize(token, get_part_of_speech(token)) for token in filtered]

# take the normalised words and bag em 
bag_of_looking_glass_words = Counter(normalized)
print(bag_of_looking_glass_words)

```

    Counter({'humpty': 19, 'dumpty': 19, 'say': 19, 'alice': 16, 'name': 7, 'like': 7, 'think': 7, 'look': 6, 'im': 5, 'know': 5, 'mean': 5, 'go': 5, 'egg': 4, 'fall': 4, 'king': 4, 'would': 4, 'dont': 4, 'come': 3, 'write': 3, 'might': 3, 'sit': 3, 'one': 3, 'didnt': 3, 'take': 3, 'must': 3, 'stand': 3, 'hand': 3, 'remark': 3, 'never': 3, 'last': 3, 'wall': 3, 'horse': 3, 'men': 3, 'almost': 3, 'ask': 3, 'shape': 3, 'good': 3, 'another': 3, 'however': 2, 'large': 2, 'saw': 2, 'eye': 2, 'mouth': 2, 'cant': 2, 'face': 2, 'time': 2, 'narrow': 2, 'quite': 2, 'could': 2, 'long': 2, 'away': 2, 'speak': 2, 'call': 2, 'gently': 2, 'explain': 2, 'add': 2, 'turn': 2, 'conversation': 2, 'couldnt': 2, 'much': 2, 'interrupt': 2, 'course': 2, 'short': 2, 'laugh': 2, 'there': 2, 'cry': 2, 'make': 2, 'riddle': 2, 'thats': 2, 'behind': 2, 'book': 2, 'may': 2, 'ear': 2, 'little': 2, 'afraid': 2, 'old': 2, 'id': 2, 'get': 1, 'human': 1, 'within': 1, 'yard': 1, 'nose': 1, 'close': 1, 'clearly': 1, 'anybody': 1, 'else': 1, 'certain': 1, 'hundred': 1, 'easily': 1, 'enormous': 1, 'leg': 1, 'cross': 1, 'turk': 1, 'top': 1, 'high': 1, 'wallsuch': 1, 'wonder': 1, 'keep': 1, 'balanceand': 1, 'steadily': 1, 'fix': 1, 'opposite': 1, 'direction': 1, 'least': 1, 'notice': 1, 'stuff': 1, 'figure': 1, 'exactly': 1, 'aloud': 1, 'ready': 1, 'catch': 1, 'every': 1, 'moment': 1, 'expect': 1, 'provoke': 1, 'silence': 1, 'eggvery': 1, 'sir': 1, 'pretty': 1, 'hop': 1, 'sort': 1, 'compliment': 1, 'people': 1, 'usual': 1, 'sense': 1, 'baby': 1, 'wasnt': 1, 'anything': 1, 'fact': 1, 'evidently': 1, 'address': 1, 'treeso': 1, 'softly': 1, 'repeat': 1, 'great': 1, 'put': 1, 'place': 1, 'line': 1, 'poetry': 1, 'loud': 1, 'forget': 1, 'hear': 1, 'chatter': 1, 'first': 1, 'tell': 1, 'business': 1, 'stupid': 1, 'enough': 1, 'impatiently': 1, 'something': 1, 'doubtfully': 1, 'amand': 1, 'handsome': 1, 'alone': 1, 'wish': 1, 'begin': 1, 'argument': 1, 'nobody': 1, 'answer': 1, 'youd': 1, 'safe': 1, 'grind': 1, 'idea': 1, 'simply': 1, 'natured': 1, 'anxiety': 1, 'queer': 1, 'creature': 1, 'tremendously': 1, 'easy': 1, 'growl': 1, 'ever': 1, 'offwhich': 1, 'chance': 1, 'ofbut': 1, 'purse': 1, 'lip': 1, 'solemn': 1, 'grand': 1, 'hardly': 1, 'help': 1, 'promise': 1, 'mewith': 1, 'mouthtoto': 1, 'send': 1, 'rather': 1, 'unwisely': 1, 'declare': 1, 'bad': 1, 'break': 1, 'sudden': 1, 'passion': 1, 'youve': 1, 'listen': 1, 'doorsand': 1, 'treesand': 1, 'chimneysor': 1, 'havent': 1, 'indeed': 1, 'ah': 1, 'well': 1, 'thing': 1, 'calm': 1, 'tone': 1, 'history': 1, 'england': 1, 'mayhap': 1, 'youll': 1, 'see': 1, 'show': 1, 'proud': 1, 'shake': 1, 'grin': 1, 'lean': 1, 'forward': 1, 'nearly': 1, 'possible': 1, 'fell': 1, 'offer': 1, 'watch': 1, 'anxiously': 1, 'smile': 1, 'end': 1, 'meet': 1, 'happen': 1, 'head': 1, 'yes': 1, 'theyd': 1, 'pick': 1, 'minute': 1, 'fast': 1, 'let': 1, 'back': 1, 'remember': 1, 'politely': 1, 'case': 1, 'start': 1, 'fresh': 1, 'choose': 1, 'subject': 1, 'talk': 1, 'game': 1, 'here': 1, 'question': 1, 'calculation': 1, 'seven': 1, 'year': 1, 'six': 1, 'month': 1, 'wrong': 1, 'exclaim': 1, 'triumphantly': 1, 'word': 1, 'though': 1})


    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/adammcmurchie/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!


# Language Models: N-Gram and NLM



```python

```
