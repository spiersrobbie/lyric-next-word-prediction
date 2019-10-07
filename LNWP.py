import numpy as np
import matplotlib.pyplot as plt
from NextWordPredict import LSTM
from DownloadLyrics import GeniusAPI
from SkipGram import *

artist = 'Cupcakke'
txt_file = 'lyricsv1.txt'
skipLearn = 0.1
skipProj = 1000
num_strings = 20
per_random = 50
epoch_length = 30
rand_mult = 0.1
skip_mult = 0.1


skipOptions = {'learning_rate': skipLearn, 'projection_size': skipProj, 'rand_mult': skip_mult}


api = GeniusAPI(artist)
api.download_to_txt(txt_file)

print("Finished Downloading Lyrics")

skip = SkipGram(txt_file, skipOptions)
word_vecs, skipCost = skip.train_to_vectors()

plt.figure
plt.plot(skipCost)
plt.savefig('skipCost.png')

print("Finished Generating Word Vectors")

NWP = LSTM(epoch_length, per_random, rand_mult)
NWPin = NWP.next_word_predict(skip.dictionary, word_vecs)

plt.figure
plt.plot(NWPin['cost'])
plt.savefig('costLSTM.png')

output_predictions = NWP.validate_strings(word_vecs, NWPin['W'], NWPin['U'], NWPin['B'], num_strings)

with open('outputStrings.txt', 'w') as f:
    for row in output_predictions:
        f.write("%s\n\n" %row)
