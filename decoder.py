from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

import time
dep_relations = ['tmod', 'vmod', 'csubjpass', 'rcmod', 'ccomp', 'poss', 'parataxis', 'appos', 'dep', 'iobj', 'pobj', 'mwe', 'quantmod', 'acomp', 'number', 'csubj', 'root', 'auxpass', 'prep', 'mark', 'expl', 'cc', 'npadvmod', 'prt', 'nsubj', 'advmod', 'conj', 'advcl', 'punct', 'aux', 'pcomp', 'discourse', 'nsubjpass', 'predet', 'cop', 'possessive', 'nn', 'xcomp', 'preconj', 'num', 'amod', 'dobj', 'neg','dt','det']

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)
        count = 0

        # print('sentence is:')
        # print (words)



        while state.buffer:
            nn_input = self.extractor.get_input_representation(words, pos, state)
            out = self.model.predict(np.expand_dims(nn_input, axis=0))
            # print('Prediction:')
            output = out[0]
            # print ('buffer is:')
            # print(state.buffer)

            # print('stack is:')
            # print(state.stack)

            # count +=1

            # if count ==1:
            #     break
            # time.sleep(2)

            pairs = []
            for i in range(0,len(output)):
                pairs.append((self.output_labels[i], output[i]))

            pairs.sort(key=lambda tup: tup[1], reverse = True)
            # print (pairs)

            for i in range(0,len(pairs)):
                if pairs[i][0][0] == 'shift':
                    # print ('enter shift')
                    if len(state.buffer) == 1 and len(state.stack) == 0:
                        state.stack.append(state.buffer[-1])
                        state.buffer = state.buffer[:-1]
                        break
                    elif  len(state.buffer) == 1 and len(state.stack) > 0:
                        continue
                    else:
                        state.stack.append(state.buffer[-1])
                        state.buffer = state.buffer[:-1]
                        break

                elif pairs[i][0][0] == 'left_arc':
                    # print ('enter left arc')
                    if len(state.stack) == 0 or len(state.buffer) == 0:
                        continue
                    target = state.stack[-1]
                    head = state.buffer[-1]

                    if target == 0:
                        continue
                    else:
                        state.deps.add((head, target, pairs[i][0][-1]))
                        state.stack = state.stack[:-1]
                        break


                elif pairs[i][0][0] == 'right_arc':
                    # print ('enter right arc')
                    if len(state.stack) == 0 or len(state.buffer) == 0:
                        continue
                    target = state.buffer[-1]
                    head = state.stack[-1]

                    if head == 0 and len(state.buffer) > 1:
                        continue

                    if target == 0:
                        continue
                    else:
                        state.deps.add((head, target, pairs[i][0][-1]))
                        state.buffer = state.buffer[:-1]
                        state.buffer.append(state.stack[-1])
                        state.stack = state.stack[:-1]
                        break

            # pass

            # TODO: Write the body of this loop for part 4 
        # print ('')
        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        # return ""
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])
    # print (parser.output_labels)
    count = 0
    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()

            # count +=1
            # if count > 2:
            #     break


