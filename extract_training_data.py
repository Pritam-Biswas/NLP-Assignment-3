from conll_reader import DependencyStructure, conll_reader
from collections import defaultdict
import copy
import sys
import keras
import numpy as np

class State(object):
    def __init__(self, sentence = []):
        self.stack = []
        self.buffer = []
        if sentence: 
            self.buffer = list(reversed(sentence))
        self.deps = set() 
    
    def shift(self):
        self.stack.append(self.buffer.pop())

    def left_arc(self, label):
        self.deps.add( (self.buffer[-1], self.stack.pop(),label) )

    def right_arc(self, label):
        parent = self.stack.pop()
        self.deps.add( (parent, self.buffer.pop(), label) )
        self.buffer.append(parent)

    def __repr__(self):
        return "{},{},{}".format(self.stack, self.buffer, self.deps)

   

def apply_sequence(seq, sentence):
    state = State(sentence)
    for rel, label in seq:
        if rel == "shift":
            state.shift()
        elif rel == "left_arc":
            state.left_arc(label) 
        elif rel == "right_arc":
            state.right_arc(label) 
         
    return state.deps
   
class RootDummy(object):
    def __init__(self):
        self.head = None
        self.id = 0
        self.deprel = None    
    def __repr__(self):
        return "<ROOT>"

     
def get_training_instances(dep_structure):

    deprels = dep_structure.deprels
    
    sorted_nodes = [k for k,v in sorted(deprels.items())]
    state = State(sorted_nodes)
    state.stack.append(0)

    childcount = defaultdict(int)
    for ident,node in deprels.items():
        childcount[node.head] += 1
 
    seq = []
    while state.buffer: 
        if not state.stack:
            seq.append((copy.deepcopy(state),("shift",None)))
            state.shift()
            continue
        if state.stack[-1] == 0:
            stackword = RootDummy() 
        else:
            stackword = deprels[state.stack[-1]]
        bufferword = deprels[state.buffer[-1]]
        if stackword.head == bufferword.id:
            childcount[bufferword.id]-=1
            seq.append((copy.deepcopy(state),("left_arc",stackword.deprel)))
            state.left_arc(stackword.deprel)
        elif bufferword.head == stackword.id and childcount[bufferword.id] == 0:
            childcount[stackword.id]-=1
            seq.append((copy.deepcopy(state),("right_arc",bufferword.deprel)))
            state.right_arc(bufferword.deprel)
        else: 
            seq.append((copy.deepcopy(state),("shift",None)))
            state.shift()
    return seq   


dep_relations = ['tmod', 'vmod', 'csubjpass', 'rcmod', 'ccomp', 'poss', 'parataxis', 'appos', 'dep', 'iobj', 'pobj', 'mwe', 'quantmod', 'acomp', 'number', 'csubj', 'root', 'auxpass', 'prep', 'mark', 'expl', 'cc', 'npadvmod', 'prt', 'nsubj', 'advmod', 'conj', 'advcl', 'punct', 'aux', 'pcomp', 'discourse', 'nsubjpass', 'predet', 'cop', 'possessive', 'nn', 'xcomp', 'preconj', 'num', 'amod', 'dobj', 'neg','dt','det']


class FeatureExtractor(object):
       
    def __init__(self, word_vocab_file, pos_vocab_file):
        self.word_vocab = self.read_vocab(word_vocab_file)        
        self.pos_vocab = self.read_vocab(pos_vocab_file)        
        self.output_labels = self.make_output_labels()

    def make_output_labels(self):
        labels = []
        labels.append(('shift',None))
    
        for rel in dep_relations:
            labels.append(("left_arc",rel))
            labels.append(("right_arc",rel))
        return dict((label, index) for (index,label) in enumerate(labels))

    def read_vocab(self,vocab_file):
        vocab = {}
        for line in vocab_file: 
            word, index_s = line.strip().split()
            index = int(index_s)
            vocab[word] = index
        return vocab     

    def get_input_representation(self, words, pos, state):
        # TODO: Write this method for Part 2

        for i in range(0, len(words)):
            if words[i] is None:
                words[i] = '<ROOT>'
            elif pos[i] == 'NNP':
                words[i] = '<NNP>'
            elif pos[i] == 'CD':
                words[i] = '<CD>'
            elif pos[i] == 'NNPS':
                words[i] = '<NNP>'
            elif pos[i] == 'UNK' or not words[i] in self.word_vocab:
                words[i] = '<UNK>'



            
        buff = state.buffer
        stk = state.stack
        buff_list  = []
        stk_list = []
        if len(buff) == 2:
            buff_list = [buff[-1],buff[-2],-1]
        elif len(buff) == 1:
            buff_list = [buff[-1],-1,-1]
        elif len(buff) == 0:
            buff_list = [-1,-1,-1]
        else:
            buff_list = [buff[-1], buff[-2], buff[-3]]


        if len(stk) == 2:
            stk_list = [stk[-1],stk[-2],-1]
        elif len(stk) == 1:
            stk_list = [stk[-1],-1,-1]
        elif len(stk) == 0:
            stk_list = [-1, -1, -1]
        else:
            stk_list = [stk[-1],stk[-2],stk[-3]]


        for i in range(0,len(buff_list)):
            if buff_list[i] == -1:
                buff_list[i] = self.word_vocab['<NULL>']
            else:
                buff_list[i] = self.word_vocab[words[buff_list[i]]]


        for i in range(0,len(stk_list)):
            if stk_list[i] == -1:
                stk_list[i] = self.word_vocab['<NULL>']
            else:
                stk_list[i] = self.word_vocab[words[stk_list[i]]]
        input_list = stk_list+buff_list
        

        # print (input_list)
        return np.array(input_list)
        # print (pos)
        # # print(words)
        # for i in range(0,len(words)):
        #     # print(len(words[i]))
        #     try:
        #         if  'UNK' in pos[i]:
        #             print ('Foudn token --------------------------------------------------')
        #     except Exception as e:
        #         # print ('null text')
        #         continue
        # # print('')
        # return np.zeros(6)

    def get_output_representation(self, output_pair):  
        # TODO: Write this method for Part 2
        # global dep_relations
        # left_rel = []
        # right_rel = []

        # for i in range(0,len(dep_relations)):
        #     left_rel.append('left_arc_'+dep_relations[i])
        #     right_rel.append('right_arc_'+dep_relations[i])

        # classes = left_rel + right_rel + ['shift']
        # int_classes = []
        # class_map = {}
        # for i in range(0,91):
            # int_classes.append(i)
            # class_map[classes[i]] = i

        # mapping = keras.utils.to_categorical(int_classes, num_classes=91, dtype='int32')
        blank = [0] * 91
        blank[self.output_labels[output_pair]] = 1
        return (np.array(blank))
        # print(len(mapping[-1]))
        # print (output_pair)
        # print (mapping)
        # return (np.array(mapping[self.output_labels[output_pair]]))
        # if output_pair[0] == 'shift':
        #     # print (np.array(mapping[-1]))
        #     return np.array(mapping[-1]) 
        # else:
        #     key = output_pair[0]+'_'+output_pair[1]
        #     # print (np.array(mapping[class_map[key]]))
        #     return np.array(mapping[class_map[key]])
        # print (output_pair)


        # return np.zeros(91)

     
    
def get_training_matrices(extractor, in_file):
    inputs = []
    outputs = []
    count = 0 
    for dtree in conll_reader(in_file): 
        words = dtree.words()
        pos = dtree.pos()
        # print (words)
        for state, output_pair in get_training_instances(dtree):
            inputs.append(extractor.get_input_representation(words, pos, state))
            outputs.append(extractor.get_output_representation(output_pair))
        if count%100 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
        count += 1
        #####my code
        # if count >= 1:
        #     break
        ##### my code
    sys.stdout.write("\n")
    return np.vstack(inputs),np.vstack(outputs)
       


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 


    with open(sys.argv[1],'r') as in_file:   

        extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
        # print (extractor.output_labels)

        # # print (extractor.word_vocab)
        # # print(extractor.pos_vocab)

        print("Starting feature extraction... (each . represents 100 sentences)")
        inputs, outputs = get_training_matrices(extractor,in_file)
        print("Writing output...")
        np.save(sys.argv[2], inputs)
        np.save(sys.argv[3], outputs)


