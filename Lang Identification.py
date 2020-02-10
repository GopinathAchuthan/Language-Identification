import io
import numpy
import pickle

def train_data(data):
	lines = data.readlines()
	data_words = []
	for line in lines:
		if line.startswith("#") or line.startswith("\n"):
			continue
		line_split = line.split("\t")
		data_words.append(line_split[0:2])

	del(lines)
	sentences = []
	sentence = []
	W = len(data_words)
	word_types = dict()
	word_types["<UNK>"] = 0

	#Changing first occurrence types to unknown word
	for word in data_words:
		if word[0] is '1':
			sentences.append(sentence)
			sentence = []
		token = word[1]
		if token not in word_types.keys():

			word_types[token] = 0
			word_types["<UNK>"] += 1
			sentence.append("<UNK>")
		else:
			word_types[token] += 1
			sentence.append(token)

	del(sentence)
	del(sentences[0])

	words = dict()

	for i,j in word_types.items():
		if j is not 0:
			words[i] = j
	del(word_types)

	V = len(words)

	pairs = dict()
	pairs_count = 0
	for sentence in sentences:
		length_sentence = len(sentence)
		pairs_count += length_sentence + 1 
		pair = ("<BOS>",sentence[0])
		if pair in pairs.keys():
			pairs[pair] += 1
		else:
			pairs[pair] = 1
		
		for i in range(0,length_sentence-1):
			pair = (sentence[i],sentence[i+1])
			if pair in pairs.keys():
				pairs[pair] += 1
			else:
				pairs[pair] = 1
		
		pair = (sentence[length_sentence-1],"<EOS>")
		if pair in pairs.keys():
			pairs[pair] += 1
		else:
			pairs[pair] = 1

	return W, words, pairs, len(sentences)

def model(eng, french, data, laplace):
	e_W, e_words, e_pairs, e_sentences = eng
	f_W, f_words, f_pairs, f_sentences = french


	e_words["<BOS>"]=e_sentences
	e_words["<EOS>"]=e_sentences
	e_W += 2*e_sentences

	f_words["<BOS>"]=f_sentences
	f_words["<EOS>"]=f_sentences
	f_W += 2*f_sentences

	e_pairs_count = len(e_pairs)
	f_pairs_count = len(f_pairs)
	t_pairs_count = e_pairs_count + f_pairs_count
	e_pairs_sum = 0
	f_pairs_sum = 0
	for value in e_pairs.values():
		e_pairs_sum += value
	for value in f_pairs.values():
		f_pairs_sum += value

	probability_eng = (e_pairs_count + laplace) / (t_pairs_count * laplace + e_pairs_sum)
	probability_french = (f_pairs_count + laplace) / (t_pairs_count * laplace + f_pairs_sum)

	sentences = create_sentences(data, e_words, f_words)

	label_file = []
	label_pair = dict()

	for sentence in sentences:
		sentence.insert(0,"<BOS>")
		sentence.append("<EOS>")
		label = []
		for i in range(len(sentence)-1):
			pair = (sentence[i], sentence[i+1])
			if pair in e_pairs.keys() or pair in f_pairs.keys():
				if pair in e_pairs.keys() and pair in f_pairs.keys():
					pair_e = e_pairs[pair]
					pair_f = f_pairs[pair]
					e_sum_x,e_count_x = start_pair_x(pair[0],e_pairs)
					f_sum_x,f_count_x = start_pair_x(pair[0],f_pairs)
					e_word_x = e_words[pair[0]]
					f_word_x = f_words[pair[0]]
					label_e = ((pair_e+laplace)/(e_sum_x + laplace * e_count_x) )  * probability_eng
					label_f = ((pair_f+laplace)/(f_sum_x + laplace * f_count_x)) * probability_french

					if label_e > label_f:
						label.append("eng")
						label_pair[pair] = "eng"
					elif label_f > label_e:
						label.append("french")
						label_pair[pair] = "french"
					else:
						label.append("unk")
						label_pair[pair] = "unk"

				elif pair in e_pairs.keys():
					label_pair[pair] = "eng"
					label.append("eng")
				else:
					label_pair[pair] = "french"
					label.append("french")

			else:
				label_pair[pair] = "unk"
				label.append("unk")	

		label_file.append(label)

	lang = dict()
	lang["eng"] = 0
	lang["french"] = 0
	lang["unk"] = 0
	for label in label_file:
		if label.count("eng") > label.count("french"):
			#print("English")
			lang["eng"] += 1
		elif label.count("french") > label.count("eng"):
			#print("French")
			lang["french"] += 1
		else:
			#print("UNKNOWN")
			lang["unk"] += 1

	print("English count:", lang["eng"])
	print("French count", lang["french"])
	print("Unknown count", lang["unk"])

	return label_pair

def start_pair_x(x,pairs):
	sum_x = 0
	count_x = 0
	for pair, value in pairs.items():
		if x == pair[0]:
			sum_x += value
			count_x += 1
	return sum_x, count_x 


def create_sentences(data, e_words, f_words):
	lines = data.readlines()
	words =[]

	for line in lines:
		if line.startswith("#") or line.startswith("\n"):
			continue
		line_split = line.split("\t")
		words.append(line_split[0:2])

	del(lines)
	sentences = []
	sentence = []

	for word in words:
		if word[0] is '1':
			sentences.append(sentence)
			sentence = []
		if word[1] in e_words.keys() or word[1] in f_words.keys()  :
			sentence.append(word[1])
		else:
			sentence.append("<UNK>")
	del(sentence)
	del(words)
	del(sentences[0])

	return sentences


def main():

	laplace = 0

	try:

		eng_train = open("train.conllu", "r", encoding="utf-8")
		french_train = open("french_train.conllu", "r", encoding="utf-8")
		pickle_file = open("q7.pickle","wb")

		dev = open("dev.conllu", "r", encoding="utf-8")

		eng = train_data(eng_train)
		french = train_data(french_train)

		label_pair = model(eng, french, dev, laplace)
		pickle.dump(label_pair,pickle_file)


	finally:
		eng_train.close()
		french_train.close()
		pickle_file.close()

if __name__ == "__main__":
	main()