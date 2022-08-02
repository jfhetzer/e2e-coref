import re

###
#    Parse German SemEval datasets into CONLL-2012 format
###

class Parser:

	def parse_line(self, line):
		# handle begin of document and word lines
		# ignore end of document and empty lines
		line = line.strip()
		if not line or line.startswith('#end document'):
			return line
		if line.startswith('#begin document'):
			self.parse_begin_line(line)
			return line
		return self.parse_word_line(line)

	def parse_begin_line(self, line):
		# document name is everything after 'begin document'
		match = re.search('#begin document (.*)', line)
		name = match.group(1)
		# document id can't contain any whitespaces
		name = name.strip()
		self.doc_id = name.replace(' ', '_')

	def parse_word_line(self, line):
		# split into columns of CoNLL-2010 Format
		cols_2010 = line.split()
		# rearrange columns into CoNLL-2012 Format
		cols_2012 = []
		cols_2012.append(self.doc_id) 	# 01 document id
		cols_2012.append(0) 		# 02 part number
		cols_2012.append(cols_2010[0])	# 03 word number
		cols_2012.append(cols_2010[1]) 	# 04 word itself
		cols_2012.append(cols_2010[4]) 	# 05 part of speach
		cols_2012.append('-') 		# 06 parse bit
		cols_2012.append(cols_2010[2]) 	# 07 predicate lemma
		cols_2012.append('-')		# 08 predicted frameset id
		cols_2012.append('-')		# 09 word sense
		cols_2012.append('-')		# 10 speaker
		cols_2012.append(cols_2010[12]) # 11 named entities
		cols_2012.append('-')		# 12 predicate arguments
		cols_2012.append(cols_2010[16]) # 13 coreference
		# join columns and return
		return ' '.join(str(c) for c in cols_2012)


	def parse(self, in_file, out_file):
		with open(in_file, 'r') as in_file:
			with open(out_file, 'w') as out_file:
				line = in_file.readline()
				while line:
					output = self.parse_line(line)
					if output:
						out_file.write(output + '\n')
					line = in_file.readline()


if __name__ == '__main__':
	parser = Parser()
	parser.parse('de.trial.txt', 'de.trial.conll')
