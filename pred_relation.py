import numpy as np
import cPickle
import sys,os
import argparse
import ast
from models.HEER import HEER
from models.DistMult import DistMult
from models.ComplEx import ComplEx

all_nodes = []
train_edge_dict = {}
in_mapping = {}
type_offset = {}

def read_config(conf_name):
	config = {}
	with open(conf_name) as IN:
		config['edges'] = ast.literal_eval(IN.readline())
		config['nodes'] = ast.literal_eval(IN.readline())
		config['types'] = ast.literal_eval(IN.readline())
		for i,x in enumerate(ast.literal_eval(IN.readline())):
			config['edges'][i].append(int(x))        
	assert len(config['edges']) == len(config['types'])
	return config

def parse_args():
	'''
	Parses the heer arguments.
	'''
	parser = argparse.ArgumentParser(description="Run heer.")

	parser.add_argument('--model', nargs='?', default='HEER',
						help='Which specific model to run')

	parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
	                    help='Input graph path')

	parser.add_argument('--gpu', nargs='?', default='0',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')
	
	parser.add_argument('--batch-size', type=int, default=50,
	                    help='Batch size. Default is 50.')

	parser.add_argument('--graph-name', type=str, default='',
                    	help='prefix of dumped data')
	parser.add_argument('--data-dir', type=str, default='',
                    	help='data directory')
	parser.add_argument('--model-dir', type=str, default='',
                    	help='model directory')
	parser.add_argument('--test-dir', type=str, default='',
                    	help='test directory')

	return parser.parse_args()

def calculate_metrics(scores, head_type, head_val, tail_type, tail_val, edge_type_str, edge_type_idx, edge_id_str_map=None):
	target=scores[edge_type_idx]
	# sort the scores in descending order, but keep indices of original array
	#desc_score_ids = np.array(batch).argsort()[::-1]
	#print('target score = ', target)
	num_less, num_grtr, num_grtr_f = 0, -1, 0
	avg_precision = 0.0
	true_pos_cnt, false_pos_cnt = 0.0, 0.0
	#print(scores)
	for pred_rank, pred_id in enumerate(scores):
		if pred_id == edge_type_idx:
			num_grtr = pred_rank
		#compute average precision
		pred_str = str(pred_id) + ':d'
		if edge_id_str_map:
			pred_str = edge_id_str_map[pred_id]
		if pred_str == edge_type_str or pred_str in train_edge_dict[head_type][head_val]:
			true_pos_cnt += 1
			avg_precision += true_pos_cnt/(true_pos_cnt + false_pos_cnt)
		else:
			false_pos_cnt += 1
			num_grtr_f += (num_grtr == -1)

	rr = 1.0/(1+num_grtr)
	rr_f = 1.0/(1+num_grtr_f)
	#print('target score rank = ', 1.0/(num_grtr+1), rr, num_less)
	hit1 = float(num_grtr<1)
	hit3 = float(num_grtr<3)
	hit10 = float(num_grtr<10)
	hit1_f = float(num_grtr_f<1)
	hit3_f = float(num_grtr_f<3)
	hit10_f = float(num_grtr_f<10)
	return rr, hit1, hit3, hit10, rr_f, hit1_f, hit3_f, hit10_f, avg_precision/true_pos_cnt

def check_connection(ko_dic,edge_dictionary,node_1_value,node_2_type,node_2_value):
	if node_1_value in edge_dictionary[node_2_type][node_2_value] :
		return True
	else:
		if node_1_value in ko_dic:
			if node_2_value in ko_dic[node_1_value]:
				return True
		return False

# Generate corrupted samples by either corrupting the head or the tail nodes
# This ignores all nodes that exist in the training set that complete the triplet
# Ex. if 
def gen_corrupt_samples(test_head_type, test_head_value, test_tail_type, test_tail_value, edge_dictionary, ko_dic, in_mapping, type_offset, corrupt_head=True):
	new_edge_dictionary={}
	for node_type in edge_dictionary:
		new_edge_dictionary[node_type]=list(edge_dictionary[node_type])

	heads = [in_mapping[test_head_type][test_head_value] + type_offset[test_head_type]]
	tails = [in_mapping[test_tail_type][test_tail_value] + type_offset[test_tail_type]]

	if corrupt_head:
		heads = range(14541)
		tails += tails * 14540
	else:
		tails = range(14541)
		heads += heads * 14540
	#print(np.array(heads), np.array(tails))
	return heads, tails
	
	"""
	test_case=test_head_type+":"+test_head_value+" "+test_tail_type+":"+test_tail_value
	#sample_number of negative edges with same node_1, but different node_2
	#content_temp.append(temp)
	count=0
	if not corrupt_head:
		for neg_test_tail_value in new_edge_dictionary[test_tail_type]:
			#random_test_tail_value=choice(new_edge_dictionary[test_tail_type])
			#while (random_test_tail_value in edge_dictionary[test_head_type][test_head_value] or random_test_tail_value in ko_dic[test_head_value]):
			#    random_test_tail_value=choice(new_edge_dictionary[test_tail_type])
			if neg_test_tail_value not in edge_dictionary[test_head_type][test_head_value] and neg_test_tail_value not in ko_dic[test_head_value]:
				tails.append(in_mapping[test_tail_type][neg_test_tail_value] + type_offset[test_tail_type])
				heads.append(in_mapping[test_head_type][test_head_value] + type_offset[test_head_type])
				#temp=test_head_type+":"+test_head_value+" "+test_tail_type+":"+neg_test_tail_value+" "+'0 '+edge+'\n'
				#content_temp.append(temp)
				count+=1
				#if count == 100:
				#	break
			#else:
			#	print('%s:%s is connected to %s:%s' % (test_tail_type, neg_test_tail_value, test_head_type, test_head_value))
		#print ('Generated %d corrupted tail triples for %s' % (count, test_case))
	#sample_number of negative edges with same node_2 but differnt node_1
	else:
		for neg_test_head_value in new_edge_dictionary[test_head_type]:
			#random_test_head_value=choice(new_edge_dictionary[test_head_type])
			#while check_connection(ko_dic,edge_dictionary,random_test_head_value,test_tail_type,test_tail_value):
			#while (random_test_head_value in edge_dictionary[test_tail_type][test_tail_value] or test_tail_value in ko_dic[test_head_value]):
			#    random_test_head_value=choice(new_edge_dictionary[test_head_type])
			if not check_connection(ko_dic,edge_dictionary,neg_test_head_value,test_tail_type,test_tail_value):
				heads.append(in_mapping[test_head_type][neg_test_head_value] + type_offset[test_head_type])
				tails.append(in_mapping[test_tail_type][test_tail_value] + type_offset[test_tail_type])
				#temp=test_tail_type+":"+test_tail_value+" "+test_head_type+":"+neg_test_head_value+" "+'0 '+edge+'-1'+'\n'
				#content_temp.append(temp)
				count+=1
				#if count == 100:
				#	break
			#else:
			#	print('%s:%s is connected to %s:%s' % (test_head_type, neg_test_head_value, test_tail_type, test_tail_value))
		#print ('Generated %d corrupted head triples for %s' % (count, test_case))
	return heads, tails"""

if __name__ == '__main__':
	args = parse_args()
	args.test_dir = os.path.abspath('./datasets/FB15K237/per_type_temp/')
	config_file = os.path.abspath('./datasets/FB15K237/fb15k237.config')
	type_offset_file = os.path.abspath('./datasets/FB15K237/fb15k237_ko_0.07_offset.p')
	config = read_config(config_file)
	#t.cuda.set_device(0)
	
	global type_offset
	type_offset = cPickle.load(open(type_offset_file))

	if args.model == 'HEER':
		model = HEER(type_offset, config)
	elif args.model == 'DistMult':
		model = DistMult('embeddings/distmult/distmult.pt')
	elif args.model == 'ComplEx':
		model = ComplEx('embeddings/complex/fb15k237_ComplEx.pt')
	else:
		model = HEER(type_offset, config)

	suffix = '_fb15k237_ko_0.07_eval.txt'

	global in_mapping
	global train_edge_dict
	in_mapping = cPickle.load(open('./datasets/FB15K237/fb15k237_ko_0.07_in_mapping.p'))
	train_edge_dict = cPickle.load(open('./datasets/FB15K237/fb15k237_train_edge_dict.p'))
	node_type_dict = cPickle.load(open('./datasets/FB15K237/fb15k237_node_type_dict.p'))
	ko_dic = cPickle.load(open('./datasets/FB15K237/fb15k237_ko_dict.p'))

	#global all_nodes
	for node_type, id_dict in in_mapping.iteritems():
		for key, index in id_dict.iteritems():
			assert int(key) == index, "key should be index in in_mapping"
	#		all_nodes.append(index + type_offset[node_type])
	#all_nodes.sort()
	#print(all_nodes)
	#all_edges = [ idx for idx, _ in enumerate(config['types']) ]
	#print('num entities = %d, num edges = %d' % (len(all_nodes), len(all_edges)))

	metrics = {
		'predict_relation': {
			'rr_sum': 0.0,
			'hit1_sum': 0.0,
			'hit3_sum': 0.0,
			'hit10_sum': 0.0,
			'rr_f_sum': 0.0,
			'hit1_f_sum': 0.0,
			'hit3_f_sum': 0.0,
			'hit10_f_sum': 0.0,
			'avg_precision': 0.0
		},
	}
	test_count = 0
	with open('./datasets/FB15K237/fb15k237_test.hin') as INPUT:
		for line in INPUT.readlines():
			node = line.strip().split(' ')
			print('processing test case %d' % test_count)
			_type_a, _id_a = node[0].split(':')
			_type_b, _id_b = node[1].split(':')
			edge_id_str_arr = None # non-HEER models map edge IDs directly to their indices
			if args.model == 'HEER':
				# HEER model needs ID of edge as stored in its config
				edge_type = config['types'].index(node[3])
				edge_id_str_arr = config['types']
			else:
				# other models use ID of edge taken directly from the file
				edge_type = int(node[3].split(':')[0])
			if _id_a not in train_edge_dict[_type_a] or _id_b not in train_edge_dict[_type_b]:
				print('test_head_value %s or test_tail_value %s not valid' % (_id_a, _id_b))
				continue

			head = in_mapping[_type_a][_id_a] + type_offset[_type_a]
			tail = in_mapping[_type_b][_id_b] + type_offset[_type_b]
			# Test relation prediction
			target_idx = head
			scores = model.predict_relation(head, tail, edge_type)
			#scores = model.predict_rel(_input, _output, all_edges)
			#print('target_idx = %d' % target_idx)
			rr, hit1, hit3, hit10, rr_f, hit1_f, hit3_f, hit10_f, avg_prec = calculate_metrics(scores, _type_a, _id_a, _type_b, _id_b, node[3], edge_type, edge_id_str_arr)
			metrics['predict_relation']['rr_sum'] += rr
			metrics['predict_relation']['hit1_sum'] += hit1
			metrics['predict_relation']['hit3_sum'] += hit3
			metrics['predict_relation']['hit10_sum'] += hit10
			metrics['predict_relation']['rr_f_sum'] += rr_f
			metrics['predict_relation']['hit1_f_sum'] += hit1_f
			metrics['predict_relation']['hit3_f_sum'] += hit3_f
			metrics['predict_relation']['hit10_f_sum'] += hit10_f
			metrics['predict_relation']['avg_precision'] += avg_prec

			test_count += 1
		#print('Finished scoring edge %s, rr_sum = %f, test_count = %d' % (node[3], rr_sum, test_count))

	print('total tests = %d' % test_count)
	print('metric:\t\t\tMRR\t\thit@1\t\thit@3\t\thit@10')
	print('relation (raw)\t\t%f\t%f\t%f\t%f' % (
		metrics['predict_relation']['rr_sum']/test_count,
		metrics['predict_relation']['hit1_sum']/test_count,
		metrics['predict_relation']['hit3_sum']/test_count,
		metrics['predict_relation']['hit10_sum']/test_count))
	print('relation (filtered)\t%f\t%f\t%f\t%f' % (
		metrics['predict_relation']['rr_f_sum']/test_count,
		metrics['predict_relation']['hit1_f_sum']/test_count,
		metrics['predict_relation']['hit3_f_sum']/test_count,
		metrics['predict_relation']['hit10_f_sum']/test_count))
	print('')
	print('relation mAP = %f' % (metrics['predict_relation']['avg_precision']/test_count))