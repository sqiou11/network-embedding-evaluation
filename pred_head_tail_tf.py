import numpy as np
import pickle
import sys,os
import argparse
import ast
from models.ProjE import ProjE

all_nodes = []
train_edge_dict = {}
in_mapping = {}
type_offset = {}

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

def calculate_metrics(batch, head_type, head_val, tail_type, tail_val, edge_type, pred_tail=True):
	if not pred_tail:
		target_idx = in_mapping[head_type][head_val] + type_offset[head_type]
	else:
		target_idx = in_mapping[tail_type][tail_val] + type_offset[tail_type]
	target=batch[target_idx]
	#print('target score = ', target)
	num_less, num_grtr, num_grtr_f = 0, 0, 0
	#print(batch)
	#print(np.argmax(batch), batch[np.argmax(batch)])
	for pred_id, s in enumerate(batch):
		#print('node %d scored %f' % (scored_node_ids[i], s))
		#if s < target:
		#	num_less += 1
		if s > target:
			#print('%d scored higher' % pred_id)
			num_grtr += 1
			if not pred_tail:
				# if this node doesn't exist as a head in training set, head is not linked with this tail, or this head and tail not linked by this edge
				if str(pred_id) not in train_edge_dict[head_type] or tail_val not in train_edge_dict[head_type][str(pred_id)] or edge_type not in train_edge_dict[head_type][str(pred_id)][tail_val]:
					num_grtr_f += 1
			else:
				# if this node doesn't exist as a tail for the given head node or these two are linked with an edge not of this type
				if str(pred_id) not in train_edge_dict[head_type][head_val] or edge_type not in train_edge_dict[head_type][head_val][str(pred_id)]:
					num_grtr_f += 1
			
	#print('target score rank = ', num_grtr+1)
	# the lower num_grtr is, the higher target is ranked (closer to 1)
	#rr_list = map(lambda x: 1./x, range(num_grtr+1, len(batch)-num_less+1))
	#rr = sum(rr_list) / (len(batch) - num_less - num_grtr)
	rr = 1.0/(1+num_grtr)
	rr_f = 1.0/(1+num_grtr_f)
	#print('target score rank = ', 1.0/(num_grtr+1), rr, num_less)
	hit1 = float(num_grtr<1)
	hit3 = float(num_grtr<3)
	hit10 = float(num_grtr<10)
	hit1_f = float(num_grtr_f<1)
	hit3_f = float(num_grtr_f<3)
	hit10_f = float(num_grtr_f<10)
	return rr, hit1, hit3, hit10, rr_f, hit1_f, hit3_f, hit10_f

if __name__ == '__main__':
	args = parse_args()
	args.test_dir = os.path.abspath('./datasets/FB15K237/per_type_temp/')
	config_file = os.path.abspath('./datasets/FB15K237/fb15k237.config')
	type_offset_file = os.path.abspath('./datasets/FB15K237/fb15k237_ko_0.07_offset.p')
	config = read_config(config_file)
	
	global type_offset
	type_offset = pickle.load(open(type_offset_file, 'rb'))
	if args.model == 'ProjE':
		model = ProjE()
	else:
		model = Proje()

	suffix = '_fb15k237_ko_0.07_eval.txt'

	global in_mapping
	global train_edge_dict
	in_mapping = pickle.load(open('./datasets/FB15K237/fb15k237_ko_0.07_in_mapping.p', 'rb'))
	train_edge_dict = pickle.load(open('./datasets/FB15K237/fb15k237_train_edge_dict.p', 'rb'))
	node_type_dict = pickle.load(open('./datasets/FB15K237/fb15k237_node_type_dict.p', 'rb'))
	ko_dic = pickle.load(open('./datasets/FB15K237/fb15k237_ko_dict.p', 'rb'))

	#global all_nodes
	for node_type, id_dict in in_mapping.items():
		for key, index in id_dict.items():
			assert int(key) == index, "key should be index in in_mapping"
	#		all_nodes.append(index + type_offset[node_type])
	#all_nodes.sort()
	#print(all_nodes)
	#all_edges = [ idx for idx, _ in enumerate(config['types']) ]
	#print('num entities = %d, num edges = %d' % (len(all_nodes), len(all_edges)))

	metrics = {
		'predict_head': {
			'rr_sum': 0.0,
			'hit1_sum': 0.0,
			'hit3_sum': 0.0,
			'hit10_sum': 0.0,
			'rr_f_sum': 0.0,
			'hit1_f_sum': 0.0,
			'hit3_f_sum': 0.0,
			'hit10_f_sum': 0.0
		},
		'predict_tail': {
			'rr_sum': 0.0,
			'hit1_sum': 0.0,
			'hit3_sum': 0.0,
			'hit10_sum': 0.0,
			'rr_f_sum': 0.0,
			'hit1_f_sum': 0.0,
			'hit3_f_sum': 0.0,
			'hit10_f_sum': 0.0
		}
	}
	test_count = 0

	with open('./datasets/FB15K237/fb15k237_test.hin') as INPUT:
		for line in INPUT.readlines():
			node = line.strip().split(' ')
			#print('processing test case %s' % line)
			_type_a, _id_a = node[0].split(':')
			_type_b, _id_b = node[1].split(':')
			if args.model == 'HEER':
				# HEER model needs ID of edge as stored in its config
				edge_type = config['types'].index(node[3])
			else:
				# other models use ID of edge taken directly from the file
				edge_type = int(node[3].split(':')[0])
			if _id_a not in train_edge_dict[_type_a] or _id_b not in train_edge_dict[_type_b]:
				print('test_head_value %s or test_tail_value %s not valid' % (_id_a, _id_b))
				continue

			head = in_mapping[_type_a][_id_a] + type_offset[_type_a]
			tail = in_mapping[_type_b][_id_b] + type_offset[_type_b]
			# Test head prediction
			#_input, _output = gen_corrupt_samples(_type_a, _id_a, _type_b, _id_b, train_edge_dict, ko_dic, in_mapping, type_offset, True)
			target_idx = head
			scores = model.predict(head, tail, edge_type, False)
			#scores = model.predict_rel(_input, _output, all_edges)
			#print('target_idx = %d' % target_idx)
			rr, hit1, hit3, hit10, rr_f, hit1_f, hit3_f, hit10_f = calculate_metrics(scores, _type_a, _id_a, _type_b, _id_b, node[3], False)
			metrics['predict_head']['rr_sum'] += rr
			metrics['predict_head']['hit1_sum'] += hit1
			metrics['predict_head']['hit3_sum'] += hit3
			metrics['predict_head']['hit10_sum'] += hit10
			metrics['predict_head']['rr_f_sum'] += rr_f
			metrics['predict_head']['hit1_f_sum'] += hit1_f
			metrics['predict_head']['hit3_f_sum'] += hit3_f
			metrics['predict_head']['hit10_f_sum'] += hit10_f

			# Test tail prediction
			#_input, _output = gen_corrupt_samples(_type_a, _id_a, _type_b, _id_b, train_edge_dict, ko_dic, in_mapping, type_offset, False)
			target_idx = tail
			scores = model.predict(head, tail, edge_type, True)
			rr, hit1, hit3, hit10, rr_f, hit1_f, hit3_f, hit10_f = calculate_metrics(scores, _type_a, _id_a, _type_b, _id_b, node[3], True)
			metrics['predict_tail']['rr_sum'] += rr
			metrics['predict_tail']['hit1_sum'] += hit1
			metrics['predict_tail']['hit3_sum'] += hit3
			metrics['predict_tail']['hit10_sum'] += hit10
			metrics['predict_tail']['rr_f_sum'] += rr_f
			metrics['predict_tail']['hit1_f_sum'] += hit1_f
			metrics['predict_tail']['hit3_f_sum'] += hit3_f
			metrics['predict_tail']['hit10_f_sum'] += hit10_f
			test_count += 1
		#print('Finished scoring edge %s, rr_sum = %f, test_count = %d' % (node[3], rr_sum, test_count))

	print ('total tests = %d' % test_count)
	print ('metric:\t\t\tMRR\t\thit@1\t\thit@3\t\thit@10')
	print ('head (raw)\t\t%f\t%f\t%f\t%f' % (
		metrics['predict_head']['rr_sum']/test_count,
		metrics['predict_head']['hit1_sum']/test_count,
		metrics['predict_head']['hit3_sum']/test_count,
		metrics['predict_head']['hit10_sum']/test_count))
	print ('tail (raw)\t\t%f\t%f\t%f\t%f' % (
		metrics['predict_tail']['rr_sum']/test_count,
		metrics['predict_tail']['hit1_sum']/test_count,
		metrics['predict_tail']['hit3_sum']/test_count,
		metrics['predict_tail']['hit10_sum']/test_count))
	print ('head (filtered)\t\t%f\t%f\t%f\t%f' % (
		metrics['predict_head']['rr_f_sum']/test_count,
		metrics['predict_head']['hit1_f_sum']/test_count,
		metrics['predict_head']['hit3_f_sum']/test_count,
		metrics['predict_head']['hit10_f_sum']/test_count))
	print ('tail (filtered)\t\t%f\t%f\t%f\t%f' % (
		metrics['predict_tail']['rr_f_sum']/test_count,
		metrics['predict_tail']['hit1_f_sum']/test_count,
		metrics['predict_tail']['hit3_f_sum']/test_count,
		metrics['predict_tail']['hit10_f_sum']/test_count))