import os
import json
import argparse
import cv2
import numpy as np
import pickle as pkl
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_v1

def init(model_path, sess):
	def get_variables_in_checkpoint_file(file_name):
		reader = tf.pywrap_tensorflow.NewCheckpointReader(file_name)
		var_to_shape_map = reader.get_variable_to_shape_map()
		return var_to_shape_map, reader

	var_keep_dic, reader = get_variables_in_checkpoint_file(model_path)
	my_var_list = tf.compat.v1.global_variables()
	sess.run(tf.compat.v1.variables_initializer(my_var_list, name='init'))

	variables_to_restore = []
	my_dict = {}
	for v in my_var_list:
		name = v.name.split(':')[0]
		my_dict[name] = 0
		if not var_keep_dic.has_key(name):
			print('He does not have', name)
		else:
			if v.shape != var_keep_dic[name]:
				print('Does not match shape: ', v.shape, var_keep_dic[name])
				continue
			variables_to_restore.append(v)

	restorer = tf.compat.v1.train.Saver(variables_to_restore)
	restorer.restore(sess, model_path)
	print('Initialized')

def image_to_labels(image_path):
	# imageNet preatrained CNN (resnet 50)
	model_path = '../pretrain_weights/resnet_v1_50.ckpt'

	# key to put this here to init global variables
	pool5, image_holder = res50()

	tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
	tfconfig.gpu_options.allow_growth = True
	session = tf.compat.v1.Session(config=tfconfig)
	init(model_path, session)

	image = preprocess_res50(image_path)
	if image is None:
		print('Image doesn\'t exist')
		return None

	# key step
	feature = run_feat(session, pool5, image_holder, image)

	topK = 10
	classifiers, labels = get_classifiers_with_labels()
	classifiers = classifiers.T

	scores = np.dot(feature, classifiers).squeeze()
	scores = scores - scores.max()
	scores = np.exp(scores)
	scores = scores / scores.sum()
	print("scores.shape", scores.shape)

	ids = np.argsort(-scores)
	topK_label_ids = []
	for sort_id in range(topK):
		lbl = labels[ids[sort_id]]
		topK_label_ids.append(lbl)
	return getNames(topK_label_ids)

def get_classifiers_with_labels():
	fc_model_path = '../model/wordnet_resnet_glove_feat_2048_1024_512_300'
	word2vec_path = '../data/word_embedding_model/glove_word2vec_wordnet.pkl'
	class_ids_path = '../data/list/corresp-2-hops.json'
	class_text_path = '../data/list/invdict_wordntext.json'

	feature_dimension = 2048

	with open(word2vec_path, 'rb') as fp:
		word2vec_feat = pkl.load(fp)

	with open(fc_model_path, 'rb') as fp:
		fc_layers_pred = pkl.load(fp)
	fc_layers_pred = np.array(fc_layers_pred)
	print('fc output', fc_layers_pred.shape)

	with open(class_ids_path) as fp:
		class_ids = json.load(fp)
	
	with open(class_text_path) as fp:
		class_names = json.load(fp)

	# process 'train' classes. they are possible candidates during inference
	cnt_zero_wv = 0
	labels_train, word2vec_train = [], []
	fc_now = []
	for j in range(len(class_ids)):
		tfc = fc_layers_pred[j]

		if class_ids[j][1] == 0:
			continue

		if class_ids[j][0] >= 0:
			twv = word2vec_feat[j]
			if np.linalg.norm(twv) == 0:
				cnt_zero_wv = cnt_zero_wv + 1
				continue
				
			# todo: use class_name to append to labels_train
			class_name = class_names[class_ids[j][0]]
			labels_train.append(class_ids[j][0])
			word2vec_train.append(twv)

			feature_len = len(tfc)
			tfc = tfc[feature_len - feature_dimension: feature_len]
			fc_now.append(tfc)

	fc_now = np.array(fc_now)
	labels = np.array(labels_train)

	print('skip candidate class due to no word embedding: %d / %d:' % (cnt_zero_wv, len(labels_train) + cnt_zero_wv))
	print('candidate class shape: ', fc_now.shape)
	return fc_now, labels

def getNames(label_indices):
	file_2_hops_path = '../data/list/2-hops.txt' 
	dict_path = '../data/list/words.pkl'

	wnids = []
	with open(file_2_hops_path) as fp:
		for line in fp:
			wnids.append(line.strip())
	
	with open(dict_path) as fp:
		wnid_word = pkl.load(fp)

	names = []
	for i in range(len(label_indices)):
		# -1000 is to offset 1K trained data set. Checkout make_corresp in prepare_list.py
		wnid_index = label_indices[i] - 1000
		wnid = wnids[wnid_index]
		names.append(wnid_word[wnid])
	return names

def preprocess_res50(image_path):
	_R_MEAN = 123.68
	_G_MEAN = 116.78
	_B_MEAN = 103.94
	image = cv2.imread(image_path)
	if image is None:
		return None
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	target_size = 256
	crop_size = 224
	im_size_min = np.min(image.shape[0:2])
	im_scale = float(target_size) / float(im_size_min)
	image = cv2.resize(image, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
	height = image.shape[0]
	width = image.shape[1]
	x = int((width - crop_size) / 2)
	y = int((height - crop_size) / 2)
	image = image[y: y + crop_size, x: x + crop_size]

	image = image.astype(np.float32)
	image[:, :, 0] -= _R_MEAN
	image[:, :, 1] -= _G_MEAN
	image[:, :, 2] -= _B_MEAN
	image = image[np.newaxis, :, :, :]
	return image

def run_feat(session, pool5, image_holder, image):
	feat = session.run(pool5, feed_dict={image_holder: image})
	feat = np.squeeze(feat)
	return feat

def res50():
	image = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, 224, 224, 3], 'image')
	with slim.arg_scope(resnet_arg_scope(is_training=False)):
		net_conv, end_point = resnet_v1.resnet_v1_50(image, global_pool=True, is_training=False)
	return net_conv, image

def resnet_arg_scope(is_training=True,
					 batch_norm_decay=0.997,
					 batch_norm_epsilon=1e-5,
					 batch_norm_scale=True):
	batch_norm_params = {
		'is_training': False,
		'decay': batch_norm_decay,
		'epsilon': batch_norm_epsilon,
		'scale': batch_norm_scale,
		'trainable': False,
		'updates_collections': tf.compat.v1.GraphKeys.UPDATE_OPS
	}
	with slim.arg_scope(
			[slim.conv2d],
			weights_initializer=slim.variance_scaling_initializer(),
			trainable=is_training,
			activation_fn=tf.nn.relu,
			normalizer_fn=slim.batch_norm,
			normalizer_params=batch_norm_params):
		with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
			return arg_sc

if __name__ == '__main__':
	
	# os.environ['CUDA_VISIBLE_DEVICES'] = '0' gpu unit
	parser = argparse.ArgumentParser(description='Process Inputs.')
	parser.add_argument('--image', type=str, default='../demo_images/tiger.PNG', help='path of iamge to classify')

	args = parser.parse_args()
	if not os.path.exists(args.image):
		print('image does not exist: %s' % args.image)
		raise NotImplementedError

	print("processing image at %s" % args.image)
	predicted_labels = image_to_labels(args.image)
	print("predicted_labels: ")
	print("\n".join(predicted_labels))

