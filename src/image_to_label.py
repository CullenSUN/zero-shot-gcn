import os
import numpy as np
import cv2
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
	# for name in var_keep_dic:
	# 	if not my_dict.has_key(name):
	# 		print('I do not have ', name)
	restorer = tf.compat.v1.train.Saver(variables_to_restore)
	restorer.restore(sess, model_path)
	print('Initialized')

def image_to_feature(image_file):
	# mageNet preatrained CNN (resnet 50)
	model_path = '../pretrain_weights/resnet_v1_50.ckpt'
	image_directory = '../demo_images/'
	feature_directory = '../demo_feats/'

	# key to put this here to init global variables
	pool5, image_holder = res50()

	tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
	tfconfig.gpu_options.allow_growth = True
	sess = tf.compat.v1.Session(config=tfconfig)
	init(model_path, sess)

	image_path = os.path.join(image_directory, image_file)
	image = preprocess_res50(image_path)
	if image is None:
		print('Image doesn\'t exist')
		return

	feature = run_feat(sess, pool5, image_holder, image)
	feature_path = os.path.join(feature_directory, image_file.split('.')[0] + '.npz')
	np.savez_compressed(feature_path, feat=feature)

def getTop10Labels():
	return None


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

def run_feat(sess, pool5, image_holder, image):
	feat = sess.run(pool5, feed_dict={image_holder: image})
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
	image_to_feature("n02112497_47.JPEG")
