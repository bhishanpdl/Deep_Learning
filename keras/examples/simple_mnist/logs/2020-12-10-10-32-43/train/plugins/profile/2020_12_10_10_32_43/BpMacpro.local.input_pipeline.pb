	�ʡE���?�ʡE���?!�ʡE���?	+�~J<C!@+�~J<C!@!+�~J<C!@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�ʡE���?
ףp=
�?A�E�����?Y�Q���?*	     @[@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatˡE����?!�Ṱ�B@)y�&1��?1���%�9@:Preprocessing2F
Iterator::ModelˡE����?!�Ṱ�B@)9��v���?1��d	l�7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice����Mb�?!����[-@)����Mb�?1����[-@:Preprocessing2U
Iterator::Model::ParallelMapV2���Q��?!^8�߅+@)���Q��?1^8�߅+@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor9��v���?!��d	l�'@)9��v���?1��d	l�'@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapZd;�O��?!�����5@)y�&1�|?1���%�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�� �rh�?!s�3R1O@)����Mbp?1����[@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 8.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t11.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9,�~J<C!@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	
ףp=
�?
ףp=
�?!
ףp=
�?      ��!       "      ��!       *      ��!       2	�E�����?�E�����?!�E�����?:      ��!       B      ��!       J	�Q���?�Q���?!�Q���?R      ��!       Z	�Q���?�Q���?!�Q���?JCPU_ONLYY,�~J<C!@b 