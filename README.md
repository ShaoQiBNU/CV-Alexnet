# Alexnet_MNIST
利用Alexnet实现MNIST
===================

利用Alexnet的5层卷积结构实现MNIST识别（注意：层数传递时影像的大小很重要，直接决定网络是否正确）
----------------------------------------------------------------------------------

对于conv和pool层，均涉及padding的类型，padding有两种类型，一种是'SAME'，一种是'VALID'。
-------------------------------------------------------------------------------

		对于SAME，the output height and width are computed as:

		out_height = ceil(float(in_height) / float(strides[1]))

		out_width = ceil(float(in_width) / float(strides[2]))


		对于VALID，the output height and width are computed as:

		out_height = ceil(float(in_height - filter_height + 1) / float(strides1))

		out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))

		ceil表示向上取整
