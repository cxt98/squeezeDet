import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import math
import torch.nn.functional as F


__all__ = ['ZynqNet_FPN', 'zynqnet_fpn']


class Conv2dSame(nn.Module):

	def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
		super(Conv2dSame, self).__init__()
		self.F = kernel_size
		self.S = stride
		self.D = dilation
		self.layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

	def forward(self, x_in):
		N, C, H, W = x_in.shape
		H2 = math.ceil(H / self.S)
		W2 = math.ceil(W / self.S)
		Pr = (H2 - 1) * self.S + (self.F - 1) * self.D + 1 - H
		Pc = (W2 - 1) * self.S + (self.F - 1) * self.D + 1 - W
		x_pad = nn.ZeroPad2d((int(Pr//2), int(Pr - Pr//2), int(Pc//2), int(Pc - Pc//2)))(x_in)
		x_out = self.layer(x_pad)
		# if H % self.S == 0:
		# 	pad = max(self.F - self.S, 0)
		# else:
		# 	pad = max(self.F - (H % self.S), 0)

		# if pad % 2 == 0:
		# 	pad_val = pad // 2
		# 	padding = (pad_val, pad_val, pad_val, pad_val)
		# else:
		# 	pad_val_start = pad // 2
		# 	pad_val_end = pad - pad_val_start
		# 	padding = (pad_val_start, pad_val_end, pad_val_start, pad_val_end)
		# x = F.pad(x_in, padding, "constant", 0)
		# x_out = self.layer(x)

		return x_out

class Fire(nn.Module):
	def __init__(self, inplanes, squeeze_planes,
				expand1x1_planes, expand3x3_planes, pool):
		super(Fire, self).__init__()
		self.inplanes = inplanes
		if pool:
			self.squeeze = Conv2dSame(inplanes, squeeze_planes, kernel_size=3, stride=2)
		else:
			self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)

		self.squeeze_activation = nn.ReLU(inplace=True)
		self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
		self.expand1x1_activation = nn.ReLU(inplace=True)
		self.expand3x3 = Conv2dSame(squeeze_planes, expand3x3_planes, kernel_size=3)
		self.expand3x3_activation = nn.ReLU(inplace=True)

	def forward(self, x):
		# print(x.size())
		x = self.squeeze_activation(self.squeeze(x))
		return torch.cat([
			self.expand1x1_activation(self.expand1x1(x)),
			self.expand3x3_activation(self.expand3x3(x)),
			], 1)


class ZynqNet_FPN(nn.Module):
	def __init__(self, num_classes=1000, num_shape=16):
		super(ZynqNet_FPN, self).__init__()
		self.num_classes = num_classes
		self.num_shape = num_shape

		self.conv1 = Conv2dSame(3, 64, kernel_size=3, stride=2)
		self.fire2 = Fire(64, 16, 64, 64, True)
		self.fire3 = Fire(128, 16, 64, 64, False)
		self.fire4 = Fire(128, 32, 128, 128, True)
		self.fire5 = Fire(256, 32, 128, 128, False)
		self.fire6 = Fire(256, 64, 256, 256, True)
		self.fire7 = Fire(512, 64, 192, 192, False)
		self.fire8 = Fire(384, 112, 256, 256, True)
		self.fire9 = Fire(512, 112, 368, 368, False)

		self.top_layer = nn.Conv2d(736, 256, kernel_size=1)
		self.lat4 = nn.Conv2d(512, 256, kernel_size=1)
		self.lat3 = nn.Conv2d(256, 256, kernel_size=1)
		self.predictor = Conv2dSame(256, 26, kernel_size=3, stride=1)

	
	def _upsample_add(self, x, y):
		'''Upsample and add two feature maps.
		Args:
		  x: (Variable) top feature map to be upsampled.
		  y: (Variable) lateral feature map.
		Returns:
		  (Variable) added feature map.
		Note in PyTorch, when input size is odd, the upsampled feature map
		with `F.upsample(..., scale_factor=2, mode='nearest')`
		maybe not equal to the lateral feature map size.
		e.g.
		original input size: [N,_,15,15] ->
		conv2d feature map size: [N,_,8,8] ->
		upsampled feature map size: [N,_,16,16]
		So we choose bilinear upsample which supports arbitrary output sizes.
		'''
		_,_,H,W = y.size()
		return F.upsample(x, size=(H,W), mode='nearest') + y

	def forward(self, x):
		c1 = self.conv1(x)
		f2 = self.fire2(c1)
		f3 = self.fire3(f2)
		f4 = self.fire4(f3)
		f5 = self.fire5(f4)
		f6 = self.fire6(f5)
		f7 = self.fire7(f6)
		f8 = self.fire8(f7)
		f9 = self.fire9(f8)

		p5 = self.top_layer(f9)
		p4 = self._upsample_add(p5, self.lat4(f6))
		p3 = self._upsample_add(p4, self.lat3(f4))
		preds = self.predictor(p5)
		preds_p4 = self.predictor(p4)
		preds_p3 = self.predictor(p3)

		return [c1, preds, preds_p4, preds_p3]

def zynqnet_fpn(pretrained=False, **kwargs):
	model = ZynqNet_FPN(**kwargs)
	model_dict = model.state_dict()
	if pretrained:
		print("using pretrained model")
		pretrained_dict = torch.load('./data/ZynqNet/zynqnet_fpn.pkl')
		# for k, v in model_dict.iteritems():
		# 	print k 
		# for k, v in pretrained_dict.iteritems():
		# 	print k
		# print(model.state_dict()['conv1.bias'])
		model.state_dict()['conv1.layer.weight'] .copy_(torch.from_numpy(pretrained_dict['conv1/kernels:0']))
		model.state_dict()['conv1.layer.bias'].copy_(torch.from_numpy(pretrained_dict['conv1/biases:0']))
		# model.state_dict()['conv1.layer.bias'].copy_(torch.zeros(64))
		# print(model.state_dict()['conv1.layer.weight'].data.numpy()[0,0,:,:])

		model.state_dict()['fire2.squeeze.layer.weight'].copy_(torch.from_numpy(pretrained_dict['fire2/squeeze3x3/kernels:0']))
		model.state_dict()['fire2.squeeze.layer.bias'].copy_(torch.from_numpy(pretrained_dict['fire2/squeeze3x3/biases:0']))
		model.state_dict()['fire2.expand1x1.weight'].copy_(torch.from_numpy(pretrained_dict['fire2/expand1x1/kernels:0']))
		model.state_dict()['fire2.expand1x1.bias'].copy_(torch.from_numpy(pretrained_dict['fire2/expand1x1/biases:0']))
		model.state_dict()['fire2.expand3x3.layer.weight'].copy_(torch.from_numpy(pretrained_dict['fire2/expand3x3/kernels:0']))
		model.state_dict()['fire2.expand3x3.layer.bias'].copy_(torch.from_numpy(pretrained_dict['fire2/expand3x3/biases:0']))

		model.state_dict()['fire3.squeeze.weight'].copy_(torch.from_numpy(pretrained_dict['fire3/squeeze1x1/kernels:0']))
		model.state_dict()['fire3.squeeze.bias'].copy_(torch.from_numpy(pretrained_dict['fire3/squeeze1x1/biases:0']))
		model.state_dict()['fire3.expand1x1.weight'].copy_(torch.from_numpy(pretrained_dict['fire3/expand1x1/kernels:0']))
		model.state_dict()['fire3.expand1x1.bias'].copy_(torch.from_numpy(pretrained_dict['fire3/expand1x1/biases:0']))
		model.state_dict()['fire3.expand3x3.layer.weight'].copy_(torch.from_numpy(pretrained_dict['fire3/expand3x3/kernels:0']))
		model.state_dict()['fire3.expand3x3.layer.bias'].copy_(torch.from_numpy(pretrained_dict['fire3/expand3x3/biases:0']))

		model.state_dict()['fire4.squeeze.layer.weight'].copy_(torch.from_numpy(pretrained_dict['fire4/squeeze3x3/kernels:0']))
		model.state_dict()['fire4.squeeze.layer.bias'].copy_(torch.from_numpy(pretrained_dict['fire4/squeeze3x3/biases:0']))
		model.state_dict()['fire4.expand1x1.weight'].copy_(torch.from_numpy(pretrained_dict['fire4/expand1x1/kernels:0']))
		model.state_dict()['fire4.expand1x1.bias'].copy_(torch.from_numpy(pretrained_dict['fire4/expand1x1/biases:0']))
		model.state_dict()['fire4.expand3x3.layer.weight'].copy_(torch.from_numpy(pretrained_dict['fire4/expand3x3/kernels:0']))
		model.state_dict()['fire4.expand3x3.layer.bias'].copy_(torch.from_numpy(pretrained_dict['fire4/expand3x3/biases:0']))

		model.state_dict()['fire5.squeeze.weight'].copy_(torch.from_numpy(pretrained_dict['fire5/squeeze1x1/kernels:0']))
		model.state_dict()['fire5.squeeze.bias'].copy_(torch.from_numpy(pretrained_dict['fire5/squeeze1x1/biases:0']))
		model.state_dict()['fire5.expand1x1.weight'].copy_(torch.from_numpy(pretrained_dict['fire5/expand1x1/kernels:0']))
		model.state_dict()['fire5.expand1x1.bias'].copy_(torch.from_numpy(pretrained_dict['fire5/expand1x1/biases:0']))
		model.state_dict()['fire5.expand3x3.layer.weight'].copy_(torch.from_numpy(pretrained_dict['fire5/expand3x3/kernels:0']))
		model.state_dict()['fire5.expand3x3.layer.bias'].copy_(torch.from_numpy(pretrained_dict['fire5/expand3x3/biases:0']))

		model.state_dict()['fire6.squeeze.layer.weight'].copy_(torch.from_numpy(pretrained_dict['fire6/squeeze3x3/kernels:0']))
		model.state_dict()['fire6.squeeze.layer.bias'].copy_(torch.from_numpy(pretrained_dict['fire6/squeeze3x3/biases:0']))
		model.state_dict()['fire6.expand1x1.weight'].copy_(torch.from_numpy(pretrained_dict['fire6/expand1x1/kernels:0']))
		model.state_dict()['fire6.expand1x1.bias'].copy_(torch.from_numpy(pretrained_dict['fire6/expand1x1/biases:0']))
		model.state_dict()['fire6.expand3x3.layer.weight'].copy_(torch.from_numpy(pretrained_dict['fire6/expand3x3/kernels:0']))
		model.state_dict()['fire6.expand3x3.layer.bias'].copy_(torch.from_numpy(pretrained_dict['fire6/expand3x3/biases:0']))

		model.state_dict()['fire7.squeeze.weight'].copy_(torch.from_numpy(pretrained_dict['fire7/squeeze1x1/kernels:0']))
		model.state_dict()['fire7.squeeze.bias'].copy_(torch.from_numpy(pretrained_dict['fire7/squeeze1x1/biases:0']))
		model.state_dict()['fire7.expand1x1.weight'].copy_(torch.from_numpy(pretrained_dict['fire7/expand1x1/kernels:0']))
		model.state_dict()['fire7.expand1x1.bias'].copy_(torch.from_numpy(pretrained_dict['fire7/expand1x1/biases:0']))
		model.state_dict()['fire7.expand3x3.layer.weight'].copy_(torch.from_numpy(pretrained_dict['fire7/expand3x3/kernels:0']))
		model.state_dict()['fire7.expand3x3.layer.bias'].copy_(torch.from_numpy(pretrained_dict['fire7/expand3x3/biases:0']))

		model.state_dict()['fire8.squeeze.layer.weight'].copy_(torch.from_numpy(pretrained_dict['fire8/squeeze3x3/kernels:0']))
		model.state_dict()['fire8.squeeze.layer.bias'].copy_(torch.from_numpy(pretrained_dict['fire8/squeeze3x3/biases:0']))
		model.state_dict()['fire8.expand1x1.weight'].copy_(torch.from_numpy(pretrained_dict['fire8/expand1x1/kernels:0']))
		model.state_dict()['fire8.expand1x1.bias'].copy_(torch.from_numpy(pretrained_dict['fire8/expand1x1/biases:0']))
		model.state_dict()['fire8.expand3x3.layer.weight'].copy_(torch.from_numpy(pretrained_dict['fire8/expand3x3/kernels:0']))
		model.state_dict()['fire8.expand3x3.layer.bias'].copy_(torch.from_numpy(pretrained_dict['fire8/expand3x3/biases:0']))

		model.state_dict()['fire9.squeeze.weight'].copy_(torch.from_numpy(pretrained_dict['fire9/squeeze1x1/kernels:0']))
		model.state_dict()['fire9.squeeze.bias'].copy_(torch.from_numpy(pretrained_dict['fire9/squeeze1x1/biases:0']))
		model.state_dict()['fire9.expand1x1.weight'].copy_(torch.from_numpy(pretrained_dict['fire9/expand1x1/kernels:0']))
		model.state_dict()['fire9.expand1x1.bias'].copy_(torch.from_numpy(pretrained_dict['fire9/expand1x1/biases:0']))
		model.state_dict()['fire9.expand3x3.layer.weight'].copy_(torch.from_numpy(pretrained_dict['fire9/expand3x3/kernels:0']))
		model.state_dict()['fire9.expand3x3.layer.bias'].copy_(torch.from_numpy(pretrained_dict['fire9/expand3x3/biases:0']))

		model.state_dict()['top_layer.weight'].copy_(torch.from_numpy(pretrained_dict['top_layer/kernels:0']))
		model.state_dict()['top_layer.bias'].copy_(torch.from_numpy(pretrained_dict['top_layer/biases:0']))
		model.state_dict()['lat4.weight'].copy_(torch.from_numpy(pretrained_dict['lat4/kernels:0']))
		model.state_dict()['lat4.bias'].copy_(torch.from_numpy(pretrained_dict['lat4/biases:0']))
		model.state_dict()['lat3.weight'].copy_(torch.from_numpy(pretrained_dict['lat3/kernels:0']))
		model.state_dict()['lat3.bias'].copy_(torch.from_numpy(pretrained_dict['lat3/biases:0']))

		model.state_dict()['predictor.layer.weight'].copy_(torch.from_numpy(pretrained_dict['kernel:0']))
		model.state_dict()['predictor.layer.bias'].copy_(torch.from_numpy(pretrained_dict['bias:0']))
	return model
