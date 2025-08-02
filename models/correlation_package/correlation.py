import correlation_cuda
import torch
from torch.autograd import Function
from torch.nn.modules.module import Module

class CorrelationFunction(Function):

    pad_size = 0
    kernel_size = 0
    max_displacement = 0
    stride1 = 1
    stride2 = 2
    corr_multiply = 1

    @staticmethod
    def forward(ctx, input1, input2):
        ctx.save_for_backward(input1, input2)

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()

            correlation_cuda.forward(input1, input2, rbot1, rbot2, output, 
                CorrelationFunction.pad_size, CorrelationFunction.kernel_size, 
                CorrelationFunction.max_displacement, CorrelationFunction.stride1, 
                CorrelationFunction.stride2, CorrelationFunction.corr_multiply)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()

            grad_input1 = input1.new()
            grad_input2 = input2.new()

            correlation_cuda.backward(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2,
                CorrelationFunction.pad_size, CorrelationFunction.kernel_size, 
                CorrelationFunction.max_displacement, CorrelationFunction.stride1, 
                CorrelationFunction.stride2, CorrelationFunction.corr_multiply)

        return grad_input1, grad_input2

class Correlation(Module):
    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        CorrelationFunction.pad_size = pad_size
        CorrelationFunction.kernel_size = kernel_size
        CorrelationFunction.max_displacement = max_displacement
        CorrelationFunction.stride1 = stride1
        CorrelationFunction.stride2 = stride2
        CorrelationFunction.corr_multiply = corr_multiply

    def forward(self, input1, input2):

        result = CorrelationFunction.apply(input1, input2)

        return result
