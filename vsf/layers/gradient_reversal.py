import torch as tr
from torch.autograd import Function
from torch.nn import Module


class RevGradFunc(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        if ctx.needs_input_grad[0]:
            lambda_ = ctx.saved_tensors[0]
            grad_input = -grad_output * lambda_
            return grad_input, None
        return None, None


class GradientReversal(Module):
    def __init__(self, lambda_=1., *args, **kwargs):
        """
        A gradient reversal layer.

        This layer has no parameters, and simply reverses the gradient
        in the backward pass.

        Args:
            lambda_: gradient scale, must > 0
        """
        assert 0 < lambda_, f'lambda must be positive, but found {lambda_}'
        super().__init__(*args, **kwargs)
        self.lambda_ = tr.tensor(lambda_, requires_grad=False)

    def forward(self, x):
        return RevGradFunc.apply(x, self.lambda_)


if __name__ == '__main__':
    tr.manual_seed(0)


    class Model(tr.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fc1 = tr.nn.Linear(128, 64)
            self.fc2 = tr.nn.Linear(64, 10)
            self.revgrad = GradientReversal()

        def forward(self, x):
            x = self.fc1(x)
            x = tr.relu(x)
            x = self.revgrad(x)

            x = self.fc2(x)
            return x


    data = tr.rand([1, 128])
    model = Model()
    print(next(model.parameters()))
    print('---------')
    loss_fn = tr.nn.MSELoss()
    optimizer = tr.optim.SGD(model.parameters(), lr=1e-2)

    # forward
    pred = model(data)
    y = tr.ones([1, 10])
    loss = loss_fn(pred, y)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(next(model.parameters()))
