import torch
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import utils.mnist_reader as mnist_reader
from utils.model import *
from utils.data import *
from utils.optim import exp_lr_scheduler
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from utils.resnet import *
#from utils.logger import Logger
from visdom import Visdom
import numpy as np

torch.manual_seed(1)    # reproducible
viz = Visdom()

# Hyper Parameters
EPOCH = 20
BATCH_SIZE = 16
LR = 0.001

cnn = resnet34()
print(cnn)
cnn.cuda()

X_train, y_train = mnist_reader.load_mnist('./fashion-mnist-data', kind='train')
X_test, y_test = mnist_reader.load_mnist('./fashion-mnist-data', kind='t10k')

X_train = X_train.reshape((-1, 1, 28 , 28))
X_test = X_test.reshape((-1, 1, 28, 28))

X_train = numpy2torch(X_train)
y_train = numpy2torch(y_train)
X_test = numpy2torch(X_test)
y_test = numpy2torch(y_test)

train_data = Data.TensorDataset(data_tensor=X_train[:55000], target_tensor=y_train[:55000])
val_data= Data.TensorDataset(data_tensor=X_train[55000:], target_tensor=y_train[55000:])
Xtest = Data.TensorDataset(data_tensor=X_test, target_tensor=y_test)

train_loader = Data.DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
val_x = Variable(val_data.data_tensor, volatile=True)
val_y = val_data.target_tensor

xtest = Variable(Xtest.data_tensor, volatile=True).type(torch.FloatTensor)
ytest = Xtest.target_tensor.type(torch.LongTensor)


#win = viz.line(Y = np.empty(1))
#win_acc = viz.line(Y = np.empty(1))
win = viz.line(Y = np.linspace(0,3,1))
win_acc = viz.line(Y = np.linspace(0,1.3,1))

#optimizer = torch.optim.SGD(cnn.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

count = 0

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = to_cuda(Variable(x).type(torch.FloatTensor))
        b_y = to_cuda(Variable(y).type(torch.LongTensor))

        output = cnn(b_x)[0]
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, argmax = torch.max(output, 1)
        train_accuracy = (b_y == argmax.squeeze()).float().mean()
        if count % 20 == 0:
            viz.line(X = np.array((count,)), Y = np.array((loss.data[0],)), win=win, update='append')
            viz.line(X = np.array((count,)), Y = np.array((train_accuracy.data[0],)), win=win_acc, update='append')
        count = count + 1
        '''if (step+1) % 100 == 0:
            info = {
                    'loss': loss.data[0],
                    'accuracy': train_accuracy.data[0]
                    }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, step+1)
            # (2) Log values and gradients of the parameters (histogram)
            for tag, value in cnn.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, to_np(value), step+1)
                logger.histo_summary(tag+'/grad', to_np(value.grad), step+1)

            # (3) Log the images
            info = {
                    'images': to_np(b_x.view(BATCH_SIZE, -1, 28, 28)[:10])
                    }

            for tag, images in info.items():
                logger.image_summary(tag, b_x, step+1)
        '''
        if step % 50 == 0:
            test_output, last_layer = cnn(to_cuda(val_x.type(torch.FloatTensor)))
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y.cpu() == val_y.type(torch.LongTensor)) / float(val_x.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' %
                    loss.data[0], '| train accuracy: %.2f' % train_accuracy.data[0], '| val accuracy: %.2f' % accuracy)
    optimizer = exp_lr_scheduler(optimizer, epoch, init_lr=LR, lr_decay_epoch=10)

test_output, _ = cnn(to_cuda(xtest.type(torch.FloatTensor)))
pred_y = torch.max(test_output, 1)[1].data.squeeze()
accuracy = sum(pred_y.cpu() == ytest.type(torch.LongTensor)) / float(xtest.size(0))
print('Test Accuracy is %.4f' % accuracy)
