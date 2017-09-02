import torch
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import utils.mnist_reader as mnist_reader
from utils.model import *
from utils.data import *
from utils.optim import exp_lr_scheduler

torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 50
LR = 0.001

cnn = LeNet()

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

#optimizer = torch.optim.SGD(cnn.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x).type(torch.FloatTensor)
        b_y = Variable(y).type(torch.LongTensor)

        output = cnn(b_x)[0]
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output, last_layer = cnn(val_x.type(torch.FloatTensor))
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == val_y.type(torch.LongTensor)) / float(val_x.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' %
                  loss.data[0], '| val accuracy: %.2f' % accuracy)
    optimizer = exp_lr_scheduler(optimizer, epoch, init_lr=LR, lr_decay_epoch=2)

test_output, _ = cnn(xtest.type(torch.FloatTensor))
pred_y = torch.max(test_output, 1)[1].data.squeeze()
accuracy = sum(pred_y == ytest.type(torch.LongTensor)) / float(xtest.size(0))
print('Test Accuracy is %.4f' % accuracy)