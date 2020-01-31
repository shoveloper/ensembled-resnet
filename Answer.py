import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(1021)
torch.cuda.manual_seed(1021)
np.random.seed(1021)

#########################################################################################################
# ----------------------------------------- SELECT DATA & TASK -----------------------------------------#
# data = {cifar, mnist}
# task = {1_imbalanced, 2_semisupervised, 3_noisy}

data = 'mnist'
task = '1_imbalanced'
# ----------------------------------------- END OF SELECTION -------------------------------------------#
#########################################################################################################

# check the data & task you selected
print('#'*50)
print('DATA: {}'.format(data))
print('TASK: {}'.format(task))
print('#'*50)


# Load the dataset
data_path = os.path.join('./data', data, task)
x_train = np.load(os.path.join(data_path, 'train_x.npy'))
y_train = np.load(os.path.join(data_path, 'train_y.npy'))
x_valid = np.load(os.path.join(data_path, 'valid_x.npy'))
y_valid = np.load(os.path.join(data_path, 'valid_y.npy'))
x_test = np.load(os.path.join(data_path, 'test_x.npy'))

# Check the shape of your data
print('x_train shape: {}'.format(x_train.shape))
print('y_train shape: {}'.format(y_train.shape))
print('x_valid shape: {}'.format(x_valid.shape))
print('y_valid shape: {}'.format(y_valid.shape))
print('x_test shape: {}'.format(x_test.shape))
print('#'*50)


#########################################################################################################
# ----------------------------------------- SAMPLE CODE ------------------------------------#
# No need to use the code below.
from datetime import datetime
import random
import easydict
import argparse
import torch.utils.data as utils
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from PIL import Image

random.seed(1021)
np.random.seed(1021)
torch.manual_seed(1021)
torch.cuda.manual_seed(1021)
torch.cuda.manual_seed_all(1021)
torch.backends.cudnn.deterministic=True

if data == 'cifar':
    default = easydict.EasyDict({
        'num_classes': 100,
        'num_models': 5,
        'batch_size': 128,
        'num_epochs': 200,
        'num_skip_epochs': 20,
        'num_workers': 4,
        'learning_rate': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'in_channels': 3,
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        'size': 32,
        'padding': 4,
        'degrees': 15,
        'critic': True,
        'trust': 1.0,
        'accept_threshold': 3.0,
        'over': [37, 73],
    })
else:
    default = easydict.EasyDict({
        'num_classes': 10,
        'num_models': 5,
        'batch_size': 128,
        'num_epochs': 200,
        'num_skip_epochs': 20,
        'num_workers': 4,
        'learning_rate': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'in_channels': 1,
        'mean': (0.5,),
        'std': (0.5,),
        'size': 28,
        'padding': 3,
        'degrees': 12,
        'critic': True,
        'trust': 1.0,
        'accept_threshold': 3.0,
        'over': [3, 7],
    })
if task == '1_imbalanced':
    default.critic = False
elif task == '2_semisupervised':
    default.trust = 10.0
    default.accept_threshold = 2.5
    default.over = []
else: # 3_noisy
    default.over = []

class LossLogger(object):
    def __init__(self):
        self.correct = 0
        self.total = 0
        self.losses = []

    def append(self, y_pred, y, loss):
        self.losses.append(loss)
        _, y_pred = y_pred.max(1)
        self.correct += y_pred.eq(y).sum().item()
        self.total += y_pred.shape[0]

    def item(self):
        return (sum(self.losses) / len(self.losses), self.correct / self.total) # loss, accuracy

class SoyuDataset(utils.Dataset):
    def __init__(self, images, targets, reals=None, in_channels=3, size=32, transform=None):
        self.transform = transform
        self.images = images
        self.targets = targets
        self.reals = reals
        self.in_channels = in_channels
        self.size = size

    def __getitem__(self, index):
        image = self.images[index]

        if self.in_channels == 1: # grayscale
            image = Image.fromarray(np.uint8(image.reshape(self.size, self.size) * 255), 'L')
        else: # rgb
            image = Image.fromarray(np.uint8(image.reshape(self.in_channels, self.size, self.size) * 255).transpose(1, 2, 0), 'RGB')

        if self.transform is not None:
            image = self.transform(image)

        if self.targets is not None: 
            target_one_hot = self.targets[index]
            target = np.argmax(target_one_hot)
            if self.reals is not None:
                real = self.reals[index]
                return image, target, target_one_hot, real
            return image, target, target_one_hot
        
        return image

    def __len__(self):
        return len(self.images)

"""resnet in pytorch (https://github.com/weiaicunzai/pytorch-cifar100)
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, in_channels=3, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output 

class SoyuPredictor(object):
    def __init__(self, args):
        global x_train
        global y_train
        global x_valid
        global y_valid
        global x_test

        # best record
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.best_pred = []

        # adaptive lr
        self.args = args
        self.lr = args.learning_rate

        # gen model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.models = []
        self.criterions = []
        self.optimizers = []
    
        for i in range(self.args.num_models):
            # resnet 18
            model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=self.args.in_channels, num_classes=self.args.num_classes) 

            model = model.to(self.device)
            if self.device == 'cuda':
                model = nn.DataParallel(model).to(self.device)
                cudnn.benchmark = True
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(),
                lr=self.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
            
            self.models.append(model)
            self.criterions.append(criterion)
            self.optimizers.append(optimizer)

        # transform
        train_transform = transforms.Compose([
            transforms.RandomCrop(self.args.size, padding=self.args.padding),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(self.args.degrees),
            transforms.ToTensor(),
            transforms.Normalize(self.args.mean, self.args.std)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.args.mean, self.args.std)
        ])

        # oversampling
        if len(self.args.over) > 0:
            new_x_train = x_train.copy()
            new_y_train = y_train.copy()
            for idx, y in enumerate(y_train):
                if y[self.args.over[0]] == 1 or y[self.args.over[1]] == 1:
                    for i in range(99):
                        np.insert(new_y_train, 0, y, axis=0)
                        np.insert(new_x_train, 0, x_train[idx], axis=0)
            y_train = new_y_train
            x_train = new_x_train

        # train set
        train_real = torch.from_numpy(np.loadtxt(os.path.join('./images', data, task, 'train_real.csv'))).type(torch.long) # TODO: delete
        train_dataset = SoyuDataset(x_train, y_train, train_real, in_channels=self.args.in_channels, size=self.args.size, transform=train_transform)
        #train_dataset = torchvision.datasets.CIFAR100('./', train=True, transform=train_transform, target_transform=None, download=True)
        self.train_loader = utils.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)

        # valid set
        valid_dataset = SoyuDataset(x_valid, y_valid, in_channels=self.args.in_channels, size=self.args.size, transform=train_transform)
        #valid_dataset = torchvision.datasets.CIFAR100('./', train=False, transform=train_transform, target_transform=None, download=True)
        self.valid_loader = utils.DataLoader(valid_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

        # test set
        test_dataset = SoyuDataset(x_test, None, in_channels=self.args.in_channels, size=self.args.size, transform=test_transform)
        self.test_loader = utils.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
        self.y_test = torch.from_numpy(np.loadtxt(os.path.join('./images', data, task, 'test_real.csv'))).type(torch.long).to(self.device) # TODO: delete

    def adjust_learning_rate(self, epoch):
        if epoch < 60:
            lr = self.args.learning_rate
        elif epoch < 120:
            lr = self.args.learning_rate * 0.2
        elif epoch < 160:
            lr = self.args.learning_rate * 0.04
        else:
            lr = self.args.learning_rate * 0.008
        if self.lr == lr:
            return
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        self.lr = lr

    def check_k_fold_condition(self, phase, batch_idx, model_idx):
        return (phase == 'train' and batch_idx % self.args.num_models != model_idx)\
            or (phase == 'validate' and batch_idx % self.args.num_models == model_idx)\
            or (phase == 'test')

    def forward(self, phase, batch_idx, x, y=None, y_one_hot=None, real=None, with_critic=False):
        x = x.to(self.device)
        if y is not None:
            y = y.to(self.device)
            y_one_hot = y_one_hot.to(self.device).type(torch.float)

        y_pred_one_hot = [None] * self.args.num_models
        normalized_y_pred_one_hot_sum = torch.zeros((x.shape[0], self.args.num_classes)).to(self.device)
        for model_idx, model in enumerate(self.models): # ensemble
            if self.check_k_fold_condition(phase, batch_idx, model_idx):
                y_pred_one_hot[model_idx] = model(x)

                # normalize
                min_y_pred = torch.min(y_pred_one_hot[model_idx])
                range_y_pred = torch.max(y_pred_one_hot[model_idx]) - min_y_pred
                if range_y_pred > 0:
                    normalised_y_pred_one_hot = (y_pred_one_hot[model_idx] - min_y_pred) / range_y_pred
                else:
                    normalised_y_pred_one_hot = torch.zeros(y_pred_one_hot[model_idx].size())
                normalized_y_pred_one_hot_sum += normalised_y_pred_one_hot

        if with_critic: # y is mandatory for critic
            _, tmp_y_pred = normalized_y_pred_one_hot_sum.max(1) # TODO: delete
            correct_y_pred = tmp_y_pred.eq(real.to(self.device)).sum().item() # TODO: delete

            # add trusted label
            weighted_one_hot = y_one_hot * self.args.trust
            normalized_y_pred_one_hot_sum += weighted_one_hot

            # filter out
            max_y_pred, tmp_y_pred = normalized_y_pred_one_hot_sum.max(1)
            accepted_idx = max_y_pred >= self.args.accept_threshold
            normalized_y_pred_one_hot_sum = normalized_y_pred_one_hot_sum[accepted_idx]
            y = y[accepted_idx]
            for model_idx, model in enumerate(self.models): # ensemble
                if self.check_k_fold_condition(phase, batch_idx, model_idx):
                    y_pred_one_hot[model_idx] = y_pred_one_hot[model_idx][accepted_idx]
            real = real[accepted_idx] # TODO: delete

            _, tmp_y_accepted = normalized_y_pred_one_hot_sum.max(1) # TODO: delete
            correct_y_accepted = tmp_y_accepted.eq(real.to(self.device)).sum().item() # TODO: delete     
            correct_y = y.eq(real.to(self.device)).sum().item() # TODO: delete
            total = y.shape[0] # TODO: delete
            print(correct_y_accepted, correct_y_pred, correct_y, total) # TODO: delete

        _, y_pred = normalized_y_pred_one_hot_sum.max(1)
        if y is not None:
            return y_pred, y, y_pred_one_hot
        return y_pred

    def evaluate(self, y, y_pred_one_hot, logger):
        loss = [None] * self.args.num_models
        for model_idx, model in enumerate(self.models): # ensemble
            if y_pred_one_hot[model_idx] is not None:
                loss[model_idx] = self.criterions[model_idx](y_pred_one_hot[model_idx], y)
                if logger is not None:
                    logger.append(y_pred_one_hot[model_idx], y, loss[model_idx].item())
        return loss

    def backward(self, loss):
        for model_idx, model in enumerate(self.models): # ensemble
            if loss[model_idx] is not None:
                self.optimizers[model_idx].zero_grad()
                loss[model_idx].backward()
                self.optimizers[model_idx].step()

    def train(self, loader, with_critic=False):
        logger = LossLogger()
        for model in self.models:
            model.train()
        for batch_idx, (x, y, y_one_hot, *extra) in enumerate(loader):
            if len(extra) > 0:
                real = extra[0]
            else:
                real = None
            _, y, y_pred_one_hot = self.forward('train', batch_idx, x, y, y_one_hot, real=real, with_critic=with_critic)
            loss = self.evaluate(y, y_pred_one_hot, logger)
            self.backward(loss)
        return logger.item()

    def validate(self, loader):
        logger = LossLogger() # valid loss / accuracy
        with torch.no_grad():
            for model in self.models:
                model.eval()
            for batch_idx, (x, y, y_one_hot) in enumerate(loader):
                _, y, y_pred_one_hot = self.forward('validate', batch_idx, x, y, y_one_hot, with_critic=False)
                self.evaluate(y, y_pred_one_hot, logger)
        return logger.item()

    def test(self):
        with torch.no_grad():
            for model in self.models:
                model.eval()
            y_preds = torch.empty((0,)).to(self.device).type(torch.long)
            for batch_idx, (x) in enumerate(self.test_loader):
                y_pred = self.forward('test', batch_idx, x, with_critic=False)
                y_preds = torch.cat((y_preds, y_pred))
        return y_preds

    def get_best(self):
        print('Best valid accuracy is {:.4f} at Epoch {}'.format(self.best_accuracy, self.best_epoch + 1))
        return self.best_pred

    def get_test_accuracy(self, epoch, test_pred): # TODO: delete
        test_pred = test_pred.to(self.device)
        total = test_pred.shape[0]
        correct = test_pred.eq(self.y_test.to(self.device)).sum().item()
        accuracy = correct / total

        # save
        save_path = os.path.join('./output', data, task)
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, 'test_pred.{:03d}.{:.4f}.npy'.format(epoch + 1, accuracy)), test_pred.clone().cpu().numpy())
        return accuracy

    def run(self):
        for epoch in range(self.args.num_epochs):
            self.adjust_learning_rate(epoch)

            if epoch + 1 > self.args.num_skip_epochs:
                self.train(self.train_loader, with_critic=self.args.critic)
            train_loss, train_accuracy = self.train(self.valid_loader)
            valid_loss, valid_accuracy = self.validate(self.valid_loader)

            test_pred = self.test()
            test_accuracy = self.get_test_accuracy(epoch, test_pred) # TODO: delete

            if valid_accuracy > self.best_accuracy:
                self.best_accuracy = valid_accuracy
                self.best_epoch = epoch
                self.best_pred = test_pred.cpu().numpy()

            print('{} [Epoch {}] LR: {:.4f}\tLoss (Train/Valid): {:.4f} / {:.4f}\tAccuracy (Train/Valid/Test): {:.4f} / {:.4f} / {:.4f}'
                .format(datetime.now(), epoch + 1, self.lr, train_loss, valid_loss, train_accuracy, valid_accuracy, test_accuracy))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num-classes', default=default.num_classes, help='', type=int)
    parser.add_argument('--num-models', default=default.num_models, help='', type=int)
    parser.add_argument('--batch-size', default=default.batch_size, help='', type=int)
    parser.add_argument('--num-epochs', default=default.num_epochs, help='', type=int)
    parser.add_argument('--num-skip-epochs', default=default.num_skip_epochs, help='', type=int)
    parser.add_argument('--num-workers', default=default.num_workers, help='', type=int)
    parser.add_argument('--learning-rate', default=default.learning_rate, help='', type=float)
    parser.add_argument('--momentum', default=default.momentum, help='', type=float)
    parser.add_argument('--weight-decay', default=default.weight_decay, help='', type=float)
    parser.add_argument('--in-channels', default=default.in_channels, help='', type=int)
    parser.add_argument('--mean', default=default.mean, help='', type=float, nargs='+')
    parser.add_argument('--std', default=default.std, help='', type=float, nargs='+')
    parser.add_argument('--size', default=default.size, help='', type=int)
    parser.add_argument('--padding', default=default.padding, help='', type=int)
    parser.add_argument('--degrees', default=default.degrees, help='', type=int)
    parser.add_argument('--critic', default=default.critic, help='', type=(lambda x: bool(int(x))))
    parser.add_argument('--trust', default=default.trust, help='', type=float)
    parser.add_argument('--accept-threshold', default=default.accept_threshold, help='', type=float)
    parser.add_argument('--over', default=default.over, help='', type=int, nargs='*')
    args = parser.parse_args()
    print(args)

    predictor = SoyuPredictor(args)
    predictor.run()

    # predict answer
    test_pred = predictor.get_best()
else:
    test_pred = []

os.makedirs(os.path.join('./output', data, task), exist_ok=True)

# ----------------------------------------- END OF CODE-------------------------------------------#
#########################################################################################################

# You need to save and submit the 'test_pred.npy' under the output directory.
save_path = os.path.join('./output', data, task)
np.save(os.path.join(save_path, 'test_pred.npy'), test_pred)
