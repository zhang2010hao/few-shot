import torch
import tqdm
from torch.autograd import Variable
from src.prototypical_loss import prototypical_loss as loss_fn


class OneShotBuilder:

    def __init__(self, data, classes_per_set, samples_per_class, lr=1e-3, lr_decay=1e-6, wd=1e-4, cuda=True, ):
        """
        Initializes an OneShotBuilder object. The OneShotBuilder object takes care of setting up our experiment
        and provides helper functions such as run_training_epoch and run_validation_epoch to simplify out training
        and evaluation procedures.
        :param data: A data provider class
        """
        self.data = data
        self.lr = lr
        self.current_lr = lr
        self.lr_decay = lr_decay
        self.wd = wd
        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class
        self.total_train_iter = 0
        self.isCudaAvailable = cuda

    def run_training_epoch_prototypical(self, total_train_batches, model, optimizer):
        """
        Runs one training epoch
        :param total_train_batches: Number of batches to train on
        :return: mean_training_categorical_crossentropy_loss and mean_training_accuracy
        """
        # Create the optimizer
        total_loss = 0.0
        total_accuracy = 0.0
        model.train()
        with tqdm.tqdm(total=total_train_batches) as pbar:
            for i in range(total_train_batches):  # train epoch
                optimizer.zero_grad()

                x_support_set, y_support_set, x_target, y_target = \
                    self.data.get_batch(str_type='train')

                x_support_set = Variable(torch.from_numpy(x_support_set)).long()
                y_support_set = Variable(torch.from_numpy(y_support_set), requires_grad=False).long()
                x_target = Variable(torch.from_numpy(x_target)).long()
                y_target = Variable(torch.from_numpy(y_target), requires_grad=False).long()

                x = torch.cat((x_support_set, x_target), dim=0)
                y = torch.cat((y_support_set, y_target), dim=0)
                # Reshape channels
                if self.isCudaAvailable:
                    model_output = model(x.cuda())
                else:
                    model_output = model(x)

                loss, acc = loss_fn(model_output, target=y,
                                    n_support=5)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_accuracy += acc.item()
                pbar.update(1)
            avg_loss = total_loss / total_train_batches
            avg_acc = total_accuracy / total_train_batches
            print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        return avg_loss, avg_acc

    def run_validation_epoch_prototypical(self, total_val_batches, model):
        """
        Runs one validation epoch
        :param total_val_batches: Number of batches to train on
        :return: mean_validation_categorical_crossentropy_loss and mean_validation_accuracy
        """

        total_loss = 0.0
        total_accuracy = 0.0
        model.eval()
        with tqdm.tqdm(total=total_val_batches) as pbar:
            for i in range(total_val_batches):  # train epoch
                x_support_set, y_support_set, x_target, y_target = \
                    self.data.get_batch(str_type='val')

                x_support_set = Variable(torch.from_numpy(x_support_set)).long()
                y_support_set = Variable(torch.from_numpy(y_support_set), requires_grad=False).long()
                x_target = Variable(torch.from_numpy(x_target)).long()
                y_target = Variable(torch.from_numpy(y_target), requires_grad=False).long()

                x = torch.cat((x_support_set, x_target), dim=0)
                y = torch.cat((y_support_set, y_target), dim=0)
                # Reshape channels
                if self.isCudaAvailable:
                    model_output = model(x.cuda())
                else:
                    model_output = model(x)

                loss, acc = loss_fn(model_output, target=y,
                                    n_support=5)
                total_loss += loss.item()
                total_accuracy += acc.item()
                pbar.update(1)
            avg_loss = total_loss / total_val_batches
            avg_acc = total_accuracy / total_val_batches
            print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        return avg_loss, avg_acc

    def run_testing_epoch_prototypical(self, total_test_batches, model):
        """
        Runs one testing epoch
        :param total_test_batches: Number of batches to train on
        :param sess: Session object
        :return: mean_testing_categorical_crossentropy_loss and mean_testing_accuracy
        """
        total_loss = 0.0
        total_accuracy = 0.0
        model.eval()
        with tqdm.tqdm(total=total_test_batches) as pbar:
            for i in range(total_test_batches):  # train epoch
                x_support_set, y_support_set, x_target, y_target = \
                    self.data.get_batch(str_type='test')

                x_support_set = Variable(torch.from_numpy(x_support_set)).long()
                y_support_set = Variable(torch.from_numpy(y_support_set), requires_grad=False).long()
                x_target = Variable(torch.from_numpy(x_target)).long()
                y_target = Variable(torch.from_numpy(y_target), requires_grad=False).long()

                x = torch.cat((x_support_set, x_target), dim=0)
                y = torch.cat((y_support_set, y_target), dim=0)
                # Reshape channels
                if self.isCudaAvailable:
                    model_output = model(x.cuda())
                else:
                    model_output = model(x)

                loss, acc = loss_fn(model_output, target=y,
                                    n_support=5)
                total_loss += loss.item()
                total_accuracy += acc.item()
                pbar.update(1)
            avg_loss = total_loss / total_test_batches
            avg_acc = total_accuracy / total_test_batches
            print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        return avg_loss, avg_acc

    def run_training_epoch_match(self, total_train_batches, matchingNet, optimizer):
        """
        Runs one training epoch
        :param total_train_batches: Number of batches to train on
        :return: mean_training_categorical_crossentropy_loss and mean_training_accuracy
        """
        total_c_loss = 0.
        total_accuracy = 0.
        # Create the optimizer
        # optimizer = self.__create_optimizer(self.matchingNet, self.lr)
        matchingNet.train()
        with tqdm.tqdm(total=total_train_batches) as pbar:
            for i in range(total_train_batches):  # train epoch
                x_support_set, y_support_set, x_target, y_target = \
                    self.data.get_batch(str_type='train')

                x_support_set = Variable(torch.from_numpy(x_support_set)).long()
                y_support_set = Variable(torch.from_numpy(y_support_set), requires_grad=False).long()
                x_target = Variable(torch.from_numpy(x_target)).long()
                y_target = Variable(torch.from_numpy(y_target), requires_grad=False).long()

                # y_support_set: Add extra dimension for the one_hot
                y_support_set = torch.unsqueeze(y_support_set, 1)
                batch_size = y_support_set.size()[0]
                y_support_set_one_hot = torch.FloatTensor(batch_size, self.classes_per_set).zero_()
                y_support_set_one_hot.scatter_(1, y_support_set.data, 1)
                y_support_set_one_hot = Variable(y_support_set_one_hot)

                # Reshape channels
                if self.isCudaAvailable:
                    acc, c_loss_value = matchingNet(x_support_set.cuda(), y_support_set_one_hot.cuda(),
                                                    x_target.cuda(), y_target.cuda())
                else:
                    acc, c_loss_value = matchingNet(x_support_set, y_support_set_one_hot,
                                                    x_target, y_target)

                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable weights
                # of the model)
                optimizer.zero_grad()

                # Backward pass: compute gradient of the loss with respect to model parameters
                c_loss_value.backward()

                # Calling the step function on an Optimizer makes an update to its parameters
                optimizer.step()

                # update the optimizer learning rate
                self.__adjust_learning_rate(optimizer)

                iter_out = "tr_loss: {}, tr_accuracy: {}".format(c_loss_value.data[0], acc.data[0])
                pbar.set_description(iter_out)

                pbar.update(1)
                total_c_loss += c_loss_value.data[0]
                total_accuracy += acc.data[0]

                self.total_train_iter += 1
                if self.total_train_iter % 2000 == 0:
                    self.lr /= 2
                    print("change learning rate", self.lr)

        total_c_loss = total_c_loss / total_train_batches
        total_accuracy = total_accuracy / total_train_batches
        return total_c_loss, total_accuracy

    def run_validation_epoch_match(self, total_val_batches, matchingNet):
        """
        Runs one validation epoch
        :param total_val_batches: Number of batches to train on
        :return: mean_validation_categorical_crossentropy_loss and mean_validation_accuracy
        """
        total_val_c_loss = 0.
        total_val_accuracy = 0.
        matchingNet.eval()
        with tqdm.tqdm(total=total_val_batches) as pbar:
            for i in range(total_val_batches):  # validation epoch
                x_support_set, y_support_set, x_target, y_target = \
                    self.data.get_batch(str_type='val')

                x_support_set = Variable(torch.from_numpy(x_support_set)).long()
                y_support_set = Variable(torch.from_numpy(y_support_set), requires_grad=False).long()
                x_target = Variable(torch.from_numpy(x_target)).long()
                y_target = Variable(torch.from_numpy(y_target), requires_grad=False).long()

                # y_support_set: Add extra dimension for the one_hot
                y_support_set = torch.unsqueeze(y_support_set, 1)
                batch_size = y_support_set.size()[0]
                y_support_set_one_hot = torch.FloatTensor(batch_size, self.classes_per_set).zero_()
                y_support_set_one_hot.scatter_(1, y_support_set.data, 1)
                y_support_set_one_hot = Variable(y_support_set_one_hot)

                # Reshape channels
                if self.isCudaAvailable:
                    acc, c_loss_value = matchingNet(x_support_set.cuda(), y_support_set_one_hot.cuda(),
                                                    x_target.cuda(), y_target.cuda())
                else:
                    acc, c_loss_value = matchingNet(x_support_set, y_support_set_one_hot,
                                                    x_target, y_target)

                iter_out = "val_loss: {}, val_accuracy: {}".format(c_loss_value.data[0], acc.data[0])
                pbar.set_description(iter_out)
                pbar.update(1)

                total_val_c_loss += c_loss_value.data[0]
                total_val_accuracy += acc.data[0]

        total_val_c_loss = total_val_c_loss / total_val_batches
        total_val_accuracy = total_val_accuracy / total_val_batches

        return total_val_c_loss, total_val_accuracy

    def run_testing_epoch_match(self, total_test_batches, matchingNet):
        """
        Runs one testing epoch
        :param total_test_batches: Number of batches to train on
        :param sess: Session object
        :return: mean_testing_categorical_crossentropy_loss and mean_testing_accuracy
        """
        total_test_c_loss = 0.
        total_test_accuracy = 0.
        with tqdm.tqdm(total=total_test_batches) as pbar:
            for i in range(total_test_batches):
                x_support_set, y_support_set, x_target, y_target = \
                    self.data.get_batch(str_type='test')

                x_support_set = Variable(torch.from_numpy(x_support_set)).long()
                y_support_set = Variable(torch.from_numpy(y_support_set), requires_grad=False).long()
                x_target = Variable(torch.from_numpy(x_target)).long()
                y_target = Variable(torch.from_numpy(y_target), requires_grad=False).long()

                # y_support_set: Add extra dimension for the one_hot
                y_support_set = torch.unsqueeze(y_support_set, 1)
                batch_size = y_support_set.size()[0]
                y_support_set_one_hot = torch.FloatTensor(batch_size, self.classes_per_set).zero_()
                y_support_set_one_hot.scatter_(1, y_support_set.data, 1)
                y_support_set_one_hot = Variable(y_support_set_one_hot)

                # Reshape channels
                if self.isCudaAvailable:
                    acc, c_loss_value = matchingNet(x_support_set.cuda(), y_support_set_one_hot.cuda(),
                                                    x_target.cuda(), y_target.cuda())
                else:
                    acc, c_loss_value = matchingNet(x_support_set, y_support_set_one_hot,
                                                    x_target, y_target)

                iter_out = "test_loss: {}, test_accuracy: {}".format(c_loss_value.data[0], acc.data[0])
                pbar.set_description(iter_out)
                pbar.update(1)

                total_test_c_loss += c_loss_value.data[0]
                total_test_accuracy += acc.data[0]
            total_test_c_loss = total_test_c_loss / total_test_batches
            total_test_accuracy = total_test_accuracy / total_test_batches
        return total_test_c_loss, total_test_accuracy

    def __adjust_learning_rate(self, optimizer):
        """Updates the learning rate given the learning rate decay.
        The routine has been implemented according to the original Lua SGD optimizer
        """
        for group in optimizer.param_groups:
            if 'step' not in group:
                group['step'] = 0
            group['step'] += 1

            group['lr'] = self.lr / (1 + group['step'] * self.lr_decay)

    def create_optimizer(self, optm, model, new_lr):
        # setup optimizer
        if optm == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=new_lr,
                                        momentum=0.9, dampening=0.9,
                                        weight_decay=self.wd)
        elif optm == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=new_lr,
                                         weight_decay=self.wd)
        else:
            raise Exception('Not supported optimizer: {0}'.format(optm))
        return optimizer
