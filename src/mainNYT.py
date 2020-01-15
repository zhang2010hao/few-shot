from src import nytNShot
from src.option import Options
from src.OneShotBuilderNYT import OneShotBuilder
from models.protonet import ProtoNet
import torch
import os
from src import config
from models.MatchingNetwork_NYT import MatchingNetwork


def load_model(path,
               model):
    if os.path.exists(path + '/model.pkl'):
        model.load_state_dict(torch.load(os.path.join(path, 'model.pkl')))
    return model


def save_model(path,
               model):
    torch.save(model.state_dict(), os.path.join(path, 'model.pkl'))


def set_all_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_device(n_gpus):
    """
    setup GPU device if available, move model into configured device
    # n_gpus，小于等于0表示使用cpu，大于0则使用range生成list
    # 如果输入的是一个list，则默认使用list[0]作为controller
     """
    if torch.cuda.is_available():
        gpu_list = n_gpus
        list_ids = n_gpus
        if isinstance(n_gpus, int):
            if n_gpus <= 0:
                device = 'cpu'
                list_ids = []

                return device, list_ids
            else:
                gpu_list = range(n_gpus)

        n_gpu = torch.cuda.device_count()
        if len(gpu_list) > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            list_ids = []
        if len(gpu_list) > n_gpu:
            msg = "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                gpu_list, range(n_gpu))
            print(msg)
            list_ids = range(n_gpu)
        device = torch.device('cuda:%d' % list_ids[0] if len(list_ids) > 0 else 'cpu')
    else:
        device = 'cpu'
        list_ids = []

    return device, list_ids


def set_model_device(model,
                     device,
                     device_ids):
    # 设置模型在GPU上还是CPU上
    if len(device_ids) > 1:
        print("current {} GPUs".format(len(device_ids)))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    if len(device_ids) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_ids[0])
    model = model.to(device)
    return model, device, len(device_ids)


'''
:param batch_size: Experiment batch_size
:param classes_per_set: Integer indicating the number of classes per set
:param samples_per_class: Integer indicating samples per class
        e.g. For a 20-way, 1-shot learning task, use classes_per_set=20 and samples_per_class=1
             For a 5-way, 10-shot learning task, use classes_per_set=5 and samples_per_class=10
'''


def fit_prototypical_net(args):
    data = nytNShot.nytNShotDataset(dataroot=args.dataroot,
                                    classes_per_set=config.classes_per_set,
                                    samples_per_class=config.samples_per_class,
                                    n_test_samples=config.n_test_samples)

    model = ProtoNet(n_word=data.word_num)

    device, device_ids = prepare_device(config.gpus)
    model, device, n_gpu = set_model_device(model,
                                            device,
                                            device_ids)

    obj_oneShotBuilder = OneShotBuilder(data, config.classes_per_set, config.samples_per_class)
    optimizer = obj_oneShotBuilder.create_optimizer('adam', model, 1e-3)
    best_val = 0.

    for e in range(0, config.total_epochs):
        total_c_loss, total_accuracy = obj_oneShotBuilder.run_training_epoch_prototypical(config.total_train_batches,
                                                                                          model,
                                                                                          optimizer)
        print("Epoch {}: train_loss: {}, train_accuracy: {}".format(e, total_c_loss, total_accuracy))

        total_val_c_loss, total_val_accuracy = obj_oneShotBuilder.run_validation_epoch_prototypical(
            config.total_val_batches, model)
        print("Epoch {}: val_loss: {}, val_accuracy: {}".format(e, total_val_c_loss, total_val_accuracy))

        if total_val_accuracy >= best_val:  # if new best val accuracy -> produce test statistics
            best_val = total_val_accuracy
            total_test_c_loss, total_test_accuracy = obj_oneShotBuilder.run_testing_epoch_prototypical(
                config.total_test_batches,
                model)
            print("Epoch {}: test_loss: {}, test_accuracy: {}".format(e, total_test_c_loss, total_test_accuracy))

            save_model(config.model_path,
                       model)
        else:
            total_test_c_loss = -1
            total_test_accuracy = -1


def fit_match_net(args):
    data = nytNShot.nytNShotDataset(dataroot=args.dataroot,
                                    classes_per_set=config.classes_per_set,
                                    samples_per_class=config.samples_per_class,
                                    n_test_samples=config.n_test_samples)

    model = MatchingNetwork(n_word=data.word_num, keep_prob=1, num_channels=config.channels,
                            fce=config.fce, num_classes_per_set=config.classes_per_set,
                            num_samples_per_class=config.samples_per_class,
                            nClasses=0, sent_len=data.max_len)

    device, device_ids = prepare_device(config.gpus)
    model, device, n_gpu = set_model_device(model,
                                            device,
                                            device_ids)

    obj_oneShotBuilder = OneShotBuilder(data, config.classes_per_set, config.samples_per_class)
    optimizer = obj_oneShotBuilder.create_optimizer('adam', model, 2e-3)
    best_val = 0.

    for e in range(0, config.total_epochs):
        total_c_loss, total_accuracy = obj_oneShotBuilder.run_training_epoch_match(config.total_train_batches,
                                                                                   model,
                                                                                   optimizer)
        print("Epoch {}: train_loss: {}, train_accuracy: {}".format(e, total_c_loss, total_accuracy))

        total_val_c_loss, total_val_accuracy = obj_oneShotBuilder.run_validation_epoch_match(
            config.total_val_batches, model)
        print("Epoch {}: val_loss: {}, val_accuracy: {}".format(e, total_val_c_loss, total_val_accuracy))

        if total_val_accuracy >= best_val:  # if new best val accuracy -> produce test statistics
            best_val = total_val_accuracy
            total_test_c_loss, total_test_accuracy = obj_oneShotBuilder.run_testing_epoch_match(
                config.total_test_batches,
                model)
            print("Epoch {}: test_loss: {}, test_accuracy: {}".format(e, total_test_c_loss, total_test_accuracy))

            save_model(config.model_path,
                       model)
        else:
            total_test_c_loss = -1
            total_test_accuracy = -1


if __name__ == '__main__':
    # Parse other options
    args = Options().parse()

    # train prototypical network
    fit_prototypical_net(args)

    # train match network
    # fit_match_net(args)
