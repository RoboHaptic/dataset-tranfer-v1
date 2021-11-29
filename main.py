import numpy as np
import torch.nn as nn
import torch.optim
from torch.optim import lr_scheduler
from model import EEGNet
from torch.utils.data import DataLoader
from mydataset import TrainingSet, ValidSet, TestSet
from args import args
import wandb
from utils import acc, data_prefetcher, to_tensor, visualizer
import time


def config():

    # CHANNELS, SAMPLES, NUM_SUBJECTS
    if args.dataset == 'iv_2a':
        args.lr, args.batch_size = 1e-5, 64
        return 21, 1000, 9
    elif args.dataset == 'sub54':
        args.lr, args.batch_size = 1e-5, 128
        return 21, 1000, 54

    wandb.config = {
      "learning_rate": args.lr,
      "epochs": args.epoch,
      "batch_size": args.batch_size}


def train(test_sub=None):

    model = EEGNet(n_classes=2, n_subjects=NUM_SUBJECTS, channels=CHANNELS, samples=SAMPLES, dropoutRate=0.25)
    print(model)

    if args.load != '':
        model.load_state_dict(torch.load(args.load))

    model.to('cuda:{}'.format(args.gpu))

    loss_ce = nn.CrossEntropyLoss().to('cuda:{}'.format(args.gpu), non_blocking=True)
    # loss_sub = loss_ce

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.set_weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=int(args.epoch / 2), gamma=0.1)

    if not args.full_train:
        valid_subject = int(np.random.rand() * NUM_SUBJECTS)
        while valid_subject == test_sub:
            valid_subject = int(np.random.rand() * NUM_SUBJECTS)

        valid_set = ValidSet(dataset=args.dataset, valid_sub=valid_subject)
        valid_loader = DataLoader(dataset=valid_set, batch_size=100)

        training_set = TrainingSet(dataset=args.dataset, test_sub=test_sub, valid_sub=valid_subject)
    else:
        training_set = TrainingSet(dataset=args.dataset)

    train_loader = DataLoader(dataset=training_set, batch_size=args.batch_size, shuffle=True, num_workers=1,
                              prefetch_factor=80, pin_memory=True)

    for i in range(args.epoch+1):

        start = time.time()

        optimizer.zero_grad()
        loss_acc, train_pred_cls_acc, train_pred_sub_acc, valid_pred_cls_acc, valid_pred_sub_acc, = 0, 0, 0, 0, 0

        # Training
        model.train(mode=True)

        for train_batch, (sub, train_data, train_label) in enumerate(train_loader):

            # train_data, train_label = to_tensor(train_data, train_label)
            train_data, train_label, sub = train_data.to('cuda:{}'.format(args.gpu), non_blocking=True),\
                                         train_label.to('cuda:{}'.format(args.gpu), dtype=torch.long, non_blocking=True),\
                                         sub.to('cuda:{}'.format(args.gpu), dtype=torch.long, non_blocking=True)

            train_cls, _, train_sub = model(train_data)
            loss = loss_ce(train_cls, train_label.squeeze(1)) # + loss_sub(train_sub, sub)
            loss_acc += loss

            train_pred_cls = acc(train_cls, train_label)
            train_pred_cls_acc += train_pred_cls

            # train_pred_sub = acc(train_sub, sub)
            # train_pred_sub_acc += train_pred_sub

            # loss_acc += (loss - loss_acc) / (batch+1)
            # pred_acc += (pred - pred_acc) / (batch+1)

            loss.backward()
            optimizer.step()
            # print(1)

        # la -> loss_acc, tca -> train_pred_cls_acc, tsa -> train_pred_sub_acc
        la, tca, tsa = loss_acc / (train_batch + 1), train_pred_cls_acc / (train_batch + 1), train_pred_sub_acc / (train_batch + 1)

        if not args.full_train:
            # Validation
            model.train(mode=False)

            for valid_batch, (sub, valid_data, valid_label) in enumerate(valid_loader):

                valid_data, valid_label, sub = valid_data.to('cuda:{}'.format(args.gpu), non_blocking=True),\
                                               valid_label.to('cuda:{}'.format(args.gpu), dtype=torch.long,
                                                              non_blocking=True),\
                                               sub.to('cuda:{}'.format(args.gpu), dtype=torch.long, non_blocking=True)

                # valid_data, valid_label = to_tensor(valid_data, valid_label)
                valid_cls, _, valid_sub = model(valid_data)
                valid_pred_cls = acc(valid_cls, valid_label)
                valid_pred_cls_acc += valid_pred_cls

                # There should be valid_pred_sub, because corresponding data/label not shown in training
                # valid_pred_sub = acc(valid_sub, sub)
                # valid_pred_sub_acc += valid_pred_sub

            vca = valid_pred_cls_acc / (valid_batch + 1)

            wandb.log({'Epoch': i,
                       'Loss': la,
                       'Training_cls_acc_{}'.format(test_sub): tca,
                       # 'Training_sub_acc_{}'.format(test_sub): tsa,
                       'Validation_cls_acc_{}'.format(test_sub): vca,
                       # 'Validation_sub_acc_{}'.format(test_sub): vsa,
                       'Learning rate': optimizer.state_dict().get('param_groups')[0].get('lr')
                       })

            print('Epoch', i,
                  'Loss {:.2f}'.format(la),
                  'TCA {:.2f}'.format(tca),
                  # 'TSA {:.2f}'.format(tsa),
                  'VCA {:.2f}'.format(vca),
                  # 'VSA {:.2f}'.format(vsa),
                  'Time Elapsed {:.2f}'.format(time.time() - start))
        else:
            print('Epoch', i,
                  'Loss {:.2f}'.format(la),
                  'TCA {:.2f}'.format(tca),
                  # 'TSA {:.2f}'.format(tsa),
                  'Time Elapsed {:.2f}'.format(time.time() - start))

        scheduler.step()

        # Save model
        if args.save & (i % args.epoch == 0) & (i > 0):
            if args.full_train:
                torch.save(model.state_dict(),
                           '/home/yk/Desktop/dataset_transfer/save//{}_fulltrain_epoch{}.pth'.format(args.dataset, i))
            else:
                torch.save(model.state_dict(),
                           '/home/yk/Desktop/dataset_transfer/save//{}_sub{}_epoch{}.pth'.format(args.dataset, test_sub, i))

    if not args.full_train:
        test(test_sub, model)


def test(test_sub, model=None):

    test_pred_cls_acc, test_pred_sub_acc = 0, 0

    test_set = TestSet(dataset=args.dataset, test_sub=test_sub)
    test_loader = DataLoader(dataset=test_set, batch_size=100)

    if args.test:
        model.load_state_dict(torch.load(args.load))

    model.eval()

    for test_batch, (sub, test_data, test_label) in enumerate(test_loader):

        test_data, test_label, sub = test_data.to('cuda:{}'.format(args.gpu), non_blocking=True),\
                                test_label.to('cuda:{}'.format(args.gpu), dtype=torch.long, non_blocking=True),\
                                sub.to('cuda:{}'.format(args.gpu), dtype=torch.long, non_blocking=True)

        test_cls, _, test_sub = model(test_data)
        test_pred_cls = acc(test_cls, test_label)
        test_pred_cls_acc += test_pred_cls

        # test_pred_sub = acc(test_sub, sub)
        # test_pred_sub_acc += test_pred_sub

    wandb.log({'Test_cls_acc_{}'.format(test_sub): test_pred_cls_acc / (test_batch + 1),
               # 'Test_sub_acc_{}'.format(test_sub): test_pred_sub_acc / (test_batch + 1)
               })

    # print('Test Accuracy {:.2f}'.format(test_pred_cls_acc.item() / (test_batch + 1)))


def visualization():

    model = EEGNet(n_classes=2, channels=CHANNELS, samples=SAMPLES, dropoutRate=0.25)
    model.to('cuda:{}'.format(args.gpu))
    model.load_state_dict(torch.load(args.load))
    model.eval()

    total_data, total_label, total_sub = None, None, None
    dataset_list = ['sub54']
    num_sub_list = [9, 54]
    for i, name in enumerate(dataset_list):

        print('Get feature embeddings for {}, and it may be slow'.format(name))

        dataset = TrainingSet(dataset=name)
        dataloader = DataLoader(dataset=dataset, batch_size=400, shuffle=False)
        
        for batch, (sub, data, label) in enumerate(dataloader):
            sub = sub.numpy()
            data = data.to('cuda:{}'.format(args.gpu), non_blocking=True)
            # label = label.to('cuda:{}'.format(args.gpu), non_blocking=True)
            if i > 0:
                sub += num_sub_list[i-1]
                # print(sub)
            _, feature = model(data)
            if total_data is None:
                total_data, total_label, total_sub = feature, label, sub[:, np.newaxis]
            else:
                total_data, total_label, total_sub = torch.cat((total_data, feature)), torch.cat((total_label, label)),\
                                                     np.vstack((total_sub, sub[:, np.newaxis]))

            visualizer(feature, label.numpy(), batch)

    print('{} points in total.'.format(total_data.shape[0]))
    visualizer(total_data, total_label.numpy())
    # visualizer(total_data, total_sub)


if __name__ == "__main__":

    # Settings for different datasets
    CHANNELS, SAMPLES, NUM_SUBJECTS = config()

    print(args)

    if args.train:
        wandb.init(project='DT_{}_{}_{}'.format(args.dataset, args.lr, args.batch_size, entity="kang97"))

    torch.cuda.set_device(args.gpu)

    # A complete training process and Leave One Subject Out for test!
    if args.train:
        if args.full_train:
            print('Use All Data To Train')
            train()
        else:
            for sub in range(NUM_SUBJECTS):
                train(sub)

    # Only test only specified datasets/subjects
    if args.test:
        for sub in range(NUM_SUBJECTS):
            test(sub)

    if args.visualizer:
        visualization()
