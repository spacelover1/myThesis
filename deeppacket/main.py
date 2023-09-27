from utils import *
from trainer import Trainer
from model import *
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import multiprocessing as mp
import time
from sklearn.metrics import confusion_matrix
from math import log2
import pickle
import os
import matplotlib.pyplot as plt
import scipy.io as sio
import itertools
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

lock = mp.Lock()
counter = mp.Value('i', 0)
    
def plot_confusion_matrix(matrix, labels,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(matrix)
    plt.figure()
    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels) #, rotation=45)
    plt.yticks(tick_marks, labels)

    fmt = '.2f' if normalize else 'd'
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    
def main():
    # load 參數
    config = get_args()
    # load 資料
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(config)
    # normalize X
    # array / 255 or max array
    X_train, X_val, X_test = np.array(X_train)/np.max(np.array(X_train)), np.array(X_val)/np.max(np.array(X_val)), np.array(X_test)/np.max(np.array(X_test))
    # 把 y 的 string 做成 one hot encoding 形式
    label_encoder = LabelBinarizer()
    y_train_onehot = label_encoder.fit_transform(y_train)
    y_val_onehot = label_encoder.transform(y_val)
    y_test_onehot = label_encoder.transform(y_test)
    # 印一些有的沒的
    print('X_train size:', len(X_train))
    max_x = 0
    for x in X_train:
        if max_x < len(x):
            max_x = len(x)
    print('max length:',max_x)
    
    
    # if config.mode == 'train':
    #     print('===== train =====')
    #     return train(config, X_train, y_train_onehot, X_val, y_val_onehot), (X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot), label_encoder
    # elif config.mode == 'test':
    #     print('===== test =====')
    #     return test(config, X_test, y_test_onehot, label_encoder), (X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot), label_encoder
    # else:
    #     pass

    if config.mode == 'train-test':
        print('===== train + test =====')
        train(config, X_train, y_train_onehot, X_val, y_val_onehot)
        test(config, X_test, y_test_onehot, label_encoder)
        
    elif config.mode == 'train':
        print('===== train =====')
        train(config, X_train, y_train_onehot, X_val, y_val_onehot)
        
    elif config.mode == 'test':
        print('===== test =====')
        test(config, X_test, y_test_onehot, label_encoder)
        
    else:
        pass

    return X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot, label_encoder

#=================================================
#  Train
#=================================================
def train(config, X_train, y_train, X_val, y_val):
    if config.model_type == 'sae':
        ae1 = AutoEncoder(1500, 400, encoder_id = 0)
        ae2 = AutoEncoder(400, 300, encoder_id = 1)
        ae3 = AutoEncoder(300, 200, encoder_id = 2)
        ae4 = AutoEncoder(200, 100, encoder_id = 3)
        ae5 = AutoEncoder(100, 50, encoder_id = 4)
        aelist = [ae1, ae2, ae3, ae4, ae5]
        sae = StackedAutoEncoder(aelist)
        next_X_train = X_train.copy()
        next_X_val = X_val.copy()
        for i, ae in enumerate(aelist):
            trainer = Trainer(config, aelist[i],
             next_X_train, next_X_train, next_X_val, next_X_val, loss_fn = 'mse', metrics = 'mse')
            trainer.name = ('{}{}'.format(config.model_name, i+1))
            trainer.train(7)
            next_X_train = encode_by_ae(ae, next_X_train, batch_size = config.batch_size)
            next_X_val = encode_by_ae(ae, next_X_val, batch_size = config.batch_size)
        trainer = Trainer(config, sae, X_train, X_train, X_val, X_val, loss_fn = 'mse', metrics = 'mse')
        trainer.train(20)
        if config.task_type == 'app':
            sae_classifier = StackedAutoEncoderClassifier(sae)
        else:
            sae_classifier = StackedAutoEncoderClassifier2(sae)
        trainer = Trainer(config, sae_classifier, X_train, y_train, X_val, y_val,
                     loss_fn = 'categorical_crossentropy', metrics = 'f1_score')
        trainer.name = ('{}_classifier'.format(config.model_name))
        trainer.train(20)
        return trainer, (sae_classifier, sae)
        #trainer.train()
    elif config.model_type == 'cnn':
        if config.task_type == 'app':
            cnn = CNN()
        else:
            cnn = CNN2()
        X_train, X_val = np.expand_dims(X_train, 2), np.expand_dims(X_val, 2)
        print('Prepare Trainer...')
        trainer = Trainer(config, cnn, X_train, y_train, X_val, y_val,
                        loss_fn = 'categorical_crossentropy', metrics = 'acc')
        print('Trainer prepared.')
        trainer.train(10)
        #trainer.train(500)
        return trainer, cnn
    else:
        print('#ERROR# invalid type \'{}\''.format(config.type))
        return -1

#=================================================
#  Test
#=================================================
def test(config, X_test, y_test, label_encoder):
    if config.model_type == 'sae':
        ae1 = AutoEncoder(1500, 400, encoder_id = 0)
        ae2 = AutoEncoder(400, 300, encoder_id = 1)
        ae3 = AutoEncoder(300, 200, encoder_id = 2)
        ae4 = AutoEncoder(200, 100, encoder_id = 3)
        ae5 = AutoEncoder(100, 50, encoder_id = 4)
        aelist = [ae1, ae2, ae3, ae4, ae5]
        #for i, ae in enumerate(aelist):
        #    ae.load_weights('models/{}{}.checkpoint.pth.h5'.format(config.model_name, i+1))
        sae = StackedAutoEncoder(aelist)
        #sae.load_weights('models/{}.checkpoint.pth.h5'.format(config.model_name))
        if config.task_type == 'app':
            sae_classifier = StackedAutoEncoderClassifier(sae)
        else:
            sae_classifier = StackedAutoEncoderClassifier2(sae)
        sae_classifier.load_weights('models/{}_classifier.checkpoint.pth.h5'.format(config.model_name))
        model = sae_classifier
    elif config.model_type == 'cnn':
        if config.task_type == 'app':
            cnn = CNN()
        else:
            cnn = CNN2()
        cnn.load_weights('models/{}.checkpoint.pth.h5'.format(config.model_name))
        X_test = np.expand_dims(X_test, 2)
        model = cnn
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [f1_score])
    print(model.evaluate(X_test, y_test, batch_size = config.batch_size, verbose = True))
    y_pred = model.predict_classes(X_test, batch_size = config.batch_size, verbose = True)
    labels = label_encoder.classes_
    cm = confusion_matrix(labels[y_test.argmax(1)], labels[y_pred], labels=labels)
    plot_confusion_matrix(cm, labels, normalize=True)

    print(classification_report(labels[y_test.argmax(1)], labels[y_pred]))
    print(accuracy_score(labels[y_test.argmax(1)], labels[y_pred]))

    file = open("report.txt", "w") 
    file.write(str(classification_report(labels[y_test.argmax(1)], labels[y_pred])))
    file.write('\n\n\n')
    file.write('acc : ' + str(accuracy_score(labels[y_test.argmax(1)], labels[y_pred])))
    file.close() 
           
    mdict = {'conf': list(cm)}
    sio.savemat('objs/cm_{}.mat'.format(config.model_name), mdict, appendmat=True, format='5',
                long_field_names=False, do_compression=False, oned_as='row')
    
    dump(cm, 'objs/cm_{}.pickle'.format(config.model_name))
    return "testing without trainer", model


def check(filename):
    return not '_class' in filename
    

def load_data(config):
    if config.debug:
        max_data_nb = 10
    else:
        max_data_nb = 10000
    directory = 'pickleFiles(org)'
    todo_list = gen_todo_list(directory, check = check)
    ### ver 1 ###
    train_rate = 0.64
    val_rate = 0.16
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []

    for counter, filename in enumerate(todo_list):

        (tmpX, tmpy) = load(filename)
        if config.task_type == 'class':
            tmpy = load('.'.join(filename.split('.')[:-1]) + '_class.pickle')
        tmpX , tmpy = tmpX[:max_data_nb], tmpy[:max_data_nb]
        assert(len(tmpX) == len(tmpy))
        tmpX = processX(tmpX)

#############################################################################
        tmpX = processX2(tmpX)               ########
#########################################################################


        train_num = int(len(tmpX) * train_rate)
        val_num = int(len(tmpX) * val_rate)
        X_train.extend(tmpX[:train_num])
        y_train.extend(tmpy[:train_num])
        X_val.extend(tmpX[train_num: train_num + val_num])
        y_val.extend(tmpy[train_num: train_num + val_num])
        X_test.extend(tmpX[train_num + val_num:])
        y_test.extend(tmpy[train_num + val_num:])
        print('\rLoading... {}/{}'.format(counter+1,len(todo_list)), end = '')
    print('\r{} Data loaded.               '.format(len(todo_list)))
    return X_train, y_train, X_val, y_val, X_test, y_test
    """
    ### ver2 ###
    cpus = mp.cpu_count() - 2
    oldtime = time.time()
    pool = mp.Pool(processes=cpus)
    manager = mp.Manager()
    ns = manager.Namespace()
    ns.X_train = []
    ns.y_train = []
    ns.X_val = []
    ns.y_val = []
    ns.X_test = []
    ns.y_test = []

    res = pool.map(task, [(ns, i, len(todo_list)) for i in todo_list])
    pool.close()
    pool.join()
    #pool.apply_async(task, (ns, i,))
    newtime = time.time()
    print('Using time:', newtime - oldtime, '(sec)')
    return ns.X_train, ns.y_train, ns.X_val, ns.y_val, ns.X_test, ns.y_test
    """


def processX(X):
    if True:
        X = np.array(X)
        lens = [len(x) for x in X] 
        maxlen = 1500
        tmpX = np.zeros((len(X), maxlen))
        mask = np.arange(maxlen) < np.array(lens)[:,None]
        tmpX[mask] = np.concatenate(X)
        return tmpX
    else:
        for i, x in enumerate(X):
            tmp_x = np.zeros((1500,))
            tmp_x[:len(x)] = x
            X[i] = tmp_x
        return X

box = 2 # bytes
overlap = 1          # bytes

def cross_entropy(p, q):
    
    for i in range(len(q)):
        if q[i] == 0:
            q[i] += 1
            #p[i] = 0
            
    return round(-sum([p[i] * log2(q[i]) for i in range(len(p))]), 2)

def processX2(X):

  for i in range(np.shape(X)[0]):
    tmp = []
    for j in range(np.shape(X)[1]-box-overlap):
      p = list(np.divide(X[i,j:j+box], 255))
      q = list(np.divide(X[i,j+overlap:j+overlap+box], 255))
      tmp.append(cross_entropy(p, q))

    for k in range(overlap+box):
      tmp.append(0)

    X[i, :] = tmp

  return X



if __name__ == '__main__':
    # (trainer, model), data, label_encoder = main()
    # (X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot) = data
    X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot, label_encoder = main()