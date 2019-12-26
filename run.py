#!/usr/bin/python3
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, GRU
from keras.optimizers import SGD

# 数据处理

# 结果标签转换为数字
dos_type = ['back','land','neptune','pod','smurf','teardrop','processtable','udpstorm','mailbomb','apache2']
probing_type = ['ipsweep','mscan','nmap','portsweep','saint','satan']
r2l_type = ['ftp_write','guess_passwd','imap','multihop','phf','warezmaster','warezclient','spy','sendmail','xlock','snmpguess','named','xsnoop','snmpgetattack','worm']
u2r_type = ['buffer_overflow','loadmodule','perl','rootkit','xterm','ps','httptunnel','sqlattack']
type2id = {'normal':0}
for i in dos_type:
    type2id[i] = 1
for i in r2l_type:
    type2id[i] = 2
for i in u2r_type:
    type2id[i] = 3
for i in probing_type:
    type2id[i] = 4
# 协议标签转换为数字
all_protocol = ['tcp', 'udp', 'icmp']
protocol_dict = {}
for id,name in enumerate(all_protocol):
    protocol_dict[name] = id
# 服务标签转换为数字
all_service = ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50']
service_dict = {}
for id,name in enumerate(all_service):
    service_dict[name] = id
# FLAG标签转换为数字 
all_flag = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
flag_dict = {}
for id,name in enumerate(all_flag):
    flag_dict[name] = id

# 读取训练集
import csv
all_train_data = []
trainX = []
trainY = []
with open('NSL-KDD/KDDTrain+.txt', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        all_train_data.append(row)
for i in all_train_data:
    i[1] = protocol_dict[i[1]]
    i[2] = service_dict[i[2]]
    i[3] = flag_dict[i[3]]
    i[-2] = type2id[i[-2]]
    trainX.append(i[:41])
    trainY.append(i[-2])
# 读取测试集
import csv
all_test_data = []
testX = []
testY = []
with open('NSL-KDD/KDDTest+.txt', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        all_test_data.append(row)
for i in all_test_data:
    i[1] = protocol_dict[i[1]]
    i[2] = service_dict[i[2]]
    i[3] = flag_dict[i[3]]
    i[-2] = type2id[i[-2]]
    testX.append(i[:41])
    testY.append(i[-2])
    
# 数据预处理
from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(trainX)
trainX = scaler.transform(trainX)
scaler = Normalizer().fit(testX)
testX = scaler.transform(testX)

# 由于RNN需要传入三维参数，这里做变换
import numpy as np
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Keras要求输出为(N,n)，其中n为分类数
trainY = keras.utils.to_categorical(trainY, num_classes=5)
testY = keras.utils.to_categorical(testY, num_classes=5)
# 到这里数据准备完成

# 开始构建模型
model = Sequential()
model.add(GRU(128, input_shape=(1,41)))
model.add(Dropout(0.2))
model.add(Dense(48, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(48, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history = model.fit(trainX, trainY,
          epochs=20,
          batch_size=32, validation_split=0.1, verbose=1)
score = model.evaluate(testX, testY, batch_size=32)

# 额外的作图
from keras.utils import plot_model
plot_model(model, to_file='model.png')

import matplotlib.pyplot as plt

# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('acc.png')

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('loss.png')

keras.utils.print_summary(model, line_length=None, positions=None, print_fn=None)