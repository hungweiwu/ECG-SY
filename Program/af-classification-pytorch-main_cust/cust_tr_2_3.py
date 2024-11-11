import torch
from torch.utils.data import DataLoader
from model.cust_m_2 import cust_net
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torchviz import make_dot

# read data from dataframe for dataload
class cust_data(torch.utils.data.Dataset):
    def __init__(self, in_data):
        self.dataframe = in_data

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, k):
        data = self.dataframe.iloc[k].ecg
        data_np = np.array(data)
        data_tf = torch.FloatTensor(data_np)
        label = self.dataframe.iloc[k].target  # integer label
        label_tf = torch.FloatTensor([label])
        return data_tf, label_tf


# load training and testing data
df1_data = pd.read_json('df_data/cust_data_cpsc_tr_n_af_pvc_0_mpvc.json', orient='records', lines=True)
df1_data_au = pd.read_json('df_data/cust_data_cpsc_tr_n_af_pvc_0_mpvc_au.json', orient='records', lines=True)
df2_data = pd.read_json('df_data/cust_data_phychk_tr_n_af_0.json', orient='records', lines=True)
df2_data_au = pd.read_json('df_data/cust_data_phychk_tr_n_af_0_au.json', orient='records', lines=True)
df3_data = pd.read_json('df_data/cust_data_phy2020_tr_n_af_pvc_0_mpvc.json', orient='records', lines=True)
df3_data_au = pd.read_json('df_data/cust_data_phy2020_tr_n_af_pvc_0_mpvc_au.json', orient='records', lines=True)
df4_data = pd.read_json('df_data/cust_data_alarge_tr_n_af_pvc_1_mpvc.json', orient='records', lines=True)
df4_data_au = pd.read_json('df_data/cust_data_alarge_tr_n_af_pvc_1_mpvc_au.json', orient='records', lines=True)
tr_data = pd.concat([df1_data, df2_data, df3_data, df4_data, df1_data_au, df2_data_au, df3_data_au, df4_data_au], axis=0)

df_ts1_data = pd.read_json('df_data/cust_data_cpsc_ts_n_af_pvc_0_mpvc.json', orient='records', lines=True)
df_ts2_data = pd.read_json('df_data/cust_data_phychk_ts_n_af_0.json', orient='records', lines=True)
df_ts3_data = pd.read_json('df_data/cust_data_phy2020_ts_n_af_pvc_0_mpvc.json', orient='records', lines=True)
df_ts4_data = pd.read_json('df_data/cust_data_alarge_ts_n_af_pvc_1_mpvc.json', orient='records', lines=True)
ts_data = pd.concat([df_ts1_data, df_ts2_data, df_ts3_data, df_ts4_data], axis=0)

# balance training data at each batch by dataload sampler
target_ar = tr_data.loc[:, "target"].values

print('training data(class0/1/2) : {}/{}/{}'.format(
    len(np.where(target_ar == 0)[0]), len(np.where(target_ar == 1)[0]), len(np.where(target_ar == 2)[0])))

class_sample_count = np.array([len(np.where(target_ar == t)[0]) for t in np.unique(target_ar)])
weight = 1. / class_sample_count
samples_weight_ar = np.array([weight[t] for t in target_ar])

sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight_ar, len(samples_weight_ar))
train_loader = DataLoader(cust_data(in_data=tr_data), batch_size=128, sampler=sampler)

for i, (data, target) in enumerate(train_loader):
    print("batch index {}, 0/1/2: {}/{}/{}".format(
        i,
        len(np.where(target.numpy() == 0)[0]),
        len(np.where(target.numpy() == 1)[0]),
        len(np.where(target.numpy() == 2)[0])))

# data loader
train_loader = DataLoader(cust_data(in_data=tr_data), batch_size=128, shuffle=True)
test_loader = DataLoader(cust_data(in_data=ts_data), batch_size=128, shuffle=False)

# model setup
model = cust_net(num_cls=3)

# cuda
model.train()
model.cuda()

# set optimization criteria
#criteria = nn.CrossEntropyLoss()
#optimizer = optim.AdamW(model.parameters(), lr=0.001)
weights = torch.tensor([1.0, 1.0, 1.0])
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criteria = nn.CrossEntropyLoss(weight=weights.cuda())

# training epoch number
num_ep = 25

# validation of model prediction and input data label
def cust_test(i_pred, i_label):

    cor_cnt = 0
    chk_cnt = len(i_label)

    pred = torch.argmax(i_pred, dim=1)

    for i in range(len(i_label)):
        if pred[i] == i_label[i]:
            cor_cnt += 1
    #print(pred)
    #print(i_label)

    return cor_cnt, chk_cnt

def cust_test_cls(i_pred, i_label):

    cor_cnt = np.zeros(3)
    chk_cnt = np.zeros(3)
    #chk_cnt = len(i_label)

    pred = torch.argmax(i_pred, dim=1)

    for i in range(len(i_label)):
        g = i_label[i]  # golden label
        chk_cnt[g] += 1
        if pred[i] == i_label[i]:
            cor_cnt[g] += 1
    #print(pred)
    #print(i_label)

    return cor_cnt, chk_cnt

# training
for epoch in range(num_ep):
    # variable initialize
    bh_cnt = 0
    loss_sum = 0
    loss_ep = 0
    cor_cls = np.zeros(3)
    chk_cls = np.zeros(3)
    cor_sum = 0
    chk_sum = 0
    ts_cor_cls = np.zeros(3)
    ts_chk_cls = np.zeros(3)
    ts_cor_sum = 0
    ts_chk_sum = 0
    # read data for training
    for inputs, labels in train_loader:
        #cuda
        inputs = inputs.cuda()
        labels = labels.cuda()

        # model prediction and loss calculation
        outputs = model(inputs)
        labels_sq = labels.squeeze(1).long()
        loss = criteria(outputs, labels_sq)

        # model parameter update
        optimizer.zero_grad()  # clear
        loss.backward()  # propagate
        optimizer.step()  # update

        bh_cnt += 1
        # loss_sum += loss.detach().numpy()
        loss_sum += loss.mean().item()

        # validation training data
        with torch.no_grad():
            model.eval()  # cuda
            #print(loss_sum)
            cor_tmp, chk_tmp = cust_test_cls(outputs, labels_sq)
            cor_cls += cor_tmp
            chk_cls += chk_tmp
            cor_sum += sum(cor_tmp)
            chk_sum += sum(chk_tmp)
            model.train()

        # # network visualization for connection check
        # v = model(inputs)
        # g = make_dot(v.mean(), params=dict(model.named_parameters()))
        # g.view()
        # print("vis")

    # loss and accuracy summary
    loss_ep = loss_sum / bh_cnt
    acc_ep = cor_sum / chk_sum
    acc_cls_ep = np.divide(cor_cls, chk_cls)

    # evaluation(testing)
    with torch.no_grad():
        model.eval()  # cuda
        for ts_inputs, ts_labels in test_loader:
            #cuda
            ts_inputs = ts_inputs.cuda()
            ts_labels = ts_labels.cuda()

            ts_outputs = model(ts_inputs)
            ts_labels_sq = ts_labels.squeeze(1).long()
            ts_cor_tmp, ts_chk_tmp = cust_test_cls(ts_outputs, ts_labels_sq)
            ts_cor_cls += ts_cor_tmp
            ts_chk_cls += ts_chk_tmp
            ts_cor_sum += sum(ts_cor_tmp)
            ts_chk_sum += sum(ts_chk_tmp)
        model.train()

    ts_acc_ep = ts_cor_sum / ts_chk_sum
    ts_acc_cls_ep = np.divide(ts_cor_cls, ts_chk_cls)

    # output iteration result
    print("epoch:{:d}  loss:{:.4f}  train_accuracy:{:.4f}{}  test_accuracy:{:.4f}{}".format(
        epoch, loss_ep, acc_ep, acc_cls_ep, ts_acc_ep, ts_acc_cls_ep))

    # tmp save
    if ts_acc_cls_ep[2] > 0.8:
        tmp_name = "../../Result/cust_model_tmp_" + str(epoch) + ".pth"
        torch.save(model, tmp_name)


# save model
torch.save(model, "../../Result/cust_model.pth")
