import torch
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np


TEXT_OUT_FLAG = True

# read data and label from dataframe
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
        filename = self.dataframe.iloc[k].file
        return data_tf, label_tf, filename

# evaluation accuracy and confusion matrix from model prediction and data label
def cust_test_m(i_pred, i_label, i_num_class):
    #NUM_CLASS = 3

    cor_cnt = 0
    chk_cnt = len(i_label)

    m = np.zeros((i_num_class, i_num_class))

    pred = torch.argmax(i_pred, dim=1)

    for i in range(len(i_label)):
        if pred[i] == i_label[i]:
            cor_cnt += 1

        tmp_label = i_label[i]
        tmp_pred = pred[i]

        m[tmp_label, tmp_pred] += 1

    #print(pred)
    #print(i_label)

    return cor_cnt, chk_cnt, m

# model read
device = torch.device('cpu')  # cpu because model deploy on cpu cloud(no gpu)
model = torch.load("../../Result/cust_model_cpsc_phychk_0_20240326.pth", map_location=device)
model.eval()

# read data
df_ts1_data = pd.read_json('df_data/cust_data_cpsc_ts_n_af_pvc_0.json', orient='records', lines=True)
df_ts2_data = pd.read_json('df_data/cust_data_phychk_ts_n_af_0.json', orient='records', lines=True)
ts_data = pd.concat([df_ts1_data, df_ts2_data], axis=0)
test_loader = DataLoader(cust_data(in_data=ts_data), batch_size=32, shuffle=False)

# model prediction
NUM_CLASS = 3
ts_cor_sum = 0
ts_chk_sum = 0
ts_m_sum = np.zeros((NUM_CLASS, NUM_CLASS))
if TEXT_OUT_FLAG:
    f = open("../../Result/output.txt", "w")

for ts_inputs, ts_labels, ts_filename in test_loader:
    ts_outputs = model(ts_inputs)
    ts_labels_sq = ts_labels.squeeze(1).long()
    ts_cor_tmp, ts_chk_tmp, ts_m_tmp = cust_test_m(ts_outputs, ts_labels_sq, NUM_CLASS)
    ts_cor_sum += ts_cor_tmp
    ts_chk_sum += ts_chk_tmp
    ts_m_sum += ts_m_tmp

    # write output text file
    if TEXT_OUT_FLAG:
        for i in range(len(ts_filename)):
            tmp_filename = ts_filename[i]
            tmp_label = ts_labels[i][0].numpy()
            tmp_output = ts_outputs[i]
            tmp_pred = torch.argmax(tmp_output).numpy()
            out_line = str(tmp_filename) + ' ' + str(int(tmp_label)) + ' ' + str(tmp_pred) + '\n'
            f.write(out_line)

if TEXT_OUT_FLAG:
    f.close()

ts_acc_ep = ts_cor_sum / ts_chk_sum

# output result accuracy and confusion matrix
print("test_accuracy:{:.4f}".format(ts_acc_ep))
print(ts_m_sum)

# calculate sensitivity and specificity
for i in range(NUM_CLASS):
    tmp_row = ts_m_sum[i]
    tmp_sum = sum(tmp_row)

    # sum of positive and negative
    tmp_sum_p = 0
    tmp_sum_n = 0
    for j in range(NUM_CLASS):
        if j == i:
            tmp_sum_p = tmp_row[j]
        else:
            tmp_sum_n += tmp_row[j]

    tmp_sens = tmp_sum_p / (tmp_sum_p + tmp_sum_n)
    tmp_spec = tmp_sum_n / (tmp_sum_p + tmp_sum_n)
    print("cls:{} sens:{:.4f} spec:{:.4f}".format(i, tmp_sens, tmp_spec))
