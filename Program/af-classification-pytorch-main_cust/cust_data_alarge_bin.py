import glob
import scipy
from scipy.io import loadmat
import pandas as pd
from scipy.ndimage import zoom
import numpy as np
import random

# list files
data_list = glob.glob("../../Data/alarge_train_20230426_1_mpvc_sel/*.mat")

# dataset
dataset = {}

# re-sample
FREQ_IN = 500  # 500Hz
FREQ_OUT = 250  # 250Hz
FREQ_R = FREQ_OUT/FREQ_IN

VOLT_R = 1000  # voltage 1mV

DATA_AU_FLAG = False
if DATA_AU_FLAG:
    Y_AUG_R = 0.25  # phy degrade (1-R*rand0~1)
    X_AUG_R = 0.05  # phy up/degrade (1+R*rand-1~1)
else:
    Y_AUG_R = 0
    X_AUG_R = 0

# open file
for i in range(len(data_list)):
    cur_mat = data_list[i]
    cur_mat_sp = cur_mat.split('\\')
    f_name = cur_mat_sp[len(cur_mat_sp)-1].replace(".mat", '')

    f = loadmat(cur_mat)
    data = f['val'][0]  #lead1

    X_AUG_F = 1 + X_AUG_R * random.uniform(-1, 1)
    FREQ_R_AU = FREQ_R * X_AUG_F
    a_intp = zoom(data, FREQ_R_AU)
    Y_AUG_F = 1 - Y_AUG_R * random.random()
    a_intp = a_intp * np.float32(Y_AUG_F)

    data_r = a_intp.astype(float) / VOLT_R

    # save
    dataset[f_name] = {
        'ecg': data_r,
    }

    #print(f_name)

#print(dataset)

# read excel
ref_df = pd.read_excel("../../DataBase/PhysioNet/ALargeScale12/WFDBRecords/REFERENCE_OUT.xlsx")
df_row, df_col = ref_df.shape
print("df row:", df_row)
print("df col:", df_col)

# cut data length
DATA_TIME = 30  # 30 second
DATA_FIX_LEN = FREQ_OUT * DATA_TIME

# data list
ecg_list = []
label_list = []
target_list = []
file_list = []
cls0_cnt = 0
cls1_cnt = 0
cls2_cnt = 0

# sel and assign
# 270492004 iavb
# 164889003 af
# 427172004 or 17338001 pvc
# abnormal
# Sinus Rhythm 426783006
# atrial_flutter 164890007
# complete_right_bundle_branch_block 713427006
# incomplete_right_bundle_branch_block 713426002
# left_anterior_fascicular_block 445118002
# left_bundle_branch_block 164909002
# nonspecific_intraventricular_conduction_disorder 698252002
# premature_atrial_contraction 284470004
# right_bundle_branch_block 59118001
# supraventricular_premature_beats 63593006
# ST drop down 429622005
# ST extension 164930006
# ST tilt up 164931005
for k, d in dataset.items():
    #print(k)
    # sel
    for idx in ref_df.index:
        tmp_record = ref_df['Recording'][idx]
        if k == tmp_record:
            # read labels
            tmp_label_list = []
            for i in range(1, df_col):
                col_name = "label" + str(i)
                tmp_item = ref_df[col_name][idx]

                if tmp_item is None:
                    i = df_col  # skip
                else:
                    tmp_label_list.append(tmp_item)

            # check class
            cls0_val = 0
            cls1_val = 0
            cls2_val = 0
            for i in range(len(tmp_label_list)):
                tmp_item = tmp_label_list[i]
                if tmp_item == 426783006:
                    cls0_val = 1
                if tmp_item == 164889003:
                    cls1_val = 1
                if tmp_item == 427172004 or tmp_item == 17338001:
                    cls2_val = 1

            # multi-hit check
            cls_sum = cls0_val + cls1_val + cls2_val
            if cls_sum > 1:
                multi_hit = True
            else:
                multi_hit = False

            # # cut data in fixed length
            # data_src = dataset[k]['ecg']
            # data_len = len(data_src)
            # data_tmp = np.zeros(DATA_FIX_LEN)
            # data_tmp_2d = []
            #
            # if data_len >= DATA_FIX_LEN:
            #     data_tmp = data_src[:DATA_FIX_LEN]
            # else:
            #     data_tmp[:data_len] = data_src
            #
            # data_tmp_2d.append(data_tmp)

            if not multi_hit:
                if cls0_val == 1 and cls0_cnt < 1445:  # down-sample
                    # cut data in fixed length
                    data_src = dataset[k]['ecg']
                    data_len = len(data_src)
                    data_tmp = np.zeros(DATA_FIX_LEN)
                    data_tmp_2d = []

                    if data_len >= DATA_FIX_LEN:
                        data_tmp = data_src[:DATA_FIX_LEN]
                    else:
                        data_tmp[:data_len] = data_src

                    data_tmp_2d.append(data_tmp)

                    # save
                    ecg_list.append(data_tmp_2d)
                    label_list.append('normal')
                    target_list.append(0)
                    file_list.append(k)
                    cls0_cnt += 1

                    #print('aa')
                elif cls1_val == 1 and cls1_cnt < 0:  # disable
                    # cut data in fixed length
                    data_src = dataset[k]['ecg']
                    data_len = len(data_src)
                    data_tmp = np.zeros(DATA_FIX_LEN)
                    data_tmp_2d = []

                    if data_len >= DATA_FIX_LEN:
                        data_tmp = data_src[:DATA_FIX_LEN]
                    else:
                        data_tmp[:data_len] = data_src

                    data_tmp_2d.append(data_tmp)

                    #save
                    ecg_list.append(data_tmp_2d)
                    label_list.append('af')
                    target_list.append(1)
                    file_list.append(k)
                    cls1_cnt += 1

                elif cls2_val == 1:
                    # cut data in fixed length
                    data_src = dataset[k]['ecg']
                    data_len = len(data_src)
                    data_tmp = np.zeros(DATA_FIX_LEN)
                    data_tmp_2d = []

                    if data_len >= DATA_FIX_LEN:
                        data_tmp = data_src[:DATA_FIX_LEN]
                    else:
                        data_tmp[:data_len] = data_src

                    data_tmp_2d.append(data_tmp)

                    #save
                    ecg_list.append(data_tmp_2d)
                    # label_list.append('pvc')
                    # target_list.append(2)
                    label_list.append('ab')  # binary normal/abnormal
                    target_list.append(1)
                    file_list.append(k)
                    cls2_cnt += 1

            break  # file hit then break
            #print('aa')


    #print("key:", dataset[k]['ecg'])
    # dataset[k]['label'] = 'normal'
    # dataset[k]['target'] = 0

print("Class0: ", cls0_cnt)
print("Class1: ", cls1_cnt)
print("Class2: ", cls2_cnt)
print("Total: ", len(ecg_list))

# save pd
df = pd.DataFrame({"ecg": ecg_list, "label": label_list, "target": target_list, "file": file_list})
df.to_json("df_data/cust_data_alarge_bin_.json", orient='records', lines=True)

#print("aaa")
