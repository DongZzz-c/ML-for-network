# -*- coding: utf-8 -*-

import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat
import os

random.seed(1234)

def generate(n,m,r,g = 10):
	click_f = []
	trust_f = []
	group_f = [[] for i in range(g)]
	group = [random.randint(0,g-1) for i in range(n)]
	for i in range(g):
		for j in range(m):
			if random.random() < 0.2:
				group_f[i].append(j)

	for i in range(n):
		for j in range(n):
			if group[i] == group[j] and random.random() < 1:
				trust_f.append([i,j])
				trust_f.append([j,i])

	for i in range(n):
		for j in range(m):
			if j in group_f[group[i]] and random.random() < 0.9:
				click_f.append([i,j,1])
			elif random.random() < 0.05:
				click_f.append([i,j,1])
			elif random.random() < 0.2:
				click_f.append([i,j,0])
	return click_f, trust_f


workdir = 'datasets/'
here = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--test_prop', default=0.1, help='the proportion of data used for test')
args = parser.parse_args()

# load data
click_f = loadmat(workdir + 'history.mat')
trust_f = loadmat(workdir + "cluster.mat")


click_list = []
trust_list = []

u_items_list = []
u_users_list = []
u_users_items_list = []
i_users_list = []

user_count = 0
item_count = 0
rate_count = 0


for s in click_f:
	uid = s[0]
	iid = s[1]

	if uid > user_count:
		user_count = uid
	if iid > item_count:
		item_count = iid

	click_list.append([uid, iid, 1])

pos_list = []
for i in range(len(click_list)):
	pos_list.append((click_list[i][0], click_list[i][1], click_list[i][2]))

# remove duplicate items in pos_list
pos_list = list(set(pos_list))



# train, valid and test data split
random.shuffle(pos_list)
num_test = int(len(pos_list) * args.test_prop)
test_set = pos_list[:num_test]
valid_set = pos_list[num_test:2 * num_test]
train_set = pos_list[2 * num_test:]
print('Train samples: {}, Valid samples: {}, Test samples: {}'.format(len(train_set), len(valid_set), len(test_set)))

with open(workdir + args.dataset + '/dataset.pkl', 'wb') as f:
	pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(valid_set, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)


train_df = pd.DataFrame(train_set, columns = ['uid', 'iid', 'label'])
valid_df = pd.DataFrame(valid_set, columns = ['uid', 'iid', 'label'])
test_df = pd.DataFrame(test_set, columns = ['uid', 'iid', 'label'])

click_df = pd.DataFrame(click_list, columns = ['uid', 'iid', 'label'])
train_df = train_df.sort_values(axis = 0, ascending = True, by = 'uid')

"""
u_items_list: 存储每个VM交互过的ip iid,没有则为[(0, 0)]
"""
for u in tqdm(range(user_count + 1)):
	hist = train_df[train_df['uid'] == u]
	u_items = hist['iid'].tolist()
	u_ratings = hist['label'].tolist()
	if u_items == []:
		u_items_list.append([(0, 0)])
	else:
		u_items_list.append([(iid, rating) for iid, rating in zip(u_items, u_ratings)])

train_df = train_df.sort_values(axis = 0, ascending = True, by = 'iid')

"""
i_users_list: 存储与每个ip相关联的VM,没有则为[(0, 0)]
"""
for i in tqdm(range(item_count + 1)):
	hist = train_df[train_df['iid'] == i]
	i_users = hist['uid'].tolist()
	i_ratings = hist['label'].tolist()
	if i_users == []:
		i_users_list.append([(0, 0)])
	else:
		i_users_list.append([(uid, rating) for uid, rating in zip(i_users, i_ratings)])

for s in trust_f:
	uid = s[0]
	fid = s[1]
	if uid > user_count or fid > user_count:
		continue
	trust_list.append([uid, fid])

trust_df = pd.DataFrame(trust_list, columns = ['uid', 'fid'])
trust_df = trust_df.sort_values(axis = 0, ascending = True, by = 'uid')


"""
u_users_list: 存储每个VM的邻居节点集合；
u_users_items_list: 存储VM每个邻居的ip iid列表
"""
for u in tqdm(range(user_count + 1)):
	hist = trust_df[trust_df['uid'] == u]
	u_users = hist['fid'].unique().tolist()
	if u_users == []:
		u_users_list.append([0])
		u_users_items_list.append([[(0,0)]])
	else:
		u_users_list.append(u_users)
		uu_items = []
		for uid in u_users:
			uu_items.append(u_items_list[uid])
		u_users_items_list.append(uu_items)
	
with open(workdir + args.dataset + '/list.pkl', 'wb') as f:
	pickle.dump(u_items_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(u_users_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(u_users_items_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(i_users_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump((user_count, item_count, rate_count), f, pickle.HIGHEST_PROTOCOL)


