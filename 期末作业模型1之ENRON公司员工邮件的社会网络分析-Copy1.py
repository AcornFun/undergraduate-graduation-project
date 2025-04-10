#!/usr/bin/env python
# coding: utf-8

# 方骁牛逼
# 

# In[2]:


import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print('GPU is available!')
else:
    print('GPU is not available.')

# 指定在GPU上执行后续代码
with tf.device('/GPU:0'):
    # your_code_here
    print("This code is executed on GPU.")


# 这么大的数据集，我想在显卡上运行而不是CPU，之前都成功了但是太久没用，不知道问题出在哪里

# In[3]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import re
import subprocess


# In[4]:


from subprocess import check_output


# 导入需要的library

# In[5]:


pd.options.mode.chained_assignment = None


chunk = pd.read_csv('E:\FINAL\emails2.csv\emails.csv',  chunksize=10000)


data = next(chunk)


data.info()
print(data.message[2]) 


# 选取510000组数据来进行实验。导入数据很成功。

# In[6]:


def get_text(Series, row_num_slicer):
    """returns a Series with text sliced from a list split from each message. Row_num_slicer
    tells function where to slice split text to find only the body of the message."""
    result = pd.Series(index=Series.index)
    for row, message in enumerate(Series):
        message_words = message.split('\n')
        del message_words[:row_num_slicer]
        result.iloc[row] = message_words
    return result

def get_row(Series, row_num):
    """returns a single row split out from each message. Row_num is the index of the specific
    row that you want the function to return."""
    result = pd.Series(index=Series.index)
    for row, message in enumerate(Series):
        message_words = message.split('\n')
        message_words = message_words[row_num]
        result.iloc[row] = message_words
    return result

def get_address(df, Series, num_cols=1):
    """returns a specified email address from each row in a Series"""
    address = re.compile('[\w\.-]+@[\w\.-]+\.\w+')
    addresses = []
    result1 = pd.Series(index=df.index)
    result2 = pd.Series(index=df.index)
    result3 = pd.Series(index=df.index)
    for i in range(len(df)):
        for message in Series:
            correspondents = re.findall(address, message)
            addresses.append(correspondents)
            result1[i] = addresses[i][0]
        if num_cols >= 2:
            if len(addresses[i]) >= 3:
                result2[i] = addresses[i][1]
                if num_cols == 3:
                    if len(addresses[i]) >= 4:
                        result3[i] = addresses[i][2]
    return result1, result2, result3

def standard_format(df, Series, string, slicer):
    """Drops rows containing messages without some specified value in the expected locations. 
    Returns original dataframe without these values. Don't forget to reindex after doing this!!!"""
    rows = []
    for row, message in enumerate(Series):
        message_words = message.split('\n')
        if string not in message_words[slicer]:
            rows.append(row)
    df = df.drop(df.index[rows])
    return df


# 所有这些消息都具有相同的结构。通过换行符 \n\拆分文本，然后简单地对结果列表进行切片以访问每封电子邮件的特定元素。

# In[7]:


#import pandas as pd

# 先删除名为level_0的列
#data = data.drop(columns=['level_0'])

# 重置索引
#data = data.reset_index(drop=True)


# In[8]:


x = len(data.index)
headers = ['Message-ID: ', 'Date: ', 'From: ', 'To: ', 'Subject: ']
for i, v in enumerate(headers):
    data = standard_format(data, data.message, v, i)
data = data.reset_index()
print("删除了{} 无用信息! 总数的{}% .".format(x - len(data.index), np.round(((x - len(data.index)) / x) * 100, decimals=2)))


# 

# 这两步的意义是
# 我的 standard_format（） 函数旨在删除缺少我进行分析所需的关键信息的消息。
# 有几条消息没有 To： 
# 当我尝试提取 To： 行信息时，我的 get_row（） 函数会很混乱，所以对数据进行筛选处理

# In[9]:


data['text'] = get_text(data.message, 15)
data['date'] = get_row(data.message, 1)
data['senders'] = get_row(data.message, 2)
data['recipients'] = get_row(data.message, 3)
data['subject'] = get_row(data.message, 4)
#从邮件消息中提取出日期、发件人、收件人和主题等信息，并将这些信息存储到新的列中。
data.date = data.date.str.replace('Date: ', '')
data.date = pd.to_datetime(data.date)

data.subject = data.subject.str.replace('Subject: ', '')
#对日期和主题进行了一些清洗操作，如去除多余的文本。
data['recipient1'], data['recipient2'], data['recipient3'] = get_address(data, data.recipients, num_cols=3)
data['sender'], x, y = get_address(data, data.senders)
#对收件人进行了拆分，将多个收件人拆分成了多个列（recipient1、recipient2、recipient3）。
del data['recipients']
del data['senders']
del data['file']
del data['message']

data = data[['date', 'sender', 'recipient1', 'recipient2', 'recipient3', 'subject', 'text']]

print(data.head())


# 文本的内容筛选的还是比较干净的,接下来开始画网络
# 

# In[10]:


#pip install nxviz


# In[11]:


import networkx as nx
import nxviz as nv

G = nx.from_pandas_dataframe(data, 'sender', 'recipient1', edge_attr=['date', 'subject'])
plot = nv.ArcPlot(G)
plot.draw()
plt.show() 


# 找的代码似乎停留在nxviz的之前的版本，现在改为from_pandas_edgelist了，而且不用plt.draw，在选取数据的时候就自动绘制了

# In[12]:


import networkx as nx
import nxviz as nv

G = nx.from_pandas_edgelist(data, 'sender', 'recipient1', edge_attr=['date', 'subject'])
plot = nv.ArcPlot(G)

plt.show() 


# In[26]:


print(G)


# 单向图，可以看到大概有六个节点比较的活跃

# In[13]:


plot = nv.CircosPlot(G)

plt.show()


# 环状图，没什么可说的，只是这样形式的可视化似乎不如人意，然后我试图通过nx来构建网络。

# In[14]:


plt.figure(figsize=(20,20))
pos = nx.spring_layout(G, k=.1)
nx.draw_networkx(G, pos, node_size=25, node_color='red', with_labels=True, edge_color='blue')
plt.show()


# 接下来分析一下度中心性

# In[15]:


cent = nx.degree_centrality(G)
name = []
centrality = []

for key, value in cent.items():
    name.append(key)
    centrality.append(value)

cent = pd.DataFrame()    
cent['name'] = name
cent['centrality'] = centrality
cent = cent.sort_values(by='centrality', ascending=False)

plt.figure(figsize=(11, 11))
_ = sns.barplot(x='centrality', y='name', data=cent[:15], orient='h')
_ = plt.xlabel('Degree Centrality')
_ = plt.ylabel('Correspondent')
_ = plt.title('Top 15 Degree Centrality Scores in Enron Email Network')
plt.show()


# In[16]:


# 选择前 15 行数据
top_15 = cent.head(15)

# 将数据保存到 CSV 文件中
top_15.to_csv('top_15_degree_centrality.csv', index=False)


# In[25]:


print(top_15)


# 
# 

# In[71]:


pd.set_option('display.max_colwidth', None)


# 模型就简单的分析完了，随便看看这些邮件内容好了，顺便检查一下自己的切片准不准确

# In[72]:


print(data['text'].head(100))


# In[18]:


from sklearn.cluster import KMeans


# In[34]:


# 创建 cent 的副本
cent_copy = top_15.copy()

# 提取用于聚类的特征
features = cent_copy[['centrality']]

# 设置要进行聚类的簇数
n_clusters = 3

# 初始化 k-means 模型
kmeans = KMeans(n_clusters=n_clusters)

# 对数据进行聚类
kmeans.fit(features)

# 获取聚类结果
cluster_labels = kmeans.labels_
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
# 将聚类结果添加到 DataFrame 中
cent_copy['cluster'] = cluster_labels

# 打印聚类结果
print(cent_copy)

# 将结果保存到 CSV 文件中
cent_copy.to_csv('cent_with_clusters.csv', index=False)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             


# In[ ]:




