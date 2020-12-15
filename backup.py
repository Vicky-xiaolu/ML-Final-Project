
df_train = pd.read_csv("train_2016.csv", sep=',',header=None, encoding='unicode_escape')
df_test = pd.read_csv("test_2016_no_label.csv", sep=',',header=None, encoding='unicode_escape')
df_graph= pd.read_csv("graph.csv", sep=',',header=None, encoding='unicode_escape')
# remove the first row which is the title of the csv file & check if any of the entry is null
df_train = df_train.iloc[1:]
df_test = df_test.iloc[1:]
df_graph = df_graph.iloc[1:]
# generate train label and convert to numpy array (1330 DEMs and 225 GOPs)
train_label = pd.DataFrame(df_train[2].astype(int) - df_train[3].astype(int)).to_numpy()
train_label[train_label>0] = 1
train_label[train_label<=0] = 0
train_label = train_label.ravel()
# drop the DEM and GOP column in the train dataset
df_train = df_train.drop(columns=[2,3])
# drop the state name column
df_train = df_train.drop(columns=[1])
df_test = df_test.drop(columns=[1])
# convert the median income column to float
df_train[4] = df_train[4].str.replace(',', '').astype(float)
df_test[2] = df_test[2].str.replace(',', '').astype(float)

########## ADD the neighbor ID
train_data = df_train.to_numpy().astype(float)
test_data = df_test.to_numpy().astype(float)
graph_data = df_graph.to_numpy().astype(int)
n,m=graph_data.shape
# print(graph_data)
# print(graph_data[12716][0])
countyToAvg={}
lastCounty=0
countySum=0
neighborNum=1
for i in range(n):
    if graph_data[i][0]==lastCounty:
        countySum+=graph_data[i][1]
        neighborNum+=1
    else:
        countyToAvg[lastCounty]=countySum/neighborNum
        lastCounty=graph_data[i][0]
        countySum=graph_data[i][0]
        neighborNum=1
x1,y1=train_data.shape
avgCounties=[]
for i in range(x1):
    avgCounty=countyToAvg[train_data[i][0]]
    avgCounties.append(avgCounty)
np_avgCounties=np.array(avgCounties)
col_avgCounties=np_avgCounties[..., None]
train_data=np.c_[train_data,col_avgCounties]

x2,y2=test_data.shape
testCounties=[]
for i in range(x2):
    avgCounty=countyToAvg[train_data[i][0]]
    testCounties.append(avgCounty)
np_testCounties=np.array(testCounties)
col_testCounties=np_testCounties[..., None]
test_data=np.c_[test_data,col_testCounties]

# normalizating features by feature standardization except the state ID(*****)
mean = np.mean(train_data, axis = 0)
std = np.std(train_data, axis = 0)
train_data = (train_data - mean)/std
test_data = (test_data - mean)/std
