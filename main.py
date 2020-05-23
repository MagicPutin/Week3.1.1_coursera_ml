from sklearn.svm import SVC
import pandas as pd

data = pd.read_csv('data/svm-data.csv', index_col=False, header=None)
X = data.iloc[:, 1:3].to_numpy()
y = data[0].to_numpy()
print(X,y)
clf = SVC(kernel='linear', random_state=241, C=100000)
clf.fit(X, y)
with open('answers/task1.txt', 'w') as task:
    ans = clf.support_ + 1
    task.write(str(ans[0]))
    for i in ans[1:]:
        task.write(' ' + str(i))
print("😄")