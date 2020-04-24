from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from readdata import load_mnist
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
import time
import os
import gc

clfs = {
        'K_neighbor': KNeighborsClassifier(n_neighbors=10, weights='distance', n_jobs=-1),
        'naive_bayes': MultinomialNB(),
        'svm': SVC(),
        'LogisticRegression': LogisticRegression(max_iter=3000, n_jobs=-1)
        }

# 读入数据，测试图像为28*28大小的，转化为长度为784的一维向量，总训练集大小为60000
# 总测试集大小为10000
train_image, train_label = load_mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
test_image, test_label = load_mnist("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
# 使用min-max标准化
train_image = [i/255.0 for i in train_image]
test_image = [i/255.0 for i in test_image]


# 为提升运行速度以进行测试，可在这里将训练集与测试集的大小进一步缩小
train_image = train_image[0:10000]
train_label = train_label[0:10000]
test_image = test_image[0:2000]
test_label = test_label[0:2000]



print('提示：由于数据集过大，部分模型的生成时间和预测时间较长，完整的数据集初次运行预计二十分钟左右')
for clf_key in clfs.keys():
    print('\nthe classifier is:', clf_key)
    clf = clfs[clf_key]

    predicted = []
    elapsed = 0
    # 训练模型的同时将模型保存，避免重复训练浪费时间
    if os.path.exists(clf_key+".pkl"):
        clf = load(clf_key+".pkl")
        predicted = clf.predict(test_image)
    else:
        begin = time.perf_counter()
        model = clf.fit(train_image, train_label.ravel())
        elapsed = time.perf_counter() - begin
        dump(model, clf_key + ".pkl")
        print("Model generation successful")
        predicted = clf.predict(test_image)

    result = [1 if y1 == y2 else 0 for y1, y2 in zip(test_label, predicted)]
    print('the accuracy is:', sum(result)/len(result))
    print('the elapsed is:', elapsed)

    gc.collect()

