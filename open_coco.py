import pickle

with open('/Users/crystal/Desktop/python-workspace/CogWorks2019/resnet18_features.pkl', 'rb') as file:
    coco = pickle.load(file)
    print(list(i for i in coco.values())[0:20])