
class Dataset(object):
    def __init__(self,name,root,train_file=None,val_file=None,test_file=None):
        self.name=name # 数据集名字
        self.root=root
        self.train_file=train_file
        self.val_file=val_file
        self.test_file=test_file


    # def load():


