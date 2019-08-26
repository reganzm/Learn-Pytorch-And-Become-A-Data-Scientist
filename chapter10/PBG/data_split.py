import os
import random


DATA_PATH = "datas/soc-LiveJournal/soc-LiveJournal-sample.txt"
FILENAMES = {
    'train': 'train.txt',
    'test': 'test.txt',
}
TRAIN_FRACTION = 0.8
TEST_FRACTION  = 0.2

def random_split_file(fpath):
    root = os.path.dirname(fpath)

    output_paths = [
        os.path.join(root, FILENAMES['train']),
        os.path.join(root, FILENAMES['test']),
    ]
    if all(os.path.exists(path) for path in output_paths):
        print("训练及测试文件已经存在,不再生成测试训练文件...")
        return

    #读取数据，并随机打乱，划分出训练数据集测试数据
    train_file = os.path.join(root, FILENAMES['train'])
    test_file = os.path.join(root, FILENAMES['test'])
    #读取数据
    with open(fpath, "rt") as in_tf:
        lines = in_tf.readlines()

    #调过soc-LiveJournal.txt文件头部的4行注解
    lines = lines[4:]
    #shuffle打乱数据
    random.shuffle(lines)
    split_len = int(len(lines) * TRAIN_FRACTION)
    #写入测试及训练文件
    with open(train_file, "wt") as out_tf_train:
        for line in lines[:split_len]:
            out_tf_train.write(line)

    with open(test_file, "wt") as out_tf_test:
        for line in lines[split_len:]:
            out_tf_test.write(line)


if __name__=="__main__":
    random_split_file(DATA_PATH)