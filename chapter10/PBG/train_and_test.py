import os
import attr

from torchbiggraph.converters.import_from_tsv import convert_input_data
from torchbiggraph.config import parse_config
from torchbiggraph.train import train
from torchbiggraph.eval import do_eval


DATA_DIR = "datas/soc-LiveJournal"
CONFIG_PATH = "config.py"
FILENAMES = {
    'train': 'train.txt',
    'test': 'test.txt',
}

def convert_path(fname):
    """
    辅助方法，用于将真实文件绝对路径文件名后缀使用_partitioned替换
    """
    basename, _ = os.path.splitext(fname)
    out_dir = basename + '_partitioned'
    return out_dir

edge_paths = [os.path.join(DATA_DIR, name) for name in FILENAMES.values()]
train_paths = [convert_path(os.path.join(DATA_DIR, FILENAMES['train']))]
eval_paths = [convert_path(os.path.join(DATA_DIR, FILENAMES['test']))]

def run_train_eval():
    #将数据转为PBG可读的分区文件
    convert_input_data(CONFIG_PATH,edge_paths,lhs_col=0,rhs_col=1,rel_col=None)
    #解析配置
    config = parse_config(CONFIG_PATH)
    #训练配置，已分区的train_paths路径替换配置文件中的edge_paths
    train_config = attr.evolve(config, edge_paths=train_paths)
    #传入训练配置文件开始训练
    train(train_config)
    #测试配置，已分区的eval_paths路径替换配置文件中的edge_paths
    eval_config = attr.evolve(config, edge_paths=eval_paths)
    #开始验证
    do_eval(eval_config)

if __name__ == "__main__":
    run_train_eval()