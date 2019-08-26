#实体节点路径
entities_base = 'datas/soc-LiveJournal'
#获取PBG的配置
def get_torchbiggraph_config():
    config = dict(
        #实体路径及模型检查点路径
        entity_path=entities_base,
        edge_paths=[],
        checkpoint_path='model/demo',
        #图结构及分区数
        entities={
            'user_id': {'num_partitions': 2},
        },
        #关系类型及左右实体节点
        relations=[{
            'name': 'follow',
            'lhs': 'user_id',
            'rhs': 'user_id',
            'operator': 'none',
        }],

        #嵌入维度
        dimension=520,
        global_emb=False,

        #训练10个epoch
        num_epochs=10,
        #学习率
        lr=0.001,
        # Misc
        hogwild_delay=2,
    )

    return config