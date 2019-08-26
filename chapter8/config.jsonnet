// jsonnet allows local variables like this
local embedding_dim = 16;
local hidden_dim = 16;
local num_epochs = 1000;
local patience = 10;
local batch_size = 2;
local learning_rate = 0.1;

{
    "train_data_path": './train.txt',
    "validation_data_path": './validate.txt',
    "dataset_reader": {
        "type": "pos-dataset-reader"
    },
    "model": {
        "type": "lstm-tagger",
        "word_embeddings": {
            // but that's the default TextFieldEmbedder, so doing so
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": embedding_dim
                }
            }
        },
        "encoder": {
            "type": "gru",
            "input_size": embedding_dim,
            "hidden_size": hidden_dim
        }
    },
    "iterator": {
        "type": "bucket",
        "batch_size": batch_size,
        "sorting_keys": [["sentence", "num_tokens"]]
    },
    "trainer": {
        "num_epochs": num_epochs,
        "optimizer": {
            "type": "sgd",
            "lr": learning_rate
        },
        "patience": patience
    }
}