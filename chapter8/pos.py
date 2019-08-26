from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor

torch.manual_seed(1)

class PosDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}
        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field
        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                pairs = line.strip().split()
                sentence, tags = zip(*(pair.split("###") for pair in pairs))
                yield self.text_to_instance([Token(word) for word in sentence], tags)


class LstmTagger(Model):
    def __init__(self,word_embeddings: TextFieldEmbedder,encoder: Seq2SeqEncoder,vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()

    def forward(self,sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)
        return output

    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


reader = PosDatasetReader()
train_dataset = reader.read(cached_path('./train.txt'))
validation_dataset = reader.read(cached_path('./validate.txt'))
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

EMBEDDING_DIM = 16
HIDDEN_DIM = 16

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
model = LstmTagger(word_embeddings, lstm, vocab)
if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    cuda_device = -1

optimizer = optim.SGD(model.parameters(), lr=0.1)

iterator = BucketIterator(batch_size=2, sorting_keys=[("sentence", "num_tokens")])
iterator.index_with(vocab)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=10,
                  num_epochs=1000,
                  cuda_device=cuda_device)

trainer.train()


predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
tag_logits = predictor.predict("骑着 狗  出去 逛 街")['tag_logits']
tag_ids = np.argmax(tag_logits, axis=-1)
print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])


with open("/tmp/model.th", 'wb') as f:
    torch.save(model.state_dict(), f)
vocab.save_to_files("/tmp/vocabulary")
vocab2 = Vocabulary.from_files("/tmp/vocabulary")
model2 = LstmTagger(word_embeddings, lstm, vocab2)
with open("/tmp/model.th", 'rb') as f:
    model2.load_state_dict(torch.load(f))
if cuda_device > -1:
    model2.cuda(cuda_device)
predictor2 = SentenceTaggerPredictor(model2, dataset_reader=reader)
tag_logits2 = predictor2.predict("骑着 狗  出去 逛 街")['tag_logits']
np.testing.assert_array_almost_equal(tag_logits2, tag_logits)
