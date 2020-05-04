import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    """Neural network for predicting the relation's direction (outgoing or incoming).

    The model uses a question encoder to classify the question as one of the two classes
    "outgoing" or "incoming".

    Attributes:
        hidden_dim (int): Size of the Bi_LSTM's hidden layer
        out_dim (int): Size of the output layer (here: 2)
        diminisher (nn.Module): Fine-tuning embedding layer, good for reducing Bi-LSTM's size
        lstm (nn.Module): Bi-LSTM for encoding a question
        hidden2tag (nn.Module): Output layer
    """

    def __init__(self, emb_dim: int, lstm_out_dim: int, num_classes: int):
        """Initialises all required elements of the neural network.

        Args:
            emb_dim: Output size of the fine-tuning embedding layer
            lstm_out_dim: Output size of the Bi-LSTM
            num_classes: Size of the output layer (in this context, always 2)
        """
        super(Classifier, self).__init__()
        self.hidden_dim = lstm_out_dim
        self.out_dim = num_classes

        self.diminisher = nn.Linear(768, emb_dim)
        self.lstm = nn.LSTM(emb_dim, lstm_out_dim, bidirectional=True)
        self.hidden2tag = nn.Linear(lstm_out_dim*2, self.out_dim)

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        """Application of the neural network on a given input question.

        Args:
            embeds: Tensor containing the embeddings of shape |Token| x |Batch| x |Embedding Size|

        Returns:
            Probabilities of the two classes "incoming" and "outgoing"
        """
        embeds = self.diminisher(embeds)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out[0])
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
