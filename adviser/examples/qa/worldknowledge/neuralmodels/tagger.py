from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

TAGS = ['B', 'I', 'O', '-']


class Tagger(nn.Module):
    """Neural network for predicting the topic entities.

    The model uses a question encoder and classifies each token using the BIO tag set.

    Attributes:
        hidden_dim (int): Size of the Bi_LSTM's hidden layer
        diminisher (nn.Module): Fine-tuning embedding layer, good for reducing Bi-LSTM's size
        lstm (nn.Module): Bi-LSTM
        hidden2label (nn.Module): Output layer
    """

    def __init__(self, emb_dim: int, hidden_dim: int):
        """Initialises all required elements of the neural network.

        Args:
            emb_dim: Output size of the fine-tuning embedding layer
            hidden_dim: Hidden layer size of the Bi-LSTM
        """
        super(Tagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.diminisher = nn.Linear(768, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim*2, len(TAGS))

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        """Application of the neural network on a given input question.

        Args:
            embeds: Tensor containing the embeddings of shape |Token| x |Batch| x |Embedding Size|

        Returns:
            Probabilities of each BIO tag for all tokens
        """
        embeds = self.diminisher(embeds)
        lstm_out, _ = self.lstm(embeds)
        label_space = self.hidden2label(lstm_out[1:])
        label_scores = F.log_softmax(label_space, dim=2)
        return label_scores


def extract_entities(tokens: List[str], tag_idxs: List[int]) -> List[List[str]]:
    """Extracts entities using the predicted BIO tags for each token

    Arguments:
        tokens: question's tokens
        tag_idxs: index of the BIO tag for each token in the question

    Returns:
        List of entities, i.e. list of connected tokens
    """
    entities = []
    curr_entity = []
    tags = [TAGS[tag_idx] for tag_idx in tag_idxs]
    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag == 'I':
            curr_entity.append(token)
            continue
        else:
            if curr_entity:
                entities.append(curr_entity)
                curr_entity = []
            if tag == 'B':
                curr_entity.append(token)
    if curr_entity:
        entities.append(curr_entity)
    return entities
