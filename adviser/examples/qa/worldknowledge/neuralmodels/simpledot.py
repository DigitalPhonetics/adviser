import sys, os, json, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


class SimpleDot(nn.Module):
    """Neural network for predicting the relation of a question.

    The simple dot approach compares a question with each possible relation candidate by
    taking the (simple) dot product between the encoded question and every encoded relation.

    Attributes:
        softmax (bool): whether or not the scores should be converted to probabilities
        hidden (int): size of the hidden layer
        relations_tensor (torch.autograd.Variable): embeddings for all relation descriptions
        diminisher (nn.Module): Fine-tuning embedding layer, good for reducing Bi-LSTMs' size
        lstm_question (nn.Module): Bi-LSTM for encoding a question
        lstm_relation (nn.Module): Bi-LSTM for encoding a relation
    """

    def __init__(self, emb_dim: int, hidden_dim: int, softmax: bool = True):
        """Initialises all required elements of the neural network.

        Args:
            emb_dim: Output size of the fine-tuning embedding layer
            hidden_dim: Output size of the Bi-LSTM
            softmax: Whether or not a softmax is applied on the output layer
        """
        super(SimpleDot, self).__init__()
        self.softmax = softmax
        self.hidden = hidden_dim

        self.relations_tensor = self._initialise_relations_tensor()
        self.diminisher = nn.Linear(768, emb_dim)
        self.lstm_question = nn.LSTM(emb_dim, hidden_dim, bidirectional=True)
        self.lstm_relation = nn.LSTM(emb_dim, hidden_dim, bidirectional=True)

    def _initialise_relations_tensor(self, max_rel_len: int = 30) -> torch.autograd.Variable:
        """Creates a tensor containing word embeddings of all relation descriptions.

        To be processable, the tensor is transformed to the shape
        |token| x |relation| x |embeddings|.

        Keyword Arguments:
            max_rel_len: maximum number of tokens in the relation descriptions

        Returns:
            A tensor containing word embeddings of all relation descriptions
        """
        relations_list = []
        tokens = []
        with open(os.path.join(get_root_dir(), 'resources', 'ontologies', 'qa', 'csqa_relation_embeddings.bin'), 'rb') as f:
            rels = pickle.load(f)
            relations_list = [emb[1] for emb in rels]
            tokens = [emb[0] for emb in rels]

        relations = []  # R x T_R x E
        for relation in relations_list:
            relation.extend([np.zeros(768, dtype='float32')] * (max_rel_len- len(relation)))
            relations.append(relation[:max_rel_len])
        # T_R x R x E
        return torch.autograd.Variable(torch.Tensor(relations).transpose(0,1), requires_grad=False)

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        """Application of the neural network on a given input question.

        Args:
            embeds: Tensor containing the embeddings of shape |Token| x |Batch| x |Embedding Size|

        Returns:
            Probabilities for the relation classes
        """
        embeds = self.diminisher(embeds)
        relations_embeds = self.diminisher(self.relations_tensor)

        question_out, _ = self.lstm_question(embeds)  # T_Q x B x H
        relation_out, _ = self.lstm_relation(relations_embeds)  # T_R x R x H
        last_question_out = question_out[0][:,self.hidden:]  # B x H
        last_relation_out = relation_out[0][:,self.hidden:]  # R x H

        # relation prediction
        rel_scores = torch.matmul(last_question_out, last_relation_out.transpose(0,1))
        if self.softmax:
            rel_scores = F.log_softmax(rel_scores, dim=1)
        return rel_scores
