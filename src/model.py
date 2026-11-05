# 학습 때 쓴 DKT 모델 클래스를 서빙용으로 옮기고, 마지막 시점의 전체 문항 정답 확률을 내는 함수를 추가

import torch
import torch.nn as nn


class DKT(nn.Module):
    def __init__(self, nq, h=128):
        super().__init__()
        self.emb = nn.Embedding(2 * nq + 1, h, padding_idx=0)
        self.rnn = nn.LSTM(h, h, batch_first=True)
        self.out = nn.Linear(h, nq + 1)

    def predict_all(self, q, r):
        tok = torch.where(q > 0, (q - 1) * 2 + r + 1, torch.zeros_like(q))
        h, _ = self.rnn(self.emb(tok))
        return torch.sigmoid(self.out(h[:, -1]))