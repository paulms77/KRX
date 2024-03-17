import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Stacked_VAE(nn.Module):
    def __init__(self, n_in: int = 784, n_hidden: int = 400, n_latent: int = 2, n_layers: int = 1):
        """
        Stacked Variational Autoencoder (VAE)를 초기화 하기 위함

        :param int n_in: input features의 수
        :param int n_hidden: 신경망 레이어의 hidden units 수
        :param int n_latent: 잠재 변수의 수
        :param int n_layers: 인코더 및 디코더 네트워크의 레이어 수
        """
        super(Stacked_VAE, self).__init__()
        self.soft_zero = 1e-10
        self.n_latent = n_latent
        self.n_in = n_in
        self.mu = None

        # Encoder layers
        encoder_layers = []
        for i in range(n_layers):
            in_features = n_in if i == 0 else n_hidden
            out_features = n_latent * 2 if i == n_layers - 1 else n_hidden
            encoder_layers.append(nn.Linear(in_features, out_features))
            if i < n_layers - 1:
                encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers
        decoder_layers = []
        for i in range(n_layers):
            in_features = n_latent if i == 0 else n_hidden
            out_features = n_in if i == n_layers - 1 else n_hidden
            decoder_layers.append(nn.Linear(in_features, out_features))
            if i < n_layers - 1:
                decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def encoded(self, x: torch.Tensor) -> tuple:
        """
        입력 데이터를 인코딩

        :param torch.Tensor x: 입력 데이터
        :return: 잠재 공간의 평균(mu)과 로그 분산(lv)을 포함하는 튜플
        :rtype: tuple
        """
        h = self.encoder(x)
        mu_lv = torch.split(h, self.n_latent, dim=1)
        return mu_lv[0], mu_lv[1]

    def decoded(self, z: torch.Tensor) -> torch.Tensor:
        """
        잠재 공간 표현을 디코딩

        :param torch.Tensor z: 잠재 공간 표현
        :return: 디코딩된 출력 값
        :rtype: torch.Tensor
        """
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:
        """
        잠재 공간을 다시 매개변수화

        :param torch.Tensor mu: 잠재 공간의 평균
        :param torch.Tensor lv: 잠재 공간의 로그 분산
        :return: 다시 매개변수화된 잠재공간
        :rtype: torch.Tensor
        """
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * lv) * eps
        return z

    def forward(self, x: torch.Tensor) -> tuple:
        """
        VAE를 통과하여 전달시키는 함수

        :param torch.Tensor x: 입력 데이터
        :return: 재구성된 출력(y)과 가중치 감소(weight loss)를 포함하는 튜플
        :rtype: tuple
        """
        mu, lv = self.encoded(x)
        z = self.reparameterize(mu, lv)
        y = self.decoded(z)

        # Compute the loss components
        KL = 0.5 * torch.sum(1 + lv - mu * mu - lv.exp(), dim=1)
        logloss = torch.sum(x * torch.log(y + self.soft_zero) + (1 - x) * torch.log(1 - y + self.soft_zero), dim=1)
        loss = -logloss - KL

        weight_loss = loss.unsqueeze(1)

        return y, weight_loss

class BiLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 100, n_layers: int = 5, 
                 dropout: float = 0.5, output_size: int = 15, device=device):
      """
      BiLSTM(양방향 LSTM) 모델을 초기화

      :param int input_size: input features의 수
      :param int hidden_size: LSTM 레이어의 hidden units 수      
      :param int n_layers: LSTM 레이어 수
      :param float dropout: Dropout 확률.
      :param int output_size: output units의 수
      """
      super(BiLSTM, self).__init__()
      self.input_size = input_size
      self.hidden_size = hidden_size
      self.n_layers = n_layers
      self.dropout = dropout
      self.output_size = output_size
      self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = n_layers, bidirectional = True, dropout = dropout, batch_first = True)
      self.fc = nn.Linear(hidden_size * 2, output_size)
      self.relu = nn.ReLU()
      self.device = device

    def forward(self, x: torch.Tensor)->torch.Tensor:
      """
      BiLSTM을 통과하여 전달시키는 함수

      :param torch.Tensor x: 입력 데이터
      :return: 출력 예측 값
      :rtype: torch.Tensor
      """
      h0 = Variable(torch.zeros(self.n_layers * 2, x.size(0), self.hidden_size)).to(device)
      c0 = Variable(torch.zeros(self.n_layers * 2, x.size(0), self.hidden_size)).to(device)
      output, (hn, cn) = self.lstm(x, (h0, c0))
      out = self.fc(output[:, -self.output_size, :])
      out = self.relu(out)
      return out