import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import weight_norm
from torch.autograd import Variable

from .model import Stacked_VAE, BiLSTM

def train_vae(train_loader: DataLoader, test_loader: DataLoader, feats_train: torch.Tensor, symbol_id: str, n_hidden: int, n_latent: int, n_layers: int, learning_rate: float, n_epoch: int, main_dir: str, is_train: bool = True) -> Stacked_VAE:
    """
    Stacked VAE 모델을 훈련하는 함수

    :param DataLoader train_loader: 학습 데이터를 로드하는 데이터 로더 객체
    :param DataLoader test_loader: 검증 데이터를 로드하는 데이터 로더 객체
    :param torch.Tensor feats_train: 훈련에 사용되는 입력 데이터
    :param str symbol_id: 종목 코드
    :param int n_hidden: hidden layers 수
    :param int n_latent: 잠재 공간의 수
    :param int n_layers: Stacked VAE 모델의 레이어 수
    :param float learning_rate: 학습률
    :param int n_epoch: 에포크 수
    :param str main_dir: 절대 경로
    :param bool is_train: 훈련 모드 여부
    :return: 훈련된 Stacked VAE 모델 객체
    :rtype: Stacked_VAE
    """
    
    model = Stacked_VAE(n_in = feats_train.shape[1], n_hidden = n_hidden, n_latent = n_latent, n_layers = n_layers)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    loss_function = nn.MSELoss()

    save_dir = 'my_path/vae'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    if is_train:
        training_loss = []
        validation_loss = []
        best_val_loss = float('inf')
        best_epoch = -1
        for epoch in range(n_epoch):
            epoch_loss = 0
            epoch_val_loss = 0
    
            model.train()
            for data in train_loader:
                optimizer.zero_grad()
                outputs, weight_loss = model(data)
                loss = loss_function(outputs, data)
                loss = loss + torch.mean(weight_loss)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
    
            model.eval()
            with torch.no_grad():
                for data in test_loader:
                    outputs, weight_loss = model(data)
                    loss = loss_function(outputs, data)
                    loss = loss + torch.mean(weight_loss)
                    epoch_val_loss += loss.item()
    
            epoch_loss /= len(train_loader)
            epoch_val_loss /= len(test_loader)
    
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch
                
                # 모델 훈련 파일 저장                
                try:
                    best_checkpoint_path = os.path.join(save_dir, f'{symbol_id}_vae.pt')
                    #best_checkpoint_path = f'my_path/vae/{symbol_id}_vae.pt'
                    
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss,
                        'validation_loss': epoch_val_loss,
                    }, best_checkpoint_path)
                except Exception as e:
                    print(f'Symbol: {symbol_id} Error: {str(e)}')
            
            training_loss.append(epoch_loss)
            validation_loss.append(epoch_val_loss)
    
            if epoch % 50 == 0:
                print('Epoch {}, Training loss {:.4f}, Validation loss {:.4f}'.format(epoch, epoch_loss, epoch_val_loss))

    # 훈련된 모델 불러오기
    else:
        try:
            #best_checkpoint_path = os.path.join(save_dir, f'{symbol_id}_vae.pt')
            #checkpoint = torch.load(best_checkpoint_path)

            best_checkpoint_path = f'my_path/vae/{symbol_id}_vae.pt'
            checkpoint = torch.load(os.path.join(main_dir, best_checkpoint_path))
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_loss = checkpoint['loss']
        except Exception as e:
            print(f'Symbol: {symbol_id} Error: {str(e)}')

    return model

def train_bilstm(train_loader: DataLoader, symbol_id: str, input_size: int, hidden_size: int, n_layers: int, dropout: float, output_size: int, num_epochs: int, learning_rate: float, device: str, len_sequence: bool = True) -> BiLSTM:
    """
    BiLSTM 모델을 훈련하는 함수

    :param DataLoader train_loader: 학습 데이터를 로드하는 데이터 로더 객체
    :param str symbol_id: 종목 코드
    :param int input_size: 입력 데이터의 특성 크기
    :param int hidden_size: hidden state 크기
    :param int n_layers: BiLSTM 모델의 레이어 수
    :param float dropout: 드롭아웃 확률
    :param int output_size: 출력 데이터의 크기
    :param int num_epochs: 에포크 수
    :param float learning_rate: 학습률
    :param str device: 디바이스 예)'cuda' or 'cpu'
    :param bool len_sequence: 시퀸스 길이를 고려할지 여부 (기본값: True)
    :return: 훈련된 BiLSTM 모델 객체
    :rtype: BiLSTM
    """
    model = BiLSTM(input_size, hidden_size, n_layers, dropout, output_size, device)
    #model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    save_dir = 'my_path/bilstm'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    
    if len_sequence:
        for epoch in range(num_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.unsqueeze(2)
                targets = targets.unsqueeze(2)
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets.squeeze())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch+1)%10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.5f}')

    else:
        for epoch in range(num_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets.squeeze())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
            if (epoch+1)%10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.5f}')

    # 모델 훈련 파일 저장
    try:
        file_path = os.path.join(save_dir, f'{symbol_id}_bilstm.pt')
        # file_path = f'my_path/bilstm/{symbol_id}_bilstm.pt'
        
        torch.save({
            'model_state_dict': model.state_dict(),
        }, file_path)
        
    except Exception as e:
        print(f'Symbol: {symbol_id} Error: {str(e)}')

    return model

def bilstm_inference(sequences: torch.Tensor, symbol_id: str, input_size: int, hidden_size: int, n_layers: int, dropout: float, output_size: int, window_size: int, device: str, main_dir: str, len_sequence: bool = True) -> torch.Tensor:
    """
    훈련된 BiLSTM 모델을 통해 추론하는 함수

    :param torch.Tensor: 입력 시퀸스 데이터
    :param str symbol_id: 종목 코드
    :param int input_size: 입력 데이터의 특성 크기
    :param int hidden_size: hidden state 크기
    :param int n_layers: BiLSTM 모델의 레이어 수
    :param float dropout: 드롭아웃 확률
    :param int output_size: 출력 데이터의 크기
    :param int window_size: 추론에 사용할 시퀸스 윈도우 크기
    :param str device: 디바이스 예)'cuda' or 'cpu'
    :param str main_dir: 절대 경로
    :param bool len_sequence: 시퀸스 길이를 고려할지 여부 (기본값: True)
    :return: 추론 결과로서의 예측된 출력 값
    :rtype: torch.Tensor
    """
    model = BiLSTM(input_size, hidden_size, n_layers, dropout, output_size, device)
    #model.to(device)
    
    save_dir = 'my_path/bilstm'

    # 훈련된 모델 불러오기
    try:
        #file_path = os.path.join(save_dir, f'{symbol_id}_bilstm.pt')
        #checkpoint = torch.load(file_path)

        file_path = f'my_path/bilstm/{symbol_id}_bilstm.pt'
        checkpoint = torch.load(os.path.join(main_dir, file_path))

        model.load_state_dict(checkpoint['model_state_dict'])
            
    except Exception as e:
        print(f'Symbol: {symbol_id} Error: {str(e)}')

    if len_sequence:
        model.eval()
        with torch.no_grad():
            last_sequences = sequences[-window_size: ]
            last_sequences = last_sequences.to(device)
            train_outputs = model(last_sequences.unsqueeze(2))
    else:
        model.eval()
        with torch.no_grad():
            last_sequences = sequences[-window_size: ]
            last_sequences = last_sequences.to(device)
            train_outputs = model(last_sequences)

    return train_outputs