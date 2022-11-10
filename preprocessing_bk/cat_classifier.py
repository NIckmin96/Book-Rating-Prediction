import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

'''
book.csv의 category 결측치를 채우기 위해 category를 제외한 다른 변수들을 lantent vector embedding한 뒤,
MLP layer를 거쳐 category에 대해 예측하는 모델입니다.

run을 하게 되면, 예측된 결측치 값이 nx1 형태의 dataframe으로 저장되고, 그 값은 category feature의 unique한 값이
indexing 되어 있는 값이므로 다시 unique한 값으로 치환해주어야 합니다.

'''


def load_data() : #books only
    return pd.read_csv('../data/preprocessed/books_lang.csv')

def var_to_index(df) : 

    isbn2idx = {v:k for k,v in enumerate(df['isbn'].unique())}
    author2idx = {v:k for k,v in enumerate(df['book_author'].unique())}
    year2idx = {v:k for k,v in enumerate(df['year_of_publication'].unique())}
    publisher2idx = {v:k for k,v in enumerate(df['publisher'].unique())}
    lang2idx = {v:k for k,v in enumerate(df['language'].unique())}
    cat2idx = {v:k for k,v in enumerate(df['category'].unique())}
    
    tmp = pd.DataFrame()
    tmp['book_author'] = df['book_author'].map(author2idx)
    tmp['year_of_publication'] = df['year_of_publication'].map(year2idx)
    tmp['publisher'] = df['publisher'].map(publisher2idx)
    tmp['language'] = df['language'].map(lang2idx)
    tmp['isbn'] = df['isbn'].map(isbn2idx)
    tmp['category'] = df['category'].map(cat2idx)
    
    return tmp

class MLP(nn.Module) : # Embedding 후 MLP
    def __init__(self, input_dim, output_dim, embedding_dim = 16) : 
        super().__init__()

        self.output_dim = output_dim
        self.input_dim = input_dim # embedding dim * len(num_features)
        
        self.mlp = nn.Sequential(
                    nn.Linear(self.input_dim, embedding_dim), # Layer 1
                    nn.BatchNorm1d(embedding_dim),
                    nn.ReLU(),
                    nn.Linear(16,128), # Layer 2
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Linear(128,512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Linear(512, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Linear(1024, self.output_dim)
        )
        
    def forward(self, x) : 
        return self.mlp(x)
        

class CategoryClassifier(nn.Module) : 
    def __init__(self, input, embed_dim=16) : 
        super().__init__()
        
        self.feature_dims = np.array([len(input[col].unique()) for col in input.columns])
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(200000, self.embed_dim) # 피처 임베딩 -> shape : (n x 16)
        self.offsets = np.array([0,*np.cumsum(self.feature_dims)[:-1]], dtype=np.compat.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        
        self.mlp_dim = len(self.feature_dims) * self.embed_dim # embedding dim * len(features)
        self.mlp = MLP(self.mlp_dim, 4292) # MLP (input dim, output dim)
        
        
    def forward(self, x) : 
        # x.shape = (128,5) : (batchsize, num_features)
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        x = self.embedding(x) # 128x5x16
        x = x.view(-1,self.mlp_dim) # 128x80
        output = self.mlp(x)
        
        return output
        
#############################################################################################################################

class TrainTest() : 
    def __init__(self, mode = None, target_feature = 'category') : 
        
        books = load_data()
        not_null = books.category.notnull()
        is_null = books.category.isnull()

        #feature dict
        train = books[not_null]
        test = books[is_null]

        train_df = var_to_index(train)
        test_df = var_to_index(test)
        test_df = test_df[test_df.columns.difference([target_feature])]

        X_train, X_valid, y_train, y_valid = train_test_split(train_df[train_df.columns.difference([target_feature])], train_df[target_feature], test_size = 0.3,
                                                            random_state=42, shuffle = True)
        
        
        self.train_ds = TensorDataset(torch.LongTensor(X_train.values), torch.LongTensor(y_train.values))
        self.valid_ds = TensorDataset(torch.LongTensor(X_valid.values), torch.LongTensor(y_valid.values))
        self.test_ds = TensorDataset(torch.LongTensor(test_df.values))

        self.train_dl = DataLoader(self.train_ds, batch_size = 128, shuffle = True)
        self.valid_dl = DataLoader(self.valid_ds, batch_size = 128, shuffle = True)
        self.test_dl = DataLoader(self.test_ds, batch_size = 128, shuffle = False)
        
        self.epoch = 100
        self.lr = 1e-2
        self.log_interval = 100
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        if mode == 'train' : 
            self.model = CategoryClassifier(X_train).to(self.device)
        else : 
            self.model = CategoryClassifier(test_df).to(self.device)
        self.optimizer = torch.optim.Adam(params = self.model.parameters(), lr = self.lr, amsgrad = True)
        
        
    def train(self) : 
        for epoch in range(self.epoch) : 
            self.model.train()
            total_loss = 0
            tk0 = tqdm(self.train_dl, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target)
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            print('epoch:', epoch, 'train loss:', total_loss.item())
            acc = self.predict_train()
            print('epoch:', epoch, f'validation acc:{acc.item():0.2f}%')
            
    def predict_train(self):
        self.model.eval()
        ans, predicts = torch.Tensor().to(self.device), torch.Tensor().to(self.device)
        with torch.no_grad():
            for fields, target in tqdm(self.valid_dl, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                ans = torch.cat([ans, target], dim = 0)
                predicts = torch.cat([predicts, y], dim = 0)
                
                acc = (torch.argmax(predicts, 1) == ans)
                acc = acc.float().mean()*100
            
        return acc
    
    def inference(self):
        self.model.eval()
        ans, predicts = torch.Tensor().to(self.device), torch.Tensor().to(self.device)
        with torch.no_grad():
            for fields in tqdm(self.test_dl, smoothing=0, mininterval=1.0):
                # print(fields[0])
                fields = fields[0].to(self.device)
                y = self.model(fields)
                # ans = torch.cat([ans, target], dim = 0)
                predicts = torch.cat([predicts, y], dim = 0)
                
                infer = torch.argmax(predicts, 1)
                
            
        return infer

import os

def run(target_feature) : 
    if os.path.exists(f'cat_model_{target_feature}.pth') : 
        
    ####################################################################

        tt = TrainTest(target_feature=target_feature)
        tt.model.load_state_dict(torch.load(f'cat_model_{target_feature}.pth'))
        model = tt.model
        
        res = tt.inference()
        res = res.to('cpu')
        res_df = pd.DataFrame(res, columns = ['predicted_cat'])
        res_df.to_csv(f'cat_predict_{target_feature}.csv', index = False)
        
    else : 
    ####################################################################
        run = TrainTest('train', target_feature=target_feature)
        run.train()

        torch.save(run.model.state_dict(), f'cat_model_{target_feature}.pth')

        tt = TrainTest(target_feature=target_feature)
        tt.model.load_state_dict(torch.load(f'cat_model{target_feature}.pth'))
        model = tt.model
        
        res = tt.inference()
        res = res.to('cpu')
        res_df = pd.DataFrame(res, columns = ['predicted_cat'])
        res_df.to_csv(f'cat_predict_{target_feature}.csv', index = False)


# run('category')