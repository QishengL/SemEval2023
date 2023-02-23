import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
import sys
from transformers import AutoTokenizer, AutoModel
from sklearn import metrics





#data processing
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
train=pd.read_csv('all_train.csv')
train=train.dropna()
dev=pd.read_csv('all_dev.csv')
dev=dev.dropna()


train['target_list'] = train[['Economic',
'Capacity_and_resources',
'Morality',
'Fairness_and_equality',
'Legality_Constitutionality_and_jurisprudence',
'Policy_prescription_and_evaluation',
'Crime_and_punishment',
'Security_and_defense',
'Health_and_safety',
'Quality_of_life',
'Cultural_identity',
'Public_opinion',
'Political',
'External_regulation_and_reputation']].values.tolist()



dev['target_list'] = dev[['Economic',
'Capacity_and_resources',
'Morality',
'Fairness_and_equality',
'Legality_Constitutionality_and_jurisprudence',
'Policy_prescription_and_evaluation',
'Crime_and_punishment',
'Security_and_defense',
'Health_and_safety',
'Quality_of_life',
'Cultural_identity',
'Public_opinion',
'Political',
'External_regulation_and_reputation']].values.tolist()



class CustomDataset():

    def __init__(self, dataframe, tokenizer):
        self.tokenizer = tokenizer

        self.title = list(dataframe['title'])
        self.content = list(dataframe['content'])
        self.targets = list(dataframe['target_list'])


    def __len__(self):
        return len(self.content)

    def __getitem__(self, index):

        inputs1 = self.tokenizer.encode_plus(
            self.title[index],
            None,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        inputs2 = self.tokenizer.encode_plus(
            self.content[index],
            None,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        ids_tilte = inputs1['input_ids']
        mask_title = inputs1['attention_mask']
        token_type_ids_title = inputs1["token_type_ids"]
        
        ids_content = inputs2['input_ids']
        mask_content = inputs2['attention_mask']
        token_type_ids_content = inputs2["token_type_ids"]


        return torch.tensor(ids_tilte, dtype=torch.long),torch.tensor(mask_title, dtype=torch.long),torch.tensor(ids_content, dtype=torch.long),torch.tensor(mask_content, dtype=torch.long),torch.tensor(self.targets[index], dtype=torch.float)
        


training_set = CustomDataset(train, tokenizer)
validation_set = CustomDataset(dev, tokenizer)

trainloader = DataLoader(training_set, batch_size=4,shuffle=True)
devloader = DataLoader(validation_set, batch_size=4,shuffle=False)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, targets):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))


        batch_size = features.shape[0]

        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        #Caculate weight for hard negative examples
        weight_mat = torch.eye(batch_size, dtype=torch.float32)
        for i in range(len(targets)):
            for j in range(len(targets)):
                weight=torch.sum(torch.abs(targets[i]-targets[j]))
                if(weight==0):
                    continue
                weight_mat[i][j]=weight
        weight_mat=weight_mat.repeat(anchor_count, contrast_count)
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        weight_mat=weight_mat.to(device)
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        exp_logits=exp_logits*weight_mat
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


#model
class MyModel(nn.Module):
    def __init__(self,feat_dim, num_classes):
        super(MyModel, self).__init__()
        
        self.model = AutoModel.from_pretrained('xlm-roberta-large') 

        self.head1 = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(1024*2, 1024*2),
                nn.LayerNorm(1024*2),
                nn.ReLU(inplace=True),
                nn.Linear(1024*2, feat_dim),
                nn.LayerNorm(feat_dim)
        )
        
        self.head2 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(1024*2, num_classes)
        )
    def forward(self, title,content):

        x1 = self.model(input_ids=title['input_ids'],attention_mask=title['attention_mask'])[0][:,0,:]
        x2 = self.model(input_ids=content['input_ids'],attention_mask=content['attention_mask'])[0][:,0,:]
        x=torch.cat((x1,x2),1)
        feat = F.normalize(self.head1(x), dim=1)
        classes = self.head2(x)
        
        return feat,classes

model = MyModel(128,14)


#hyperparameter
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=1e-06)
criterion1 = SupConLoss(temperature=0.07)



#training function
def training(train_loader, model,criterion, optimizer):
    epoch_loss = 0   
    model.train()


    for idx, (ids1,mask1,ids2,mask2,targets) in enumerate(train_loader):
        
        
        

        ids1=ids1.to(device)
        mask1=mask1.to(device)
        ids2=ids2.to(device)
        mask2=mask2.to(device)
        targets=targets.to(device)
        batch1={}
        batch1['input_ids']=ids1
        batch1['attention_mask']=mask1
        batch2={}
        batch2['input_ids']=ids2
        batch2['attention_mask']=mask2
        fet1,outputs = model(batch1,batch2)
        fet2,outputs = model(batch1,batch2)
        features = torch.cat([fet1.unsqueeze(1), fet2.unsqueeze(1)], dim=1)
        
        loss1 = criterion(outputs, targets)
        loss2=criterion1(features,targets)
        loss=loss1+loss2
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        



    return epoch_loss / len(train_loader)








#evaluate function
def evaluate2(train_loader, model):

    model.eval()
    pred=[]
    targets_list=[]
    with torch.no_grad():
        for idx, (ids1,mask1,ids2,mask2,targets) in enumerate(train_loader):
        

            ids1=ids1.to(device)
            mask1=mask1.to(device)
            ids2=ids2.to(device)
            mask2=mask2.to(device)

            targets=targets.to(device)
            batch1={}
            batch1['input_ids']=ids1
            batch1['attention_mask']=mask1
            batch2={}
            batch2['input_ids']=ids2
            batch2['attention_mask']=mask2

            _,outputs = model(batch1,batch2)
            pred.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            targets_list.extend(targets.cpu().detach().numpy().tolist())

    
    return pred,targets_list



#Brute Force to get the best threshold.
def get_best(pred,tar):
    best_score=0
    best_value=0
    i=0.1
    while i<=0.6:
        outputs = np.array(pred) >= i
        f1_score_micro = metrics.f1_score(outputs, tar, average='micro')
        if(f1_score_micro>best_score):
            best_score=f1_score_micro
            best_value=i
            
        i=i+0.01
    return best_score,best_value



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
for i in range(300):
    print('epochs:'+ str(i+1))
    tr_loss=training(trainloader, model, criterion, optimizer)
    print('training_loss:'+str(round(tr_loss, 5)))
    pred,tar=evaluate2(devloader, model)
    best_score,best_value=get_best(pred,tar)
    print('best_score:'+str(round(best_score, 5))+' '+'best_value:'+str(round(best_value, 5)))
    sys.stdout.flush()
    torch.save(model.state_dict(),"contrastive_model.pth")