import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoBrainFusionNet(nn.Module):
    def __init__(self, eeg_feat_dim=128, seq_len=100, num_classes=2):
        super(AutoBrainFusionNet, self).__init__()

        self.fnirs_trans_1 = nn.TransformerEncoderLayer(d_model=seq_len, nhead=4, batch_first=True)
        self.fnirs_trans_2 = nn.TransformerEncoderLayer(d_model=seq_len, nhead=4, batch_first=True)
        

        self.eeg_proj = nn.Sequential(
            nn.Linear(eeg_feat_dim, seq_len),
            nn.LayerNorm(seq_len),
            nn.GELU()
        )
        

        fusion_dim = 36 * seq_len + 36 * seq_len + seq_len
        self.attention_weights = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.Tanh(),
            nn.Linear(fusion_dim // 2, 3), 
            nn.Softmax(dim=-1)
        )
        

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, eeg_data, fnirs_data):

        # fnirs_data shape: [Batch, 72, 100]
        fnirs_stream_1 = fnirs_data[:, :36, :] 
        fnirs_stream_2 = fnirs_data[:, 36:, :] 
        
        f1_encoded = self.fnirs_trans_1(fnirs_stream_1).reshape(fnirs_data.size(0), -1)
        f2_encoded = self.fnirs_trans_2(fnirs_stream_2).reshape(fnirs_data.size(0), -1)
        
        eeg_encoded = self.eeg_proj(eeg_data).view(eeg_data.size(0), -1)
        
        concat_features = torch.cat([f1_encoded, f2_encoded, eeg_encoded], dim=1)
        
        logits = self.classifier(concat_features)
        
        return logits, concat_features