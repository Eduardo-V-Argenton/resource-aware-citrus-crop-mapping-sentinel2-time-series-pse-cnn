import torch
import torch.nn as nn

class PhenologyPSE(nn.Module):
    def __init__(self, in_channels=10): 
        super(PhenologyPSE, self).__init__()

        self.input_norm = nn.LayerNorm(in_channels) 
        
        self.spatial_mlp1 = nn.Conv1d(in_channels, 32, kernel_size=1)
        self.spatial_bn1 = nn.BatchNorm1d(32)

        self.spatial_mlp2 = nn.Conv1d(32, 64, kernel_size=1)
        self.spatial_bn2 = nn.BatchNorm1d(64)

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 32), 
            nn.GELU(), 
            nn.Dropout(0.4), 
            nn.Linear(32, 1)
        )

    def forward(self, x):
        B, T, C, S = x.size()
        
        x = x.permute(0, 1, 3, 2) # [B, T, S, C]
        x = self.input_norm(x)
        x = x.permute(0, 1, 3, 2).reshape(B * T, C, S)
                  
        x = F.gelu(self.spatial_bn1(self.spatial_mlp1(x)))
        x = F.gelu(self.spatial_bn2(self.spatial_mlp2(x)))

        x_max = torch.max(x, dim=2)[0]
        x_mean = torch.mean(x, dim=2)
        x_pooled = torch.cat([x_max, x_mean], dim=1) 

        x_temp = x_pooled.view(B, T, -1).permute(0, 2, 1)
        
        x_temp = self.temporal_conv(x_temp) 
        
        x_flat = x_temp.view(B, -1)
        out = self.classifier(x_flat)

        return out
        
model = PhenologyPSE(in_channels=10) 
model.load_state_dict(torch.load("results/recall_free/paper_results/models/weights_year_2023_seed_42.pth"))

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total:", total_params)
print("Trainable:", trainable_params)