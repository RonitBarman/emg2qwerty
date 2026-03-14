import torch
import torch.nn as nn
import torch.nn.functional as F

class RollingTimeNorm(nn.Module):
    def __init__(self, warmup_frames=125, eps=1e-5):
        super().__init__()
        self.warmup_frames = warmup_frames
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Time, Channels, Freqs)
        B, T, C, F_bins = x.shape
        
        # Causal mean and variance calculation
        cumsum = torch.cumsum(x, dim=1)
        cumsum_sq = torch.cumsum(x ** 2, dim=1)
        count = torch.arange(1, T + 1, device=x.device).view(1, -1, 1, 1).float()
        
        mu_t = cumsum / count
        var_t = (cumsum_sq / count) - mu_t**2
        
        # Apply warmup freezing if the sequence is long enough
        if T >= self.warmup_frames:
            mu_warmup = mu_t[:, self.warmup_frames - 1:self.warmup_frames, :, :]
            var_warmup = var_t[:, self.warmup_frames - 1:self.warmup_frames, :, :]
            
            mu_t = mu_t.clone()
            var_t = var_t.clone()
            mu_t[:, :self.warmup_frames, :, :] = mu_warmup
            var_t[:, :self.warmup_frames, :, :] = var_warmup
            
        sigma_t = torch.sqrt(torch.clamp(var_t, min=self.eps))
        return (x - mu_t) / sigma_t

class RotationInvariantMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x shape: (B, T, C, F)
        B, T, C, F_bins = x.shape
        out = 0
        # Average over rotations of -1, 0, and 1
        for shift in [-1, 0, 1]:
            x_shifted = torch.roll(x, shifts=shift, dims=2)
            x_flat = x_shifted.view(B, T, C * F_bins)
            out = out + self.mlp(x_flat)
        return out / 3.0

class TDSConvBlock(nn.Module):
    def __init__(self, channels=24, kernel_size=32, hidden_dim=384):
        super().__init__()
        self.channels = channels
        self.w = kernel_size
        self.h = hidden_dim // channels
        
        # 1xW causal convolution over time
        self.conv = nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), padding=(0, kernel_size - 1))
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Fully connected block
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x shape: (B, T, D) 
        B, T, D = x.shape
        
        res = x
        # Reshape to (B, Channels, hidden_per_channel, Time) for Conv2d
        x_conv = x.view(B, T, self.channels, self.h).permute(0, 2, 3, 1)
        
        # Apply convolution and trim padding to keep it causal
        x_conv = self.conv(x_conv)[:, :, :, :T]
        x_conv = F.relu(x_conv)
        
        # Revert back to (B, T, D)
        x_conv = x_conv.permute(0, 3, 1, 2).reshape(B, T, D)
        x = self.norm1(x_conv + res)
        
        # Feedforward and residual
        res = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.norm2(x + res)
        
        return x

class SplashNet(nn.Module):
    def __init__(self, num_classes, input_freqs=33, hidden_dim=384, num_blocks=4):
        super().__init__()
        self.rtn = RollingTimeNorm()
        
        # Shared Encoder setup
        self.rot_mlp = RotationInvariantMLP(input_dim=16 * input_freqs, hidden_dim=hidden_dim)
        
        self.tds_blocks = nn.ModuleList([
            TDSConvBlock(channels=24, kernel_size=32, hidden_dim=hidden_dim) 
            for _ in range(num_blocks)
        ])
        
        # Final prediction layer takes the concatenated left and right embeddings
        self.fc_out = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, left_x, right_x):
        # Expected shape for left_x and right_x: (Batch, Time, 16 electrodes, 33 freqs)
        
        # 1. Rolling Time Normalization
        left_x = self.rtn(left_x)
        right_x = self.rtn(right_x)
        
        # 2. Process through Shared Rotation-Invariant MLP
        left_emb = self.rot_mlp(left_x)
        right_emb = self.rot_mlp(right_x)
        
        # 3. Process through Shared TDS Blocks
        for block in self.tds_blocks:
            left_emb = block(left_emb)
            right_emb = block(right_emb)
            
        # 4. Concatenate and predict
        combined = torch.cat([left_emb, right_emb], dim=-1)
        logits = self.fc_out(combined)
        
        return logits