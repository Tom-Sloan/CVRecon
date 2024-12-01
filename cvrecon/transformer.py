import torch
import torch.nn as nn
import torch.nn.functional as F
from cvrecon.utils import debug_print


class MlpBlock(nn.Module):
    """Transformer MLP block"""

    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.0, debug=False):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.ReLU(True)
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize weights
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_dim, num_heads, mlp_dim, dropout_rate=0.0, debug=False):
        super().__init__()
        self.debug = debug
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate, debug=debug)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        """
        x: [n_imgs, n_voxels, dim] 
        mask: Attention mask
        """
        debug_print("\n[DEBUG] EncoderBlock input:", self.debug)
        debug_print(f"  x shape: {x.shape}", self.debug)
        if mask is not None:
            debug_print(f"  mask shape: {mask.shape}", self.debug)
            debug_print(f"  mask dtype: {mask.dtype}", self.debug)
            
            n_imgs = x.shape[0]
            target_len = x.shape[1]  # L: target/query sequence length
            
            # For self-attention, mask comes as [n_voxels, n_imgs, n_imgs]
            if len(mask.shape) == 3 and mask.shape[-1] == n_imgs:
                # Convert to [L, L] format for broadcasting
                mask = mask.permute(1, 0, 2)  # [n_imgs, n_voxels, n_imgs]
                mask = mask.unsqueeze(2).expand(-1, -1, target_len, -1)  # [n_imgs, n_voxels, n_voxels, n_imgs]
                mask = mask.max(dim=-1)[0]  # [n_imgs, n_voxels, n_voxels]
                # Take first image's mask for broadcasting
                mask = mask[0]  # [L, L]
            else:
                # For cross-attention, mask comes as [S, 1, n_imgs] where S is key/value sequence length
                source_len = mask.shape[0]  # S: source/key sequence length
                # Reshape to [L, S] format
                mask = mask.squeeze(1)  # [S, n_imgs]
                mask = mask.t()  # [n_imgs, S]
                mask = mask[0]  # Take first image's mask [S]
                # Create full attention mask [L, S]
                # Each query position (L) can attend to valid key positions (S)
                full_mask = torch.ones((target_len, source_len), dtype=torch.bool, device=mask.device)
                # Broadcast the source mask across all target positions
                full_mask = full_mask & mask  # [L, S]
                mask = full_mask.t()  # [S, L]
                mask = mask.t()  # [L, S] - final shape matches PyTorch's expectation
                
            debug_print(f"  reshaped mask shape: {mask.shape}", self.debug)

        residual = x
        x = self.norm1(x)
        
        # Handle attention mask
        if mask is not None:
            # Convert to float and replace False with -inf
            attn_mask = mask.float()
            attn_mask = attn_mask.masked_fill(mask == 0, float('-inf'))
            x, attn_weights = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        else:
            x, attn_weights = self.attn(x, x, x, need_weights=False)
            
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        
        debug_print(f"[DEBUG] EncoderBlock output shape: {x.shape}", self.debug)
        return x


class Transformer(nn.Module):
    def __init__(self, emb_dim, mlp_dim, num_layers=1, num_heads=1, dropout_rate=0.0, debug=False):
        super().__init__()
        self.debug = debug
        self.layers = nn.ModuleList([
            EncoderBlock(emb_dim, num_heads, mlp_dim, dropout_rate, debug=debug)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        debug_print("\n[DEBUG] Transformer input:", self.debug)
        debug_print(f"  x shape: {x.shape}", self.debug)
        if mask is not None:
            debug_print(f"  mask shape: {mask.shape}", self.debug)
        
        # Process through transformer layers
        for i, layer in enumerate(self.layers):
            debug_print(f"\n[DEBUG] Processing transformer layer {i}", self.debug)
            x = layer(x, mask)
            
        debug_print(f"[DEBUG] Transformer output shape: {x.shape}", self.debug)
        return x, None


class CrossTransformer(nn.Module):
    def __init__(self, emb_dim, mlp_dim, num_layers=1, num_heads=1, dropout_rate=0.0):
        super().__init__()
        self.debug = False
        self.layers = nn.ModuleList([
            CrossAttentionBlock(emb_dim, num_heads, mlp_dim, dropout_rate)
            for _ in range(num_layers)
        ])

    def forward(self, query, key, value, depth_query=None, depth_key=None, mask=None):
        """Cross attention between query and key/value pairs"""
        debug_print("\n[DEBUG] CrossTransformer input:", self.debug)
        debug_print(f"  query shape: {query.shape}", self.debug)
        debug_print(f"  key shape: {key.shape}", self.debug)
        debug_print(f"  value shape: {value.shape}", self.debug)
        if mask is not None:
            debug_print(f"  mask shape: {mask.shape}", self.debug)
            
        # Ensure query and key have compatible shapes for attention
        if query.shape[1] != key.shape[1]:
            # Expand query to match key's sequence length
            query = query.unsqueeze(1).expand(-1, key.shape[1], -1)
            
        x = query
        for i, layer in enumerate(self.layers):
            debug_print(f"\n[DEBUG] Processing cross-transformer layer {i}", self.debug)
            x = layer(query=x, key=key, value=value, mask=mask)
            
        debug_print(f"[DEBUG] CrossTransformer output shape: {x.shape}", self.debug)
        return x, None

class CrossAttentionBlock(nn.Module):
    def __init__(self, in_dim, num_heads, mlp_dim, dropout_rate=0.0):
        super().__init__()
        self.debug = False
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask=None):
        """
        query: [n_imgs, L, dim] where L is target sequence length
        key, value: [n_imgs, S, dim] where S is source sequence length
        mask: [S, 1, n_imgs] boolean attention mask
        """
        debug_print("\n[DEBUG] CrossAttentionBlock input:", self.debug)
        debug_print(f"  query shape: {query.shape}", self.debug)
        debug_print(f"  key shape: {key.shape}", self.debug)
        if mask is not None:
            debug_print(f"  mask shape: {mask.shape}", self.debug)
            debug_print(f"  mask dtype: {mask.dtype}", self.debug)
            
            # For cross-attention:
            # - L is target sequence length (query length)
            # - S is source sequence length (key length)
            L = query.shape[1]
            S = key.shape[1]
            
            # Reshape mask to [L, S] format for attention
            # Each query position (L) can attend to masked key positions (S)
            mask = mask.squeeze(1)  # [S, n_imgs]
            mask = mask.t()  # [n_imgs, S]
            mask = mask[0]  # Take first image's mask [S]
            
            # Create attention mask [L, S]
            attn_mask = torch.ones((L, S), dtype=torch.bool, device=mask.device)
            attn_mask = attn_mask & mask.unsqueeze(0)  # Broadcast mask across all queries
                
            debug_print(f"  reshaped mask shape: {attn_mask.shape}", self.debug)

        residual = query
        query = self.norm1(query)
        key = self.norm1(key)
        value = self.norm1(value)
        
        # Handle attention mask
        if mask is not None:
            # Convert to float and replace False with -inf
            attn_mask = attn_mask.float()
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
            x, _ = self.attn(query, key, value, attn_mask=attn_mask, need_weights=False)
        else:
            x, _ = self.attn(query, key, value, need_weights=False)
            
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        
        debug_print(f"[DEBUG] CrossAttentionBlock output shape: {x.shape}", self.debug)
        return x