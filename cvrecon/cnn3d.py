import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
import torch
import numpy as np
import itertools


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.ks = ks
        self.stride = stride
        self.dilation = dilation
        
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc, 
                outc, 
                kernel_size=ks, 
                dilation=dilation, 
                stride=1,
                padding=(ks - 1) // 2
            ),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        try:
            # Debug prints for input
            print(f"\nBasicConvolutionBlock (in={self.inc}, out={self.outc}, ks={self.ks}, stride={self.stride}):")
            print(f"Input - F: {x.F.shape}, C: {x.C.shape}, stride: {x.stride}")
            
            # If stride=2, adjust coordinates before convolution
            if self.stride > 1:
                print("Applying stride...")
                # Floor divide coordinates by stride
                new_coords = x.C.clone()
                new_coords[:, :3] = new_coords[:, :3] // self.stride
                
                # Remove duplicate coordinates after striding
                unique_coords, inverse_indices = torch.unique(new_coords, dim=0, return_inverse=True)
                print(f"After striding - unique coords: {len(unique_coords)}")
                
                # Aggregate features for duplicate coordinates
                new_feats = torch.zeros(
                    (len(unique_coords), x.F.shape[1]), 
                    dtype=x.F.dtype, 
                    device=x.F.device
                )
                new_feats.index_add_(0, inverse_indices, x.F)
                
                x = torchsparse.SparseTensor(
                    feats=new_feats,
                    coords=unique_coords,
                    stride=(self.stride, self.stride, self.stride)
                )
                print(f"After stride adjustment - F: {x.F.shape}, C: {x.C.shape}")
            
            # Apply network
            print("Applying convolution...")
            out = self.net(x)
            
            print(f"Output - F: {out.F.shape}, C: {out.C.shape}, stride: {out.stride}")
            return out
            
        except Exception as e:
            print(f"Error in BasicConvolutionBlock:")
            print(f"- inc: {self.inc}, outc: {self.outc}, ks: {self.ks}, stride: {self.stride}")
            print(f"- Input tensor F shape: {x.F.shape}")
            print(f"- Input tensor C shape: {x.C.shape}")
            print(f"- Input tensor stride: {x.stride}")
            print(f"- Input tensor C range: {x.C.min().item()} to {x.C.max().item()}")
            print(f"- Input tensor F range: {x.F.min().item()} to {x.F.max().item()}")
            print(f"- Number of unique coordinates: {len(torch.unique(x.C, dim=0))}")
            raise e


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.ks = ks
        self.stride = stride
        
        # Use regular convolution with fixed dimensions
        self.conv = spnn.Conv3d(
            inc, 
            outc, 
            kernel_size=3,  # Fixed kernel size
            stride=1,
            padding=1
        )
        self.bn = spnn.BatchNorm(outc)
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        try:
            # Step 1: Compute upsampled coordinates
            coords = x.C.clone()
            batch_inds = coords[:, 3:]
            spatial_coords = coords[:, :3] * self.stride
            
            # Step 2: Generate fixed offsets
            offsets = torch.tensor([
                [i, j, k] for i in range(self.stride) 
                for j in range(self.stride) 
                for k in range(self.stride)
            ], device=coords.device)
            
            # Step 3: Generate new coordinates
            new_coords = (spatial_coords.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1, 3)
            new_batch_inds = batch_inds.repeat_interleave(self.stride**3, dim=0)
            up_coords = torch.cat([new_coords, new_batch_inds], dim=1)
            
            # Step 4: Repeat features
            up_feats = x.F.repeat_interleave(self.stride**3, dim=0)
            
            # Step 5: Create new sparse tensor
            x = torchsparse.SparseTensor(
                feats=up_feats,
                coords=up_coords.to(torch.int32),
                stride=(1, 1, 1)
            )
            
            # Step 6: Apply convolution
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            
            return x
            
        except Exception as e:
            print(f"\nError in BasicDeconvolutionBlock:")
            print(f"Error details: {str(e)}")
            print(f"Input shapes - F: {x.F.shape}, C: {x.C.shape}")
            raise e


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation, stride=1),
            spnn.BatchNorm(outc),
        )

        self.downsample = (
            nn.Sequential()
            if (inc == outc and stride == 1)
            else nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(outc),
            )
        )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        print("\nResidualBlock input:")
        print("x.F shape:", x.F.shape)
        print("x.C shape:", x.C.shape)
        print("x.C unique values count:", len(torch.unique(x.C)))
        net_out = self.net(x)
        down_out = self.downsample(x)
        print("ResidualBlock after net and downsample:")
        print("net_out.C shape:", net_out.C.shape)
        print("down_out.C shape:", down_out.C.shape)
        out = self.relu(net_out + down_out)
        print("ResidualBlock output:")
        print("out.F shape:", out.F.shape)
        print("out.C shape:", out.C.shape)
        print("out.C unique values count:", len(torch.unique(out.C)))
        return out


def process_coordinates_in_chunks(coords1, coords2, features, chunk_size=100):
    """Process coordinate matching in smaller chunks to avoid memory issues"""
    device = coords1.device
    n1, n2 = len(coords1), len(coords2)
    new_features = torch.zeros(
        (n1, features.shape[1]), 
        dtype=features.dtype, 
        device=device
    )
    
    # Convert coordinates to hashable format for faster matching
    coords1_tuples = [tuple(c.tolist()) for c in coords1[:, :3]]
    coords2_dict = {tuple(c.tolist()): i for i, c in enumerate(coords2[:, :3])}
    
    # Process in small chunks
    for i in range(0, n1, chunk_size):
        end_idx = min(i + chunk_size, n1)
        chunk_coords = coords1_tuples[i:end_idx]
        
        # Find matches using dictionary lookup
        for j, coord in enumerate(chunk_coords):
            if coord in coords2_dict:
                new_features[i + j] = features[coords2_dict[coord]]
    
    return new_features


class SPVCNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.dropout = kwargs["dropout"]

        base_depth = kwargs["base_depth"]
        cs = np.array([1, 2, 4, 3, 3]) * base_depth
        self.output_depth = cs[-1]

        self.stem = nn.Sequential(
            spnn.Conv3d(kwargs["in_channels"], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True),
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[2], cs[3], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlock(cs[3] + cs[1], cs[3], ks=3, stride=1, dilation=1),
                    ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
                ),
            ]
        )

        self.up2 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(cs[3], cs[4], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlock(cs[4] + cs[0], cs[4], ks=3, stride=1, dilation=1),
                    ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
                ),
            ]
        )

        self.weight_initialization()

        if self.dropout:
            self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x0):
        try:
            print("\nSPVCNN input:")
            print(f"x0.F shape: {x0.F.shape}")
            print(f"x0.C shape: {x0.C.shape}")
            print(f"x0.C min/max: {x0.C.min()}, {x0.C.max()}")
            print(f"x0.F min/max: {x0.F.min()}, {x0.F.max()}")
            print(f"x0.C unique values: {torch.unique(x0.C)}")
            print(f"x0 stride: {x0.stride}")

            device = x0.F.device  # Get device from input tensor
            x0 = self.stem(x0)
            print("\nAfter stem:")
            print(f"x0.F shape: {x0.F.shape}")
            print(f"x0.C shape: {x0.C.shape}")
            print(f"x0 stride: {x0.stride}")

            # Save original coordinates for later upsampling
            orig_coords = x0.C.clone()

            x1 = self.stage1(x0)
            print("\nAfter stage1:")
            print(f"x1.F shape: {x1.F.shape}")
            print(f"x1.C shape: {x1.C.shape}")
            print(f"x1 stride: {x1.stride}")

            x2 = self.stage2(x1)
            print("\nAfter stage2:")
            print(f"x2.F shape: {x2.F.shape}")
            print(f"x2.C shape: {x2.C.shape}")
            print(f"x2 stride: {x2.stride}")

            print("\nBefore up1:")
            print(f"x2 stats before upsampling:")
            print(f"- F shape: {x2.F.shape}")
            print(f"- C shape: {x2.C.shape}")
            print(f"- stride: {x2.stride}")
            print(f"- C range: {x2.C.min().item()} to {x2.C.max().item()}")

            # First upsampling stage
            y3 = self.up1[0](x2)
            
            # Match coordinates using chunked processing
            x1_matched_feats = process_coordinates_in_chunks(
                y3.C[:, :3],
                x1.C[:, :3],
                x1.F,
                chunk_size=100
            )
            
            # Create new sparse tensor with matched features
            x1_matched = torchsparse.SparseTensor(
                feats=x1_matched_feats,
                coords=y3.C.clone(),
                stride=y3.stride
            )

            # Concatenate features
            y3 = torchsparse.cat([y3, x1_matched])
            y3 = self.up1[1](y3)

            # Second upsampling stage
            y4 = self.up2[0](y3)
            
            # Match coordinates for second upsampling
            x0_matched_feats = process_coordinates_in_chunks(
                y4.C[:, :3],
                orig_coords[:, :3],
                x0.F,
                chunk_size=100
            )
            
            x0_matched = torchsparse.SparseTensor(
                feats=x0_matched_feats,
                coords=y4.C.clone(),
                stride=y4.stride
            )

            y4 = torchsparse.cat([y4, x0_matched])
            y4 = self.up2[1](y4)

            print("\nFinal output:")
            print(f"y4.F shape: {y4.F.shape}")
            print(f"y4.C shape: {y4.C.shape}")
            print(f"y4 stride: {y4.stride}")

            return y4

        except Exception as e:
            print(f"\nError in SPVCNN forward:")
            print(f"Error location: {e.__traceback__.tb_lineno}")
            print(f"Last successful tensor shape: {locals().get('y4', locals().get('y3', locals().get('x2', locals().get('x1', x0)))).F.shape}")
            print(f"Error: {str(e)}")
            print(f"Device check:")
            print(f"- Input device: {x0.F.device}")
            print(f"- Current tensor device: {locals().get('y4', locals().get('y3', locals().get('x2', locals().get('x1', x0)))).F.device}")
            raise e
