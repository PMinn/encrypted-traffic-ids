import torch.nn as nn

class MLPHead(nn.Module):
    def __init__(self, dim, num_classes):
        super(MLPHead, self).__init__()
        # self.hidden_dim1 = 16   #16 32 48 32 10
        # self.hidden_dim2 = 32
        self.hidden_dim3 = 48
        # self.hidden_dim4 = 32
        
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)     # original one
            
            # nn.Linear(dim, self.hidden_dim2),               # add 1 layer - 32
            # nn.Linear(self.hidden_dim2, num_classes)
            
            # nn.Linear(dim, self.hidden_dim3),                 # add 1 layer - 48
            # nn.Linear(self.hidden_dim3, num_classes)
            
            # nn.Linear(dim, self.hidden_dim1),
            # nn.Linear(self.hidden_dim1, self.hidden_dim2),
            # nn.Linear(self.hidden_dim2, self.hidden_dim3),
            # nn.Linear(self.hidden_dim3, self.hidden_dim4),
            # nn.Linear(self.hidden_dim4, num_classes)
            
        )

    def forward(self, x):
        return self.head(x)
