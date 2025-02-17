from torch import nn


class Block(nn.Module):
    def __init__(self, dim, kernel_size=5, expansion=2):
        super().__init__()
        assert kernel_size % 2 == 1
        padding = kernel_size // 2
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, padding)
        self.norm = nn.BatchNorm2d(dim)
        hidden_dim = int(expansion * dim)
        self.ln1 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.act = nn.GELU()
        self.ln2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.ln1(out)
        out = self.act(out)
        out = self.ln2(out)
        out += x
        return out


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        def blocks(dim, n):
            return [Block(dim) for _ in range(n)]

        self.encoder = nn.Sequential(
            nn.Conv2d(2, 128, (4, 4), (4, 4), (1, 1)),
            nn.BatchNorm2d(128),
            *blocks(128, n=1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, (3, 4), (3, 4), (1, 1)),
            *blocks(256, n=3),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, (3, 4), (2, 4), (1, 0)),
        )
        self.decoder = nn.Sequential(
            *blocks(512, n=6),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, 2, 2, 0),
            *blocks(256, n=3),
            nn.ConvTranspose2d(256, 128, 2, 2, 0),
            *blocks(128, n=3),
            nn.ConvTranspose2d(128, 32, 2, 2, 0),
            *blocks(32, n=1),
            nn.ConvTranspose2d(32, 16, 2, 2, 0),
            *blocks(16, n=1),
            nn.ConvTranspose2d(16, 16, 2, 2, 0),
            *blocks(16, n=1),
            nn.ConvTranspose2d(16, 16, 2, 2, 0),
            *blocks(16, n=1),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out