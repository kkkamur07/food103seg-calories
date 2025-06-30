import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.data import data_loaders

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UnetPlus(nn.Module):
    def __init__(self, n_classes=103, filters=[64,128,256,512,512]):
        super().__init__()
        
        # Encoder
        self.conv0_0 = ConvBlock(3, filters[0])
        self.pool0 = nn.MaxPool2d(2)
        self.conv1_0 = ConvBlock(filters[0], filters[1])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2_0 = ConvBlock(filters[1], filters[2])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3_0 = ConvBlock(filters[2], filters[3])
        self.pool3 = nn.MaxPool2d(2)
        self.conv4_0 = ConvBlock(filters[3], filters[4])

        # Decoder
        self.conv0_1 = ConvBlock(filters[0]+filters[1], filters[0])
        self.conv1_1 = ConvBlock(filters[1]+filters[2], filters[1])
        self.conv2_1 = ConvBlock(filters[2]+filters[3], filters[2])
        self.conv3_1 = ConvBlock(filters[3]+filters[4], filters[3])

        self.conv0_2 = ConvBlock(filters[0]*2 + filters[1], filters[0])
        self.conv1_2 = ConvBlock(filters[1]*2 + filters[2], filters[1])
        self.conv2_2 = ConvBlock(filters[2]*2 + filters[3], filters[2])

        self.conv0_3 = ConvBlock(filters[0]*3 + filters[1], filters[0])
        self.conv1_3 = ConvBlock(filters[1]*3 + filters[2], filters[1])

        self.conv0_4 = ConvBlock(filters[0]*4 + filters[1], filters[0])

        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool0(x0_0))
        x2_0 = self.conv2_0(self.pool1(x1_0))
        x3_0 = self.conv3_0(self.pool2(x2_0))
        x4_0 = self.conv4_0(self.pool3(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, F.interpolate(x1_0, x0_0.size()[2:], mode='bilinear', align_corners=False)],1))
        x1_1 = self.conv1_1(torch.cat([x1_0, F.interpolate(x2_0, x1_0.size()[2:], mode='bilinear', align_corners=False)],1))
        x2_1 = self.conv2_1(torch.cat([x2_0, F.interpolate(x3_0, x2_0.size()[2:], mode='bilinear', align_corners=False)],1))
        x3_1 = self.conv3_1(torch.cat([x3_0, F.interpolate(x4_0, x3_0.size()[2:], mode='bilinear', align_corners=False)],1))

        # Skip connectionss
        x0_2 = self.conv0_2(torch.cat([
            x0_0, x0_1, F.interpolate(x1_1, x0_0.size()[2:], mode='bilinear', align_corners=False)
        ],1))

        x1_2 = self.conv1_2(torch.cat([
            x1_0, x1_1, F.interpolate(x2_1, x1_0.size()[2:], mode='bilinear', align_corners=False)
        ],1))

        x2_2 = self.conv2_2(torch.cat([
            x2_0, x2_1, F.interpolate(x3_1, x2_0.size()[2:], mode='bilinear', align_corners=False)
        ],1))

        x0_3 = self.conv0_3(torch.cat([
            x0_0, x0_1, x0_2, F.interpolate(x1_2, x0_0.size()[2:], mode='bilinear', align_corners=False)
        ],1))

        x1_3 = self.conv1_3(torch.cat([
            x1_0, x1_1, x1_2, F.interpolate(x2_2, x1_0.size()[2:], mode='bilinear', align_corners=False)
        ],1))

        x0_4 = self.conv0_4(torch.cat([
            x0_0, x0_1, x0_2, x0_3, F.interpolate(x1_3, x0_0.size()[2:], mode='bilinear', align_corners=False)
        ],1))

        out = self.final(x0_4)
        out = F.softmax(out, dim=1)
        return out



class FoodSegmentation(nn.Module):
    def __init__(self, n_classes = 104, lr = 1e-4, base_dir = "/home/krrish/home/desktop/sensor-behaviour/data", epochs = 10, batch_size = 16):
        
        super().__init__()
        
        self.n_classes = n_classes
        self.model = UnetPlus(n_classes=n_classes).to('cuda' if torch.cuda.is_available() else 'cpu')

        self.parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Training Parameters
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        
        # Base Directory
        self.base_dir = base_dir
        
        # Data Loaders
        self.train_loader, self.val_loader, self.test_loader = data_loaders(base_dir=self.base_dir, validation_split=0.1, batch_size=self.batch_size)

        self.loss = nn.CrossEntropyLoss() 
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        print(f"Total parameters in the model: {self.parameters}")
        
    def forward(self, x):
        return self.model(x)
    
    def train_step(self, image, mask):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        image = image.to('cuda' if torch.cuda.is_available() else 'cpu')
        mask = mask.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        outputs = self.forward(image)
        loss = self.loss(outputs, mask)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def eval_step(self, image, mask):
        self.model.eval()
        with torch.no_grad():
            
            # Move data to device
            image = image.to('cuda' if torch.cuda.is_available() else 'cpu')
            mask = mask.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            outputs = self.forward(image)
            loss = self.loss(outputs, mask)

        return loss.item()
    
    def train(self):
        for epoch in range(self.epochs):
            
            train_loss = 0.0
            for images, masks in self.train_loader:
                loss = self.train_step(images, masks)
                train_loss += loss
                
            train_loss /= len(self.train_loader)
            print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {train_loss:.4f}")
            
            val_loss = 0.0
            for images, masks in self.val_loader:
                loss = self.eval_step(images, masks)
                val_loss += loss
                
            val_loss /= len(self.val_loader)
            print(f"Epoch [{epoch+1}/{self.epochs}], Validation Loss: {val_loss:.4f}")
            
        print("Training complete.")
        
        
if __name__ == "__main__":
    model = FoodSegmentation()
    model.train()