# Import the libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import neighbors
from torch_geometric.nn import GCNConv
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

# Define the config. Don't change this
config = {
    'img_size': 256,
    'depth' : 64,
    'batch_size' : 1,  # For training it was 8. For single image testing set it to 1
    'kc' : 16,
    'kh' : 16,
    'kw' : 16,
    'dc' : 16,
    'dh' : 16,
    'dw' : 16,
    'num_classes' : 2,
    'num_node_features' : 256,
    'hidden_channels' : 32,
    'linear_channels' : 128
}

kc, kh, kw = config['kc'], config['kh'], config['kw']
dc, dh, dw = config['dc'], config['dh'], config['dw']

# Define the model class
class GCN(nn.Module):

    # Base paper: https://arxiv.org/abs/1609.02907

    
    def __init__(self, num_node_features, num_classes, hidden_channels, linear_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)

        # 3D Convolutional layer
        self.conv3d = nn.Conv3d(1, 16, 3, stride=1, padding=1)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)
        self.conv3d_2 = nn.Conv3d(16, 32, 3, stride=2, padding=1)
        self.maxpool_2 = nn.MaxPool3d(3, stride=2, padding=1)
        
        # Graph convolution layer
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels//2)
        self.conv3 = GCNConv(hidden_channels//2, hidden_channels//4)
        self.maxpool1d = nn.MaxPool1d(64)
        self.fc1 = nn.Linear(linear_channels, linear_channels*2)
        self.fc2 = nn.Linear(linear_channels*2, num_classes)
        
    
    def forward(self, x, adj, batch_size):
        # 1. Obtain Conv features
        x = x.view(x.shape[0]*x.shape[1], 1, x.shape[2], x.shape[3], x.shape[4])
        x = F.relu(self.conv3d(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3d_2(x))
        x = self.maxpool_2(x)
        xs = x.shape  # Save the shape for reshaping later. format -> (batch_size * num_of_patch_in_each_image, 4, 4, 2)
        x = x.view(x.shape[0]*x.shape[1], -1)
        x = x.view(xs[0], -1)
        
        # 2. Obtain Diagonal Blocks. This is required for the GCNConv layer to work
        block = adj[0]
        for i in range(1, adj.shape[0]):
            block = torch.block_diag(block, adj[i])
        edge_index = block.to_sparse()._indices()
        
        # 2. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 3. Readout layer
        x = x.view(batch_size, -1)
        x = self.maxpool1d(x)
        
        # 4. Apply a final classifier
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        
        return x

            

class AlzheimerPrediction:
    def __init__(self):

        # Specify the device and model
        self.device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = GCN(num_node_features=config['num_node_features'], 
                            num_classes=config['num_classes'],
                            hidden_channels=config['hidden_channels'],
                            linear_channels=config['linear_channels']).to(self.device)

        self.model_path = os.getcwd() + '/final_model_2023-09-16 14:41:14.661342.pt'
        try:
            self.model.load_state_dict(torch.load('/home/arindam/Alzheimer/GraphVig' + '/final_model_2023-09-16 14:41:14.661342.pt'))
            print('Model loaded successfully from {}'.format(self.model_path))
        except:
            print("Model loading failed. Please check the model path")
        super().__init__()


    def build_graph(self, batched_images):
        print('Building graph ...')
        batch, patch_batch = [], []
        for img in batched_images:
            patches = img.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
            patches = patches.contiguous().view(patches.size(0), -1, kc, kh, kw)
            patch_batch.append(patches)
            patches = patches.contiguous().view(patches.size(0), -1, kc*kh*kw)
            batch.append(patches)
            
        print('Patching done ...')
        patched_images = torch.cat(batch, dim=0)  # Shape -> (batch_size, num_of_patch_in_each_image, num_features)
        patch_batch = torch.cat(patch_batch, dim=0)  # Shape -> (batch_size, num_of_patch_in_each_image, kc, kh, kw) This is required for local 3D CNNs
        batch_adj = []
        print('Building adjacency matrix ...')
        for i in range(patched_images.shape[0]):
            patches = patched_images[i]
            adj = torch.as_tensor(neighbors.kneighbors_graph(patches, n_neighbors = 64).toarray(), dtype=torch.float32)  # No of neighbors = 64
            adj = adj.reshape(1, adj.shape[0], adj.shape[1])
            batch_adj.append(adj)
            
        adj = torch.cat(batch_adj, dim=0)
        return patch_batch.type(torch.FloatTensor), adj.type(torch.FloatTensor)
    

    # Load the image
    def image_loader(self, image_path):
        # 1. Load the image
        image = nib.load(image_path)
        image = image.get_fdata()
        # 2. Normalize the image
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = image.astype('float32')
        # 3. Resize the image
        height, width, depth = image.shape
        height_factor, width_factor, depth_factor = config['img_size']/ height, config['img_size']/ width, config['depth']/ depth
        image = zoom(image, (width_factor, height_factor, depth_factor), order=1)
        # 4. Convert to tensor
        image = torch.from_numpy(image)
        image = image.view(1, 1, image.shape[0], image.shape[1], image.shape[2])  # Shape -> (batch_size, channels, height, width, depth)
        
        return image

    def prediction(self, image):
        self.model.eval()
        # 1. Load the image
        image = self.image_loader(image)
        # 2. Build the graph
        patched_image, adj = self.build_graph(image)
        patched_image, adj = patched_image.to(self.device), adj.to(self.device)
        self.model.eval()
        output = self.model(patched_image, adj, batch_size=1)
        pred = torch.argmax(output.data)
        
        if pred == 0:
            print('The image is normal')
        else:
            print('The image is abnormal (Alzheimer)')
        
        if self.device == 'cuda':
            return output.data.cpu().numpy()[0], pred.cpu().numpy()
        else:
            return output.data.numpy()[0], pred.numpy()


if __name__ == '__main__':
    # Run the model
    image_path = '/home/arindam/Alzheimer/GraphVig/WebAppUploaded.nii'
    scores, label = AlzheimerPrediction.prediction(image_path)
    print(scores, label)


