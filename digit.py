import math
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,))])

# Load the MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_network(net, trainloader, pruning_mask, epochs):
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if pruning_mask is not None:
                net = apply_pruning_mask(net, pruning_mask)

            running_loss += loss.item()
            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

def test_network(net, testloader):
    # Test the network
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

def prune_network(net, pruning_percentage, pruning_threshold=1e-3):
    # Define the pruning parameters
    for name, param in net.named_parameters():
        if 'weight' in name:

            pruned_param = param

            num_weights = param.data.numel()

            num_weights_to_prune = int(pruning_percentage * num_weights)
            print(f"Pruning layer {name} with {num_weights_to_prune} weights to prune.")
            if num_weights_to_prune > 0:
                pruned_param = prune_tensor(pruned_param, num_weights_to_prune)
                param.data = pruned_param

    # Save the pruning mask
    pruning_mask = {}
    for name, param in net.named_parameters():
        if 'weight' in name:
            pruning_mask[name] = param.data.ne(0)

    return pruning_mask, net

def prune_tensor(tensor, num_weights_to_prune):
    """
    Prune the tensor by removing the num_weights_to_prune
    smallest entries in the tensor.
    """
    tensor_sorted = tensor.view(-1).abs().sort()[0]
    pruning_threshold = tensor_sorted[num_weights_to_prune].item()
    mask = tensor.abs().ge(pruning_threshold).int()
    return tensor.mul(mask)

def apply_pruning_mask(net, pruning_mask):
    for name, param in net.named_parameters():
        if 'weight' in name:
            tensor = param.data
            mask = pruning_mask[name]
            pruned_tensor = tensor.masked_fill(~mask, 0)
            param.data = pruned_tensor
    return net

def loop_pruning(net, target_sparsity, epochs, prune):
    cycles = math.log(1 - target_sparsity) / math.log(1 - prune)
    cycles = round(cycles)
    pruning_factor = 1 - prune
    for i in range(cycles):
        pruning_percent = 1-(pruning_factor**(i+1))
        print("Pruning percentage: {:.2f}%".format(100 * pruning_percent))
        pruning_mask, net = prune_network(net, pruning_percent)
        net = Net()
        net.load_state_dict(torch.load('init.pt'))

        train_network(net, trainloader, pruning_mask, epochs)
        test_network(net, testloader)
        print("Sparsity in model: {:.2f}%".format(100 * float(torch.sum(net.fc1.weight == 0) \
                + torch.sum(net.fc2.weight == 0) + torch.sum(net.fc3.weight == 0)) / float(net.fc1.weight.nelement() \
                + net.fc2.weight.nelement() + net.fc3.weight.nelement())))

    print("Target sparsity: {:.2f}%".format(100 * target_sparsity))
    return net, pruning_mask

# Make a new network and save initialization
if __name__ == '__main__':
    net = Net()
    torch.save(net.state_dict(), 'init.pt')

    # Get training rolling
    train_network(net, trainloader, None, 1)

    net, pruning_mask = loop_pruning(net, 0.9, 2, 0.2)

    train_network(net, trainloader, pruning_mask, 10)
    test_network(net, testloader)

    PATH = './mnist_net.pth'
    torch.save(net.state_dict(), PATH)
    print("Sparsity in model: {:.2f}%".format(100 * float(torch.sum(net.fc1.weight == 0) + torch.sum(net.fc2.weight == 0) + torch.sum(net.fc3.weight == 0)) / float(net.fc1.weight.nelement() + net.fc2.weight.nelement() + net.fc3.weight.nelement())))
