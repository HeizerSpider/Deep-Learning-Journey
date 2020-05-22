import torch.nn.functional as F
from torch import optim


#setting up the model (feed forward neural network)
class FMNIST(nn.module):
    def __init__(self):
        super().__init__(self)
        self.fc1=nn.Linear(784,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)

    def forward():
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

model = FMNIST()

#------------------------------------------------------------------------------------------------------------------

#Training the network (feed forward, backpropogation)
criterion = nn.NLLLoss() #can use CrossEntropyLoss as well (includes LogSoftMax)
optimizer = optim.SGD(model.parameters(), lr=0.01) #stochastic gradient descent, with lr as learning rate
num_epochs = 3 #increase number of epochs MAY increase accuracy
#one epoch = batch_1 (forward+back prop) + batch_2 (forward+back prop) + ...
for i in range(num_epochs):
    cum_loss = 0 #track total amount of loss for each batch
    for images,labels in trainloader: #taking one batch at a time
        optimizer.zero_grad() #zeroing the gradient every time we go through a new batch
        output = model(images) #running each batch through the neural network
        loss = criterion(output, labels)
        loss.backward() #calculate the loss of neural network for that batch, difference between output and labels
        optimizer.step #update weights of neural network
        cum_loss += loss.item() 