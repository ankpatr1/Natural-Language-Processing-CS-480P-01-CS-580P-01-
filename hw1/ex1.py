#Name: Ankita Patra
#B:Number: B01101280
#Mail id : apatra@binghamton.edu


import torch
import torchvision.transforms as transforms
from torchvision import datasets

# loading training data
train_dataset = datasets.MNIST(root='./data', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)
#loading test data
test_dataset = datasets.MNIST(root='./data', 
                              train=False, 
                              transform=transforms.ToTensor())
from torch.utils.data import DataLoader

# load train and test data samples into dataloader
batach_size = 32
train_loader = DataLoader(dataset=train_dataset, batch_size=batach_size, shuffle=True) 
test_loader = DataLoader(dataset=test_dataset, batch_size=batach_size, shuffle=False)

# TODO: implement a Logistic Regression
class LogisticRegression(torch.nn.Module):    
    # build the constructor
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.linear(x)

        # Accept (N, 1, 28, 28) or (N, 784)


# TODO train the classsifier on MNIST dataset
log_regr = LogisticRegression(n_inputs=28*28, n_outputs=10)

# TODO: defining the optimizer
optimizer = torch.optim.SGD(log_regr.parameters(), lr=0.1)

# defining Cross-Entropy loss
criterion = torch.nn.CrossEntropyLoss()

epochs = 50
Loss = []
acc = []
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        # TODO: Work on something here
        outputs = log_regr(images.view(-1, 28*28))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    Loss.append(loss.item())
    correct = 0
    for images, labels in test_loader:
        outputs = log_regr(images.view(-1, 28*28))
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum()
    accuracy = 100 * (correct.item()) / len(test_dataset)
    acc.append(accuracy)
    print('Epoch: {}. Loss: {}. Accuracy: {}'.format(epoch, loss.item(), accuracy))

# Double check the result by comparing to scikit-learn results
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

#from sklearn.linear_model import LogisticRegression
# X = train_dataset.data.numpy().reshape(60000, -1)
# y = train_dataset.targets.numpy()
#clf = LogisticRegression().fit(X, y)


# print("sklearn test acc:", clf.score(
#     test_dataset.data.numpy().reshape(10000, -1),
#     test_dataset.targets.numpy()
# ))


"""
# without scikit : (.venv) ankitapatra@Ankitas-MacBook-Pro nlp % ls
# LSTM-Language-Generator hw1
# (.venv) ankitapatra@Ankitas-MacBook-Pro nlp % cd hw1
##(.venv) ankitapatra@Ankitas-MacBook-Pro hw1 % python ex1.py

Epoch: 0. Loss: 0.33739495277404785. Accuracy: 91.35
Epoch: 1. Loss: 0.12182888388633728. Accuracy: 91.67
Epoch: 2. Loss: 0.2347509115934372. Accuracy: 92.05
Epoch: 3. Loss: 0.3658404052257538. Accuracy: 92.09
Epoch: 4. Loss: 0.17209982872009277. Accuracy: 92.18
Epoch: 5. Loss: 0.15114115178585052. Accuracy: 92.2
Epoch: 6. Loss: 0.22559598088264465. Accuracy: 92.2
Epoch: 7. Loss: 0.0882604569196701. Accuracy: 92.38
Epoch: 8. Loss: 0.32450225949287415. Accuracy: 92.25
Epoch: 9. Loss: 0.14988693594932556. Accuracy: 92.28
Epoch: 10. Loss: 0.5146739482879639. Accuracy: 92.36
Epoch: 11. Loss: 0.6339536905288696. Accuracy: 92.38
Epoch: 12. Loss: 0.22229544818401337. Accuracy: 92.32
Epoch: 13. Loss: 0.18164746463298798. Accuracy: 92.56
Epoch: 14. Loss: 0.2085380256175995. Accuracy: 92.33
Epoch: 15. Loss: 0.1391950100660324. Accuracy: 92.35
Epoch: 16. Loss: 0.28814831376075745. Accuracy: 92.06
Epoch: 17. Loss: 0.18575870990753174. Accuracy: 92.61
Epoch: 18. Loss: 0.3563176691532135. Accuracy: 92.56
Epoch: 19. Loss: 0.1557013988494873. Accuracy: 92.54
Epoch: 20. Loss: 0.16231034696102142. Accuracy: 92.46
Epoch: 21. Loss: 0.06282387673854828. Accuracy: 92.52
Epoch: 22. Loss: 0.17193792760372162. Accuracy: 92.58
Epoch: 23. Loss: 0.12995879352092743. Accuracy: 92.47
Epoch: 24. Loss: 0.06451379507780075. Accuracy: 92.41
Epoch: 25. Loss: 0.5173261165618896. Accuracy: 92.52
Epoch: 26. Loss: 0.280183881521225. Accuracy: 92.55
Epoch: 27. Loss: 0.5396600961685181. Accuracy: 92.32
Epoch: 28. Loss: 0.24603141844272614. Accuracy: 92.27
Epoch: 29. Loss: 0.08476096391677856. Accuracy: 92.55
Epoch: 30. Loss: 0.42968064546585083. Accuracy: 92.48
Epoch: 31. Loss: 0.04287872090935707. Accuracy: 92.37
Epoch: 32. Loss: 0.11685658991336823. Accuracy: 92.5
Epoch: 33. Loss: 0.09638021141290665. Accuracy: 92.59
Epoch: 34. Loss: 0.0779917910695076. Accuracy: 92.24
Epoch: 35. Loss: 0.1738472431898117. Accuracy: 92.52
Epoch: 36. Loss: 0.14298957586288452. Accuracy: 92.45
Epoch: 37. Loss: 0.8282263875007629. Accuracy: 92.64
Epoch: 38. Loss: 0.0602586567401886. Accuracy: 92.52
Epoch: 39. Loss: 0.27431997656822205. Accuracy: 92.57
Epoch: 40. Loss: 0.2379942536354065. Accuracy: 92.45
Epoch: 41. Loss: 0.30286699533462524. Accuracy: 92.42
Epoch: 42. Loss: 0.05798884108662605. Accuracy: 92.61
Epoch: 43. Loss: 0.11035870760679245. Accuracy: 92.59
Epoch: 44. Loss: 0.21805450320243835. Accuracy: 92.23
Epoch: 45. Loss: 0.23832279443740845. Accuracy: 92.39
Epoch: 46. Loss: 0.3095756769180298. Accuracy: 92.51
Epoch: 47. Loss: 0.15907511115074158. Accuracy: 92.58
Epoch: 48. Loss: 0.5101621150970459. Accuracy: 92.43
Epoch: 49. Loss: 0.05284545570611954. Accuracy: 92.53
# (.venv) ankitapatra@Ankitas-MacBook-Pro hw1 % 
## O/P: 
# (.venv) ankitapatra@Ankitas-MacBook-Pro nlp % pip install torch torchvision
#(.venv) ankitapatra@Ankitas-MacBook-Pro nlp % pip install scikit-learn
# (.venv) ankitapatra@Ankitas-MacBook-Pro nlp % python ex1.py
#
# 100.0%
# 100.0%
# 100.0%
# 100.0%
# Epoch: 0. Loss: 0.48237255215644836. Accuracy: 91.08
# Epoch: 1. Loss: 0.28941667079925537. Accuracy: 91.36
# Epoch: 2. Loss: 0.38507160544395447. Accuracy: 91.99
# Epoch: 3. Loss: 0.5694723725318909. Accuracy: 92.09
# Epoch: 4. Loss: 0.395939439535141. Accuracy: 92.11
# Epoch: 5. Loss: 0.48296740651130676. Accuracy: 92.31
# Epoch: 6. Loss: 0.283181369304657. Accuracy: 92.36
# Epoch: 7. Loss: 0.3426980674266815. Accuracy: 92.36
# Epoch: 8. Loss: 03861846327781677. Accuracy: 92.14
# Epoch: 9. Loss: 0.5513550639152527. Accuracy: 92.45
# Epoch: 10. Loss: 0.19756020605564117. Accuracy: 92.31
# Epoch: 11. Loss: 0.19502460956573486. Accuracy: 92.27
# Epoch: 12. Loss: 0.22119677066802979. Accuracy: 92.39
# Epoch: 13. Loss: 0.07563676685094833. Accuracy: 92.29
# Epoch: 14. Loss: 0.2214275300502777. Accuracy: 92.31
# Epoch: 15. Loss: 0.21185295283794403. Accuracy: 92.49
# Epoch: 16. Loss: 0.1210915595293045. Accuracy: 92.44
# Epoch: 17. Loss: 0.23013806343078613. Accuracy: 92.34
# Epoch: 18. Loss: 0.5404743552207947. Accuracy: 92.57
# Epoch: 19. Loss: 0.34365323185920715. Accuracy: 92.58
# Epoch: 20. Loss: 0.19733381271362305. Accuracy: 92.44
# Epoch: 21. Loss: 0.3526330590248108. Accuracy: 92.72
# Epoch: 22. Loss: 0.0788537934422493. Accuracy: 92.65
# Epoch: 23. Loss: 0.05600835382938385. Accuracy: 92.44
# Epoch: 24. Loss: 0.30678609013557434. Accuracy: 92.37
# Epoch: 25. Loss: 0.1992693990468979. Accuracy: 92.46
# Epoch: 26. Loss: 0.14886032044887543. Accuracy: 92.5
# Epoch: 27. Loss: 0.35290682315826416. Accuracy: 92.5
# Epoch: 28. Loss: 0.18025070428848267. Accuracy: 92.52
# Epoch: 29. Loss: 0.33533987402915955. Accuracy: 92.64
# Epoch: 30. Loss: 0.10078015923500061. Accuracy: 92.65
# Epoch: 31. Loss: 0.10145442932844162. Accuracy: 92.57
# Epoch: 32. Loss: 0.44652634859085083. Accuracy: 92.7
# Epoch: 33. Loss: 0.318610280752182. Accuracy: 92.43
# Epoch: 34. Loss: 0.05808462202548981. Accuracy: 92.44
# Epoch: 35. Loss: 0.12365058064460754. Accuracy: 92.58
# Epoch: 36. Loss: 0.04259800165891647. Accuracy: 92.58
# Epoch: 37. Loss: 0.39061176776885986. Accuracy: 92.55
# Epoch: 38. Loss: 0.15112923085689545. Accuracy: 92.58
# Epoch: 39. Loss: 0.840149998664856. Accuracy: 92.37
# Epoch: 40. Loss: 0.5034855604171753. Accuracy: 92.51
# Epoch: 41. Loss: 0.5105100274085999. Accuracy: 92.44
# Epoch: 42. Loss: 0.11851473152637482. Accuracy: 92.67
# Epoch: 43. Loss: 0.28970497846603394. Accuracy: 92.74
# Epoch: 44. Loss: 0.2377086579799652. Accuracy: 92.59
# Epoch: 45. Loss: 0.25672367215156555. Accuracy: 92.45
# Epoch: 46. Loss: 0.18315061926841736. Accuracy: 92.54
# Epoch: 47. Loss: 0.6673844456672668. Accuracy: 92.53
# Epoch: 48. Loss: 0.24277451634407043. Accuracy: 92.49
# Epoch: 49. Loss: 0.11376694589853287. Accuracy: 92.49

"""