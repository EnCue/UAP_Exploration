
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import collections
import copy

# from torch.autograd.gradcheck import zero_gradients
def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)

# DEEPFOOL ALGORITHM
def deepfool(image, net, num_classes, overshoot, max_iter):

    """
       :param image:
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    f_image = net.forward(Variable(image, requires_grad=True)).data.cpu().numpy().flatten()
    I = f_image.argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    # input_shape = image.to(device).numpy().shape
    
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    # x = Variable(pert_image[None, :], requires_grad=True)
    x = Variable(pert_image, requires_grad=True)
    fs = net.forward(x)
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()
        # grad_orig = x.grad.data.to(device).numpy().copy()

        for k in range(1, num_classes):
            # ! FLAG
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()
            # cur_grad = x.grad.data.to(device).numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()
            # f_k = (fs[0, I[k]] - fs[0, I[0]]).data.to(device).numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        pert_image = image.cpu() + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x.to(device).view(1, 1, 28, 28))
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    return (1+overshoot)*r_tot, loop_i, label, k_i, pert_image


def project_perturbation(perturbation, r, p):

    if p == 2:
        # perturbation = perturbation * min(1, r / np.linalg.norm(perturbation.flatten(1)))
        perturbation = perturbation * min(1, r / np.linalg.norm(perturbation.flatten()))
    elif p == np.inf:
        perturbation = np.sign(perturbation) * np.minimum(abs(perturbation), r)
    return perturbation

# Function to get fooling rate

def get_foolingrate(device, net, testset_loader, perturbation):
    
    with torch.no_grad():
        
        # Finding labels for original images
        clean_correct, total, disagreement = 0, 0, 0
        for images, labels in testset_loader:
            images, labels = images.to(device), labels.to(device)
            
            perturbed_images = images + torch.tensor(perturbation).to(device)
        
            clean_outputs = net(images)
            perturbed_outputs = net(perturbed_images)
        
            clean_predictions = torch.max(clean_outputs, 1)[1].to(device)
            perturbed_predictions = torch.max(perturbed_outputs, 1)[1].to(device)
            # predictions_list.append(predictions)
            clean_correct += (clean_predictions == labels).sum().item()
            # perturbed_correct += (perturbed_predictions == labels).sum().item()
            disagreement += (clean_predictions != perturbed_predictions).sum().item()
        
            total += len(labels)
        
        clean_accuracy = 100 * clean_correct / total
        fooling_rate = 100 * disagreement / total

        return clean_accuracy, fooling_rate

def generate_uap(device, traindata, testset_loader, net, delta=0.2, max_iter_uni=10, xi=1, p=2, num_classes=10, overshoot=0.2, max_iter_df=np.inf):
    '''
    :param trainset: Pytorch Dataloader with train data
    :param testset: Pytorch Dataloader with test data
    :param net: Network to be fooled by the adversarial examples
    :param delta: 1-delta represents the fooling_rate, and the objective
    :param max_iter_uni: Maximum number of iterations of the main algorithm
    :param p: Only p==2 or p==infinity are supported
    :param num_class: Number of classes on the dataset
    :param overshoot: Parameter to the Deep_fool algorithm
    :param max_iter_df: Maximum iterations of the deep fool algorithm
    :return: perturbation found (not always the same on every run of the algorithm)
    '''

    net.eval()

    # USE DATASETS INSTEAD OF DATALOADERS
    # ---------------
    # DIVIDE TO NORMALIZE
    trainset_np = traindata.data.numpy().astype(np.float32) / 255

    # Setting the number of images to 300  (A much lower number than the total number of instances on the training set)
    # To verify the generalization power of the approach
    # 100? 
    num_img_trn = 500
    # index_order = np.arange(num_img_trn)
    index_order = np.random.randint(0, trainset_np.shape[0], size=num_img_trn)

    # Initializing the perturbation
    v=np.zeros((1, 1, 28, 28), dtype=np.float32)
    v_optim = v

    #Initializing fooling rate and iteration count
    fooling_rate = 0.0
    max_fooling_rate = fooling_rate
    iter = 0

    # fooling_rates=[0]
    # clean_accuracies = [accuracy]
    # perturbed_accuracies = [accuracy]
    # Begin of the main loop on Universal Adversarial Perturbations algorithm
    while fooling_rate < (1-delta) * 100 and iter < max_iter_uni:
        # Shuffling to randomize initial conditions
        np.random.shuffle(index_order)

        for index in index_order:
            
            np_i = trainset_np[index, :, :]
            x_i = Variable(torch.tensor(np_i).to(device).view(1, 1, 28, 28))
            xhat_i = x_i + torch.tensor(v).to(device)
            # Feeding the original image to the network and storing the label returned
           
            y_i = torch.max(net(x_i), 1)[1].to(device).item()
            yhat_i = torch.max(net(xhat_i), 1)[1].to(device).item()

            # If the label of both images is the same, the perturbation v needs to be updated
            if y_i == yhat_i:
                # print(">> k =", np.where(index==index_order)[0][0], ', pass #', iter, end='; ')

                # Finding a new minimal perturbation with deepfool to fool the network on this image
                dr, iter_k, label, k_i, pert_image = deepfool(xhat_i, net, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df)

                # Adding the new perturbation found and projecting the perturbation v and data point xi on p.
                if iter_k < max_iter_df-1:
                    
                    v += dr

                    v = project_perturbation(v, xi, p)

        iter = iter + 1

        # CHECKING FOOLING RATE
        # ---------------------

        _, fooling_rate = get_foolingrate(device, net, testset_loader, v)

        # print("FOOLING RATE: ", fooling_rate)
        # print("ACCURACY: ", accuracy)

        if fooling_rate > max_fooling_rate:
            # print('UPDATING v_optim')
            v_optim = np.copy(v)
            max_fooling_rate = fooling_rate
    
    print("Maximal fooling rate after %s iterations: %s" % (iter, max_fooling_rate))

    
    return v_optim