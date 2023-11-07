import torch
import model_CNN
import data_files
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import gc
import csv

"""
the main program
here we can do different things through the menu() function

"""


"""
some info about the global variables below:
device -> specifying on what device we can run the model
          generally it is better to run it on a GPU supporting CUDA

model -> assigning the model as an object ( and passing it to the device )

optimizer -> ADAM Optimizer
            refer here: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

loss_fn -> Using the CrossEntropyLoss function to calculate the mean loss of the model
        refer here: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

loaders -> calling the loaders function to create dataloaders

dict_of_dates -> creating datasets

checkpoint -> calling it as a global function, will be useful in the future
                Will be used to save the state of the model

"""
# we unzip the files before starting
data_files.unziping()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model_CNN.Net().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.CrossEntropyLoss()

loaders = data_files.loader()

dict_of_datas = data_files.datasets()

checkpoint = None

def show_image_for_hoomans(matrix, value, message='Label: '):
    """

    this function takes a tensor and a value
    ->
    prints a graph ( using pyplot ) with the image
    and the value as a TITLE
    ...

    :param matrix: a matrix containing the pixels to graph
    :param value: The value to print as a title
    :param message: an optional attribute to change the title
    """
    plt.suptitle(f'{message}{chr(value + 65)}')
    plt.imshow(matrix, cmap='gray')
    plt.show()


def train(epoch):
    """

    this function performs the TRAINING of the
    model with every batch of a loader

    STEP 0 - setting the model to training mode!


    LOOPING THROUGH THE BATCHES ( using enumerate() )
    STEP BY STEP:

    1. assigning the data, and the target to the used device
    2. zero_grad() to ensure that updating the parameter is done correctly
    3. output -> passing the data attribute to the model, for its training
    4. loss -> calling the loss_fn object ( CrossEntropyLoss )
    5. loss.backward() -> doing the backpropagation to teach the model
    6. optimizer.step() -> performs a single optimization step

    in the end a log line as well :)
    ...

    :param epoch: the num of batches for
    :return: no return
    """

    model.train()

    for batch_idx, (data, target) in enumerate(loaders['train']):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = loss_fn(output, target)

        loss.backward()
        # backpropagation
        optimizer.step()
        # optimization step
        if batch_idx % 20 == 0:
            # (long) log line:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loaders['train'].dataset)} ({100 * batch_idx / len(loaders['train']):.0f}%)]\t Loss: {loss.item():.6}")
            # pardon me for too long log lines


def validation():
    """

    this function performs the VALIDATION of the model
    using the validation dataloader and spits out an accuracy percentage

    STEP 0 -> setting the model to eval mode

    Having torch.no_grad() ON !
    We pass the validation data to the model,
    which PREDICTS the target
    then we check whether it fits the REAL TARGET
    correct -> the amount of good predictions

    then we do a log line showing us how accurate the model is


    in the end we use the global variable checkpoint to save the current state of the model

    :return:
    """
    model.eval()

    val_loss = 0
    correct = 0
    global checkpoint

    with torch.no_grad():
        for data, target in loaders['valid']:

            data, target = data.to(device), target.to(device)

            output = model(data)

            val_loss += loss_fn(output, target).item()

            pred = output.argmax(dim=1, keepdim=True)
            # below we match every pred to its correspondend targe and sum it up

            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(loaders['valid'].dataset)

    accuracy = (100. * correct) / len(loaders['valid'].dataset)

    print(f"\nValidation set: Average loss: {val_loss:.4f}, Accuracy {correct}/{len(loaders['valid'].dataset)} ({accuracy:.0f}%)\n")
    # pardon me for too long log lines
    if accuracy > 85.0:
        checkpoint = save_checkpoint(model, optimizer)


def save_checkpoint(the_model, the_optimizer) -> dict:
    """

    function used to save the current state of the model and the optimizer
    we save it in a dict of object.state_dict()

    :param the_model: just the model
    :param the_optimizer: just the optimizer
    :return checkpoint_dict: the checkpoint of the model and the optimizer ( as dict )

    """
    print("<= Checkpoint saved!")
    checkpoint_dict = {"model_state_dict": the_model.state_dict(),
                       "optimizer_state_dict": the_optimizer.state_dict()
                       }
    return checkpoint_dict


def train_val(epoch_num: int):
    """

    Training and evaluation afterwards :)

    ...

    :param epoch_num: the num of how many times should the model train from a different batch
    :type epoch_num: int
    """

    for epoch in range(1, epoch_num):
        train(epoch)
        validation()


def test_one_value(index: int):
    """

    this function tests just one value of the validation dataset with a given index

    STEP 0 -> model into eval mode

    STEP_BY_STEP:
    STEP 1 -> get the VALIDATION data and targets from the dict_of_datas
    STEP 2 ->  we add one more dimension to make up for the channel
    STEP 3 -> output is the model fed with the validation data
    STEP 4 -> prediction <=> made from the argmax() function as only the item() of it
    STEP 5 -> to generate the image, we have to change the dimension to a 28x28
                we pass it to the cpu and create a numpy array again
    STEP 6 -> visualization through the show_image_for_hoomans() function


    :param index: the index of a dataset
    :type index: int
    """
    model.eval()

    val_datset = dict_of_datas['validation_dataset']
    data, target = val_datset[index]

    data = data.unsqueeze(0).to(device)

    output = model(data)

    prediction = output.argmax(dim=1, keepdims=True).item()

    image = data.squeeze(0).squeeze(0).cpu().numpy()

    show_image_for_hoomans(image, prediction, message='Prediction: ')


def the_test():
    """

    this function provides predictions for the "test_data" tensor imported from the data_files.py
    It stores its predictions in a csv file in:
        "answer_folder//submission.csv"

    the csv contains values such as:
    index, class
    0, 6
    1, 8
    etc...

    index -> the index of the sample from test_data tensor
    class -> the target class predicted by the model
    to test the data we use the same steps as in the function test_one_value() but with a whole tensor of samples

    """

    model.eval()

    test_datset = dict_of_datas['test_data']

    with open('answer_folder//submission.csv', 'w', newline='') as file:

        writer = csv.writer(file)
        writer.writerow(['index', 'class'])
        with torch.no_grad():
            for idx, test_data in enumerate(test_datset):

                test_data = test_data.unsqueeze(0).to(device)

                test_output = model(test_data)

                prediction = test_output.argmax(dim=1, keepdims=True).item()

                writer.writerow([idx, prediction])


def saving_model(state, filename="model_checkpoints//saved_model.pth.ptr"):
    """

    this function is responsible for saving the model as a pth.ptr file
    ...

    :param state: the state of the model
    :param filename: the name of the file, with its directory
    ( might cause issues after changing it, wouldn't recommend)
    """

    print("<<< Model saved!")
    torch.save(state, filename)


def load_model(model_checkpoint):
    """

    this function is responsible for loading in a saved model from a given directory
    ...

    :param model_checkpoint: a torch.load() function should be passed
    """
    print(">>> Model loaded")
    model.load_state_dict(model_checkpoint['model_state_dict'])
    optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])


def menu():
    """

    this logging function is a small interface in the command line
    CONTAINS AN INFINITE LOOP STATEMENT, WATCH OUT !!!
    ...

    if and elif statements:

    choice == 1 -> putting in the num of epochs used in training, then training the model
    choice == 2 -> One value testing, should input an integer index
    choice == 3 -> evaluating the test data tensor
    choice == 4 -> saving and loading up a state of the model
    choice == 5 -> print the used device

    :return:
    """

    while True:
        print('\t\t\t -#- Handwritten uppercase letters recog -#-')
        print("0 - quit ",
              "\n1 - train model ",
              "\n2 - one value test ",
              "\n3 - test prediction in csv ",
              "\n4 - save or load model",
              "\n5 - check device")
        choice = input("\nChoice: ")

        if choice == '0':
            break
        elif choice == '1':
            print("\t\t--- Training ---")
            num = int(input("How many Epochs: "))
            train_val(num + 1)
        elif choice == '2':
            print('\t\t--- One value ---')
            idx = int(input("Index for the value to predict: "))
            test_one_value(idx)
        elif choice == '3':
            print("\t\t--- Using X_test to predict ---")
            the_test()
        elif choice == '4':
            print("1 - Save model\n2 - Load model")
            choice_2 = input("Choice: ")
            if choice_2 == '1':
                saving_model(checkpoint)
            elif choice_2 == '2':
                load_model(torch.load("model_checkpoints//saved_model.pth.ptr"))
        elif choice == '5':
            print("\n\t\tDevice: ")
            print("\t", device)
        else:
            continue


def main():
    """
    the main function
    it first unzips, then the menu() is called
    in the end, as a safety measure we use a garbage collector
    No one knows, what could happen ...
    :return:
    """
    print("INITIALIZING!")
    menu()
    gc.collect()


if __name__ == '__main__':
    main()
