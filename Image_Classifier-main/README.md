# Image_Classifier
## In-class Practice: Gemstone Image Classification

### Practical Platform and Environment Setup:
1. **Alibaba Magic:** You don't need to configure the environment separately
2. **Others:**
- Python Environment: Ensure you have Python installed (preferably Python 3.x).
- *Dependencies: Install required dependencies using pip:

```
pip install opencv-python numpy pillow matplotlib torch torchvision
```

### Code Flow:
1. **Data Preprocessing:**
- Select the computing device (GPU or CPU) through torch.device ("CUDA" if torch.cuda.is_availability () else "CPU").
- Set training related parameters, such as input image size, number of categories , dataset path , label dictionary , etc.
- Code check and decompress the dataset archive_trainzip. If the data does not exist, decompress it.
- Generate train.txt and eval.txt, respectively storing the image paths and corresponding labels for the training set and validation set.

2. **Model Initialization:**
- Use torch.nn Module defines a Convolutional Neural Network (CNN) for image classification.
- Mainly composed of multiple nn Conv2d,  nn.MaxPool2d, and nn Composed of Linear layers, it achieves feature extraction and classification.
- Using CrossEntropyLoss as the loss function and Adam as the optimizer.

3. **Training and Evaluation:**
- The code iterates through num_ epochs, performs forward propagation, loss calculation, gradient feedback on the training data, and updates the model parameters using an optimizer.
- After each epoch, calculate and print the loss and accuracy of the training sets.
- Save the best model weights using torch.
- Load the optimal model parameters, evaluate the model on the validation set, and output the accuracy. Use models to predict and visualize data on the test set.

### Filling in the Blanks:
1. **MyCNN Class:**
- Fill in the initialization for the Convolutional Neural Network model.

2. **Model training:**
- Fill in the missing code in the model training module to enable the program to run smoothly.


### Experimental Requirements:

1. **Implementing Convolutional Neural Network (CNN) with Supplementary Code - 1.5 points**

2. **Supplement the model training code to ensure the program runs smoothly, avg_acc ≥ 0.4 - 1.5 points**
    
**Total Score: 3 points**

### Other exploration:
1. **On the basis of the successful construction of convolutional neural networks, other networks can be attempted to be constructed**
    
2. **Using accuracy on the validation set as the evaluation metric, consider optimizing to improve accuracy**
  

### Submission:
Submit both the code and the running results in a single zip archive named **"学号_姓名_课堂练习2.ipynb"**.

**ddl: 2025/4/14 23:59**
