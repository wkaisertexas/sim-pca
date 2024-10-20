import torch

def basic_operations():
    # Creating two random tensors
    A = torch.randn(3, 3)
    B = torch.randn(3, 3)

    print("Matrix A:\n", A)
    print("Matrix B:\n", B)

    # Matrix multiplication (A * B)
    result = torch.matmul(A, B)
    print("Matrix Multiplication A * B:\n", result)

    # Element-wise multiplication
    elementwise_mult = A * B
    print("Element-wise Multiplication A * B:\n", elementwise_mult)

    # Adding matrices
    matrix_sum = A + B
    print("Matrix Sum A + B:\n", matrix_sum)

def neural_network_example():
    # Creating a random input tensor
    input_tensor = torch.randn(4, 10)

    # Creating a linear layer with input size 10 and output size 5
    linear_layer = torch.nn.Linear(10, 5)

    # Passing input through the linear layer
    output_tensor = linear_layer(input_tensor)

    print("Input Tensor:\n", input_tensor)
    print("Output Tensor after Linear Layer:\n", output_tensor)

def main():
    print("Running basic matrix operations...\n")
    basic_operations()

    print("\nRunning a simple neural network layer computation...\n")
    neural_network_example()

if __name__ == "__main__":
    main()

