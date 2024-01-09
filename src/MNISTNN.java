import java.util.List;

public class MNISTNN{
    NeuralNetwork nn;

    MNISTNN() {
        int[] NN_layer_config = {
                //Inputs
                784,
                //Hidden layers
                64,
                //Output layer
                10
        };

        nn = new NeuralNetwork(NN_layer_config);

        NNActivationFunction AF = new NNActivationFunction();
        AF.activation_function = SmallMath::sigmoid;
        AF.fake_der_activation_function = SmallMath::dsigmoid;

        nn.setAF(AF);
    }

    void train() {
        List<List<VectorF>> data = Utils.parseNNData("NNData\\mnist_train.txt", 10, 28*28, true);

        nn.train(data, 200000);

        nn.saveState("NNSaveStates\\MNIST_state.txt");
    }

    void test() {
        List<List<VectorF>> data = Utils.parseNNData("NNData\\mnist_test.txt", 10, 28*28, true);

        nn.test(data, 10000);

    }

    void restore() {
        nn.restoreState("NNSaveStates\\MNIST_state.txt");
    }
}
