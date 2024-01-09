import java.util.List;

public class BTCNN {
    NeuralNetwork nn;

    BTCNN() {
        int[] NN_layer_config = {
                //Inputs
                100,
                //Hidden layers
                64,
                //Output layer
                1
        };

        nn = new NeuralNetwork(NN_layer_config);

        NNActivationFunction AF = new NNActivationFunction();
        AF.activation_function = SmallMath::tanh;
        AF.fake_der_activation_function = SmallMath::dtanh;

        nn.setAF(AF);

        nn.setPhaseLength(10000);

        nn.setLearningRate(0.1f);
    }

    void train() {
        List<List<VectorF>> data = Utils.parseNNData("NNData\\BTC_train.txt", 1, 100, false);

        nn.train(data, 200000);

        nn.saveState("NNSaveStates\\BTC_state.txt");
    }

    void test() {
        List<List<VectorF>> data = Utils.parseNNData("NNData\\BTC_test.txt", 1, 100, false);

        nn.test(data, 5000);
    }

    void restore() {
        nn.restoreState("NNSaveStates\\BTC_state.txt");
    }
}
