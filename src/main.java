import java.rmi.server.ExportException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class main {
    public static void main(String[] args) {
        List<List<VectorF>> data = Utils.parseNNData("NNData\\mnist_test.txt", 10, 28*28, true);

        int[] NN_layer_config = {
                //Inputs
                784,
                //Hidden layers
                64,
                //Output layer
                10
        };
        NeuralNetwork nn = new NeuralNetwork(NN_layer_config);

        double start = System.currentTimeMillis();
        //Train
//        for (int i = 0; i < 200000; i++) {
//            int random_index = (int)(Math.random() * (data.get(0).size()-1));
//            VectorF input = data.get(0).get(random_index);
//            VectorF target = data.get(1).get(random_index);
//            nn.train(input, target);
//        }
//
//        System.out.println("The training time was (s): ");
//        System.out.println((System.currentTimeMillis()- start)/1000.0);
//        System.out.println();

        //nn.saveState("NNSaveStates\\MNIST_state.txt");
        nn.restoreState("NNSaveStates\\MNIST_state.txt");
//        System.out.println("The result of the prediction was: ");
        nn.predict(data.get(0).get(130)).print();
    }

    void trainNN(int laps) {

    }

}
