import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class main {
    public static void main(String[] args) {
        float[][] training_data = {
                {0.0f, 0.0f},
                {1.0f, 0.0f},
                {0.0f, 1.0f},
                {1.0f, 1.0f}
        };
        List<VectorF> data_vectors = new ArrayList<>();

        float[][] training_targets = {
                {0.0f},
                {1.0f},
                {1.0f},
                {0.0f}
        };
        List<VectorF> target_vectors = new ArrayList<>();

        for (int i = 0; i < training_data.length; i++) {
            data_vectors.add(new VectorF(training_data[i]));
            target_vectors.add(new VectorF(training_targets[i]));
        }

        int[] hidden_layer_config = {
                //Inputs
                2,
                //Hidden layers
                4,
                //Output layer
                1
        };
        NeuralNetwork nn = new NeuralNetwork( hidden_layer_config);

        double start = System.currentTimeMillis();
        //Train
        for (int i = 0; i < 1000000; i++) {
            int random_index = (int)(Math.random() * ((3) + 1));
            VectorF input = data_vectors.get(random_index);
            VectorF target = target_vectors.get(random_index);
            nn.train(input, target);
        }

        System.out.println("The trainging time was (s): ");
        System.out.println((System.currentTimeMillis()- start)/1000.0);
        System.out.println();

        nn.saveState("test.txt");
        //nn.restoreState("test.txt");
        //Test
        System.out.println("0 and 0:");
        nn.predict(data_vectors.get(0)).print();
        System.out.println("1 and 0:");
        nn.predict(data_vectors.get(1)).print();
        System.out.println("0 and 1:");
        nn.predict(data_vectors.get(2)).print();
        System.out.println("1 and 1:");
        nn.predict(data_vectors.get(3)).print();
        //MatrixF.test();
//        double start = System.currentTimeMillis();
//        MatrixF m1 = new MatrixF(50,50);
//        MatrixF m2 = new MatrixF(50,50);
//        for (int i = 0; i < 100000; i++) {
//            MatrixF.multiply(m1, m2);
//        }
//        System.out.println((System.currentTimeMillis() - start)/1000);
    }
}
