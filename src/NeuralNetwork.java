import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    private List<MatrixF> weights;

    private List<VectorF> biases;

    private float learning_rate = 0.1f;

    NeuralNetwork(int[] layer_config) {
        //Initialize all weights
        weights = new ArrayList<>();
        //Start from index 1 since the weights are in relation to the previous layer
        for (int i = 1; i < layer_config.length; i++) {
            weights.add(new MatrixF(layer_config[i], layer_config[i-1]));
        }

        //Randomize all weights
        for (MatrixF weight : weights) {
            weight.randomize(-1.0f, 1.0f);
        }

        //Initialize biases
        biases = new ArrayList<>();
        //Start the bias creating from the first hidden layer, the inputs don't have biases
        for (int i = 1; i < layer_config.length; i++) {
            biases.add(new VectorF(layer_config[i]));
        }

        //Randomize all biases
        for (VectorF bias : biases) {
            bias.randomize(-1.0f, 1.0f);
        }

    }

    List<VectorF> feedForward(VectorF input) {
        List<VectorF> results = new ArrayList<>();

        for (int i = 0; i < weights.size(); i++) {
            //Use the previous layers output as a base for the current layers output
            MatrixF result;
            if (i == 0) {
                //If we are at the first iteration, use the inputs as the "input" value
                result = MatrixF.multiply(weights.get(i), input);
            } else {
                //Else use the results from the previous layer as "input" value
                result = MatrixF.multiply(weights.get(i), results.get(results.size()-1));
            }
            result.add(biases.get(i));
            result.map(SmallMath::sigmoid);
            //Make the result into a vector before adding it to the list
            results.add(VectorF.fromMatrixF(result));
        }
        return results;
    }

    void train(VectorF input, VectorF target) {
        List<VectorF> values = feedForward(input);
        //Add the inputs to the values to calculate correctly
        values.add(0, input);

        List<VectorF> errors = new ArrayList<>();
        //Go backwards from the output towards the input
        for (int i = weights.size() - 1; i >= 0; i--) {
            VectorF error;
            if (i == weights.size() - 1) {
                error = VectorF.subtract(target, values.get(i + 1));
            } else {
                MatrixF weight_t = MatrixF.transpose(weights.get(i+1));
                error = VectorF.fromMatrixF(MatrixF.multiply(weight_t, errors.get(0)));
            }
            errors.add(0, error);

            VectorF gradients = VectorF.map(values.get(i + 1), SmallMath::dsigmoid);
            gradients = VectorF.elementMultiply(gradients, errors.get(0));
            gradients.multiply(learning_rate);

            MatrixF value_t = MatrixF.transpose(values.get(i));
            MatrixF weight_deltas = MatrixF.multiply(gradients, value_t);

            weights.get(i).add(weight_deltas);
            biases.get(i).add(gradients);
        }

//        MatrixF output = MatrixF.multiply(weights.get(1), values.get(1));
//        output.add(biases.get(1));
//        output.map(SmallMath::sigmoid);
//
//        values.add(VectorF.fromMatrixF(output));
//
//        MatrixF output_errors = MatrixF.subtract(target, values.get(2));
//
//        MatrixF gradients = MatrixF.map(values.get(2), SmallMath::dsigmoid);
//        gradients = MatrixF.elementMultiply(gradients, output_errors);
//        gradients.multiply(learning_rate);
//
//        MatrixF hidden_t = MatrixF.transpose(values.get(1));
//        MatrixF weight_ho_deltas = MatrixF.multiply(gradients, hidden_t);
//
//        weights.get(1).add(weight_ho_deltas);
//        biases.get(1).add(gradients);
//
//        MatrixF who_t = MatrixF.transpose(weights.get(1));
//        MatrixF hidden_errors = MatrixF.multiply(who_t, output_errors);
//
//        MatrixF hidden_gradient = MatrixF.map(values.get(1), SmallMath::dsigmoid);
//        hidden_gradient = MatrixF.elementMultiply(hidden_gradient, hidden_errors);
//        hidden_gradient.multiply(learning_rate);
//
//        MatrixF inputs_t = MatrixF.transpose(values.get(0));
//        MatrixF weight_ih_deltas = MatrixF.multiply(hidden_gradient, inputs_t);
//
//        weights.get(0).add(weight_ih_deltas);
//        biases.get(0).add(hidden_gradient);
    }

    VectorF predict(VectorF input) {
        List<VectorF> results = feedForward(input);
        return results.get(results.size()-1);
    }

    void saveState(String file_path) {
        try {
            FileWriter writer = new FileWriter(file_path, false);
            for (int i = 0; i < weights.size(); i++) {
                writer.write(weights.get(i).toCSV());
                if (i != weights.size() - 1) {
                    writer.write("&\n");
                }
            }
            writer.write("---\n");
            for (int i = 0; i < biases.size(); i++) {
                writer.write(biases.get(i).toCSV());
                if (i != biases.size() - 1) {
                    writer.write("&\n");
                }
            }
            writer.close();
        } catch (Exception e) {
            System.out.println("Problem when writing to file!");
            System.out.println(e);
        }
    }

    void restoreState(String file_path) {
        //Read the data
        String file_content = "";
        try {
            file_content = Utils.readFile(file_path);
        } catch (Exception e) {
            System.out.println("Could not read weight file!");
            System.out.println(e);
        }

        if (!file_content.equals("")) {
            //Start the treatment of the data
            String[] data_parts = file_content.split("---\n");
            String[] weight_matrix_strings = data_parts[0].split("&\n");
            //Check the dimensions to ensure that they match with the current weight setup
            if (weight_matrix_strings.length != weights.size()) {
                throw new IllegalArgumentException("Wrong amount of weights!");
            }
            for (int i = 0; i < weight_matrix_strings.length; i++) {
                weights.get(i).fromCSV(weight_matrix_strings[i]);
            }

            //Biases
            String[] bias_matrix_strings = data_parts[1].split("&\n");
            //Check the dimensions to ensure that they match with the current weight setup
            if (bias_matrix_strings.length != weights.size()) {
                throw new IllegalArgumentException("Wrong amount of biases!");
            }
            for (int i = 0; i < bias_matrix_strings.length; i++) {
                biases.get(i).fromCSV(bias_matrix_strings[i]);
            }
        }
    }
}
