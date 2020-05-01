import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

class NeuralNetwork {
    private List<MatrixF> weights;

    private List<VectorF> biases;

    private float learning_rate = 0.1f;

    private int lapping_phase_length = 10000;

    //Default to the sigmoid activation function
    private NNActivationFunction AF = new NNActivationFunction();

    private float current_phase_error = 0.0f;

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
            result.map(AF.activation_function);
            //Make the result into a vector before adding it to the list
            results.add(VectorF.fromMatrixF(result));
        }
        return results;
    }

    void train(List<List<VectorF>> data, int laps) {
        System.out.println("Training network for " + laps + " laps");
        double start = System.currentTimeMillis();

        //Start at 1 for the error checking to understand what's going on
        for (int i = 1; i <= laps; i++) {
            int random_index = (int)(Math.random() * (data.get(0).size()-1));
            VectorF input = data.get(0).get(random_index);
            VectorF target = data.get(1).get(random_index);

            List<VectorF> values = feedForward(input);
            //Add the inputs to the values to calculate correctly
            values.add(0, input);

            List<VectorF> errors = new ArrayList<>();
            //Go backwards from the output towards the input
            for (int j = weights.size() - 1; j >= 0; j--) {
                VectorF error;
                if (j == weights.size() - 1) {
                    error = VectorF.subtract(target, values.get(j + 1));
                } else {
                    MatrixF weight_t = MatrixF.transpose(weights.get(j + 1));
                    error = VectorF.fromMatrixF(MatrixF.multiply(weight_t, errors.get(0)));
                }
                errors.add(0, error);

                VectorF gradients = VectorF.map(values.get(j + 1), AF.fake_der_activation_function);
                gradients = VectorF.elementMultiply(gradients, errors.get(0));
                gradients.multiply(learning_rate);

                MatrixF value_t = MatrixF.transpose(values.get(j));
                MatrixF weight_deltas = MatrixF.multiply(gradients, value_t);

                weights.get(j).add(weight_deltas);
                biases.get(j).add(gradients);
            }

            updateError(i, errors.get(biases.size() - 1), lapping_phase_length);
        }
        current_phase_error = 0.0f;

        System.out.print("The training time was (s): ");
        System.out.println((System.currentTimeMillis()- start)/1000.0);
        System.out.println();
    }

    void test(List<List<VectorF>> data, int laps) {
        System.out.println("Testing network for " + laps + " laps");
        double start = System.currentTimeMillis();

        //Start at 1 for the error checking to understand what's going on
        for (int i = 1; i <= laps; i++) {
            int random_index = (int)(Math.random() * (data.get(0).size()-1));
            VectorF input = data.get(0).get(random_index);
            VectorF target = data.get(1).get(random_index);

            List<VectorF> values = feedForward(input);
            //Add the inputs to the values to calculate correctly
            values.add(0, input);

            List<VectorF> errors = new ArrayList<>();
            //Go backwards from the output towards the input
            for (int j = weights.size() - 1; j >= 0; j--) {
                VectorF error;
                if (j == weights.size() - 1) {
                    error = VectorF.subtract(target, values.get(j + 1));
                } else {
                    MatrixF weight_t = MatrixF.transpose(weights.get(j + 1));
                    error = VectorF.fromMatrixF(MatrixF.multiply(weight_t, errors.get(0)));
                }
                errors.add(0, error);
            }
            //For testing the phase is as long as the amount of laps, no need to do it more than once
            updateError(i, errors.get(biases.size() - 1), laps);
        }
        current_phase_error = 0.0f;

        System.out.print("The testing time was (s): ");
        System.out.println((System.currentTimeMillis()- start)/1000.0);
        System.out.println();
    }

    private void updateError(int lap, VectorF output_error, int phase_length) {
        float error = 0.0f;
        for (float[] e : output_error.data) {
            error += Math.abs(e[0]);
        }
        error = error / output_error.rows;

        current_phase_error += error / phase_length;

        if (lap % phase_length == 0) {
            System.out.println("Error for " + phase_length + " laps: " + current_phase_error);
            current_phase_error = 0.0f;
        }
    }

    VectorF predict(VectorF input) {
        List<VectorF> results = feedForward(input);
        return results.get(results.size()-1);
    }

    void setLearningRate(float lR) {
        learning_rate = lR;
    }

    void setPhaseLength(int phase_length) {
        lapping_phase_length = phase_length;
    }

    void setAF(NNActivationFunction activation_function) {
        AF = activation_function;
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
