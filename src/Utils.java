import java.io.*;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

public class Utils {
    static String readFile(String path) {
        StringBuilder stringBuilder = new StringBuilder();
        try {
            InputStream file_i = new FileInputStream(path);
            BufferedInputStream file = new BufferedInputStream(file_i);
            int data = file.read();
            while (data != -1) {
                stringBuilder.append((char) data);
                data = file.read();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return stringBuilder.toString();
    }

    static List<List<VectorF>> parseNNData(String path, int num_outputs, int num_inputs, boolean one_hot) {
        double start_loading_time = System.currentTimeMillis();

        String training_data = Utils.readFile(path);
        String datapoints[] = training_data.split("\n");

        if (one_hot) {
            System.out.println("Handling " + datapoints.length + " datapoints with one hot enabled");
        } else {
            System.out.println("Handling " + datapoints.length + " datapoints");
        }

        float[][] inputs = new float[datapoints.length][num_inputs];
        float[][] targets = new float[datapoints.length][num_outputs];

        for (int i = 0; i < datapoints.length;i++) {
            String parts[] = datapoints[i].split(":");
            if (parts.length != 2) {
                throw new IllegalArgumentException("The network data was malformed! Wrong number of parts!");
            }

            //Parse targets
            String target_datapoints[] = parts[0].split(",");
            if (one_hot) {
                //One hot encoding creates a vector with all zeros except for one element which is one
                if (target_datapoints.length != 1) {
                    throw new IllegalArgumentException("The network data was malformed! Wrong number of target datapoints for one hot encoding!");
                }
                targets[i][Integer.parseInt(parts[0])] = 1.0f;
            } else {
                if (target_datapoints.length != num_outputs) {
                    throw new IllegalArgumentException("The network data was malformed! Wrong number of target datapoints!");
                }

                for (int j = 0; j < target_datapoints.length-1; j++) {
                    targets[i][j] = Float.parseFloat(target_datapoints[j]);
                }
            }

            //Parse inputs
            String input_datapoints[] = parts[1].split(",");
            if (input_datapoints.length != num_inputs) {
                throw new IllegalArgumentException("The network data was malformed! Wrong number of input datapoints!");
            }

            for (int j = 0; j < input_datapoints.length-1; j++) {
                inputs[i][j] = Float.parseFloat(input_datapoints[j]);
            }
        }

        List<VectorF> input_vectors = new ArrayList<>();
        List<VectorF> target_vectors = new ArrayList<>();

        for (int i = 0; i < datapoints.length; i++) {
            input_vectors.add(new VectorF(inputs[i]));
            target_vectors.add(new VectorF(targets[i]));
        }

        System.out.println("The data loading took: ");
        System.out.println((System.currentTimeMillis() - start_loading_time) / 1000.0f);

        List<List<VectorF>> parsed_data = new ArrayList<>();
        parsed_data.add(input_vectors);
        parsed_data.add(target_vectors);

        return parsed_data;
    }
}
