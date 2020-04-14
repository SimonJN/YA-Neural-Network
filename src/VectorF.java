import java.util.function.Function;

public class VectorF extends MatrixF {
    VectorF(int num_rows) {
        super(num_rows, 1);
    }

    VectorF(float[] input) {
        super(input.length, 1);
        float[][] data = new float[input.length][1];
        for (int i = 0; i < input.length; i++) {
            data[i][0] = input[i];
        }
        super.setData(data);
    }

    static VectorF fromMatrixF(MatrixF m) {
        if (m.cols != 1) {
            throw  new IllegalArgumentException("Can't create vector if the number of columns in the matrix is not equal to 1!");
        }
        float[] vector_values = new float[m.rows];
        for (int i = 0; i < vector_values.length; i++) {
            vector_values[i] = m.data[i][0];
        }
        return new VectorF(vector_values);
    }

    //Overrides to return vectors
    static VectorF map(VectorF v, Function<Float, Float> method) {
        return fromMatrixF(MatrixF.map(v, method));
    }

    static VectorF randomize(VectorF v, float min, float max) {
        return fromMatrixF(MatrixF.randomize(v, min, max));
    }

    static VectorF add(VectorF v1, VectorF v2) {
        return fromMatrixF(MatrixF.add(v1, v2));
    }

    static VectorF subtract(VectorF v1, VectorF v2) {
        return fromMatrixF(MatrixF.subtract(v1, v2));
    }

    static VectorF multiply(VectorF v, float scalar) {
        return fromMatrixF(MatrixF.multiply(v, scalar));
    }

    static VectorF elementMultiply(VectorF v1, VectorF v2) {
        return fromMatrixF(MatrixF.elementMultiply(v1, v2));
    }
}
