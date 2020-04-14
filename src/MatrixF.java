import java.util.function.Function;

public class MatrixF {
    protected int rows;
    protected int cols;
    protected float[][] data;

    MatrixF(int num_rows, int num_columns) {
        rows = num_rows;
        cols = num_columns;
        data = new float[num_rows][num_columns];
    }

    MatrixF(float[][] input_data) {
        rows = input_data.length;
        cols = input_data[0].length;
        data = input_data;
    }

    void copy(MatrixF m) {
        rows = m.rows;
        cols = m.cols;
        data = m.data;
    }

    void print() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                System.out.print(data[i][j] + "\t\t");
            }
            System.out.println();
        }
    }

    public String toCSV() {
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.append(data[i][j]).append(",");
            }
            result.append("\n");
        }
        return result.toString();
    }

    void fromCSV(String csv) {
        //Assemble the data into the matrix
        String[] row_strings = csv.split("\n");
        if (row_strings.length != rows){
            throw new IllegalArgumentException("Wrong amount of rows!");
        }
        for (int i = 0; i < row_strings.length; i++) {
            String[] col_strings = row_strings[i].split(",");
            if (col_strings.length != cols){
                throw new IllegalArgumentException("Wrong amount of cols!");
            };

            for (int j = 0; j < col_strings.length; j++) {
                data[i][j] = Float.parseFloat(col_strings[j]);
            }
        }
    }

    //Any function that might change the dimensions of the matrix are static, if the rows and cols stay the same the function can have both a static and a non-static version

    //Applies a supplied method to all values in the matrix
    void map(Function <Float, Float> method) {
        copy(map(this, method));
    }

    static MatrixF map(MatrixF m, Function <Float, Float> method) {
        MatrixF result = new MatrixF(m.rows, m.cols);
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                result.data[i][j] = method.apply(m.data[i][j]);
            }
        }
        return result;
    }

    //Randomize all values in the matrix
    void randomize(float min, float max) {
        copy(randomize(this, min, max));
    }

    static MatrixF randomize(MatrixF m, float min, float max) {
        return map(m, e -> ((float)Math.random()) * (max + 1) + min);
    }

    //Adds the supplied matrix to this matrix using element-wise addition
    void add(MatrixF m) {
        copy(add(this, m));
    }

    static MatrixF add(MatrixF m1, MatrixF m2) {
        if (m1.cols != m2.cols || m1.rows != m2.rows) {
            throw new IllegalArgumentException("Both matrices must be of equal size!");
        }
        MatrixF result = new MatrixF(m1.rows, m1.cols);
        for (int i = 0; i < m1.rows; i++) {
            for (int j = 0; j < m1.cols; j++) {
                result.data[i][j] = m1.data[i][j] + m2.data[i][j];
            }
        }
        return result;
    }

    //Subtracts the supplied matrix from this matrix using element-wise subtraction
    void subtract(MatrixF m) {
        copy(subtract(this, m));
    }

    static MatrixF subtract(MatrixF m1, MatrixF m2) {
        if (m1.cols != m2.cols || m1.rows != m2.rows) {
            throw new IllegalArgumentException("Both matrices must be of equal size!");
        }
        MatrixF result = new MatrixF(m1.rows, m1.cols);
        for (int i = 0; i < m1.rows; i++) {
            for (int j = 0; j < m1.cols; j++) {
                result.data[i][j] = m1.data[i][j] - m2.data[i][j];
            }
        }
        return result;
    }

    //Scale the matrix using a scalar
    void multiply(float scalar) {
        copy(multiply(this, scalar));
    }

    static MatrixF multiply(MatrixF m, float scalar) {
        return map(m, e -> e * scalar);
    }

    //Do matrix multiplication(i.e. NOT SCALING) with the supplied matrices and return the result
    static MatrixF multiply(MatrixF m1, MatrixF m2) {
//        System.out.println("Multiplying:");
//        m1.print();
//        System.out.println("With:");
//        m2.print();
        if (m1.cols != m2.rows) {
            throw new IllegalArgumentException("m1.cols must be equal to m2.rows!");
        }
        MatrixF out = new MatrixF(m1.rows, m2.cols);
        for (int i = 0; i < m1.rows; i++) {
            for (int j = 0; j < m2.cols; j++) {
                float value = 0.0f;
                for (int k = 0; k < m1.cols; k++) {
                    value += m1.data[i][k] * m2.data[k][j];
                }
                out.data[i][j] = value;
            }
        }
//        System.out.println("The multiplication resulted in:");
//        out.print();
        return out;
    }

    void elementMultiply(MatrixF m) {
        copy(elementMultiply(this, m));
    }

    static MatrixF elementMultiply(MatrixF m1, MatrixF m2) {
        if (m1.cols != m2.cols ||m1.rows != m2.rows) {
            throw new IllegalArgumentException("Both matrices must have equal size!");
        }
        MatrixF result = new MatrixF(m1.rows, m1.cols);
        for (int i = 0; i < m1.rows; i++) {
            for (int j = 0; j < m1.cols; j++) {
                result.data[i][j] = m1.data[i][j] * m2.data[i][j];
            }
        }
        return result;
    }

    //Transpose the matrix, meaning to swap rows and columns, and return the result
    static MatrixF transpose(MatrixF m) {
        //Swap the rows and the columns
        MatrixF out = new MatrixF(m.cols, m.rows);
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                out.data[j][i] = m.data[i][j];
            }
        }
        return out;
    }

    void setData(float[][] data) {
        if (this.rows != data.length || this.cols != data[0].length) {
            throw new IllegalArgumentException("Size must be equal to the set rows and columns!");
        }
        this.data = data;
    }

    static void test() {
        //Matrix tests
        float[][] m1 = {
                {1.0f, 0.0f, 2.0f},
                {1.0f, 0.0f, 2.0f},
                {1.0f, 0.0f, 2.0f}
        };
        float[][] m2 = {
                {0.0f, 2.0f, 1.0f},
                {0.0f, 2.0f, 1.0f},
                {0.0f, 2.0f, 1.0f}
        };
        MatrixF m_1 = new MatrixF(m1);
        MatrixF m_2 = new MatrixF(m2);

        System.out.println("Matrix 1:");
        m_1.print();
        System.out.println("Matrix 2:");
        m_2.print();

        System.out.println("Mapping m1 to *2");
        MatrixF.map(m_1, e -> e * 2).print();

        System.out.println("Adding m1 and m2");
        MatrixF.add(m_1, m_2).print();

        System.out.println("Subtracting m2 from m1");
        MatrixF.subtract(m_1, m_2).print();

        System.out.println("Multiplying m1 by 3");
        MatrixF.multiply(m_1, 3).print();

        System.out.println("Matrix multiply m1 with m2");
        MatrixF.multiply(m_1, m_2).print();

        System.out.println("Element wise multiply m1 * m2");
        MatrixF.elementMultiply(m_1, m_2).print();

        System.out.println("Transpose m1");
        MatrixF.transpose(m_1).print();

        System.out.println("Randomizing -1.0f to 1.0f");
        m_1.randomize(-1.0f, 1.0f);
        m_1.print();
    }
}
