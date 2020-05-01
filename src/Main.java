public class Main {
    public static void main(String[] args) {
        MNISTNN mnistnn = new MNISTNN();
        mnistnn.train();
        mnistnn.test();
    }
}
