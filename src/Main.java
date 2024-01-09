public class Main {
    public static void main(String[] args) {
        MNISTNN mnistnn = new MNISTNN();
        mnistnn.train();
        //mnistnn.restore();
        mnistnn.test();

        /*BTCNN btcnn = new BTCNN();
        btcnn.train();
        btcnn.test();*/
    }
}
