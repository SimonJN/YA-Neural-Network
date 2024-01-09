public class SmallMath {
    static float sigmoid(float x) {
        return (float)(1 / (1 + Math.exp(-x)));
    }

    static float dsigmoid(float x) {
        //Fake derivative, assuming that x has already been put through sigmoid
        return x * (1 - x);
    }

    static float tanh(float x) {
        return (float)((Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x)));
    }

    static float dtanh(float x) {
        return 1 - x * x;
    }
}
