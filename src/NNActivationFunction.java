import java.util.function.Function;

public class NNActivationFunction {
    //Default to sigmoid activation function
    public Function<Float, Float> activation_function = SmallMath::sigmoid;
    //This function is called a fake derivative because we assume that the values have already been put through the activation_function
    public Function<Float, Float> fake_der_activation_function = SmallMath::dsigmoid;
}
