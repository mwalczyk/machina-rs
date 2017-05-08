use rulinalg::matrix::{Matrix, BaseMatrix, BaseMatrixMut};
use rulinalg::vector::Vector;

extern crate rand;
use rand::Rng;
use rand::distributions::{Normal, IndependentSample};

use std::f64;
use model::Supervised;

pub struct NeuralNetwork {
    /// The number of input neurons
    pub inputs: usize,

    /// A list that contains the number of neurons in each hidden layer, i.e. &[2, 3, 4]
    /// would create a neural network with 3 hidden layers that contain 2, 3, and 4 neurons,
    /// respectively
    pub hidden: usize, //&'a [usize],

    /// The number of output neurons
    pub outputs: usize,

    /// A matrix that contains all of the weights that will be used to feed-forward the inputs
    /// to the first hidden layer
    pub weights_ih: Matrix<f64>,

    /// A matrix that contains all of the weights that will be used to feed-forward the outputs
    /// of the first hidden layer to the output nodes
    pub weights_ho: Matrix<f64>,

    /// A hyperparameter controlling how quickly the network learns during backpropagation
    pub learning_rate: f64,
}

impl NeuralNetwork {
    pub fn new(inputs: usize, hidden: usize, outputs: usize, learning_rate: f64) -> NeuralNetwork {
        let (weights_ih, weights_ho) = NeuralNetwork::generate_weight_matrices(inputs, hidden, outputs);
        NeuralNetwork{inputs, hidden, outputs, weights_ih, weights_ho, learning_rate}
    }

    /// When initializing network weights, it is often beneficial to sample
    /// from a normal distribution that is centered around zero with a standard
    /// deviation of 1 / sqrt(number of incoming links)
    fn create_normal_distribution(number_in_next_layer: usize) -> Normal {
        let standard_deviation_ih = (number_in_next_layer as f64).powf(-0.5) * 0.5;
        Normal::new(0.0, standard_deviation_ih)
    }

    fn generate_weight_matrices(inputs: usize, hidden: usize, outputs: usize) -> (Matrix<f64>, Matrix<f64>) {
        // Generate samples from a standard normal distribution
        let normal_ih = NeuralNetwork::create_normal_distribution(hidden);
        let normal_ho = NeuralNetwork::create_normal_distribution(outputs);

        // Each entry w_ij represents the weight between the i-th neuron in the input
        // layer and the j-th neuron in the hidden layer
        let weights_ih: Matrix<f64> = Matrix::from_fn(hidden, inputs, |_, _| {
            normal_ih.ind_sample(&mut rand::thread_rng())
        });

        let weights_ho: Matrix<f64> = Matrix::from_fn(outputs, hidden, |_, _| {
            normal_ho.ind_sample(&mut rand::thread_rng())
        });

        (weights_ih, weights_ho)
    }

    fn feed_forward(&self, xte: &Matrix<f64>) -> Matrix<f64> {
        // Signals into the hidden layer
        let hidden_inputs = &self.weights_ih * xte;

        // Signals emerging from the hidden layer
        let hidden_ouputs = hidden_inputs.apply(&NeuralNetwork::sigmoid_activation);

        // Signals into the output layer
        let final_inputs = &self.weights_ho * hidden_ouputs;

        // Signals emerging from the output layer
        let final_outputs = final_inputs.apply(&NeuralNetwork::sigmoid_activation);

        final_outputs
    }

    fn sigmoid_activation(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}

impl Supervised<Matrix<f64>, Matrix<f64>> for NeuralNetwork {
    fn train(&mut self, xtr: &Matrix<f64>, ytr: &Matrix<f64>) {
        // The training data should be formatted as a matrix where each
        // column represents a single training example. So, the number of
        // rows (features) in the training data matrix should equal the
        // number of inputs to the neural network.
        //
        // Likewise, the matrix containing all of the labels for the training
        // data should have a column (label) for each training example. The
        // number of rows in this matrix should equal the number of outputs
        // produced by the neural network.
        if xtr.rows() != self.inputs {
            panic!("The number of features in the provided training data
                    is not equal to the number of inputs to this network");
        }
        if ytr.rows() != self.outputs {
            panic!("The number of classes/outputs in the provided training labels
                    is not equal to the number of outputs to this network");
        }
        if xtr.cols() != ytr.cols() {
            panic!("The number of training examples does not equal the number
                    of training labels");
        }

        // Signals into the hidden layer
        let hidden_inputs = &self.weights_ih * xtr;

        // Signals emerging from the hidden layer
        let hidden_ouputs = hidden_inputs.apply(&NeuralNetwork::sigmoid_activation);

        // Signals into the output layer
        let final_inputs = &self.weights_ho * &hidden_ouputs;

        // Signals emerging from the output layer
        let final_outputs = final_inputs.apply(&NeuralNetwork::sigmoid_activation);

        // The error (element-wise difference) between the network output
        // and the expected output
        let output_errors = ytr - &final_outputs;

        //println!("Current error: {}", output_errors.sum());

        // The matrix that holds all of the weights between the hidden and
        // output layer has dimensions `ouputs x hidden`. So, its transpose
        // has dimensions `hidden x outputs`. Obviously, the matrix containing
        // the error from the output layer has dimensions `outputs x 1`, so
        // the matrix multiplication below results in a matrix with dimensions
        // `hidden x 1`.
        let hidden_errors = self.weights_ho.transpose() * &output_errors;

        //println!("\nCurrent weights (hidden -> output):");
        //println!("{}", self.weights_ho

        //Matrix::<f64>::ones(final_outputs.rows(), final_outputs.cols())
        // Update the weights for the connections between the hidden layer
        // and output layer
        self.weights_ho += output_errors.elemul(&final_outputs).elemul(&(-final_outputs + 1.0)) *
                            hidden_ouputs.transpose() *
                            self.learning_rate;

        self.weights_ih += hidden_errors.elemul(&hidden_ouputs).elemul(&(-hidden_ouputs + 1.0)) *
                            xtr.transpose() *
                            self.learning_rate;
        //println!("\nUpdated weights (hidden -> output):");
        //println!("{}", self.weights_ho);
    }

    fn predict(&self, xte: &Matrix<f64>) -> Matrix<f64> {
        self.feed_forward(xte)
    }
}
