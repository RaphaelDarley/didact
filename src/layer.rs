use crate::activation::*;
use ndarray::{Array1, Array2};
use ndarray_rand::{rand_distr::Normal, RandomExt};
use std::{error::Error, fmt};

pub trait Layer {
    fn forward(&self, input: &Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>>;
}

#[derive(Debug)]
pub struct LayerDense {
    pub input_num: usize,
    pub neuron_num: usize,
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub activation: Activation,
}

impl fmt::Display for LayerDense {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "inputs: {}, neurons: {}, weights: {}, biases: {}, activation: {}",
            self.input_num, self.neuron_num, self.weights, self.biases, self.activation
        )
    }
}

impl LayerDense {
    pub fn new_rand(
        input_num: usize,
        neuron_num: usize,
        activation_opt: Option<Activation>,
    ) -> LayerDense {
        let activation = activation_opt.unwrap_or(Activation::LINEAR);
        LayerDense {
            input_num,
            neuron_num,
            weights: Array2::random((input_num, neuron_num), Normal::new(0., 0.1).unwrap()),
            biases: Array1::zeros(neuron_num),
            activation,
        }
    }

    pub fn new(
        weights: Array2<f64>,
        biases: Array1<f64>,
        activation_opt: Option<Activation>,
    ) -> LayerDense {
        let activation = activation_opt.unwrap_or(Activation::LINEAR);
        LayerDense {
            input_num: weights.shape()[0],
            neuron_num: biases.shape()[0],
            weights,
            biases,
            activation,
        }
    }
}

impl Layer for LayerDense {
    fn forward<'a>(&self, input: &Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
        let activation = self.activation.function;
        let mut acc = input.dot(&self.weights) + &self.biases;
        for sample in acc.outer_iter_mut() {
            activation(sample)?;
        }
        Ok(acc)
    }
}
