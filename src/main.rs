use didact::layer::Layer;
use didact::*;
use ndarray::{arr1, arr2};

fn main() {
    // println!("Hello, world!");

    // // let inputs = arr1(&[1., 2., 3., 2.5]);

    let batch_inputs = arr2(&[
        [1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8],
    ]);

    let weights1 = arr2(&[
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ]);

    let weights1_t = arr2(&[
        [0.2, 0.5, -0.26],
        [0.8, -0.91, -0.27],
        [-0.5, 0.26, 0.17],
        [1.0, -0.5, 0.87],
    ]);

    let biases1 = arr1(&[2.0, 3.0, 0.5]);

    // let weights2 = arr2(&[[0.1, -0.14, 0.5], [-0.5, 0.12, -0.33], [0.44, 0.73, -0.13]]);
    // let biases2 = arr1(&[-1.0, 2.0, -0.5]);

    // let layer1_output = batch_inputs.dot(&weights1.t()) + biases1;

    // let layer2_output = layer1_output.dot(&weights2.t()) + biases2;

    // println!("{}", layer2_output);

    // for x in batch_inputs.axis_iter(ndarray::Axis(0)) {
    //     println!("{}", x);
    // }

    let layer1 = layer::LayerDense {
        input_num: 4,
        neuron_num: 3,
        weights: weights1_t.clone(),
        biases: arr1(&[2.0, 3.0, 0.5]),
        activation: activation::Activation::LINEAR,
    };

    // println!("{}", layer1);
    // println!("{}", layer1.forward(&batch_inputs).unwrap());

    // println!("{:?}", Array2::<i32>::zeros((5, 2)));

    let layer1_1 =
        layer::LayerDense::new(weights1_t, biases1, Some(activation::Activation::LINEAR));

    // println!("{}", layer1_1);

    // println!("output: {}", layer1_1.forward(&batch_inputs).unwrap());

    let test_vals = arr1(&[1.0, 0.9, 0.7]);

    activation::Activation::SOFTMAX.function(test_vals);

    println!("{}");
}
