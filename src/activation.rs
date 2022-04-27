use ndarray::ArrayViewMut1;
use std::error::Error;
use std::fmt;

pub type ActivationFunction = fn(ArrayViewMut1<f64>) -> Result<(), Box<dyn Error>>;

// #[derive(Debug)]
pub struct Activation {
    pub name: &'static str,
    pub function: ActivationFunction,
}
impl fmt::Debug for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name)
    }
}
impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name)
    }
}

impl Activation {
    fn new(name: &'static str, function: ActivationFunction) -> Activation {
        Activation { function, name }
    }

    pub const LINEAR: Activation = Activation {
        name: "linear",
        function: |_| Ok(()),
    };

    pub const RELU: Activation = Activation {
        name: "ReLU (Rectified Linear Units)",
        function: |mut xs| {
            for x in xs.iter_mut() {
                *x = *x * { *x > 0.0 } as usize as f64;
            }
            Ok(())
        },
    };

    pub const SIGMOID: Activation = Activation {
        name: "Sigmoid",
        function: |mut xs| {
            for x in xs.iter_mut() {
                *x = ((-*x).exp() + 1.0).recip();
            }
            Ok(())
        },
    };

    pub const SOFTMAX: Activation = Activation {
        name: "Softmax",
        function: |mut xs| {
            let exp_vals: Vec<f64> = xs.iter().map(|x| x.exp()).collect();
            let total_inv: f64 = exp_vals.iter().sum::<f64>().recip();
            xs.iter_mut()
                .zip(exp_vals)
                .for_each(|(x, exp)| *x = exp / total_inv);
            Ok(())
        },
    };
}

pub mod functions {
    use super::ActivationFunction;

    pub const LINEAR: ActivationFunction = |_| Ok(());
}
