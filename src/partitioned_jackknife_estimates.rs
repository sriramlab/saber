use std::collections::HashSet;
use std::fmt;

use analytic::set::ordered_integer_set::OrderedIntegerSet;
use analytic::traits::ToIterator;

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Estimate<T> {
    pub bias_corrected_estimate: T,
    pub jackknife_mean: T,
    pub point_estimate_without_jackknife: T,
    pub standard_error: T,
}

impl<T> Estimate<T> {
    pub fn new(
        bias_corrected_estimate: T,
        jackknife_mean: T,
        point_estimate_without_jackknife: T,
        standard_error: T,
    ) -> Estimate<T> {
        Estimate {
            bias_corrected_estimate,
            jackknife_mean,
            point_estimate_without_jackknife,
            standard_error,
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct PartitionedJackknifeEstimates {
    pub partition_names: Option<Vec<String>>,
    pub partition_estimates: Vec<Estimate<f64>>,
    pub sum_estimate: Option<Estimate<f64>>,
    pub subset_sum_estimates: Option<Vec<(String, Estimate<f64>)>>,
}

fn get_jackknife_mean_and_std(point_estimate_without_jackknife: f64, estimates: &Vec<f64>) -> Estimate<f64> {
    let n = estimates.len() as f64;
    let n_minus_one = n - 1.;

    let s: f64 = estimates.iter().sum();
    let x_avg = s / n;
    let x_i: Vec<f64> = estimates.iter().map(|&e| (s - e) / n_minus_one).collect();
    let standard_error = (
        x_i.into_iter()
           .map(|x| {
               let delta = x - x_avg;
               delta * delta
           })
           .sum::<f64>()
            * n_minus_one
            / n
    ).sqrt();

    let bias_corrected_estimate = n * point_estimate_without_jackknife - (n - 1.) * x_avg;
    Estimate {
        bias_corrected_estimate,
        jackknife_mean: x_avg,
        point_estimate_without_jackknife,
        standard_error,
    }
}

impl PartitionedJackknifeEstimates {
    pub fn from_jackknife_estimates(
        point_estimate_without_jackknife: &Vec<f64>,
        jackknife_iteration_estimates: &Vec<Vec<f64>>,
        partition_names: Option<Vec<String>>,
        subset_sum_indices: Option<Vec<(String, OrderedIntegerSet<usize>)>>)
        -> Result<PartitionedJackknifeEstimates, String> {
        if jackknife_iteration_estimates.iter().map(|estimates| estimates.len()).collect::<HashSet<usize>>().len() > 1 {
            return Err(format!("inconsistent number of partitioned estimates across Jackknife iterations"));
        }
        if jackknife_iteration_estimates.len() == 0 {
            return Ok(PartitionedJackknifeEstimates {
                partition_names: None,
                partition_estimates: Vec::new(),
                sum_estimate: None,
                subset_sum_estimates: None,
            });
        }
        let num_partitions = point_estimate_without_jackknife.len();
        if let Some(names) = &partition_names {
            if names.len() != num_partitions {
                return Err(format!("partition_names.len() {} != the number of partitions in the jackknife estimates {}",
                                   names.len(), num_partitions));
            }
        }
        let mut partition_raw_estimates = vec![vec![0f64; jackknife_iteration_estimates.len()]; num_partitions];
        for (i, estimates) in jackknife_iteration_estimates.iter().enumerate() {
            assert_eq!(estimates.len(), num_partitions,
                       "the number of partitions in the Jackknife iteration {} \
                       != the number of partitions {} in the point estimate",
                       estimates.len(), num_partitions);
            for p in 0..num_partitions {
                partition_raw_estimates[p][i] = estimates[p];
            }
        }
        let partition_estimates = point_estimate_without_jackknife
            .iter()
            .zip(partition_raw_estimates.iter())
            .map(|(&point_estimate, estimates)| {
                get_jackknife_mean_and_std(point_estimate, estimates)
            })
            .collect();

        let total_variance_estimates: Vec<f64> = jackknife_iteration_estimates
            .iter()
            .map(|point_estimate| point_estimate.iter().sum())
            .collect();

        let sum_estimate = {
            if total_variance_estimates.len() > 1 {
                Some(get_jackknife_mean_and_std(
                    point_estimate_without_jackknife.iter().sum(),
                    &total_variance_estimates))
            } else {
                None
            }
        };

        let subset_sum_estimates = match subset_sum_indices {
            None => None,
            Some(indices_list) => {
                Some(indices_list.iter().map(|(subset_key, subset_indices)| {
                    (subset_key.to_string(),
                     get_jackknife_mean_and_std(
                         subset_indices.to_iter().fold(0f64, |acc, i| acc + point_estimate_without_jackknife[i]),
                         &jackknife_iteration_estimates
                             .iter()
                             .map(|point_estimate| {
                                 subset_indices.to_iter().fold(0f64, |acc, i| acc + point_estimate[i])
                             })
                             .collect::<Vec<f64>>()))
                }).collect::<Vec<(String, Estimate<f64>)>>())
            }
        };

        Ok(PartitionedJackknifeEstimates {
            partition_names,
            partition_estimates,
            sum_estimate,
            subset_sum_estimates,
        })
    }

    pub fn get_partition_names(&self) -> Option<&Vec<String>> {
        match &self.partition_names {
            Some(names) => Some(names),
            None => None,
        }
    }
}

const NUM_DISPLAY_DECIMALS: usize = 5;

impl<T: fmt::Display> fmt::Display for Estimate<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let indent = f.width().unwrap_or(0);
        let fill = if indent > 0 { f.fill().to_string() } else { "".to_string() };
        write!(
            f,
            "{:indent$}point_estimate_without_jackknife: {:.*}\n\
            {:indent$}Jackknife mean: {:.*}\n\
            {:indent$}bias-corrected estimate: {:.*} (probably over-corrected)\n\
            {:indent$}standard error: {:.*}",
            fill, NUM_DISPLAY_DECIMALS, self.point_estimate_without_jackknife,
            fill, NUM_DISPLAY_DECIMALS, self.jackknife_mean,
            fill, NUM_DISPLAY_DECIMALS, self.bias_corrected_estimate,
            fill, NUM_DISPLAY_DECIMALS, self.standard_error,
            indent = indent
        )?;
        Ok(())
    }
}

impl fmt::Display for PartitionedJackknifeEstimates {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let indent: usize = 4;
        if let Some(partition_names) = &self.partition_names {
            for (name, estimate) in partition_names.iter().zip(self.partition_estimates.iter()) {
                writeln!(f, "\npartition named {}\n{:indent$}", name, estimate, indent = indent)?;
            }
        } else {
            for (i, estimate) in self.partition_estimates.iter().enumerate() {
                writeln!(f, "\npartition {}\n{:indent$}", i, estimate, indent = indent)?;
            }
        }

        if let Some(subset_sum_estimates) = &self.subset_sum_estimates {
            for (key, estimate) in subset_sum_estimates.iter() {
                writeln!(f, "\nestimate for subset {}\n{:indent$}", key, estimate, indent = indent)?;
            }
        }
        if let Some(sum_estimate) = self.sum_estimate {
            writeln!(f, "\ntotal estimate\n{:indent$}", sum_estimate, indent = indent)?;
        }
        Ok(())
    }
}
