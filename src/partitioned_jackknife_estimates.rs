use std::collections::HashSet;
use std::fmt;

use analytic::set::ordered_integer_set::OrderedIntegerSet;
use analytic::traits::ToIterator;

#[derive(Clone, PartialEq, Debug)]
pub struct PartitionedJackknifeEstimates {
    partition_names: Option<Vec<String>>,
    pub partition_means_and_stds: Vec<(f64, f64)>,
    pub sum_estimates: Option<(f64, f64)>,
    pub subset_sum_estimates: Option<Vec<(String, (f64, f64))>>,
}

fn get_jackknife_mean_and_std(estimates: &Vec<f64>) -> (f64, f64) {
    let num_knives = estimates.len();
    let s: f64 = estimates.iter().sum();
    let num_knives_minus_one_f64 = (num_knives - 1) as f64;
    let mut x_i = Vec::new();
    for i in 0..num_knives {
        x_i.push((s - estimates[i]) / num_knives_minus_one_f64);
    }
    let x_avg = s / num_knives as f64;
    let standard_error = (x_i.into_iter().map(|x| {
        let delta = x - x_avg;
        delta * delta
    }).sum::<f64>() * num_knives_minus_one_f64 / num_knives as f64).sqrt();
    (x_avg, standard_error)
}

impl PartitionedJackknifeEstimates {
    pub fn from_jackknife_estimates(jackknife_iteration_estimates: &Vec<Vec<f64>>,
                                    partition_names: Option<Vec<String>>,
                                    subset_sum_indices: Option<Vec<(String, OrderedIntegerSet<usize>)>>)
        -> Result<PartitionedJackknifeEstimates, String> {
        if jackknife_iteration_estimates.iter().map(|estimates| estimates.len()).collect::<HashSet<usize>>().len() > 1 {
            return Err(format!("inconsistent number of partitioned estimates across Jackknife iterations"));
        }
        if jackknife_iteration_estimates.len() == 0 {
            return Ok(PartitionedJackknifeEstimates {
                partition_names: None,
                partition_means_and_stds: Vec::new(),
                sum_estimates: None,
                subset_sum_estimates: None,
            });
        }
        let num_partitions = jackknife_iteration_estimates[0].len();
        if let Some(names) = &partition_names {
            if names.len() != num_partitions {
                return Err(format!("partition_names.len() {} != the number of partitions in the jackknife estimates {}",
                                   names.len(), num_partitions));
            }
        }
        let mut partition_estimates = vec![vec![0f64; jackknife_iteration_estimates.len()]; num_partitions];
        for (i, estimates) in jackknife_iteration_estimates.iter().enumerate() {
            for p in 0..num_partitions {
                partition_estimates[p][i] = estimates[p];
            }
        }
        let total_variance_estimates: Vec<f64> = jackknife_iteration_estimates.iter()
                                                                              .map(|v| v.iter()
                                                                                        .map(|&x| x as f64)
                                                                                        .sum())
                                                                              .collect();
        let sum_estimates = {
            if total_variance_estimates.len() > 1 {
                Some(get_jackknife_mean_and_std(&total_variance_estimates))
            } else {
                None
            }
        };

        let subset_sum_estimates = match subset_sum_indices {
            None => None,
            Some(indices_list) => {
                Some(indices_list.iter().map(|(subset_key, subset_indices)| {
                    (subset_key.to_string(),
                     get_jackknife_mean_and_std(&jackknife_iteration_estimates
                         .iter()
                         .map(|single_iter_estimates| {
                             subset_indices.to_iter().fold(0f64, |acc, i| acc + single_iter_estimates[i])
                         })
                         .collect::<Vec<f64>>()))
                }).collect::<Vec<(String, (f64, f64))>>())
            }
        };
        Ok(PartitionedJackknifeEstimates {
            partition_names,
            partition_means_and_stds: partition_estimates.iter()
                                                         .map(|estimates| get_jackknife_mean_and_std(estimates))
                                                         .collect(),
            sum_estimates,
            subset_sum_estimates,
        })
    }
}

impl std::fmt::Display for PartitionedJackknifeEstimates {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let num_decimals = 7;
        if let Some(partition_names) = &self.partition_names {
            for (name, (m, s)) in partition_names.iter().zip(self.partition_means_and_stds.iter()) {
                writeln!(f, "estimate for partition named {}: {:.*} standard error: {:.*}",
                         name, num_decimals, m, num_decimals, s)?;
            }
        } else {
            for (i, (m, s)) in self.partition_means_and_stds.iter().enumerate() {
                writeln!(f, "estimate for partition {}: {:.*} standard error: {:.*}",
                         i, num_decimals, m, num_decimals, s)?;
            }
        }

        if let Some(subset_sum_estimates) = &self.subset_sum_estimates {
            for (key, (m, s)) in subset_sum_estimates.iter() {
                writeln!(f, "estimate for subset {}: {:.*} standard error: {:.*}",
                         key, num_decimals, m, num_decimals, s)?;
            }
        }
        if let Some(sum_estimates) = self.sum_estimates {
            writeln!(f, "total estimate: {:.*} standard error: {:.*}",
                     num_decimals, sum_estimates.0, num_decimals, sum_estimates.1)?;
        }
        Ok(())
    }
}
