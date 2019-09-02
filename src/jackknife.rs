use std::marker::{Send, Sync};
use std::ops::{Add, Index, Sub};

use analytic::partition::integer_partitions::{IntegerPartitionIter, IntegerPartitions, Partition};
use analytic::sample::Sample;
use analytic::set::ordered_integer_set::{ContiguousIntegerSet, OrderedIntegerSet};
use analytic::set::traits::Finite;
use rayon::prelude::*;

pub struct Jackknife<C> {
    pub components: Vec<C>,
}

impl<C: Send> Jackknife<C> {
    pub fn from_op_over_jackknife_partitions<F>(jackknife_partitions: &JackknifePartitions, op: F) -> Jackknife<C>
        where F: Fn(&Partition) -> C + Send + Sync {
        Jackknife {
            components: jackknife_partitions.iter().into_par_iter().map(|p| op(&p)).collect()
        }
    }
}

pub struct AdditiveJackknife<C> {
    pub additive_components: Vec<C>,
    sum: Option<C>,
}

impl<C: Send> AdditiveJackknife<C> {
    pub fn from_op_over_jackknife_partitions<F>(jackknife_partitions: &JackknifePartitions, op: F) -> AdditiveJackknife<C>
        where F: Fn(usize, &Partition) -> C + Send + Sync,
              C: for<'a> Add<&'a C, Output=C> + Clone {
        let additive_components: Vec<C> = jackknife_partitions.iter().into_par_iter().enumerate().map(|(i, p)| op(i, &p)).collect();
        let sum = match additive_components.first() {
            Some(first) => Some(additive_components.iter().skip(1).fold(first.clone(), |acc, x| acc + x)),
            None => None
        };
        AdditiveJackknife {
            additive_components,
            sum,
        }
    }

    #[inline]
    pub fn get_component_sum(&self) -> Option<&C> {
        match &self.sum {
            Some(s) => Some(s),
            None => None,
        }
    }

    #[inline]
    pub fn into_component_sum(self) -> Option<C> {
        match self.sum {
            Some(s) => Some(s),
            None => None,
        }
    }

    #[inline]
    pub fn sum_minus_component<'a>(&'a self, component_index: usize) -> C
        where &'a C: Sub<Output=C> {
        self.sum.as_ref().unwrap() - &self.additive_components[component_index]
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct JackknifePartitions {
    partitions: IntegerPartitions,
}

impl JackknifePartitions {
    pub fn from_partitions(partitions: IntegerPartitions) -> JackknifePartitions {
        JackknifePartitions {
            partitions
        }
    }

    /// partitions each of the set in the `integer_sets` into `num_partitions` partitions
    /// and combines the i-th partition from each set into a single Jackknife partition, for all i
    pub fn from_integer_set(mut integer_sets: Vec<OrderedIntegerSet<usize>>, num_partitions: usize, randomize: bool) -> JackknifePartitions {
        let partition_size: Vec<usize> = integer_sets.iter().map(|s| s.size() / num_partitions).collect();
        let mut partitions = Vec::new();
        for _ in 0..num_partitions - 1 {
            let mut merged_partition = Vec::new();
            for (i, s) in integer_sets.iter_mut().enumerate() {
                let p;
                if randomize {
                    p = s.sample_subset_without_replacement(partition_size[i]).unwrap();
                } else {
                    p = s.slice(0..partition_size[i]);
                }
                *s -= &p;
                merged_partition.append(&mut p.into_intervals());
            }
            partitions.push(OrderedIntegerSet::from(merged_partition));
        }
        partitions.push(
            OrderedIntegerSet::from(
                integer_sets.into_iter()
                            .flat_map(|s| s.into_intervals())
                            .collect::<Vec<ContiguousIntegerSet<usize>>>()
            )
        );
        JackknifePartitions {
            partitions: IntegerPartitions::new(partitions)
        }
    }

    #[inline]
    pub fn iter(&self) -> IntegerPartitionIter {
        self.partitions.iter()
    }

    #[inline]
    pub fn num_partitions(&self) -> usize {
        self.partitions.num_partitions()
    }

    #[inline]
    pub fn union(&self) -> Partition {
        self.partitions.union()
    }
}

impl Index<usize> for JackknifePartitions {
    type Output = Partition;
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.partitions[index]
    }
}

#[cfg(test)]
mod tests {
    use analytic::set::ordered_integer_set::OrderedIntegerSet;
    use analytic::set::traits::Finite;

    use super::JackknifePartitions;

    #[test]
    fn test_jackknife_config_from_integer_set() {
        let num_partitions = 7;
        let integer_set = OrderedIntegerSet::from_slice(&[[1, 5], [8, 12], [14, 20], [25, 32]]);
        let size = integer_set.size();
        let config = JackknifePartitions::from_integer_set(vec![integer_set.clone()], num_partitions, true);
        for (i, p) in config.partitions.iter().enumerate() {
            if i == num_partitions - 1 {
                assert!(p.size() >= size / num_partitions);
            } else {
                assert_eq!(p.size(), size / num_partitions);
            }
        }
        let config = JackknifePartitions::from_integer_set(vec![integer_set], num_partitions, false);
        for (i, p) in config.partitions.iter().enumerate() {
            if i == num_partitions - 1 {
                assert!(p.size() >= size / num_partitions);
            } else {
                assert_eq!(p.size(), size / num_partitions);
            }
        }
    }
}
