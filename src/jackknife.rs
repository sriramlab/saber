use std::{
    fs::OpenOptions,
    io::{BufReader, BufWriter},
    marker::{Send, Sync},
    ops::{Add, Deref, Index, Sub},
};

use math::{
    partition::integer_partitions::{
        IntegerPartitionIter, IntegerPartitions, Partition,
    },
    sample::Sample,
    set::{
        contiguous_integer_set::ContiguousIntegerSet,
        ordered_integer_set::OrderedIntegerSet, traits::Finite,
    },
};
use num::{FromPrimitive, Integer, ToPrimitive};
use rayon::prelude::*;

use crate::error::Error;
use std::{fmt::Debug, iter::Sum};

pub struct Jackknife<C> {
    pub components: Vec<C>,
}

impl<C: Send> Jackknife<C> {
    pub fn from_op_over_jackknife_partitions<
        F,
        T: Copy + Debug + FromPrimitive + Integer + Send + Sum + ToPrimitive,
    >(
        jackknife_partitions: &JackknifePartitions<T>,
        op: F,
    ) -> Jackknife<C>
    where
        F: Fn(&Partition<T>) -> C + Send + Sync, {
        Jackknife {
            components: jackknife_partitions
                .iter()
                .into_par_iter()
                .map(|p| op(&p))
                .collect(),
        }
    }
}

pub struct AdditiveJackknife<C> {
    pub additive_components: Vec<C>,
    sum: Option<C>,
}

impl<C: Send> AdditiveJackknife<C> {
    pub fn from_op_over_jackknife_partitions<
        F,
        T: Copy + Debug + FromPrimitive + Integer + Send + Sum + ToPrimitive,
    >(
        jackknife_partitions: &JackknifePartitions<T>,
        op: F,
    ) -> AdditiveJackknife<C>
    where
        F: Fn(usize, &Partition<T>) -> C + Send + Sync,
        C: for<'a> Add<&'a C, Output = C> + Clone, {
        let additive_components: Vec<C> = jackknife_partitions
            .iter()
            .into_par_iter()
            .enumerate()
            .map(|(i, p)| op(i, &p))
            .collect();
        let sum = match additive_components.first() {
            Some(first) => Some(
                additive_components
                    .iter()
                    .skip(1)
                    .fold(first.clone(), |acc, x| acc + x),
            ),
            None => None,
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
    where
        &'a C: Sub<Output = C>, {
        self.sum.as_ref().unwrap() - &self.additive_components[component_index]
    }

    pub fn sum_minus_component_or_sum<'a>(
        &'a self,
        component_index: Option<usize>,
    ) -> Result<C, String>
    where
        &'a C: Sub<Output = C> + Deref,
        C: Clone, {
        match component_index {
            Some(k) => Ok(self.sum_minus_component(k)),
            None => match &self.sum {
                Some(s) => Ok((*s).clone()),
                None => Err("component sum is None".to_string()),
            },
        }
    }

    fn get_sum_minus_component_filepath(
        file_prefix: &str,
        component_index: usize,
    ) -> String {
        format!("{}_s-{}.jackknife", file_prefix, component_index)
    }

    pub fn serialize_to_file<'a>(
        &'a self,
        file_prefix: &str,
    ) -> Result<(), Error>
    where
        &'a C: Sub<Output = C>,
        C: serde::Serialize, {
        for i in 0..self.additive_components.len() {
            let buf_writer = BufWriter::new(
                OpenOptions::new()
                    .create(true)
                    .truncate(true)
                    .write(true)
                    .open(
                    AdditiveJackknife::<C>::get_sum_minus_component_filepath(
                        file_prefix,
                        i,
                    ),
                )?,
            );
            let data = self.sum_minus_component(i);
            bincode::serialize_into(buf_writer, &data)?;
        }
        Ok(())
    }

    pub fn deserialize_sum_minus_component(
        &self,
        component_index: usize,
        file_prefix: &str,
    ) -> Result<C, Error>
    where
        for<'a> C: serde::de::Deserialize<'a>, {
        let buf_reader = BufReader::new(OpenOptions::new().read(true).open(
            AdditiveJackknife::<C>::get_sum_minus_component_filepath(
                file_prefix,
                component_index,
            ),
        )?);
        let decoded: C = bincode::deserialize_from(buf_reader)?;
        Ok(decoded)
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct JackknifePartitions<
    T: Copy + Debug + FromPrimitive + Integer + Sum + ToPrimitive,
> {
    partitions: IntegerPartitions<T>,
}

impl<T: Copy + Debug + FromPrimitive + Integer + Sum + ToPrimitive>
    JackknifePartitions<T>
{
    pub fn from_partitions(
        partitions: IntegerPartitions<T>,
    ) -> JackknifePartitions<T> {
        JackknifePartitions {
            partitions,
        }
    }

    /// partitions each of the set in the `integer_sets` into `num_partitions`
    /// partitions and combines the i-th partition from each set into a
    /// single Jackknife partition, for all i
    pub fn from_integer_set(
        mut integer_sets: Vec<OrderedIntegerSet<T>>,
        num_partitions: usize,
        randomize: bool,
    ) -> JackknifePartitions<T> {
        let partition_size: Vec<usize> = integer_sets
            .iter()
            .map(|s| s.size() / num_partitions)
            .collect();
        let mut partitions = Vec::new();
        for _ in 0..num_partitions - 1 {
            let mut merged_partition = Vec::new();
            for (i, s) in integer_sets.iter_mut().enumerate() {
                let p;
                if randomize {
                    p = s
                        .sample_subset_without_replacement(partition_size[i])
                        .unwrap();
                } else {
                    p = s.slice(0..partition_size[i]);
                }
                *s -= &p;
                merged_partition.append(&mut p.into_intervals());
            }
            partitions.push(OrderedIntegerSet::from(merged_partition));
        }
        partitions.push(OrderedIntegerSet::from(
            integer_sets
                .into_iter()
                .flat_map(|s| s.into_intervals())
                .collect::<Vec<ContiguousIntegerSet<T>>>(),
        ));
        JackknifePartitions {
            partitions: IntegerPartitions::new(partitions),
        }
    }

    #[inline]
    pub fn iter(&self) -> IntegerPartitionIter<T> {
        self.partitions.iter()
    }

    #[inline]
    pub fn num_partitions(&self) -> usize {
        self.partitions.num_partitions()
    }

    #[inline]
    pub fn union(&self) -> Partition<T> {
        self.partitions.union()
    }
}

impl<T: Copy + Debug + FromPrimitive + Integer + Sum + ToPrimitive> Index<usize>
    for JackknifePartitions<T>
{
    type Output = Partition<T>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.partitions[index]
    }
}

#[cfg(test)]
mod tests {
    use math::set::{ordered_integer_set::OrderedIntegerSet, traits::Finite};
    use ndarray::{Array, Ix2};

    use crate::jackknife::AdditiveJackknife;

    use super::JackknifePartitions;
    use math::traits::ToIterator;

    #[test]
    fn test_jackknife_config_from_integer_set() {
        let num_partitions = 7;
        let integer_set =
            OrderedIntegerSet::from_slice(&[[1, 5], [8, 12], [14, 20], [
                25, 32,
            ]]);
        let size = integer_set.size();
        let config = JackknifePartitions::from_integer_set(
            vec![integer_set.clone()],
            num_partitions,
            true,
        );
        for (i, p) in config.partitions.iter().enumerate() {
            if i == num_partitions - 1 {
                assert!(p.size() >= size / num_partitions);
            } else {
                assert_eq!(p.size(), size / num_partitions);
            }
        }
        let config = JackknifePartitions::from_integer_set(
            vec![integer_set],
            num_partitions,
            false,
        );
        for (i, p) in config.partitions.iter().enumerate() {
            if i == num_partitions - 1 {
                assert!(p.size() >= size / num_partitions);
            } else {
                assert_eq!(p.size(), size / num_partitions);
            }
        }
    }

    #[test]
    fn test_serialize_jackknife() {
        let num_partitions = 7;
        let integer_set =
            OrderedIntegerSet::from_slice(&[[1, 5], [8, 12], [14, 20], [
                25, 32,
            ]]);
        let config = JackknifePartitions::from_integer_set(
            vec![integer_set.clone()],
            num_partitions,
            false,
        );

        let file_prefix = "test_serialize_jackknife";
        let jackknife = AdditiveJackknife::from_op_over_jackknife_partitions(
            &config,
            |_k, knife| {
                let s = knife.to_iter().sum::<usize>();
                Array::<f32, Ix2>::ones((2, 2)) * s as f32
            },
        );
        jackknife.serialize_to_file(file_prefix).unwrap();
        for i in 0..num_partitions {
            let decoded = jackknife
                .deserialize_sum_minus_component(i, file_prefix)
                .unwrap();
            assert_eq!(decoded, jackknife.sum_minus_component(i));
        }
    }
}
