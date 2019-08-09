use std::ops::{Add, Sub, Index};

use math::integer_partitions::{IntegerPartitionIter, IntegerPartitions, Partition};
use math::sample::Sample;
use math::set::ordered_integer_set::OrderedIntegerSet;
use math::set::traits::Finite;

pub struct Jackknife<C> {
    pub components: Vec<C>,
    component_union: C,
}

impl<C> Jackknife<C> {
    pub fn from_op_over_jackknife_partitions<F>(jackknife_partitions: &JackknifePartitions, mut op: F) -> Jackknife<C>
        where F: FnMut(&Partition) -> C {
        let components: Vec<C> = jackknife_partitions.iter().map(|p| op(p)).collect();
        let component_union = op(&jackknife_partitions.union());
        Jackknife {
            components,
            component_union,
        }
    }

    #[inline]
    pub fn get_component_union(&self) -> &C {
        &self.component_union
    }
}

pub struct AdditiveJackknife<C> {
    pub additive_components: Vec<C>,
    sum: Option<C>,
}

impl<C> AdditiveJackknife<C> {
    pub fn from_op_and_add_over_jackknife_partitions<F>(jackknife_partitions: &JackknifePartitions, mut op: F) -> AdditiveJackknife<C>
        where F: FnMut(usize, &Partition) -> C,
              C: for<'a> Add<&'a C, Output=C> + Clone {
        let additive_components: Vec<C> = jackknife_partitions.iter().enumerate().map(|(i, p)| op(i, p)).collect();
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
            Some(matrix_sum) => Some(matrix_sum),
            None => None,
        }
    }

    #[inline]
    pub fn into_component_sum(self) -> Option<C> {
        match self.sum {
            Some(matrix_sum) => Some(matrix_sum),
            None => None,
        }
    }

    #[inline]
    pub fn sum_minus_component<'a>(&'a self, component_index: usize) -> C
        where &'a C: Sub<Output=C> {
        self.sum.as_ref().unwrap() - &self.additive_components[component_index]
    }
}

pub struct BipartiteAdditiveJackknife<C> {
    bipartite_additive_components: Vec<Vec<C>>,
    sum: Option<C>,
}

impl<C> BipartiteAdditiveJackknife<C>
{
    pub fn from_op_over_jackknife_partitions<F>(jackknife_partitions: &JackknifePartitions, mut op: F) -> BipartiteAdditiveJackknife<C>
        where F: FnMut(usize, &Partition, usize, &Partition) -> C,
              C: for<'a> Add<&'a C, Output=C> + Clone {
        let bipartite_additive_components: Vec<Vec<C>> = jackknife_partitions.iter().enumerate().map(|(k1, knife_i)|
            jackknife_partitions.iter().enumerate().map(|(k2, knife_j)|
                op(k1, knife_i, k2, knife_j)
            ).collect::<Vec<C>>()
        ).collect();

        let mut sum = None;
        if let Some(first_row) = bipartite_additive_components.first() {
            if let Some(first) = first_row.first() {
                let init = first_row.iter().skip(1).fold(first.clone(), |acc, x| acc + x);
                sum = Some(bipartite_additive_components.iter()
                                                        .skip(1)
                                                        .fold(init.clone(), |acc, v|
                                                            acc + &v.iter()
                                                                    .skip(1)
                                                                    .fold(v[0].clone(), |acc, x| acc + x)));
            }
        }

        BipartiteAdditiveJackknife {
            bipartite_additive_components,
            sum,
        }
    }

    #[inline]
    pub fn sum_minus_component(&self, component_index: usize) -> C
        where C: for<'a> Sub<&'a C, Output=C>, C: Clone {
        let mut s = self.sum.as_ref().unwrap().clone();
        for j in &self.bipartite_additive_components[component_index] {
            s = s - j;
        }
        for i in 0..component_index {
            s = s - &self.bipartite_additive_components[i][component_index]
        }
        for i in component_index + 1..self.bipartite_additive_components.len() {
            s = s - &self.bipartite_additive_components[i][component_index]
        }
        s
    }

    #[inline]
    pub fn get_sum(&self) -> Option<&C> {
        match &self.sum {
            Some(sum) => Some(sum),
            None => None,
        }
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

    pub fn from_integer_set(mut integer_set: OrderedIntegerSet<usize>, num_partitions: usize) -> JackknifePartitions {
        let partition_size = integer_set.size() / num_partitions;
        let mut partitions = Vec::new();
        for _ in 0..num_partitions - 1 {
            let p = integer_set.sample_subset_without_replacement(partition_size).unwrap();
            integer_set -= &p;
            partitions.push(p);
        }
        partitions.push(integer_set);
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
    use math::set::ordered_integer_set::OrderedIntegerSet;
    use math::set::traits::Finite;

    use super::JackknifePartitions;

    #[test]
    fn test_jackknife_config_from_integer_set() {
        let num_partitions = 7;
        let integer_set = OrderedIntegerSet::from_slice(&[[1, 5], [8, 12], [14, 20], [25, 32]]);
        let size = integer_set.size();
        let config = JackknifePartitions::from_integer_set(integer_set, num_partitions);
        for (i, p) in config.partitions.iter().enumerate() {
            if i == num_partitions - 1 {
                assert!(p.size() >= size / num_partitions);
            } else {
                assert_eq!(p.size(), size / num_partitions);
            }
        }
    }
}
