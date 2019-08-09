use ndarray::{Array, Ix2};

use math::integer_partitions::{IntegerPartitionIter, IntegerPartitions, Partition};
use math::sample::Sample;
use math::set::ordered_integer_set::OrderedIntegerSet;
use math::set::traits::Finite;

pub struct JackknifeMatrix {
    pub additive_components: Vec<Array<f32, Ix2>>,
    matrix_sum: Option<Array<f32, Ix2>>,
}

impl JackknifeMatrix {
    pub fn from_op_over_jackknife_partitions<F>(jackknife_partitions: &JackknifePartitions, mut op: F) -> JackknifeMatrix
        where F: FnMut(&Partition) -> Array<f32, Ix2> {
        let additive_components: Vec<Array<f32, Ix2>> = jackknife_partitions.iter().map(|p| op(p)).collect();
        let matrix_sum = match additive_components.first() {
            Some(first) => Some(additive_components.iter().fold(Array::zeros(first.dim()), |acc, x| acc + x)),
            None => None
        };
        JackknifeMatrix {
            additive_components,
            matrix_sum,
        }
    }

    pub fn get_matrix_sum(&self) -> Option<&Array<f32, Ix2>> {
        match &self.matrix_sum {
            Some(matrix_sum) => Some(matrix_sum),
            None => None,
        }
    }

    pub fn into_matrix_sum(self) -> Option<Array<f32, Ix2>> {
        match self.matrix_sum {
            Some(matrix_sum) => Some(matrix_sum),
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
        for i in 0..num_partitions - 1 {
            let p = integer_set.sample_subset_without_replacement(partition_size).unwrap();
            integer_set -= &p;
            partitions.push(p);
        }
        partitions.push(integer_set);
        JackknifePartitions {
            partitions: IntegerPartitions::new(partitions)
        }
    }
    pub fn iter(&self) -> IntegerPartitionIter {
        self.partitions.iter()
    }

    pub fn num_partitions(&self) -> usize {
        self.partitions.num_partitions()
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
