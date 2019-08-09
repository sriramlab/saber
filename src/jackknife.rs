use math::integer_partition::IntegerPartition;
use math::sample::Sample;
use math::set::ordered_integer_set::OrderedIntegerSet;
use math::set::traits::Finite;

#[derive(Clone, PartialEq, Debug)]
pub struct JackknifeConfig {
    pub partitions: IntegerPartition,
}

impl JackknifeConfig {
    pub fn from_partitions(partitions: IntegerPartition) -> JackknifeConfig {
        JackknifeConfig {
            partitions
        }
    }

    pub fn from_integer_set(mut integer_set: OrderedIntegerSet<usize>, num_partitions: usize) -> JackknifeConfig {
        let partition_size = integer_set.size() / num_partitions;
        let mut partitions = Vec::new();
        for _ in 0..num_partitions - 1 {
            let p = integer_set.sample_subset_without_replacement(partition_size).unwrap();
            integer_set -= &p;
            partitions.push(p);
        }
        partitions.push(integer_set);
        JackknifeConfig {
            partitions: IntegerPartition::new(partitions)
        }
    }

    pub fn num_partitions(&self) -> usize {
        self.partitions.num_partitions()
    }
}

#[cfg(test)]
mod tests {
    use super::JackknifeConfig;
    use math::set::ordered_integer_set::OrderedIntegerSet;

    fn test_jackknife_config_from_integer_set() {
        let integer_set = OrderedIntegerSet::from_slice(&[[1, 5], [8, 12], [14, 20], [25, 32]]);
        let config = JackknifeConfig::from_integer_set(integer_set, 7);
        println!("{:?}", config);
        assert!(false);
    }
}
