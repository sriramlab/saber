use rand::distributions::{Distribution, Uniform};

use crate::set::traits::Finite;
use crate::traits::{Collecting, Constructable, ToIterator};

pub trait Sample<I: Iterator<Item=E>, E: Clone>: Finite + ToIterator<I, E> {
    type Output: Collecting<E> + Constructable<Output=Self::Output>;
    /// samples `size` elements without replacement
    /// `size`: the number of samples to be drawn
    /// returns Err if `size` is larger than the population size
    fn sample_subset_without_replacement(&self, size: usize) -> Result<Self::Output, String> {
        let mut remaining = self.size();
        if size > remaining {
            return Err(format!("desired sample size {} > population size {}", size, remaining));
        }
        let mut samples = Self::Output::new();
        let mut needed = size;
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(0., 1.);

        for element in self.iter() {
            if uniform.sample(&mut rng) <= (needed as f64 / remaining as f64) {
                samples.collect(element.clone());
                needed -= 1;
            }
            remaining -= 1;
        }
        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use crate::set::ordered_integer_set::{ContiguousIntegerSet, OrderedIntegerSet};

    use super::Sample;

    #[test]
    fn test_sample() {
        let interval = ContiguousIntegerSet::new(0, 100);
        let num_samples = 25;
        let samples = interval.sample_subset_without_replacement(num_samples).unwrap();
        assert_eq!(samples.size(), num_samples);

        let set = OrderedIntegerSet::from_slice(&[[-89, -23], [-2, 100], [300, 345]]);
        let num_samples = 18;
        let samples = set.sample_subset_without_replacement(num_samples).unwrap();
        assert_eq!(samples.size(), num_samples);
    }
}