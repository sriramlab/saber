extern crate num_traits;

use num_traits::{NumOps, NumAssign};
use num_traits::cast::ToPrimitive;
use std::slice::Iter;

fn naive_mean<T: NumOps + NumAssign + Copy + ToPrimitive>(mut iter: Iter<T>) -> f64 {
    let mut sum = T::zero();
    let mut count = 0usize;
    for a in iter {
        sum += *a;
        count += 1;
    }
    sum.to_f64().unwrap() / count as f64
}

fn get_mean<T: NumOps + NumAssign + Copy + ToPrimitive>(mut iter: Iter<T>) -> f64 {
    let mut current_mean = match iter.next() {
        None => return 0f64,
        Some(a) => (*a).to_f64().unwrap()
    };

    for (i, a) in iter.enumerate() {
        current_mean = current_mean + ((*a).to_f64().unwrap() - current_mean) / (i + 2) as f64
    }
    current_mean
}

#[cfg(test)]
mod tests {
    extern crate rand;

    use crate::stats_util::{get_mean, naive_mean};
    use rand::Rng;

    #[test]
    fn test_mean() {
        let mut numbers = Vec::<i64>::with_capacity(100);
        let mut rng = rand::thread_rng();
        for _ in 0..1000000 {
            numbers.push(rng.gen_range(1, 21));
        }
        println!("{}", get_mean(numbers.iter()));
        println!("{}", numbers.iter().sum::<i64>() as f64 / numbers.len() as f64);
        println!("{}", naive_mean(numbers.iter()));
        assert_eq!(2, 3);
    }
}
