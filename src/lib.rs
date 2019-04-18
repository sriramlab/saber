#![feature(test)]
#[macro_use]
extern crate clap;
#[cfg(feature = "cuda")]
extern crate collenchyma as co;
#[cfg(feature = "cuda")]
extern crate collenchyma_blas as co_blas;
extern crate colored;
#[macro_use]
extern crate ndarray;
extern crate ndarray_linalg;
extern crate ndarray_rand;
extern crate num_traits;
extern crate rand;
extern crate time;

pub mod gg_estimates;
pub mod histogram;
pub mod mailman;
pub mod matrix_util;
pub mod program_flow;
pub mod sparsity_stats;
pub mod timer;
pub mod simulation;
pub mod stats_util;
#[cfg(feature = "cuda")]
pub mod cublas;
