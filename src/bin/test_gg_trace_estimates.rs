#[macro_use]
extern crate clap;
#[macro_use]
extern crate ndarray;
extern crate ndarray_parallel;
extern crate saber;

use ndarray::Array;
use ndarray_parallel::prelude::*;
use ndarray_rand::RandomExt;
use rand::distributions::{Normal, Uniform};

use saber::program_flow::OrExit;
use saber::simulation::sim_geno::get_gxg_arr;
use saber::util::timer::Timer;
use saber::trace_estimator::{estimate_gxg_dot_y_norm_sq, estimate_gxg_gram_trace, estimate_gxg_kk_trace,
                             estimate_tr_k_gxg_k};
use saber::util::extract_str_arg;

fn main() {
    let matches = clap_app!(Saber =>
        (version: "0.1")
        (author: "Aaron Zhou")
        (@arg num_rows: -r +takes_value "number of rows, i.e. individuals; required")
        (@arg num_cols: -c +takes_value "number of columns, i.e. SNPs; required")
        (@arg num_random_vecs: -n +takes_value "number of random vectors; required")
    ).get_matches();
    let num_rows = extract_str_arg(&matches, "num_rows")
        .parse::<usize>().unwrap_or_exit(Some("failed to parse num_rows"));
    let num_cols = extract_str_arg(&matches, "num_cols")
        .parse::<usize>().unwrap_or_exit(Some("failed to parse num_cols"));
    let num_random_vecs = extract_str_arg(&matches, "num_random_vecs")
        .parse::<usize>().unwrap_or_exit(Some("failed to parse num_random_vecs"));
    println!("num_rows: {}\nnum_cols: {}\nnum_random_vecs: {}", num_rows, num_cols, num_random_vecs);

    let gxg_basis = Array::random((num_rows, num_cols), Uniform::from(0..3))
        .mapv(|e| e as f32);

    println!("\n=> generating the GxG matrix");
    let gxg = get_gxg_arr(&gxg_basis);
    println!("GxG dim: {:?}", gxg.dim());

    let mut timer = Timer::new();
    println!("\n=> calculating tr_k_true");
    let tr_k_true = (&gxg * &gxg).sum() as f64;
    println!("tr_k_true: {}", tr_k_true);
    timer.print();

    println!("\n=> estimating the trace of GxG.dot(GxG.T)");
    let mut ratio_list = Vec::new();
    for iter in 0..50 {
        let tr_k_est = match estimate_gxg_gram_trace(&gxg_basis, num_random_vecs) {
            Ok(t) => t,
            Err(why) => {
                eprintln!("{}", why);
                return;
            }
        };
        let r = (tr_k_est - tr_k_true) / tr_k_true;
        println!("\niter: {} tr_k_est: {}\nk_trace_est error ratio: {:.5}%", iter + 1, tr_k_est, r * 100.);
        ratio_list.push(r.abs());
    }
    timer.print();

    let abs_k_err_avg = ratio_list.iter().sum::<f64>() / ratio_list.len() as f64;
    let abs_k_err_std = (ratio_list.iter().map(|r| (r - abs_k_err_avg) * (r - abs_k_err_avg)).sum::<f64>() / ratio_list.len() as f64).sqrt();
    println!("\nabs_ratio_avg: {}%\nabs_ratio_std: {}%", abs_k_err_avg * 100., abs_k_err_std * 100.);

    println!("\n=> calculating tr_kk_true");
    let k = gxg.dot(&gxg.t()) / gxg.dim().1 as f32;
    println!("k dim: {:?}", k.dim());
    let tr_kk_true = (&k * &k).sum() as f64;
    println!("tr_kk_true: {}", tr_kk_true);
    timer.print();

    println!("\n=> computing tr_kk_est");
    let mut kk_abs_ratio_list = Vec::new();
    for iter in 0..5 {
        let tr_kk_est = match estimate_gxg_kk_trace(&gxg_basis, num_random_vecs) {
            Ok(t) => t,
            Err(why) => {
                eprintln!("{}", why);
                return;
            }
        };
        let kk_ratio = (tr_kk_est - tr_kk_true) / tr_kk_true;
        println!("\niter: {} tr_kk_est: {}\ntr_kk_est error ratio: {:.5}%", iter + 1, tr_kk_est, kk_ratio * 100.);
        kk_abs_ratio_list.push(kk_ratio.abs());
    }
    timer.print();

    let kk_abs_ratio_avg = kk_abs_ratio_list.iter().sum::<f64>() / kk_abs_ratio_list.len() as f64;
    let kk_abs_ratio_std = (kk_abs_ratio_list.iter()
                                             .map(|r| (r - kk_abs_ratio_avg) * (r - kk_abs_ratio_avg))
                                             .sum::<f64>() / kk_abs_ratio_list.len() as f64).sqrt();
    println!("\nkk_abs_ratio_avg: {}%\nkk_abs_ratio_std: {}%", kk_abs_ratio_avg * 100., kk_abs_ratio_std * 100.);

    println!("\n=> test estimate_tr_k_gxg_k");
    let rand_geno = Array::random((num_rows, num_cols), Uniform::from(4..7))
        .mapv(|e| e as f32);
    let tr_k_gxg_k_est = estimate_tr_k_gxg_k(&rand_geno, &gxg_basis, num_random_vecs);
    println!("tr_k_gxg_k_est: {}", tr_k_gxg_k_est);

    let k_gxg_k = rand_geno.t().dot(&gxg);
    let tr_k_gxg_k_true = (&k_gxg_k * &k_gxg_k).sum() / (gxg.dim().1 * rand_geno.dim().1) as f32;
    println!("tr_k_gxg_k_true: {}", tr_k_gxg_k_true);
    println!("error ratio: {:.5}%", (tr_k_gxg_k_est as f32 - tr_k_gxg_k_true) / tr_k_gxg_k_true * 100.);

    println!("\n=> test estimate_gxg_dot_y_norm_sq");
    let y = Array::random(num_rows, Normal::new(0., 1.))
        .mapv(|e| e as f32);
    let gxg_dot_y_norm_sq_est = estimate_gxg_dot_y_norm_sq(&gxg_basis, &y, 1000);
    println!("gxg_dot_y_norm_sq_est: {}", gxg_dot_y_norm_sq_est);
    let mut gxg_dot_y = gxg.t().dot(&y);
    gxg_dot_y.par_iter_mut().for_each(|x| *x = (*x) * (*x));
    let gxg_dot_y_norm_sq = gxg_dot_y.sum();
    println!("gxg_dot_y_norm_sq true: {}", gxg_dot_y_norm_sq);
    println!("error ratio: {:.5}%", (gxg_dot_y_norm_sq_est as f32 - gxg_dot_y_norm_sq) / gxg_dot_y_norm_sq * 100.);
}
