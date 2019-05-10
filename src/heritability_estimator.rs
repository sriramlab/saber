use ndarray::{Array, Ix1, Ix2};
use ndarray_linalg::Solve;
use bio_file_reader::plink_bed::MatrixIR;
use crate::timer::Timer;
use crate::matrix_util::{generate_plus_minus_one_bernoulli_matrix, matrix_ir_to_ndarray, mean_center_vector,
                         normalize_matrix_row_wise_inplace, row_mean_vec, row_std_vec};
use crate::stats_util::{sum, sum_of_squares, std};
use crate::gxg_trace_estimators::{estimate_kk_trace, estimate_gxg_gram_trace, estimate_gxg_dot_y_norm_sq, estimate_tr_k_gxg_k};
use colored::Colorize;

fn bold_print(msg: &String) {
    println!("{}", msg.bold());
}

pub fn estimate_heritability(mut geno_arr: Array<f32, Ix2>, mut pheno_arr: Array<f32, Ix1>, num_random_vecs: usize) -> Result<f64, String> {
    println!("\n=> creating the genotype ndarray and starting the timer for profiling");
    let mut timer = Timer::new();
    // geno_arr is num_snps x num_people
    let (num_rows, num_cols) = geno_arr.dim();
    println!("geno_arr dim: {:?}", geno_arr.dim());
    timer.print();

    println!("\n=> normalizing the genotype matrix row-wise");
    geno_arr = normalize_matrix_row_wise_inplace(geno_arr, 1);
    timer.print();

    println!("\n=> mean centering the phenotype vector");
    pheno_arr = mean_center_vector(pheno_arr);
    timer.print();

    println!("\n=> generating random estimators");
    let rand_mat = generate_plus_minus_one_bernoulli_matrix(num_cols, num_random_vecs);
    timer.print();

    println!("\n=> MatMul geno_arr{:?} with rand_mat{:?}", geno_arr.dim(), rand_mat.dim());
    let xz_arr = geno_arr.dot(&rand_mat);
    println!("xz_arr: {:?}", xz_arr.dim());
    timer.print();

    println!("\n=> MatMul geno_arr{:?}.T with xz_arr{:?}", geno_arr.dim(), xz_arr.dim());
    let xxz = geno_arr.t().dot(&xz_arr);
    println!("xxz dim: {:?}", xxz.dim());
    timer.print();

    println!("\n=> calculating trace estimate through L2 squared");
    let trace_kk_est = sum_of_squares(xxz.iter()) / (num_rows * num_rows * num_random_vecs) as f64;
    println!("trace_kk_est: {}", trace_kk_est);
    timer.print();

    println!("\n=> calculating Xy");
    let xy = geno_arr.dot(&pheno_arr);
    timer.print();

    println!("\n=> calculating yKy and yy");
    let yky = sum_of_squares(xy.iter()) / num_rows as f64;
    let yy = sum_of_squares(pheno_arr.iter());
    timer.print();

    println!("\n=> solving for heritability");
    let a = array![[trace_kk_est, (num_cols - 1) as f64],[(num_cols - 1) as f64, num_cols as f64]];
    let b = array![yky, yy];
    println!("solving ax=b\na = {:?}\nb = {:?}", a, b);
    let sig_sq = a.solve_into(b).unwrap();

    println!("sig_sq: {:?}", sig_sq);
    let s_y_sq = yy / (num_cols - 1) as f64;
    let heritability = sig_sq[0] as f64 / s_y_sq;
    println!("heritability: {}  s_y^2: {}", heritability, s_y_sq);

    let standard_error = (2. / (trace_kk_est - num_cols as f64)).sqrt();
    println!("standard error: {}", standard_error);

    Ok(heritability)
}

use crate::simulation::get_gxg_arr;

pub fn estimate_joint_heritability(mut geno_arr: Array<f32, Ix2>, mut independent_snps_arr: Array<f32, Ix2>,
    mut pheno_arr: Array<f32, Ix1>, num_random_vecs: usize) -> Result<(f64, f64, f64), String> {
    independent_snps_arr = independent_snps_arr.slice(s![..,..748]).to_owned();
    let (num_people, num_snps) = geno_arr.dim();
    let num_independent_snps = independent_snps_arr.dim().1;
    println!("num_people: {}\nnum_snps: {}\nnum_independent_snps: {}",
             num_people, num_snps, num_independent_snps);

    println!("\n=> normalizing the genotype matrix row-wise");
    geno_arr = normalize_matrix_row_wise_inplace(geno_arr.t().to_owned(), 0).t().to_owned();
    independent_snps_arr = normalize_matrix_row_wise_inplace(independent_snps_arr.t().to_owned(), 0).t().to_owned();

    println!("\n=> mean centering the phenotype vector");
    pheno_arr = mean_center_vector(pheno_arr);
    pheno_arr /= std(pheno_arr.iter(), 0) as f32;

    println!("\n=> estimating traces related to the G matrix");
    let rand_mat = generate_plus_minus_one_bernoulli_matrix(num_people, num_random_vecs);
    let xz_arr = geno_arr.t().dot(&rand_mat);
    let xxz = geno_arr.dot(&xz_arr);
    let tr_kk_est = sum_of_squares(xxz.iter()) / (num_snps * num_snps * num_random_vecs) as f64;
    println!("tr_kk_est: {}", tr_kk_est);
    let xy = geno_arr.t().dot(&pheno_arr);
    let yky = sum_of_squares(xy.iter()) / num_snps as f64;
    let yy = sum_of_squares(pheno_arr.iter());

    println!("\n=> estimating traces related to the GxG matrix");
    let num_snp_pairs = num_independent_snps * (num_independent_snps - 1) / 2;
    let mm = num_snp_pairs as f64;

    let gxg_tr_kk_est = estimate_kk_trace(&independent_snps_arr, num_random_vecs)? / (mm * mm);
    let gxg_tr_k_est = estimate_gxg_gram_trace(&independent_snps_arr, num_random_vecs)? / mm;

    println!("gxg_tr_k_est: {}", gxg_tr_k_est);
    println!("gxg_tr_kk_est: {}", gxg_tr_kk_est);

    let gxg_yky = estimate_gxg_dot_y_norm_sq(&independent_snps_arr, &pheno_arr, 1000) / mm;
    println!("gxg_yky: {}", gxg_yky);

    let tr_gk_est = estimate_tr_k_gxg_k(&geno_arr, &independent_snps_arr, num_random_vecs) / (mm * num_snps as f64);
    println!("tr_gk_est: {}", tr_gk_est);

    let n = num_people as f64;
    let a = array![[tr_kk_est, tr_gk_est, n], [tr_gk_est, gxg_tr_kk_est, gxg_tr_k_est], [n, gxg_tr_k_est, n]];
    let b = array![yky, gxg_yky, yy];
    println!("solving ax=b\na = {:?}\nb = {:?}", a, b);
    let sig_sq = a.solve_into(b).unwrap();

    println!("variance estimates: {:?}", sig_sq);
    Ok((sig_sq[0], sig_sq[1], sig_sq[2]))
}

pub fn estimate_gxg_heritability(geno_arr: Array<f32, Ix2>, mut pheno_arr: Array<f32, Ix1>, num_random_vecs: usize) -> Result<f64, String> {
    println!("\n=> estimate_gxg_heritability");
//    println!("\n=> creating the genotype ndarray and starting the timer for profiling");
//    let mut timer = Timer::new();
    // geno_arr is num_people x num_snps
    let (num_people, num_snps) = geno_arr.dim();
    println!("geno_arr dim: {:?}", geno_arr.dim());

    let num_snp_pairs = num_snps * (num_snps - 1) / 2;
    let mm = num_snp_pairs as f64;

//    println!("\n=> mean centering the phenotype vector");
//    pheno_arr = mean_center_vector(pheno_arr);
    //////
//    let gxg = get_gxg_arr(&geno_arr);
//    println!("GxG dim: {:?}", gxg.dim());
//    println!("\n=> calculating tr_k_true");
//    let tr_k_true = (&gxg * &gxg).dot(&Array::<f32, Ix1>::ones(gxg.dim().1)).sum() as f64 / mm;
//    println!("tr_k_true: {}", tr_k_true);
//    println!("\n=> calculating tr_kk_true");
//    let k = gxg.dot(&gxg.t());
//    let yky = pheno_arr.dot(&k.dot(&pheno_arr)) as f64 / mm;
//    let tr_kk_true = (&k * &k).dot(&Array::<f32, Ix1>::ones(k.dim().1)).sum() as f64 / (mm * mm);
//    println!("tr_kk_true: {}", tr_kk_true);
    //////

    let gxg_kk_trace_est = estimate_kk_trace(&geno_arr, num_random_vecs)? / (mm * mm);
    let gxg_k_trace_est = estimate_gxg_gram_trace(&geno_arr, num_random_vecs)? / mm;

    println!("gxg_k_trace_est: {}", gxg_k_trace_est);
    println!("gxg_kk_trace_est: {}", gxg_kk_trace_est);

    let yky = estimate_gxg_dot_y_norm_sq(&geno_arr, &pheno_arr, 1000) / mm;
    let yy = sum_of_squares(pheno_arr.iter());
    println!("yky: {}", yky);
    println!("yy: {}", yy);

    let a = array![[gxg_kk_trace_est, gxg_k_trace_est],[gxg_k_trace_est, num_people as f64]];
    let b = array![yky, yy];
    println!("solving ax=b\na = {:?}\nb = {:?}", a, b);
    let sig_sq = a.solve_into(b).unwrap();

    let sig_sq_g = sig_sq[0] as f64;
    let sig_sq_e = sig_sq[1] as f64;
    println!("\nsig_sq: {} {}", sig_sq_g, sig_sq_e);
    let heritability = sig_sq_g / (sig_sq_g + sig_sq_e);
    bold_print(&format!("heritability: {}", heritability));

    Ok(heritability)
}
