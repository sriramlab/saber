use clap::clap_app;

use bio_file_reader::plink_bed::PlinkBed;
use saber::heritability_estimator::estimate_heritability;
use saber::program_flow::OrExit;
use saber::util::{extract_str_arg, get_pheno_arr};

fn main() {
    let matches = clap_app!(estimate_heritability =>
        (version: "0.1")
        (author: "Aaron Zhou")
        (@arg plink_filename_prefix: --bfile -b <BFILE> "required; the prefix for x.bed, x.bim, x.fam is x")
        (@arg pheno_filename: --pheno -p <PHENO> "required; each row is one individual containing one phenotype value")
        (@arg num_random_vecs: --nrv +takes_value "number of random vectors used to estimate traces; required")
    ).get_matches();

    let plink_filename_prefix = extract_str_arg(&matches, "plink_filename_prefix");
    let pheno_filename = extract_str_arg(&matches, "pheno_filename");

    let plink_bed_path = format!("{}.bed", plink_filename_prefix);
    let plink_bim_path = format!("{}.bim", plink_filename_prefix);
    let plink_fam_path = format!("{}.fam", plink_filename_prefix);

    let num_random_vecs = extract_str_arg(&matches, "num_random_vecs")
        .parse::<usize>()
        .unwrap_or_exit(Some("failed to parse num_random_vecs"));

    println!("PLINK bed path: {}\nPLINK bim path: {}\nPLINK fam path: {}",
             plink_bed_path, plink_bim_path, plink_fam_path);
    println!("pheno_filepath: {}\nnum_random_vecs: {}", pheno_filename, num_random_vecs);

    let pheno_arr = get_pheno_arr(&pheno_filename)
        .unwrap_or_exit(None::<String>);

    let bed = PlinkBed::new(&plink_bed_path,
                            &plink_bim_path,
                            &plink_fam_path)
        .unwrap_or_exit(None::<String>);

    match estimate_heritability(bed,
                                pheno_arr,
                                num_random_vecs) {
        Ok(h) => {
            println!("\nheritability estimate: {}", h);
        }
        Err(why) => {
            eprintln!("{}", why);
            return ();
        }
    };
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use rand::distributions::StandardNormal;

    use saber::util::matrix_util::generate_plus_minus_one_bernoulli_matrix;
    use saber::util::stats_util::sum_of_squares;

    #[test]
    fn test_trace_estimator() {
        let n = 1000;
        let num_random_vecs = 40;
        let x = Array::random(
            (n, n),
            StandardNormal).mapv(|e| e as i32 as f32);
        // want to estimate the trace of x.t().dot(&x)
        let true_trace = sum_of_squares(x.iter());
        println!("true trace: {}", true_trace);

        let rand_mat = generate_plus_minus_one_bernoulli_matrix(n, num_random_vecs);

        let trace_est = sum_of_squares(x.dot(&rand_mat).iter()) / num_random_vecs as f64;
        println!("trace_est: {}", trace_est);
    }

    #[test]
    fn test_bernoulli_matrix() {
        let n = 1000;
        let num_random_vecs = 100;
        let rand_mat = generate_plus_minus_one_bernoulli_matrix(n, num_random_vecs);
        assert_eq!((n, num_random_vecs), rand_mat.dim());
        let mut value_set = HashSet::<i32>::new();
        for a in rand_mat.iter() {
            value_set.insert(*a as i32);
        }
        // almost certainly this will contain the two values 1 and -1
        assert_eq!(2, value_set.len());
        assert_eq!(true, value_set.contains(&-1));
        assert_eq!(true, value_set.contains(&1));
    }
}
