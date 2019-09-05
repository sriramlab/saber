use analytic::set::ordered_integer_set::OrderedIntegerSet;
use analytic::set::traits::Finite;
use analytic::traits::Collecting;
use biofile::plink_bed::PlinkBed;
use biofile::plink_bim::{FilelinePartitions, PlinkBim};
use clap::{Arg, clap_app};

use saber::heritability_estimator::DEFAULT_PARTITION_NAME;
use saber::heritability_estimator::estimate_heritability;
use saber::program_flow::OrExit;
use saber::util::{
    extract_numeric_arg, extract_optional_numeric_arg, extract_optional_str_arg, extract_str_arg,
    get_pheno_arr,
};

fn main() {
    let mut app = clap_app!(estimate_heritability =>
        (version: "0.1")
    );
    app = app
        .arg(
            Arg::with_name("plink_filename_prefix")
                .long("bfile").short("b").takes_value(true).required(true)
                .help(
                    "If we have files named \n\
                    PATH/TO/x.bed PATH/TO/x.bim PATH/TO/x.fam \n\
                    then the <plink_filename_prefix> should be path/to/x"
                )
        )
        .arg(
            Arg::with_name("pheno_path")
                .long("pheno").short("p").takes_value(true).required(true)
                .help(
                    "The header line should be\n\
                    FID IID PHENOTYPE_NAME\n\
                    where PHENOTYPE_NAME can be any string without white spaces.\n\
                    The rest of the lines are of the form:\n\
                    1000011 1000011 -12.11363"
                )
        )
        .arg(
            Arg::with_name("num_random_vecs")
                .long("nrv").takes_value(true).required(true)
                .help(
                    "The number of random vectors used to estimate traces\n\
                    Recommends at least 100 for small datasets, and 10 for huge datasets"
                )
        )
        .arg(
            Arg::with_name("num_jackknife_partitions")
                .long("--num-jackknifes").short("k").takes_value(true).default_value("20")
                .help(
                    "The number of jackknife partitions\n\
                    SNPs will be divided into <num_jackknife_partitions> partitions\n\
                    where each partition will be treated as a single point of observation"
                )
        )
        .arg(
            Arg::with_name("partition_file")
                .long("partition").takes_value(true)
                .help(
                    "A file to partition the SNPs into multiple components.\n\
                    Each line consists of two values of the form:\n\
                    SNP_ID PARTITION\n\
                    For example,\n\
                    rs3115860 1\n\
                    will assign SNP with ID rs3115860 in the BIM file to a partition named 1"
                )
        )
        .arg(
            Arg::with_name("lowest_allowed_maf")
                .long("lowest-maf").takes_value(true)
                .help(
                    "Lowest allowed minor allele frequency (MAF)\n\
                    Any SNPs with a MAF less than <lowest_allowed_maf> will be ignored"
                )
        );
    let matches = app.get_matches();

    let plink_filename_prefix = extract_str_arg(&matches, "plink_filename_prefix");
    let pheno_path = extract_str_arg(&matches, "pheno_path");
    let partition_filepath = extract_optional_str_arg(&matches, "partition_file");

    let num_jackknife_partitions = extract_numeric_arg::<usize>(
        &matches, "num_jackknife_partitions",
    ).unwrap_or_exit(Some("failed to extract num_jackknife_partitions"));

    let lowest_allowed_maf = extract_optional_numeric_arg::<f32>(
        &matches, "lowest_allowed_maf",
    ).unwrap_or_exit(Some("failed to extract lowest_allowed_maf"));

    let num_random_vecs = extract_str_arg(&matches, "num_random_vecs")
        .parse::<usize>()
        .unwrap_or_exit(Some("failed to parse num_random_vecs"));

    let plink_bed_path = format!("{}.bed", plink_filename_prefix);
    let plink_bim_path = format!("{}.bim", plink_filename_prefix);
    let plink_fam_path = format!("{}.fam", plink_filename_prefix);

    println!(
        "PLINK bed path: {}\n\
        PLINK bim path: {}\n\
        PLINK fam path: {}",
        plink_bed_path,
        plink_bim_path,
        plink_fam_path
    );
    println!(
        "pheno_filepath: {}\n\
        num_random_vecs: {}",
        pheno_path,
        num_random_vecs
    );
    println!(
        "partition_filepath: {}\n\
        num_jackknife_partitions: {}",
        partition_filepath.as_ref().unwrap_or(&"".to_string()),
        num_jackknife_partitions
    );

    let pheno_arr = get_pheno_arr(&pheno_path)
        .unwrap_or_exit(None::<String>);

    let bed = PlinkBed::new(
        &plink_bed_path,
        &plink_bim_path,
        &plink_fam_path,
    ).unwrap_or_exit(None::<String>);

    let maf = bed.get_minor_allele_frequencies(None);
    let mut low_maf = OrderedIntegerSet::new();
    match lowest_allowed_maf {
        None => {
            maf.into_iter().enumerate().for_each(|(i, f)| {
                if f == 0. {
                    low_maf.collect(i);
                }
            })
        }
        Some(l) => {
            maf.into_iter().enumerate().for_each(|(i, f)| {
                if f < l {
                    low_maf.collect(i);
                }
            })
        }
    };
    println!(
        "removing {} alleles with frequency < {}",
        low_maf.size(),
        lowest_allowed_maf.unwrap_or(0.)
    );

    let mut bim = match &partition_filepath {
        Some(partition_filepath) => PlinkBim::new_with_partition_file(
            &plink_bim_path,
            partition_filepath,
        ).unwrap_or_exit(
            Some(format!(
                "failed to create PlinkBim from bim file: {} and partition file: {}",
                &plink_bim_path, partition_filepath
            ))
        ),
        None => PlinkBim::new(&plink_bim_path)
            .unwrap_or_exit(Some(format!("failed to create PlinkBim from {}", &plink_bim_path))),
    };
    let mut filtered_partitions = bim
        .get_fileline_partitions_or(
            DEFAULT_PARTITION_NAME,
            OrderedIntegerSet::from_slice(&[[0, bed.num_snps - 1]]),
        )
        .into_hash_map();
    filtered_partitions.values_mut().for_each(|v| *v -= &low_maf);
    bim.set_fileline_partitions(Some(FilelinePartitions::new(filtered_partitions)));

    match estimate_heritability(bed,
                                bim,
                                pheno_arr,
                                num_random_vecs,
                                num_jackknife_partitions) {
        Ok(h) => {
            println!("\nheritability estimates:\n{}", h);
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
