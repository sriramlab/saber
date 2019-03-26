#[macro_use]
extern crate clap;

use clap::ArgMatches;
use bio_file_reader::plink_bed::PlinkBed;

fn extract_filename_arg(matches: &ArgMatches, arg_name: &str) -> String {
    match matches.value_of(arg_name) {
        Some(filename) => filename.to_string(),
        None => {
            eprintln!("the argument {} is required", arg_name);
            std::process::exit(1);
        }
    }
}

struct SparsityStats {
    sparsity_vec: Vec<f32>,
}

impl SparsityStats {
    fn new(matrix: &Vec<Vec<u8>>) -> SparsityStats {
        let mut sparsity_vec = Vec::new();
        for snp_variants in matrix.iter() {
            let num_zeros = snp_variants.iter().fold(0, |acc, &a| acc + (a == 0u8) as usize);
            sparsity_vec.push(num_zeros as f32 / snp_variants.len() as f32);
        }
        SparsityStats { sparsity_vec }
    }

    fn histogram(&self, num_intervals: usize) -> Vec<f32> {
        let mut histogram = vec![0; num_intervals + 1];
        for s in self.sparsity_vec.iter() {
            let index = (*s * num_intervals as f32) as usize;
            histogram[index] += 1;
        }
        let mut ratio_distribution = vec![0f32; num_intervals + 1];
        for i in 0..num_intervals + 1 {
            ratio_distribution[i] = histogram[i] as f32 / self.sparsity_vec.len() as f32;
        }
        ratio_distribution
    }

    fn avg_sparsity(&self) -> f32 {
        let mut iter = self.sparsity_vec.iter();
        let mut current_mean = match iter.next() {
            None => return 0f32,
            Some(a) => *a
        };

        for (i, a) in iter.enumerate() {
            let ratio = i as f32 / (i + 1) as f32;
            current_mean = current_mean * ratio + a / (i + 1) as f32
        }
        current_mean
    }
}

fn main() {
    let matches = clap_app!(Saber =>
        (version: "0.1")
        (author: "Aaron Zhou")
        (@arg plink_bed_filename: --bed <BED> "required")
        (@arg plink_bim_filename: --bim <BIM> "required")
        (@arg plink_fam_filename: --fam <FAM> "required")
    ).get_matches();

    let plink_bed_filename = extract_filename_arg(&matches, "plink_bed_filename");
    let plink_bim_filename = extract_filename_arg(&matches, "plink_bim_filename");
    let plink_fam_filename = extract_filename_arg(&matches, "plink_fam_filename");

    println!("PLINK bed filename: {}\nPLINK bim filename: {}\nPLINK fam filename: {}",
             plink_bed_filename, plink_bim_filename, plink_fam_filename);

    let mut bed = match PlinkBed::new(&plink_bed_filename, &plink_bim_filename, &plink_fam_filename) {
        Err(why) => {
            println!("{}", why);
            std::process::exit(1);
        }
        Ok(bed) => bed
    };

    println!("=> generating the genotype matrix");
    let genotype_matrix = match bed.get_genotype_matrix() {
        Err(io_error) => {
            eprintln!("failed to get the genotype matrix: {}", io_error);
            std::process::exit(1);
        }
        Ok(matrix) => matrix
    };
    println!("genotype_matrix.shape: ({}, {})", genotype_matrix.len(), genotype_matrix.first().unwrap_or(&Vec::new()).len());

    let stats = SparsityStats::new(&genotype_matrix);
    println!("avg sparsity: {}", stats.avg_sparsity());
    let num_intervals = 20usize;
    let histogram = stats.histogram(num_intervals);
    let mut cum = 0f32;
    let delta = 1f32 / num_intervals as f32;
    for i in (0..num_intervals).rev() {
        cum += histogram[i];
        println!("[{:<3.2}, {:>3.2}): {:>6.2} {:>6.2}", i as f32 * delta, (i + 1) as f32 * delta, histogram[i], cum);
    }
}
