use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader};
use std::path::Path;

use analytic::stats::percentile_by;
use analytic::traits::HasDuplicate;
use clap::{clap_app, Arg};
use program_flow::argparse::{
    extract_boolean_flag, extract_numeric_arg, extract_optional_numeric_arg,
    extract_optional_str_arg, extract_optional_str_vec_arg, extract_str_arg, extract_str_vec_arg
};
use program_flow::OrExit;

use saber::simulation::sim_pheno::{generate_g_contribution_from_bed_bim, write_effects_to_file};
use saber::util::{get_bed_bim_from_prefix_and_partition, get_fid_iid_list, get_file_line_tokens};

fn main() {
    let mut app = clap_app!(generate_g_effects =>
        (version: "0.1")
        (author: "Aaron Zhou")
    );
    app = app
        .arg(
            Arg::with_name("plink_filename_prefix")
                .long("bfile")
                .short("b")
                .takes_value(true)
                .required(true)
                .multiple(true)
                .number_of_values(1)
                .help(
                    "If we have files named \n\
                     PATH/TO/x.bed PATH/TO/x.bim PATH/TO/x.fam \n\
                     then the <plink_filename_prefix> should be path/to/x"
                )
        )
        .arg(
            Arg::with_name("plink_dominance_prefix")
                .long("dominance-bfile")
                .short("d")
                .takes_value(true)
                .multiple(true)
                .number_of_values(1)
                .help(
                    "The SNPs for the dominance component. Same format as plink_filename_prefix."
                )
        )
        .arg(
            Arg::with_name("partition_filepath")
                .long("partition")
                .short("p")
                .takes_value(true)
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
            Arg::with_name("partition_variance_file")
                .long("--partition-var")
                .short("v")
                .takes_value(true)
                .multiple(true)
                .number_of_values(1)
                .help(
                    "Each line in the file has two tokens:\n\
                     partition_name total_partition_variance"
                )
        )
        .arg(
            Arg::with_name("partition_variance_paths_file")
                .long("--variance-pathfile")
                .short("f")
                .takes_value(true)
                .help(
                    "Each line in the file has two tokens:\n\
                     partition_variance_file num_reps\n\
                     where num_reps is the number of replicates to generate for that file"
                )
        )
        .arg(
            Arg::with_name("fill_noise")
                .long("fill-noise")
                .short("z")
                .help("This will generate noise so that the total phenotypic variance is 1.")
        )
        .arg(
            Arg::with_name("binary_ratio")
                .long("--binary")
                .takes_value(true)
                .value_name("BINARY_RATIO")
                .help("generates binary output with BINARY_RATIO ones in expectation.")
        )
        .arg(
            Arg::with_name("out_dir")
                .long("out-dir")
                .short("o")
                .takes_value(true)
                .help("output file directory")
        )
        .arg(
            Arg::with_name("chunk_size")
                .long("chunk-size")
                .takes_value(true)
                .default_value("100")
        );
    let matches = app.get_matches();

    let plink_filename_prefixes = extract_str_vec_arg(&matches, "plink_filename_prefix")
        .unwrap_or_exit(Some("failed to parse the bfile list".to_string()));

    let plink_dominance_prefixes = extract_optional_str_vec_arg(&matches, "plink_dominance_prefix");
    let partition_filepath = extract_optional_str_arg(&matches, "partition_filepath");
    let partition_variance_filepaths: Vec<(String, usize)> =
        extract_optional_str_vec_arg(&matches, "partition_variance_file")
            .unwrap_or(Vec::<String>::new())
            .into_iter()
            .map(|p| (p, 1usize))
            .collect();

    let partition_variance_paths_file =
        extract_optional_str_arg(&matches, "partition_variance_paths_file");

    let fill_noise = extract_boolean_flag(&matches, "fill_noise");
    let binary_ratio = extract_optional_numeric_arg::<f64>(&matches, "binary_ratio")
        .unwrap_or_exit(None::<String>);

    let out_dir = extract_str_arg(&matches, "out_dir");
    let chunk_size = extract_numeric_arg::<usize>(&matches, "chunk_size")
        .unwrap_or_exit(Some(format!("failed to extract chunk_size")));

    println!(
        "partition_filepath: {}\n\
         partition_variance_paths_file: {}\n\
         fill_noise: {}\n\
         out_dir: {}\n\
         binary_ratio: {:?}",
        partition_filepath.as_ref().unwrap_or(&"".to_string()),
        partition_variance_paths_file
            .as_ref()
            .unwrap_or(&"".to_string()),
        fill_noise,
        out_dir,
        binary_ratio
    );
    let partition_variance_filepaths_and_reps = match partition_variance_paths_file {
        None => partition_variance_filepaths,
        Some(partition_variance_paths_file) => {
            let mut paths: Vec<(String, usize)> =
                get_file_line_tokens(&partition_variance_paths_file, 2)
                    .unwrap_or_exit(Some(format!(
                        "failed to read the lines from {}",
                        partition_variance_paths_file
                    )))
                    .into_iter()
                    .map(|toks| {
                        let reps = toks[1]
                            .parse::<usize>()
                            .unwrap_or_exit(Some(format!("failed to parse {} as usize", toks[1])));
                        (toks.into_iter().nth(0).unwrap(), reps)
                    })
                    .collect();
            paths.extend(partition_variance_filepaths.into_iter());
            paths
        }
    };
    let num_paths = partition_variance_filepaths_and_reps.len();
    if num_paths == 0 {
        eprintln!("No partition_variance_file provided. Please provide them through -f or -v");
        std::process::exit(1);
    }
    partition_variance_filepaths_and_reps
        .iter()
        .enumerate()
        .for_each(|(i, (p, reps))| {
            println!("[{}/{}, reps: {}] {}", i + 1, num_paths, reps, p);
        });

    let out_paths = partition_variance_filepaths_and_reps
        .iter()
        .flat_map(|(path, reps)| {
            let basename = match Path::new(path).file_name() {
                None => {
                    eprintln!("Invalid variance filename: {}", path);
                    std::process::exit(1);
                }
                Some(path) => path
            };
            let out_prefix = match Path::new(&out_dir).join(basename).to_str() {
                Some(s) => s.to_string(),
                None => {
                    eprintln!(
                        "failed to create output filepath for outdir: {} and filename: {}",
                        out_dir, path
                    );
                    std::process::exit(1);
                }
            };
            (0..*reps)
                .into_iter()
                .map(|i| format!("{}_rep{}.effects", out_prefix, i + 1))
                .collect::<Vec<String>>()
        })
        .collect::<Vec<String>>();

    let num_out_paths = out_paths.len();
    println!("\nout_paths:");
    out_paths.iter().enumerate().for_each(|(i, p)| {
        println!("[{}/{}] {}", i + 1, num_out_paths, p);
    });
    if out_paths.has_duplicate() {
        eprintln!(
            "{}",
            "The default-created output paths for the simulated effects have duplicates. \
             Please make sure the basenames of all the variance files are distinct."
        );
        std::process::exit(1);
    }

    if let Some(r) = binary_ratio {
        if r < 0. || r > 1. {
            eprintln!(
                "{}",
                format!("binary ratio has to be between 0 and 1, received {}", r)
            );
            std::process::exit(1);
        }
    }

    let (bed, bim) = get_bed_bim_from_prefix_and_partition(
        &plink_filename_prefixes,
        &plink_dominance_prefixes,
        &partition_filepath
    )
    .unwrap_or_exit(None::<String>);

    type PartitionKey = String;
    type VarianceValue = f64;
    let partition_to_variances = partition_variance_filepaths_and_reps.iter().fold(
        HashMap::<PartitionKey, Vec<VarianceValue>>::new(),
        |mut acc_map, (path, reps)| {
            let partition_to_variances = get_partition_to_variance(path)
                .unwrap_or_exit(Some(format!("failed to get partition_to_variance_map")));
            for (partition_name, variance) in partition_to_variances.iter() {
                let mut vars = vec![*variance; *reps];
                acc_map
                    .entry(partition_name.to_string())
                    .or_insert(Vec::new())
                    .append(&mut vars);
            }
            acc_map
        }
    );

    println!("\n=> generating G effects");
    let effects = generate_g_contribution_from_bed_bim(
        &bed,
        &bim,
        &partition_to_variances,
        fill_noise,
        chunk_size
    )
    .unwrap_or_exit(None::<String>);
    let fid_iid_list = get_fid_iid_list(&format!("{}.fam", plink_filename_prefixes[0]))
        .unwrap_or_exit(None::<String>);

    assert_eq!(effects.dim().1, num_out_paths);
    for (i, y) in effects.gencolumns().into_iter().enumerate() {
        let pheno_output = match binary_ratio {
            None => y.to_owned(),
            Some(r) => {
                let lowest_positive_score =
                    percentile_by(y.to_vec(), 1. - r, |a, b| a.partial_cmp(b).unwrap())
                        .unwrap_or_exit(Some(format!(
                            "failed to get percentile {} for the generated effects of length {}",
                            1. - r,
                            y.len()
                        )));
                println!(
                    "lowest positive score: {} to achieve a ratio of {} \
                     for the number of positive labels",
                    lowest_positive_score, r
                );
                y.mapv(|e| if e >= lowest_positive_score { 1. } else { 0. })
            }
        };
        let path = &out_paths[i];
        println!("=> writing the effects due to {}", path);
        write_effects_to_file(&pheno_output, &fid_iid_list, path).unwrap_or_exit(Some(format!(
            "failed to write the simulated effects to file: {}",
            path
        )));
    }
}

fn get_partition_to_variance(
    partition_variance_filepath: &str
) -> Result<HashMap<String, f64>, String> {
    let buf = match OpenOptions::new()
        .read(true)
        .open(partition_variance_filepath)
    {
        Err(why) => {
            return Err(format!(
                "failed to open {}: {}",
                partition_variance_filepath, why
            ));
        }
        Ok(f) => BufReader::new(f)
    };
    Ok(buf
        .lines()
        .map(|l| {
            let toks: Vec<String> = l
                .unwrap()
                .split_whitespace()
                .map(|t| t.to_string())
                .collect();
            if toks.len() != 2 {
                Err(format!(
                    "Each line in the partition variance file should have 2 tokens, found {}",
                    toks.len()
                ))
            } else {
                let variance = toks[1].parse::<f64>().unwrap();
                Ok((toks[0].to_owned(), variance))
            }
        })
        .collect::<Result<HashMap<String, f64>, String>>()?)
}

#[cfg(test)]
mod tests {
    use std::fs::OpenOptions;
    use std::io::{BufWriter, Write};

    use tempfile::NamedTempFile;

    use crate::get_partition_to_variance;

    #[test]
    fn test_get_partition_to_variance() {
        let partition_to_var_path = NamedTempFile::new().unwrap().into_temp_path();
        {
            let mut buf = BufWriter::new(
                OpenOptions::new()
                    .write(true)
                    .truncate(true)
                    .create(true)
                    .open(partition_to_var_path.to_str().unwrap())
                    .unwrap()
            );
            buf.write_fmt(format_args!(
                "{} {}\n\
                 {} {}\n\
                 {} {}\n\
                 {} {}\n",
                "p1", 0.02, "p2", 0., "p3", 0.425, "p4", 0.01,
            ))
            .unwrap();
        }
        let partition_to_var =
            get_partition_to_variance(partition_to_var_path.to_str().unwrap()).unwrap();
        assert_eq!(partition_to_var["p1"], 0.02);
        assert_eq!(partition_to_var["p2"], 0.);
        assert_eq!(partition_to_var["p3"], 0.425);
        assert_eq!(partition_to_var["p4"], 0.01);
    }
}
