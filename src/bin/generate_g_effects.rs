use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader};

use clap::{Arg, clap_app};
use program_flow::argparse::{extract_numeric_arg, extract_optional_str_arg, extract_optional_str_vec_arg, extract_str_arg, extract_str_vec_arg, extract_boolean_flag, extract_optional_numeric_arg};
use program_flow::OrExit;

use saber::simulation::sim_pheno::{
    generate_g_contribution_from_bed_bim, get_sim_output_path, SimEffectMechanism,
    write_effects_to_file,
};
use saber::util::{get_bed_bim_from_prefix_and_partition, get_fid_iid_list};

fn main() {
    let mut app = clap_app!(generate_g_effects =>
        (version: "0.1")
        (author: "Aaron Zhou")
    );
    app = app
        .arg(
            Arg::with_name("plink_filename_prefix")
                .long("bfile").short("b").takes_value(true).required(true)
                .multiple(true).number_of_values(1)
                .help(
                    "If we have files named \n\
                    PATH/TO/x.bed PATH/TO/x.bim PATH/TO/x.fam \n\
                    then the <plink_filename_prefix> should be path/to/x"
                )
        )
        .arg(
            Arg::with_name("plink_dominance_prefix")
                .long("dominance-bfile").short("d").takes_value(true)
                .multiple(true).number_of_values(1)
                .help(
                    "The SNPs for the dominance component. Same format as plink_filename_prefix."
                )
        )
        .arg(
            Arg::with_name("partition_filepath")
                .long("partition").short("p").takes_value(true)
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
                .long("--partition-var").short("v").takes_value(true).required(true)
                .help(
                    "Each line in the file has two tokens:\n\
                    partition_name total_partition_variance"
                )
        )
        .arg(
            Arg::with_name("binary_ratio")
                .long("--binary").takes_value(true).value_name("BINARY_RATIO")
                .help("generates binary output with BINARY_RATIO ones in expectation.")
        )
        .arg(
            Arg::with_name("fill_noise")
                .long("fill-noise").short("z")
                .help("This will generate noise so that the total phenotypic variance is 1.")
        )
        .arg(
            Arg::with_name("out_path_prefix")
                .long("out-prefix").short("o").takes_value(true)
                .help("output file path prefix")
        )
        .arg(
            Arg::with_name("chunk_size")
                .long("chunk-size").takes_value(true).default_value("1000")
        );
    let matches = app.get_matches();

    let plink_filename_prefixes = extract_str_vec_arg(&matches, "plink_filename_prefix")
        .unwrap_or_exit(Some("failed to parse the bfile list".to_string()));

    let plink_dominance_prefixes = extract_optional_str_vec_arg(&matches, "plink_dominance_prefix");
    let partition_filepath = extract_optional_str_arg(&matches, "partition_filepath");
    let partition_variance_filepath = extract_str_arg(&matches, "partition_variance_file");
    let out_path_prefix = extract_str_arg(&matches, "out_path_prefix");
    let binary_ratio = extract_optional_numeric_arg(&matches, "binary_ratio")
        .unwrap_or_exit(None::<String>);
    let fill_noise = extract_boolean_flag(&matches, "fill_noise");
    let chunk_size = extract_numeric_arg::<usize>(&matches, "chunk_size")
        .unwrap_or_exit(Some(format!("failed to extract chunk_size")));

    println!(
        "partition_filepath: {}\n\
        partition_variance_filepath: {}\n\
        fill_noise: {}\n\
        out_path_prefix: {}",
        partition_filepath.as_ref().unwrap_or(&"".to_string()),
        partition_variance_filepath,
        fill_noise,
        out_path_prefix
    );

    let (bed, bim) = get_bed_bim_from_prefix_and_partition(
        &plink_filename_prefixes,
        &plink_dominance_prefixes,
        &partition_filepath,
    ).unwrap_or_exit(None::<String>);

    let partition_to_variance = get_partition_to_variance(&partition_variance_filepath)
        .unwrap_or_exit(Some(format!("failed to get the partition_to_variance map")));

    println!("\n=> generating G effects");
    let out_path = get_sim_output_path(&out_path_prefix, SimEffectMechanism::G);
    let effects = generate_g_contribution_from_bed_bim(
        &bed,
        &bim,
        &partition_to_variance,
        fill_noise,
        chunk_size,
    ).unwrap_or_exit(None::<String>);

    let pheno_output = match binary_ratio {
        None => effects,
        Some(r) => effects.mapv(|e| if e > r { 1. } else { 0. })
    };

    let fid_iid_list = get_fid_iid_list(&format!("{}.fam", plink_filename_prefixes[0]))
        .unwrap_or_exit(None::<String>);

    println!("\n=> writing the effects due to G to {}", out_path);
    write_effects_to_file(&pheno_output, &fid_iid_list, &out_path)
        .unwrap_or_exit(Some(format!("failed to write the simulated effects to file: {}", out_path)));
}

fn get_partition_to_variance(partition_variance_filepath: &str) -> Result<HashMap<String, f64>, String> {
    let buf = match OpenOptions::new().read(true).open(partition_variance_filepath) {
        Err(why) => return Err(format!("failed to open {}: {}", partition_variance_filepath, why)),
        Ok(f) => BufReader::new(f)
    };
    Ok(
        buf.lines()
           .map(|l| {
               let toks: Vec<String> =
                   l.unwrap()
                    .split_whitespace()
                    .map(|t| t.to_string())
                    .collect();
               if toks.len() != 2 {
                   Err(format!("Each line in the partition variance file should have 2 tokens, found {}", toks.len()))
               } else {
                   let variance = toks[1].parse::<f64>().unwrap();
                   Ok((toks[0].to_owned(), variance))
               }
           })
           .collect::<Result<HashMap<String, f64>, String>>()?
    )
}

#[cfg(test)]
mod tests {
    use std::io::{BufWriter, Write};

    use tempfile::NamedTempFile;

    use crate::get_partition_to_variance;
    use std::fs::OpenOptions;

    #[test]
    fn test_get_partition_to_variance() {
        let partition_to_var_path = NamedTempFile::new().unwrap().into_temp_path();
        {
            let mut buf = BufWriter::new(
                OpenOptions::new().write(true).truncate(true).create(true)
                                  .open(partition_to_var_path.to_str().unwrap()).unwrap()
            );
            buf.write_fmt(format_args!(
                "{} {}\n\
                {} {}\n\
                {} {}\n\
                {} {}\n",
                "p1", 0.02,
                "p2", 0.,
                "p3", 0.425,
                "p4", 0.01,
            )).unwrap();
        }
        let partition_to_var = get_partition_to_variance(partition_to_var_path.to_str().unwrap()).unwrap();
        assert_eq!(partition_to_var["p1"], 0.02);
        assert_eq!(partition_to_var["p2"], 0.);
        assert_eq!(partition_to_var["p3"], 0.425);
        assert_eq!(partition_to_var["p4"], 0.01);
    }
}
