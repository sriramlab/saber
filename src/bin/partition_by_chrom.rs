use std::{
    fs::OpenOptions,
    io::{BufRead, BufReader, BufWriter, Write},
};

use biofile::plink_bim::{CHROM_FIELD_INDEX, VARIANT_ID_FIELD_INDEX};
use clap::{clap_app, Arg};
use program_flow::{
    argparse::{extract_str_arg, extract_str_vec_arg},
    OrExit,
};

fn main() {
    let mut app = clap_app!(partition_by_chrom =>
        (version: "0.1")
        (author: "Aaron Zhou")
    );
    app = app
        .arg(
            Arg::with_name("bim")
                .long("bim").short("b").takes_value(true).required(true).multiple(true)
                .help("Plink BIM files")
        )
        .arg(
            Arg::with_name("out_path")
                .long("out")
                .short("o")
                .takes_value(true)
                .required(true)
                .help("output path; each line will have two fields: variant_id chrom_partition_assignment")
        );
    let matches = app.get_matches();

    let bim_path_list = extract_str_vec_arg(&matches, "bim")
        .unwrap_or_exit(Some("failed to parse the bim paths"));
    let out_path = extract_str_arg(&matches, "out_path");

    println!(
        "PLINK bim path: {:?}\nout_path: {}",
        bim_path_list, out_path
    );
    let mut writer = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&out_path)
            .unwrap_or_exit(Some(format!("failed to create {}", out_path))),
    );
    assert_eq!(CHROM_FIELD_INDEX, 0);
    assert_eq!(VARIANT_ID_FIELD_INDEX, 1);
    for path in bim_path_list.iter() {
        let file = OpenOptions::new()
            .read(true)
            .open(path)
            .unwrap_or_exit(Some(format!("failed to open {}", path)));
        for (i, line) in BufReader::new(file).lines().enumerate() {
            let l = line.unwrap_or_exit(Some(
                "failed to get lines from the bim file object",
            ));
            let mut toks = l.split_whitespace();

            let partition = toks
                .next()
                .unwrap_or_exit(Some(format!(
                    "failed to extract the chrom from line {} in {}",
                    i, path
                )))
                .to_string();

            let variant_id = toks
                .next()
                .unwrap_or_exit(Some(format!(
                    "failed to extract variant id from line {} in {}",
                    i, path
                )))
                .to_string();

            writer
                .write_fmt(format_args!("{} {}\n", variant_id, partition))
                .unwrap_or_exit(Some(format!(
                    "failed to write to file: {}",
                    out_path
                )));
        }
    }
}
