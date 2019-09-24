use std::fs::OpenOptions;
use std::io::{BufWriter, Write};

use analytic::set::ordered_integer_set::OrderedIntegerSet;
use biofile::plink_bed::PlinkBed;
use clap::clap_app;
use program_flow::argparse::extract_str_arg;

use program_flow::OrExit;
use saber::util::get_bed_bim_fam_path;

fn main() {
    let app = clap_app!(generate_gxg_bed =>
        (version: "0.1")
        (author: "Aaron Zhou")
        (@arg bfile: --bfile -b <BFILE> "required; the prefix for x.bed, x.bim, x.fam is x")
        (@arg out_path: --out <OUT> "required; output file path")
    );
    let matches = app.get_matches();

    let out_path = extract_str_arg(&matches, "out_path");
    let bfile = extract_str_arg(&matches, "bfile");
    let [bed_path, bim_path, fam_path] = get_bed_bim_fam_path(&bfile);
    println!(
        "PLINK bed path: {}\n\
        PLINK bim path: {}\n\
        PLINK fam path: {}\n\
        out_path: {}",
        bed_path, bim_path, fam_path, out_path);
    let bed = PlinkBed::new(&vec![(bed_path, bim_path, fam_path)])
        .unwrap_or_exit(None::<String>);

    println!("\n=> writing gxg bed to {}", out_path);
    let num_people = bed.num_people;
    let num_g_snps = bed.total_num_snps();
    let mut buf_writer = BufWriter::new(OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(out_path)
        .unwrap_or_exit(None::<String>)
    );
    buf_writer.write(&[0x6c, 0x1b, 0x1]).unwrap_or_exit(None::<String>);

    let chunk_size = 100;
    let total_chunks = num_g_snps / chunk_size + (num_g_snps % chunk_size > 0) as usize;
    bed.col_chunk_iter(chunk_size, None)
       .enumerate()
       .for_each(|(chunk_index_i, chunk_i)| {
           println!("processing chunk [{}/{}]", chunk_index_i + 1, total_chunks);
           let chunk_i = chunk_i.mapv(|e| e as u8);
           bed
               .col_chunk_iter(
                   chunk_size,
                   Some(OrderedIntegerSet::from_slice(
                       &[[chunk_index_i * chunk_size, num_g_snps - 1]])
                   ),
               )
               .enumerate()
               .for_each(|(chunk_index_j, chunk_j)| {
                   let chunk_j = chunk_j.mapv(|e| e as u8);
                   let same_chunk = chunk_index_j == 0;
                   for (i, col_i) in chunk_i.gencolumns().into_iter().enumerate() {
                       for (j, col_j) in chunk_j.gencolumns().into_iter().enumerate() {
                           if same_chunk && j <= i {
                               continue;
                           }
                           let mut k = 0;
                           for _ in 0..num_people / 4 {
                               buf_writer.write(&[
                                   PlinkBed::geno_to_lowest_two_bits(col_i[k] * col_j[k])
                                       | (PlinkBed::geno_to_lowest_two_bits(col_i[k + 1] * col_j[k + 1]) << 2)
                                       | (PlinkBed::geno_to_lowest_two_bits(col_i[k + 2] * col_j[k + 2]) << 4)
                                       | (PlinkBed::geno_to_lowest_two_bits(col_i[k + 3] * col_j[k + 3]) << 6)
                               ]).unwrap_or_exit(None::<String>);
                               k += 4;
                           }
                           let remainder = num_people % 4;
                           if remainder > 0 {
                               let mut byte = 0u8;
                               for j in 0..remainder {
                                   byte |= PlinkBed::geno_to_lowest_two_bits(col_i[k + j] * col_j[k + j]) << (j * 2);
                               }
                               buf_writer.write(&[byte]).unwrap_or_exit(None::<String>);
                           }
                       }
                   }
               });
       });
}
