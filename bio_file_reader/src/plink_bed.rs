use std::cmp::min;
use std::fs::{File, OpenOptions};
use std::io;
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};

use ndarray::{Array, Ix2, ShapeBuilder};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};

use crate::byte_chunk_iter::ByteChunkIter;
use crate::error::Error;

pub struct PlinkBed {
    bed_buf: BufReader<File>,
    pub num_people: usize,
    pub num_snps: usize,
    pub num_bytes_per_snp: usize,
    pub filepath: String,
}

impl PlinkBed {
    pub fn get_magic_bytes() -> [u8; 3] {
        [0x6c_u8, 0x1b_u8, 0x01_u8]
    }

    fn get_buf(filename: &str) -> Result<BufReader<File>, Error> {
        match OpenOptions::new().read(true).open(filename) {
            Err(io_error) => Err(Error::IO { why: format!("failed to open {}: {}", filename, io_error), io_error }),
            Ok(f) => Ok(BufReader::new(f))
        }
    }

    fn get_line_count(filename: &str) -> Result<usize, Error> {
        let fam_buf = PlinkBed::get_buf(filename)?;
        Ok(fam_buf.lines().count())
    }

    fn usize_div_ceil(a: usize, divisor: usize) -> usize {
        a / divisor + (a % divisor != 0) as usize
    }

    pub fn new(bed_filename: &str, bim_filename: &str, fam_filename: &str) -> Result<PlinkBed, Error> {
        let mut bed_buf = PlinkBed::get_buf(bed_filename)?;

        // check if PLINK bed file has the correct file signature
        let expected_bytes = [0x6c_u8, 0x1b_u8, 0x01_u8];
        let mut magic_bytes = [0u8; 3];
        if let Err(io_error) = bed_buf.read_exact(&mut magic_bytes) {
            return Err(Error::IO { why: format!("Failed to read the first three bytes of {}: {}", bed_filename, io_error), io_error });
        }
        if magic_bytes != expected_bytes {
            return Err(Error::BadFormat(
                format!("The first three bytes of the PLINK bed file are supposed to be 0x{:x?}, but found 0x{:x?}",
                        expected_bytes, magic_bytes)
            ));
        }

        let num_people = PlinkBed::get_line_count(fam_filename)?;
        let num_snps = PlinkBed::get_line_count(bim_filename)?;
        let num_bytes_per_snp = PlinkBed::usize_div_ceil(num_people, 4);
        println!("{} stats:\nnum_snps: {}\nnum_people: {}\nnum_bytes_per_block: {}\n----------",
                 bed_filename, num_snps, num_people, num_bytes_per_snp);

        Ok(PlinkBed { bed_buf, num_people, num_snps, num_bytes_per_snp, filepath: bed_filename.to_string() })
    }

    // the first person is the lowest two bits
    // 00 -> 2 homozygous for the first allele in the .bim file (usually the minor allele)
    // 01 -> 0 missing genotype
    // 10 -> 1 heterozygous
    // 11 -> 0 homozygous for the second allele in the .bim file (usually the major allele)
    pub fn create_bed(arr: &Array<u8, Ix2>, out_path: &str) -> Result<(), Error> {
        let (num_peope, _num_snps) = arr.dim();
        let mut buf_writer = BufWriter::new(OpenOptions::new().create(true).truncate(true).write(true).open(out_path)?);
        buf_writer.write(&[0x6c, 0x1b, 0x1])?;
        for col in arr.gencolumns() {
            let mut i = 0;
            for _ in 0..num_peope / 4 {
                buf_writer.write(&[
                    PlinkBed::geno_to_lowest_two_bits(col[i])
                        | (PlinkBed::geno_to_lowest_two_bits(col[i + 1]) << 2)
                        | (PlinkBed::geno_to_lowest_two_bits(col[i + 2]) << 4)
                        | (PlinkBed::geno_to_lowest_two_bits(col[i + 3]) << 6)
                ])?;
                i += 4;
            }
            let remainder = num_peope % 4;
            if remainder > 0 {
                let mut byte = 0u8;
                for j in 0..remainder {
                    byte |= PlinkBed::geno_to_lowest_two_bits(col[i + j]) << (j * 2);
                }
                buf_writer.write(&[byte])?;
            }
        }
        Ok(())
    }

    pub fn reset_bed_buf(&mut self) -> Result<(), io::Error> {
        // the first three bytes are the file signature
        self.bed_buf.seek(SeekFrom::Start(3))?;
        Ok(())
    }

    /// makes the BufReader point to the start of the byte containing the SNP i individual j
    /// 0-indexing
    fn seek_to_byte_containing_snp_i_person_j(&mut self, snp_i: usize, person_j: usize) -> Result<(), io::Error> {
        // the first three bytes are the file signature
        self.bed_buf.seek(SeekFrom::Start((3 + self.num_bytes_per_snp * snp_i + person_j / 4) as u64))?;
        Ok(())
    }

    fn geno_to_lowest_two_bits(geno: u8) -> u8 {
        // 00 -> 2 homozygous for the first allele in the .bim file (usually the minor allele)
        // 01 -> 0 missing genotype
        // 10 -> 1 heterozygous
        // 11 -> 0 homozygous for the second allele in the .bim file (usually the major allele)
        let not_a = ((geno & 0b10) >> 1) ^ 1;
        let not_b = (geno & 1) ^ 1;
        (not_a << 1) | (not_b & not_a)
    }

    fn lowest_two_bits_to_geno(byte: u8) -> u8 {
        // 00 -> 2 homozygous for the first allele in the .bim file (usually the minor allele)
        // 01 -> 0 missing genotype
        // 10 -> 1 heterozygous
        // 11 -> 0 homozygous for the second allele in the .bim file (usually the major allele)
        let a = (byte & 0b10) >> 1;
        let b = byte & 1;
        (((a | b) ^ 1) << 1) | (a & (!b))
    }

    pub fn get_genotype_matrix(&mut self) -> Result<Array<f32, Ix2>, io::Error> {
        self.reset_bed_buf()?;

        let last_byte_index = self.num_bytes_per_snp - 1;
        let num_people_last_byte = match self.num_people % 4 {
            0 => 4,
            x => x
        };

        let mut v = Vec::with_capacity(self.num_people * self.num_snps);
        unsafe {
            v.set_len(self.num_people * self.num_snps);
        }
        let mut vi = 0usize;

        let mut snp_bytes = vec![0u8; self.num_bytes_per_snp];
        for _ in 0..self.num_snps {
            self.bed_buf.read_exact(&mut snp_bytes)?;
            for i in 0..last_byte_index {
                v[vi] = PlinkBed::lowest_two_bits_to_geno(snp_bytes[i]) as f32;
                v[vi + 1] = PlinkBed::lowest_two_bits_to_geno(snp_bytes[i] >> 2) as f32;
                v[vi + 2] = PlinkBed::lowest_two_bits_to_geno(snp_bytes[i] >> 4) as f32;
                v[vi + 3] = PlinkBed::lowest_two_bits_to_geno(snp_bytes[i] >> 6) as f32;
                vi += 4;
            }
            // last byte
            for k in 0..num_people_last_byte {
                v[vi] = PlinkBed::lowest_two_bits_to_geno(snp_bytes[last_byte_index] >> (k << 1)) as f32;
                vi += 1;
            }
        }
        let geno_arr = Array::from_shape_vec((self.num_people, self.num_snps)
                                                 .strides((1, self.num_people)), v).unwrap();
        Ok(geno_arr)
    }

    pub fn col_chunk_iter(&mut self, snp_stride: usize) -> PlinkColChunkIter {
        let buf = PlinkBed::get_buf(&self.filepath).unwrap();
        PlinkColChunkIter::new(buf,
                               0,
                               self.num_snps,
                               snp_stride,
                               self.num_people,
                               &self.filepath)
    }

    pub fn byte_chunk_iter(&mut self, start_byte_index: usize, end_byte_index_exclusive: usize,
        chunk_size: usize) -> Result<ByteChunkIter<File>, Error> {
        let buf = match OpenOptions::new().read(true).open(&self.filepath) {
            Ok(file) => BufReader::new(file),
            Err(io_error) => return Err(Error::IO { why: format!("failed to open {}: {}", self.filepath, io_error), io_error }),
        };
        Ok(ByteChunkIter::new(buf, start_byte_index, end_byte_index_exclusive, chunk_size))
    }

    /// save the transpose of the BED file into `out_path`, which should have an extension of .bedt
    /// wherein the n-th sequence of bytes corresponds to the SNPs for the n-th person
    /// larger values of `snp_byte_chunk_size` lead to faster performance, at the cost of higher memory requirement
    pub fn create_bed_t(&mut self, out_path: &str, snp_byte_chunk_size: usize) -> Result<(), io::Error> {
        let mut buf_writer = BufWriter::new(OpenOptions::new().create(true).truncate(true).write(true).open(out_path)?);
        let num_bytes_per_person = PlinkBed::usize_div_ceil(self.num_snps, 4);

        let people_stride = snp_byte_chunk_size * 4;
        let mut snp_bytes = vec![0u8; snp_byte_chunk_size];

        // write people_stride people at a time
        for j in (0..self.num_people).step_by(people_stride) {
            let mut people_buf = vec![vec![0u8; num_bytes_per_person]; people_stride];
            if self.num_people - j < people_stride {
                let remaining_people = self.num_people % people_stride;
                snp_bytes = vec![0u8; PlinkBed::usize_div_ceil(remaining_people, 4)];
            }
            let relative_seek_offset = (self.num_bytes_per_snp - snp_bytes.len()) as i64;
            // read 4 SNPs to the buffers at a time
            self.seek_to_byte_containing_snp_i_person_j(0, j)?;
            for (snp_byte_index, k) in (0..self.num_snps).step_by(4).enumerate() {
                for (snp_offset, _) in (k..min(k + 4, self.num_snps)).enumerate() {
                    self.bed_buf.read_exact(&mut snp_bytes)?;
                    for w in 0..snp_bytes.len() {
                        people_buf[w + 0][snp_byte_index] |= (snp_bytes[w] & 0b11) << (snp_offset << 1);
                        people_buf[w + 1][snp_byte_index] |= ((snp_bytes[w] >> 2) & 0b11) << (snp_offset << 1);
                        people_buf[w + 2][snp_byte_index] |= ((snp_bytes[w] >> 4) & 0b11) << (snp_offset << 1);
                        people_buf[w + 3][snp_byte_index] |= ((snp_bytes[w] >> 6) & 0b11) << (snp_offset << 1);
                    }
                    self.bed_buf.seek_relative(relative_seek_offset)?;
                }
            }
            for (p, buf) in people_buf.iter().enumerate() {
                if j + p < self.num_people {
                    buf_writer.write(buf.as_slice())?;
                }
            }
        }
        Ok(())
    }
}

pub struct PlinkColChunkIter {
    start_snp_index: usize,
    end_snp_index: usize,
    current_snp_index: usize,
    num_people: usize,
    num_snps_per_iter: usize,
    bed_filename: String,
    buf: BufReader<File>,
}

impl PlinkColChunkIter {
    pub fn new(buf: BufReader<File>, start_snp_index: usize, end_snp_index: usize, num_snps_per_iter: usize,
        num_people: usize, bed_filename: &str) -> PlinkColChunkIter {
        let mut iter = PlinkColChunkIter {
            start_snp_index,
            end_snp_index,
            current_snp_index: start_snp_index,
            num_people,
            num_snps_per_iter,
            bed_filename: bed_filename.to_string(),
            buf,
        };
        iter.seek_to_snp(start_snp_index);
        iter
    }

    fn num_bytes_per_snp(&self) -> usize {
        PlinkBed::usize_div_ceil(self.num_people, 4)
    }

    fn seek_to_snp(&mut self, snp_index: usize) {
        assert!(snp_index >= self.start_snp_index, "can only seek to snp with index >= self.start_snp_index: {}. Received {}",
                self.start_snp_index, snp_index);
        // skip the first 3 magic bytes
        self.buf.seek(SeekFrom::Start(3 + (self.num_bytes_per_snp() * snp_index) as u64)).unwrap();
    }

    /// indices are 0 based
    /// end_snp_index is exclusive
    fn clone_with_start_end_index(&self, start_snp_index: usize, end_snp_index: usize) -> PlinkColChunkIter {
        PlinkColChunkIter::new(
            PlinkBed::get_buf(&self.bed_filename).unwrap(),
            start_snp_index,
            end_snp_index,
            self.num_snps_per_iter,
            self.num_people,
            &self.bed_filename,
        )
    }

    fn read_chunk(&mut self, chunk_size: usize) -> Array<f32, Ix2> {
        let num_bytes_per_snp = self.num_bytes_per_snp();
        let num_people_last_byte = match self.num_people % 4 {
            0 => 4,
            x => x
        };

        let mut v = Vec::with_capacity(self.num_people * chunk_size);
        unsafe {
            v.set_len(self.num_people * chunk_size);
        }
        let mut current_index = 0usize;

        let mut snp_bytes = vec![0u8; num_bytes_per_snp];
        for _ in 0..chunk_size {
            self.buf.read_exact(&mut snp_bytes).unwrap();
            for i in 0..num_bytes_per_snp - 1 {
                v[current_index] = PlinkBed::lowest_two_bits_to_geno(snp_bytes[i]) as f32;
                v[current_index + 1] = PlinkBed::lowest_two_bits_to_geno(snp_bytes[i] >> 2) as f32;
                v[current_index + 2] = PlinkBed::lowest_two_bits_to_geno(snp_bytes[i] >> 4) as f32;
                v[current_index + 3] = PlinkBed::lowest_two_bits_to_geno(snp_bytes[i] >> 6) as f32;
                current_index += 4;
            }
            // last byte
            for k in 0..num_people_last_byte {
                v[current_index] = PlinkBed::lowest_two_bits_to_geno(snp_bytes[num_bytes_per_snp - 1] >> (k << 1)) as f32;
                current_index += 1;
            }
        }
        self.current_snp_index += chunk_size;
        Array::from_shape_vec((self.num_people, chunk_size)
                                  .strides((1, self.num_people)), v).unwrap()
    }
}

impl IntoParallelIterator for PlinkColChunkIter {
    type Iter = PlinkColChunkParallelIter;
    type Item = <PlinkColChunkParallelIter as ParallelIterator>::Item;

    fn into_par_iter(self) -> Self::Iter {
        PlinkColChunkParallelIter { iter: self }
    }
}

impl Iterator for PlinkColChunkIter {
    type Item = Array<f32, Ix2>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_snp_index >= self.end_snp_index {
            return None;
        }
        let chunk_size = min(self.num_snps_per_iter, self.end_snp_index - self.current_snp_index);
        Some(self.read_chunk(chunk_size))
    }
}

impl ExactSizeIterator for PlinkColChunkIter {
    fn len(&self) -> usize {
        PlinkBed::usize_div_ceil(self.end_snp_index - self.current_snp_index, self.num_snps_per_iter)
    }
}

impl DoubleEndedIterator for PlinkColChunkIter {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.current_snp_index >= self.end_snp_index {
            return None;
        }
        let chunk_size = min(self.num_snps_per_iter, self.end_snp_index - self.current_snp_index);
        self.end_snp_index -= chunk_size;
        self.seek_to_snp(self.end_snp_index);
        let chunk = self.read_chunk(chunk_size);
        self.seek_to_snp(self.current_snp_index);
        Some(chunk)
    }
}

struct ColChunkIterProducer {
    iter: PlinkColChunkIter,
}

impl Producer for ColChunkIterProducer {
    type Item = <PlinkColChunkIter as Iterator>::Item;
    type IntoIter = PlinkColChunkIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let mid_snp_index = min(self.iter.start_snp_index + self.iter.num_snps_per_iter * index, self.iter.end_snp_index);
        (ColChunkIterProducer { iter: self.iter.clone_with_start_end_index(self.iter.start_snp_index, mid_snp_index) },
         ColChunkIterProducer { iter: self.iter.clone_with_start_end_index(mid_snp_index, self.iter.end_snp_index) })
    }
}

impl IntoIterator for ColChunkIterProducer {
    type Item = <PlinkColChunkIter as Iterator>::Item;
    type IntoIter = PlinkColChunkIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter
    }
}

pub struct PlinkColChunkParallelIter {
    iter: PlinkColChunkIter
}

impl ParallelIterator for PlinkColChunkParallelIter {
    type Item = <PlinkColChunkIter as Iterator>::Item;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where C: UnindexedConsumer<Self::Item>
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.iter.len())
    }
}

impl IndexedParallelIterator for PlinkColChunkParallelIter {
    fn len(&self) -> usize {
        self.iter.len()
    }

    fn drive<C>(self, consumer: C) -> C::Result
        where C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
        where CB: ProducerCallback<Self::Item>,
    {
        callback.callback(ColChunkIterProducer { iter: self.iter })
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::min;
    use std::io;
    use std::io::Write;

    use ndarray::{array, Array, s};
    use ndarray_rand::RandomExt;
    use rand::distributions::Uniform;
    use tempfile::NamedTempFile;

    use super::PlinkBed;

    fn create_dummy_bim_fam(bim: &mut NamedTempFile, fam: &mut NamedTempFile, num_people: usize, num_snps: usize) -> Result<(), io::Error> {
        for i in 1..=num_people {
            fam.write_fmt(format_args!("{}\n", i))?;
        }
        for i in 1..=num_snps {
            bim.write_fmt(format_args!("{}\n", i))?;
        }
        Ok(())
    }

    #[test]
    fn test_create_bed() {
        let geno = array![
            [0,0,1,2],
            [1,1,2,1],
            [2,0,0,0],
            [1,0,0,2],
            [0,2,1,0]
        ];
        let mut bim = NamedTempFile::new().unwrap();
        let mut fam = NamedTempFile::new().unwrap();
        create_dummy_bim_fam(&mut bim, &mut fam, geno.dim().0, geno.dim().1).unwrap();
        let path = NamedTempFile::new().unwrap().into_temp_path().to_str().unwrap().to_string();
        PlinkBed::create_bed(&geno, &path).unwrap();
        let mut geno_bed = PlinkBed::new(&path,
                                         bim.into_temp_path().to_str().unwrap(),
                                         fam.into_temp_path().to_str().unwrap()).unwrap();
        assert_eq!(geno.mapv(|x| x as f32), geno_bed.get_genotype_matrix().unwrap());
    }

    #[test]
    fn test_chunk_iter() {
        let (num_people, num_snps) = (137usize, 71usize);
        let geno = Array::random((num_people, num_snps), Uniform::from(0..3));

        let mut bim = NamedTempFile::new().unwrap();
        let mut fam = NamedTempFile::new().unwrap();
        create_dummy_bim_fam(&mut bim, &mut fam, num_people, num_snps).unwrap();
        let bed_file = NamedTempFile::new().unwrap();
        let bed_path = bed_file.into_temp_path().to_str().unwrap().to_string();
        PlinkBed::create_bed(&geno, &bed_path).unwrap();

        let mut bed = PlinkBed::new(
            &bed_path,
            bim.into_temp_path().to_str().unwrap(),
            fam.into_temp_path().to_str().unwrap(),
        ).unwrap();
        let true_geno_arr = bed.get_genotype_matrix().unwrap();
        let chunk_size = 5;
        for (i, snps) in bed.col_chunk_iter(chunk_size).enumerate() {
            let end_index = min((i + 1) * chunk_size, true_geno_arr.dim().1);
            assert!(true_geno_arr.slice(s![..,i * chunk_size..end_index]) == snps);
        }
    }
}

