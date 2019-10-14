use std::{fmt, io};
use biofile::error::Error as BiofileError;

#[derive(Debug)]
pub enum Error {
    IO { why: String, io_error: io::Error },
    Generic(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::IO { why, .. } => write!(f, "IO error: {}", why),
            Error::Generic(why) => write!(f, "Generic Error: {}", why)
        }
    }
}

impl From<BiofileError> for Error {
    fn from(err: BiofileError) -> Error {
        match err {
            BiofileError::BadFormat(why) => Error::Generic(why),
            BiofileError::Generic(why) => Error::Generic(why),
            BiofileError::IO { why, io_error } => Error::IO { why, io_error },
        }
    }
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Error {
        Error::IO { why: "IO Error: ".to_string(), io_error: err }
    }
}

impl From<String> for Error {
    fn from(err: String) -> Error {
        Error::Generic(err)
    }
}

impl From<bincode::Error> for Error {
    fn from(err: bincode::Error) -> Error {
        Error::Generic(format!("bincode::error: {}", *err))
    }
}
