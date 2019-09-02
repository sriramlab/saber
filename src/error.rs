use std::{fmt, io};
use biofile::error::Error as PlinkBedError;

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

impl From<PlinkBedError> for Error {
    fn from(err: PlinkBedError) -> Error {
        match err {
            PlinkBedError::BadFormat(why) => Error::Generic(why),
            PlinkBedError::Generic(why) => Error::Generic(why),
            PlinkBedError::IO { why, io_error } => Error::IO { why, io_error },
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
