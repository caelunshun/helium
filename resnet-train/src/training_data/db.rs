use anyhow::{bail, Context};
use redb::{
    backends::FileBackend, AccessGuard, Durability, ReadableTable, StorageBackend, TableDefinition,
    WriteTransaction,
};
use serde::{Deserialize, Serialize};
use std::{fs, fs::File, io::Error, path::Path};
use uuid::Uuid;

pub struct Database {
    db: redb::Database,
}

static METADATA_TABLE: TableDefinition<u128, &str> = TableDefinition::new("metadata");
static IMAGE_DATA_TABLE: TableDefinition<u128, &[u8]> = TableDefinition::new("data");

impl Database {
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        Ok(Self {
            db: redb::Database::open(path.as_ref())?,
        })
    }

    pub fn open_read_only(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        Ok(Self {
            db: redb::Builder::new()
                .create_with_backend(ReadOnlyFileBackend(FileBackend::new(File::open(path)?)?))?,
        })
    }

    pub fn create(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        if path.as_ref().exists() {
            bail!("refusing to overwrite existing database");
        }
        if let Some(parent) = path.as_ref().parent() {
            fs::create_dir_all(parent).ok();
        }
        Ok(Self {
            db: redb::Database::create(path.as_ref())?,
        })
    }

    pub fn begin_transaction(&self) -> anyhow::Result<Transaction> {
        Ok(Transaction {
            tx: self.db.begin_write()?,
        })
    }

    pub fn list_image_ids(&self, split: Split) -> anyhow::Result<Vec<Uuid>> {
        let tx = self.db.begin_read()?;
        let table = tx.open_table(METADATA_TABLE)?;

        let mut ids = Vec::new();
        for entry in table.iter()? {
            let (id, metadata) = entry?;
            let metadata: ImageMetadata = serde_json::from_str(metadata.value())?;
            if metadata.split == split {
                ids.push(Uuid::from_u128_le(id.value()));
            }
        }
        Ok(ids)
    }

    pub fn get_metadata(&self, id: Uuid) -> anyhow::Result<ImageMetadata> {
        let tx = self.db.begin_read()?;
        let table = tx.open_table(METADATA_TABLE)?;

        let json = table.get(id.to_u128_le())?.context("missing key")?;
        serde_json::from_str(json.value()).map_err(anyhow::Error::from)
    }

    pub fn get_encoded_image_data(&self, id: Uuid) -> anyhow::Result<AccessGuard<&'static [u8]>> {
        let tx = self.db.begin_read()?;
        let table = tx.open_table(IMAGE_DATA_TABLE)?;

        table.get(id.to_u128_le())?.context("missing key")
    }

    pub fn compact(&mut self) -> anyhow::Result<()> {
        self.db.compact()?;
        self.db.check_integrity()?;
        let mut tx = self.db.begin_write()?;
        tx.set_durability(Durability::Immediate);
        tx.commit()?;
        Ok(())
    }
}

pub struct Transaction {
    tx: WriteTransaction,
}

impl Transaction {
    pub fn insert_image(
        &mut self,
        metadata: ImageMetadata,
        encoded_bytes: &[u8],
    ) -> anyhow::Result<Uuid> {
        let id = Uuid::new_v4();

        {
            let mut table = self.tx.open_table(METADATA_TABLE)?;
            table.insert(id.to_u128_le(), serde_json::to_string(&metadata)?.as_str())?;
        }
        {
            let mut table = self.tx.open_table(IMAGE_DATA_TABLE)?;
            table.insert(id.to_u128_le(), encoded_bytes)?;
        }

        Ok(id)
    }

    pub fn commit(self) -> anyhow::Result<()> {
        self.tx.commit()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageMetadata {
    pub split: Split,
    pub label: i32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Split {
    Training,
    Validation,
    Test,
}

#[derive(Debug)]
struct ReadOnlyFileBackend(FileBackend);

impl StorageBackend for ReadOnlyFileBackend {
    fn len(&self) -> Result<u64, Error> {
        self.0.len()
    }

    fn read(&self, offset: u64, len: usize) -> Result<Vec<u8>, Error> {
        self.0.read(offset, len)
    }

    fn set_len(&self, _len: u64) -> Result<(), Error> {
        Ok(())
    }

    fn sync_data(&self, _eventual: bool) -> Result<(), Error> {
        Ok(())
    }

    fn write(&self, _offset: u64, _data: &[u8]) -> Result<(), Error> {
        Ok(())
    }
}
