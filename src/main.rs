use anyhow::{Context, Result, anyhow};
use arrow::{
    array::{Array, BooleanArray, Float64Array, Int32Array, Int64Array, StringArray},
    datatypes::DataType as ArrowDataType,
    record_batch::RecordBatch,
};
use datafusion::execution::context::SessionContext;
use deltalake::{
    kernel::{DataType as KernelDataType, PrimitiveType},
    open_table,
};
use futures::{StreamExt, pin_mut};
use native_tls::TlsConnector;
use postgres_native_tls::MakeTlsConnector;
use std::{collections::BTreeMap, env, path::PathBuf, pin::Pin, sync::Arc};
use tokio::sync::Semaphore;
use tokio_postgres::{
    binary_copy::BinaryCopyInWriter,
    types::{ToSql, Type as PgType},
};
use walkdir::WalkDir;

const CONCURRENCY_LIMIT: usize = 12;

#[derive(Clone)]
struct Config {
    pg_url: String,
    target_table: String,
    tls: MakeTlsConnector,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    let root = env::var("DELTA_ROOT")?;

    let cfg = Config {
        pg_url: env::var("PG_URL")?,
        target_table: env::var("TARGET_TABLE")?,
        tls: MakeTlsConnector::new(TlsConnector::new()?),
    };

    let mut map: BTreeMap<(i32, u32), Vec<PathBuf>> = BTreeMap::new();

    for entry in WalkDir::new(&root)
        .min_depth(3)
        .max_depth(3)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|d| d.file_type().is_dir())
    {
        let rel = entry.path().strip_prefix(&root)?;
        let mut comps = rel.components();
        let year: i32 = comps
            .next()
            .unwrap()
            .as_os_str()
            .to_str()
            .unwrap()
            .parse()?;

        let month: u32 = comps
            .next()
            .unwrap()
            .as_os_str()
            .to_str()
            .unwrap()
            .parse()?;

        map.entry((year, month))
            .or_default()
            .push(entry.path().to_path_buf());
    }

    let mut jobs: Vec<((i32, u32), Vec<PathBuf>)> = map
        .into_iter()
        .map(|(yearMonth, mut ymPaths)| {
            ymPaths.sort();
            (yearMonth, ymPaths)
        })
        .collect();

    jobs.sort_by_key(|(yearMonth, _)| *yearMonth);
    // starting with latest dates and moving backwards
    jobs.reverse();

    let sem = Arc::new(Semaphore::new(CONCURRENCY_LIMIT));

    futures::stream::iter(jobs)
        .for_each_concurrent(CONCURRENCY_LIMIT, |((y, m), day_dirs)| {
            let cfg = cfg.clone();
            let sem = sem.clone();

            async move {
                let _permit = sem.acquire_owned().await.unwrap();

                if let Err(e) = import_month(y, m, day_dirs, &cfg).await {
                    eprintln!("error importing {}-{:02}: {e}", y, m);
                } else {
                    println!("Finished {}-{:02}", y, m);
                }
            }
        })
        .await;

    Ok(())
}

async fn import_month(
	year: i32, month: u32, day_dirs: Vec<PathBuf>, cfg: &Config
) -> Result<()> {
    let (client, connection) = tokio_postgres::connect(
	    &cfg.pg_url,
	    cfg.tls.clone(),
    ).await?;

    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("pgsql connection error: {e}");
        }
    });

    println!("importing {}-{:02}", year, month);

    let first_dt = open_table(day_dirs[0].to_str().unwrap())
        .await
        .context("opening first delta table")?;

    let struct_type = first_dt
        .schema()
        .ok_or_else(|| anyhow!("Delta table has no StructType schema"))?;

    let pg_types: Vec<PgType> = struct_type
        .fields()
        .map(|f| delta_to_pg(f.data_type()))
        .collect();

    let stmt = format!("COPY public.{} FROM STDIN BINARY", cfg.target_table);
    let sink = client.copy_in(&stmt).await?;
    let mut writer = BinaryCopyInWriter::new(sink, &pg_types);
    pin_mut!(writer);

    let ctx = SessionContext::new();

    for dpath in day_dirs {
        let dt = open_table(dpath.to_str().unwrap())
            .await
            .with_context(|| format!("opening delta {}", dpath.display()))?;
        ctx.register_table("t", Arc::new(dt))?;

        for batch in ctx
            .sql("SELECT * FROM t")
            .await?
            .collect()
            .await
            .context("executing DataFusion query")?
        {
            write_batch_binary(&batch, writer.as_mut()).await?;
        }
        ctx.deregister_table("t")?;
    }

    writer.as_mut().finish().await?;
    Ok(())
}

async fn write_batch_binary(
    batch: &RecordBatch,
    mut w: Pin<&mut BinaryCopyInWriter>,
) -> Result<()> {
    for row in 0..batch.num_rows() {
        let mut vals: Vec<Box<dyn ToSql + Sync>> = Vec::with_capacity(
	        batch.num_columns()
        );

        for col in 0..batch.num_columns() {
            vals.push(arrow_cell(batch.column(col), row));
        }

        let refs: Vec<&(dyn ToSql + Sync)> = vals.iter().map(|b| &**b).collect();

        w.as_mut().write(&refs).await?;
    }
    Ok(())
}

// maps a give arrow type to pg type
fn delta_to_pg(dt: &KernelDataType) -> PgType {
    use PrimitiveType::*;

    match dt {
        KernelDataType::Primitive(p) => match p {
            Byte | Short => PgType::INT2,
            Integer => PgType::INT4,
            Long => PgType::INT8,
            Float => PgType::FLOAT4,
            Double => PgType::FLOAT8,
            Boolean => PgType::BOOL,
            String => PgType::TEXT,
            Binary => PgType::BYTEA,
            Date => PgType::DATE,
            Timestamp => PgType::TIMESTAMPTZ,
            TimestampNtz => PgType::TIMESTAMP,
            Decimal(_) => PgType::NUMERIC,
            _ => unimplemented!("unsupported primitive {:?}", p),
        },
        _ => unimplemented!("non-primitive type {:?}", dt),
    }
}

fn arrow_cell(arr: &dyn Array, row: usize) -> Box<dyn ToSql + Sync> {
    match arr.data_type() {
        ArrowDataType::Int32 => {
            let a = arr.as_any().downcast_ref::<Int32Array>().unwrap();

            if a.is_null(row) {
                Box::new(None::<i32>)
            } else {
                Box::new(Some(a.value(row)))
            }
        }

        ArrowDataType::Int64 => {
            let a = arr.as_any().downcast_ref::<Int64Array>().unwrap();

            if a.is_null(row) {
                Box::new(None::<i64>)
            } else {
                Box::new(Some(a.value(row)))
            }
        }

        ArrowDataType::Float64 => {
            let a = arr.as_any().downcast_ref::<Float64Array>().unwrap();

            if a.is_null(row) {
                Box::new(None::<f64>)
            } else {
                Box::new(Some(a.value(row)))
            }
        }

        ArrowDataType::Boolean => {
            let a = arr.as_any().downcast_ref::<BooleanArray>().unwrap();

            if a.is_null(row) {
                Box::new(None::<bool>)
            } else {
                Box::new(Some(a.value(row)))
            }
        }

        ArrowDataType::Utf8 => {
            let a = arr.as_any().downcast_ref::<StringArray>().unwrap();

            if a.is_null(row) {
                Box::new(None::<String>)
            } else {
                Box::new(Some(a.value(row).to_owned()))
            }
        }
        _ => unimplemented!("Arrow cell {:?}", arr.data_type()),
    }
}
