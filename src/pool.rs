use std::collections::HashMap;
use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender, TryRecvError};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const LOGIN_REQUEST_ID: u64 = 1;
const RECONNECT_DELAY: Duration = Duration::from_secs(2);
const SOCKET_IO_TIMEOUT: Duration = Duration::from_millis(250);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);
const CHANNEL_CAPACITY: usize = 4096;

#[derive(Debug, Clone, Deserialize)]
pub struct PoolJob {
    pub job_id: String,
    pub header_base: String,
    pub target: String,
    pub height: u64,
    pub nonce_start: u64,
    pub nonce_end: u64,
}

impl PoolJob {
    pub fn nonce_count(&self) -> u64 {
        if self.nonce_end < self.nonce_start {
            return 0;
        }
        self.nonce_end
            .saturating_sub(self.nonce_start)
            .saturating_add(1)
    }
}

#[derive(Debug, Clone)]
pub struct PoolSubmitAck {
    pub job_id: String,
    pub nonce: u64,
    pub accepted: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub enum PoolEvent {
    Connected,
    Disconnected(String),
    LoginAccepted,
    LoginRejected(String),
    Job(PoolJob),
    SubmitAck(PoolSubmitAck),
}

#[derive(Debug, Clone)]
struct PoolSubmit {
    job_id: String,
    nonce: u64,
}

#[derive(Debug)]
pub struct PoolClient {
    submit_tx: Sender<PoolSubmit>,
    event_rx: Receiver<PoolEvent>,
}

impl PoolClient {
    pub fn connect(
        pool_url: &str,
        address: String,
        worker: String,
        shutdown: Arc<AtomicBool>,
    ) -> Result<Self> {
        let endpoint = parse_pool_endpoint(pool_url)?;
        let (submit_tx, submit_rx) = bounded::<PoolSubmit>(CHANNEL_CAPACITY);
        let (event_tx, event_rx) = bounded::<PoolEvent>(CHANNEL_CAPACITY);

        std::thread::Builder::new()
            .name("pool-stratum".to_string())
            .spawn(move || {
                run_pool_client_thread(endpoint, address, worker, submit_rx, event_tx, shutdown)
            })
            .context("failed to spawn pool client thread")?;

        Ok(Self {
            submit_tx,
            event_rx,
        })
    }

    pub fn submit_share(&self, job_id: String, nonce: u64) -> Result<()> {
        self.submit_tx
            .send(PoolSubmit { job_id, nonce })
            .map_err(|_| anyhow!("pool submit channel closed"))
    }

    pub fn recv_event_timeout(&self, timeout: Duration) -> Option<PoolEvent> {
        match self.event_rx.recv_timeout(timeout) {
            Ok(event) => Some(event),
            Err(RecvTimeoutError::Timeout) | Err(RecvTimeoutError::Disconnected) => None,
        }
    }

    pub fn drain_events(&self) -> Vec<PoolEvent> {
        let mut out = Vec::new();
        loop {
            match self.event_rx.try_recv() {
                Ok(event) => out.push(event),
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
            }
        }
        out
    }
}

#[derive(Debug, Serialize)]
struct StratumRequest<'a, T: Serialize> {
    id: u64,
    method: &'a str,
    params: T,
}

#[derive(Debug, Serialize)]
struct LoginParams<'a> {
    address: &'a str,
    worker: &'a str,
}

#[derive(Debug, Serialize)]
struct SubmitParams<'a> {
    job_id: &'a str,
    nonce: u64,
}

#[derive(Debug, Deserialize)]
struct StratumMessage {
    #[serde(default)]
    id: Option<u64>,
    #[serde(default)]
    method: Option<String>,
    #[serde(default)]
    params: Option<Value>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    result: Option<Value>,
}

fn run_pool_client_thread(
    endpoint: String,
    address: String,
    worker: String,
    submit_rx: Receiver<PoolSubmit>,
    event_tx: Sender<PoolEvent>,
    shutdown: Arc<AtomicBool>,
) {
    let mut next_request_id = LOGIN_REQUEST_ID + 1;
    while !shutdown.load(Ordering::Relaxed) {
        match connect_pool_stream(&endpoint) {
            Ok((mut stream, mut reader)) => {
                let _ = event_tx.try_send(PoolEvent::Connected);
                if let Err(err) = send_login(&mut stream, &address, &worker) {
                    let _ = event_tx.try_send(PoolEvent::Disconnected(format!(
                        "failed to send login request: {err:#}"
                    )));
                    sleep_with_shutdown(&shutdown, RECONNECT_DELAY);
                    continue;
                }
                let mut pending_submits = HashMap::<u64, PoolSubmit>::new();
                let mut login_confirmed = false;

                loop {
                    if shutdown.load(Ordering::Relaxed) {
                        return;
                    }

                    let mut reconnect_requested = false;
                    while let Ok(submit) = submit_rx.try_recv() {
                        let request_id = next_request_id;
                        next_request_id = next_request_id.wrapping_add(1).max(LOGIN_REQUEST_ID + 1);
                        if send_submit(&mut stream, request_id, &submit).is_err() {
                            let _ = event_tx.try_send(PoolEvent::Disconnected(
                                "failed to send submit request".to_string(),
                            ));
                            pending_submits.clear();
                            reconnect_requested = true;
                            break;
                        }
                        pending_submits.insert(request_id, submit);
                    }
                    if reconnect_requested {
                        sleep_with_shutdown(&shutdown, RECONNECT_DELAY);
                        break;
                    }

                    let mut line = String::new();
                    match reader.read_line(&mut line) {
                        Ok(0) => {
                            let _ = event_tx.try_send(PoolEvent::Disconnected(
                                "pool connection closed".to_string(),
                            ));
                            break;
                        }
                        Ok(_) => {
                            let trimmed = line.trim();
                            if trimmed.is_empty() {
                                continue;
                            }
                            if let Some(event) = decode_pool_message(
                                trimmed,
                                &mut login_confirmed,
                                &mut pending_submits,
                            ) {
                                let _ = event_tx.try_send(event);
                            }
                        }
                        Err(err)
                            if matches!(
                                err.kind(),
                                ErrorKind::WouldBlock
                                    | ErrorKind::TimedOut
                                    | ErrorKind::Interrupted
                            ) => {}
                        Err(err) => {
                            let _ = event_tx.try_send(PoolEvent::Disconnected(format!(
                                "pool read error: {err}"
                            )));
                            break;
                        }
                    }
                }
            }
            Err(err) => {
                let _ = event_tx.try_send(PoolEvent::Disconnected(format!(
                    "pool connect error: {err:#}"
                )));
                sleep_with_shutdown(&shutdown, RECONNECT_DELAY);
            }
        }
    }
}

fn connect_pool_stream(endpoint: &str) -> Result<(TcpStream, BufReader<TcpStream>)> {
    let mut addrs: Vec<SocketAddr> = endpoint
        .to_socket_addrs()
        .with_context(|| format!("failed to resolve pool endpoint {endpoint}"))?
        .collect();
    if addrs.is_empty() {
        bail!("pool endpoint resolved to no addresses: {endpoint}");
    }
    addrs.sort_by_key(|addr| !addr.is_ipv4());

    let mut last_err: Option<std::io::Error> = None;
    let mut stream = None;
    for addr in addrs {
        match TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT) {
            Ok(connected) => {
                stream = Some(connected);
                break;
            }
            Err(err) => {
                last_err = Some(err);
            }
        }
    }
    let stream = stream.ok_or_else(|| {
        anyhow!(
            "failed to connect to pool at {endpoint}: {}",
            last_err
                .map(|err| err.to_string())
                .unwrap_or_else(|| "no addresses available".to_string())
        )
    })?;

    stream
        .set_read_timeout(Some(SOCKET_IO_TIMEOUT))
        .context("failed to set pool read timeout")?;
    stream
        .set_write_timeout(Some(SOCKET_IO_TIMEOUT))
        .context("failed to set pool write timeout")?;
    let reader = BufReader::new(
        stream
            .try_clone()
            .context("failed to clone pool stream for reader")?,
    );
    Ok((stream, reader))
}

fn send_login(stream: &mut TcpStream, address: &str, worker: &str) -> Result<()> {
    let request = StratumRequest {
        id: LOGIN_REQUEST_ID,
        method: "login",
        params: LoginParams { address, worker },
    };
    send_json_line(stream, &request)
}

fn send_submit(stream: &mut TcpStream, request_id: u64, submit: &PoolSubmit) -> Result<()> {
    let request = StratumRequest {
        id: request_id,
        method: "submit",
        params: SubmitParams {
            job_id: &submit.job_id,
            nonce: submit.nonce,
        },
    };
    send_json_line(stream, &request)
}

fn send_json_line<T: Serialize>(stream: &mut TcpStream, payload: &T) -> Result<()> {
    let mut line = serde_json::to_vec(payload).context("failed to serialize stratum payload")?;
    line.push(b'\n');
    stream
        .write_all(&line)
        .context("failed writing stratum payload")
}

fn decode_pool_message(
    raw: &str,
    login_confirmed: &mut bool,
    pending_submits: &mut HashMap<u64, PoolSubmit>,
) -> Option<PoolEvent> {
    let msg: StratumMessage = serde_json::from_str(raw).ok()?;

    if let Some(method) = msg.method.as_deref() {
        if method == "job" {
            let params = msg.params?;
            let job: PoolJob = serde_json::from_value(params).ok()?;
            return Some(PoolEvent::Job(job));
        }
    }

    let id = msg.id?;
    if id == LOGIN_REQUEST_ID {
        if let Some(err) = msg.error {
            *login_confirmed = false;
            return Some(PoolEvent::LoginRejected(err));
        }
        let status_ok = msg
            .status
            .as_deref()
            .map(|status| status.eq_ignore_ascii_case("ok"))
            .unwrap_or(false);
        if status_ok {
            *login_confirmed = true;
            return Some(PoolEvent::LoginAccepted);
        }
        *login_confirmed = false;
        return Some(PoolEvent::LoginRejected("pool rejected login".to_string()));
    }

    let submit = pending_submits.remove(&id)?;
    let accepted = if msg.error.is_some() {
        false
    } else if let Some(result) = msg.result.as_ref() {
        result
            .get("accepted")
            .and_then(Value::as_bool)
            .unwrap_or(true)
    } else {
        msg.status
            .as_deref()
            .map(|status| status.eq_ignore_ascii_case("ok"))
            .unwrap_or(true)
    };
    let error = if accepted {
        None
    } else {
        Some(
            msg.error
                .or_else(|| msg.status)
                .unwrap_or_else(|| "pool rejected share".to_string()),
        )
    };

    Some(PoolEvent::SubmitAck(PoolSubmitAck {
        job_id: submit.job_id,
        nonce: submit.nonce,
        accepted,
        error,
    }))
}

fn parse_pool_endpoint(pool_url: &str) -> Result<String> {
    let trimmed = pool_url.trim();
    if trimmed.is_empty() {
        bail!("pool URL is empty");
    }
    let authority = trimmed
        .strip_prefix("stratum+tcp://")
        .unwrap_or(trimmed)
        .split('/')
        .next()
        .unwrap_or(trimmed)
        .trim();
    if authority.is_empty() {
        bail!("pool URL authority is empty");
    }
    if authority.contains("://") {
        bail!("unsupported pool URL scheme");
    }
    Ok(authority.to_string())
}

fn sleep_with_shutdown(shutdown: &AtomicBool, duration: Duration) {
    let deadline = std::time::Instant::now() + duration;
    while !shutdown.load(Ordering::Relaxed) {
        if std::time::Instant::now() >= deadline {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}
