//! Reading GPU buffers back: map futures, queue completion, blocking helpers.

use crate::{Gpu, Tensor};
use anyhow::Result;

impl Gpu {
    /// A tensor (or view) back as f32 `[c][h][w]`.
    pub fn readback_t(&self, t: &Tensor) -> Vec<f32> {
        let plane = (t.h * t.w) as usize;
        let all = self.readback(&t.buf, (t.c0 as usize + t.c as usize) * plane);
        all[t.c0 as usize * plane..].iter().map(|v| v * t.scale).collect()
    }
}

impl Gpu {
    pub fn wait(&self) {
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
    }
    pub fn readback(&self, buf: &wgpu::Buffer, n: usize) -> Vec<f32> {
        let stage = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * self.esize()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(buf, 0, &stage, 0, (n * self.esize()) as u64);
        self.queue.submit(Some(enc.finish()));
        let slice = stage.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.wait();
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range().unwrap();
        let v: Vec<f32> = if self.f32_mode {
            bytemuck::cast_slice::<u8, f32>(&data).to_vec()
        } else {
            bytemuck::cast_slice::<u8, u16>(&data).iter().map(|&b| half::f16::from_bits(b).to_f32()).collect()
        };
        drop(data);
        stage.unmap();
        v
    }
}

// the map_async result and the waker of whoever awaits it
type MapState =
    std::sync::Arc<std::sync::Mutex<(Option<Result<(), wgpu::BufferAsyncError>>, Option<std::task::Waker>)>>;
// one-shot future resolved by the map_async callback (no executor dependency; works on native and wasm)
pub struct MapWait {
    st: MapState,
}
// request a read mapping of the whole buffer; await the returned future (native: after device.poll)
pub fn map_read(buf: &wgpu::Buffer) -> MapWait {
    let st: MapState = std::sync::Arc::new(std::sync::Mutex::new((None, None)));
    let st2 = st.clone();
    buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let mut g = st2.lock().unwrap();
        g.0 = Some(r);
        if let Some(w) = g.1.take() {
            w.wake();
        }
    });
    MapWait { st }
}
impl std::future::Future for MapWait {
    type Output = Result<(), wgpu::BufferAsyncError>;
    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Self::Output> {
        let mut g = self.st.lock().unwrap();
        if let Some(r) = g.0.take() {
            std::task::Poll::Ready(r)
        } else {
            g.1 = Some(cx.waker().clone());
            std::task::Poll::Pending
        }
    }
}

/// A small GPU result on its way to the CPU: the staging buffer's mapping is awaited, the bytes decoded into `T` and
/// the buffer handed back to whoever lent it. Native hosts `block` on it, the browser awaits `finish` (the mapAsync
/// promise resolves it).
pub struct Readback<T> {
    stage: wgpu::Buffer,
    fut: MapWait,
    decode: Decode<T>,
    done: Option<Done>,
}
/// Turns the mapped bytes into the result.
pub type Decode<T> = Box<dyn FnOnce(&[u8]) -> T>;
/// Runs once the staging buffer is unmapped again (hands it back to its pool).
pub type Done = Box<dyn FnOnce()>;
impl<T> Readback<T> {
    /// `stage` must be `MAP_READ` and already the destination of a submitted copy; `done` runs after the decode.
    pub fn new(stage: wgpu::Buffer, decode: Decode<T>, done: Option<Done>) -> Self {
        let fut = map_read(&stage);
        Readback { stage, fut, decode, done }
    }
    pub async fn finish(self) -> Result<T> {
        self.fut.await.map_err(|e| anyhow::anyhow!("map_async: {e:?}"))?;
        let v = {
            let d = self.stage.slice(..).get_mapped_range().map_err(|e| anyhow::anyhow!("mapped range: {e:?}"))?;
            (self.decode)(&d)
        };
        self.stage.unmap();
        if let Some(done) = self.done {
            done();
        }
        Ok(v)
    }
    /// The result if the mapping has completed, else the readback itself to try again later: a host that cannot await
    /// (a render loop) checks once per frame. Native needs a device poll first for the mapping to complete.
    pub fn try_finish(self) -> std::result::Result<Result<T>, Self> {
        let r = self.fut.st.lock().unwrap().0.take();
        let Some(r) = r else { return Err(self) };
        let v = match r {
            Err(e) => Err(anyhow::anyhow!("map_async: {e:?}")),
            Ok(()) => {
                let v = match self.stage.slice(..).get_mapped_range() {
                    Ok(d) => Ok((self.decode)(&d)),
                    Err(e) => Err(anyhow::anyhow!("mapped range: {e:?}")),
                };
                self.stage.unmap();
                v
            }
        };
        if let Some(done) = self.done {
            done();
        }
        Ok(v)
    }
    /// Native: poll the device and wait for the result.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn block(self, gpu: &Gpu) -> Result<T> {
        gpu.wait();
        pollster::block_on(self.finish())
    }
}

// one-shot future resolved when everything submitted so far has completed on the GPU (queue.onSubmittedWorkDone)
pub struct QueueWait {
    st: std::sync::Arc<std::sync::Mutex<(bool, Option<std::task::Waker>)>>,
}
pub fn queue_done(queue: &wgpu::Queue) -> QueueWait {
    let st: std::sync::Arc<std::sync::Mutex<(bool, Option<std::task::Waker>)>> =
        std::sync::Arc::new(std::sync::Mutex::new((false, None)));
    let st2 = st.clone();
    queue.on_submitted_work_done(move || {
        let mut g = st2.lock().unwrap();
        g.0 = true;
        if let Some(w) = g.1.take() {
            w.wake();
        }
    });
    QueueWait { st }
}
impl std::future::Future for QueueWait {
    type Output = ();
    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<()> {
        let mut g = self.st.lock().unwrap();
        if g.0 {
            std::task::Poll::Ready(())
        } else {
            g.1 = Some(cx.waker().clone());
            std::task::Poll::Pending
        }
    }
}
