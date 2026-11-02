//! The HLSL compiler behind wgpu's DX12 backend. wgpu compiles the kernels for D3D12 through FXC unless it finds
//! `dxcompiler.dll` (with `dxil.dll` beside it): FXC takes 139 s for the kernel set where DXC takes 4 s, and it has no
//! shader-f16, so the f32 kernels run (docs/platforms.md). wgpu looks for the DLL on the process's search path, which
//! is the executable's directory and PATH, not the directory of `flyhigh.dll`; this module looks there as well.
use serde::Serialize;
#[cfg(windows)]
use std::path::{Path, PathBuf};

/// Which compiler the DX12 backend loaded (`DeviceInfo::shader_compiler`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ShaderCompiler {
    Dxc,
    Fxc,
}

/// The compiler for a new instance: `dxc_path` (a directory, or the DLL itself) when given; else `dxcompiler.dll`
/// beside this module when it is there; else wgpu's own search (`Auto`: the DLL on the search path, then FXC).
#[cfg(windows)]
pub fn choose(dxc_path: Option<&Path>) -> wgpu::Dx12Compiler {
    let found = dxc_path
        .map(Path::to_path_buf)
        .or_else(|| module_dir().map(|d| d.join("dxcompiler.dll")).filter(|p| p.is_file()));
    match found {
        Some(p) => {
            let dll = if p.is_dir() { p.join("dxcompiler.dll") } else { p };
            wgpu::Dx12Compiler::DynamicDxc { dxc_path: dll.to_string_lossy().into_owned() }
        }
        None => wgpu::Dx12Compiler::Auto,
    }
}

/// What the DX12 backend ended up with: wgpu keeps `dxcompiler.dll` loaded for the instance's lifetime, so its presence
/// in the process means DXC.
#[cfg(windows)]
pub fn loaded() -> ShaderCompiler {
    use windows::Win32::System::LibraryLoader::GetModuleHandleW;
    match unsafe { GetModuleHandleW(windows::core::w!("dxcompiler.dll")) } {
        Ok(_) => ShaderCompiler::Dxc,
        Err(_) => ShaderCompiler::Fxc,
    }
}

/// The directory of the module this code is linked into: `flyhigh.dll`, or the executable when the engine is static.
#[cfg(windows)]
fn module_dir() -> Option<PathBuf> {
    use std::os::windows::ffi::OsStringExt;
    use windows::Win32::Foundation::HMODULE;
    use windows::Win32::System::LibraryLoader::{
        GetModuleFileNameW, GetModuleHandleExW, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
    };
    let mut module = HMODULE::default();
    unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            windows::core::PCWSTR(module_dir as *const () as *const u16),
            &mut module,
        )
    }
    .ok()?;
    let mut name = vec![0u16; 32768];
    let n = unsafe { GetModuleFileNameW(Some(module), &mut name) } as usize;
    if n == 0 || n >= name.len() {
        return None;
    }
    PathBuf::from(std::ffi::OsString::from_wide(&name[..n])).parent().map(Path::to_path_buf)
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    fn dll_of(c: wgpu::Dx12Compiler) -> Option<String> {
        match c {
            wgpu::Dx12Compiler::DynamicDxc { dxc_path } => Some(dxc_path),
            _ => None,
        }
    }

    #[test]
    fn explicit_path_is_used_as_given() {
        let dir = std::env::temp_dir().join("flyhigh-dxc-test");
        std::fs::create_dir_all(&dir).unwrap();
        let dll = dir.join("dxcompiler.dll");
        std::fs::write(&dll, b"").unwrap();
        assert_eq!(dll_of(choose(Some(&dir))), Some(dll.to_string_lossy().into_owned()));
        assert_eq!(dll_of(choose(Some(&dll))), Some(dll.to_string_lossy().into_owned()));
        let missing = dir.join("nowhere");
        assert_eq!(dll_of(choose(Some(&missing))), Some(missing.to_string_lossy().into_owned()));
    }

    #[test]
    fn module_dir_is_where_the_test_binary_lives() {
        let exe = std::env::current_exe().unwrap();
        assert_eq!(module_dir().as_deref(), exe.parent());
    }
}
