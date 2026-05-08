use candle_core::{utils, DType, Device};

use crate::{Result, WaxError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceChoice {
    Auto,
    Cpu,
    Cuda,
    Metal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DTypeChoice {
    Auto,
    F32,
    F16,
    BF16,
}

pub fn select_device(choice: DeviceChoice) -> Result<Device> {
    match choice {
        DeviceChoice::Auto => {
            if utils::cuda_is_available() {
                Ok(Device::new_cuda(0)?)
            } else if utils::metal_is_available() {
                Ok(Device::new_metal(0)?)
            } else {
                Ok(Device::Cpu)
            }
        }
        DeviceChoice::Cpu => Ok(Device::Cpu),
        DeviceChoice::Cuda => {
            if !utils::cuda_is_available() {
                return Err(WaxError::InvalidRequest(
                    "CUDA was requested but is not available in this build/runtime".to_string(),
                ));
            }
            Ok(Device::new_cuda(0)?)
        }
        DeviceChoice::Metal => {
            if !utils::metal_is_available() {
                return Err(WaxError::InvalidRequest(
                    "Metal was requested but is not available in this build/runtime".to_string(),
                ));
            }
            Ok(Device::new_metal(0)?)
        }
    }
}

pub fn select_dtype(choice: DTypeChoice, device: &Device) -> DType {
    match choice {
        DTypeChoice::Auto => match device {
            Device::Cpu => DType::F32,
            Device::Cuda(_) | Device::Metal(_) => DType::F16,
        },
        DTypeChoice::F32 => DType::F32,
        DTypeChoice::F16 => DType::F16,
        DTypeChoice::BF16 => DType::BF16,
    }
}

pub fn device_label(device: &Device) -> String {
    match device {
        Device::Cpu => "cpu".to_string(),
        Device::Cuda(_) => "cuda:0".to_string(),
        Device::Metal(_) => "metal:0".to_string(),
    }
}

pub fn dtype_label(dtype: DType) -> String {
    format!("{dtype:?}").to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};

    use super::{device_label, dtype_label, select_dtype, DTypeChoice};

    #[test]
    fn cpu_auto_dtype_defaults_to_f32() {
        assert_eq!(select_dtype(DTypeChoice::Auto, &Device::Cpu), DType::F32);
    }

    #[test]
    fn explicit_dtype_overrides_device_default() {
        assert_eq!(select_dtype(DTypeChoice::F16, &Device::Cpu), DType::F16);
        assert_eq!(select_dtype(DTypeChoice::BF16, &Device::Cpu), DType::BF16);
    }

    #[test]
    fn labels_are_stable_for_stats() {
        assert_eq!(device_label(&Device::Cpu), "cpu");
        assert_eq!(dtype_label(DType::F32), "f32");
        assert_eq!(dtype_label(DType::F16), "f16");
    }
}
