use candle_core::{CpuStorage, Device::Cpu, Layout, Shape, Tensor};
use std::ops::Deref;

pub struct Layer {
    pub point2splat: Tensor,
    pub img_shape: (usize, usize),
    pub tile2jdx: Vec<usize>,
    pub jdx2idx: Vec<usize>,
    pub idx2point: Vec<usize>,
    pub tile_size: usize,
    pub mvp: [f32; 16],
}

impl candle_core::CustomOp1 for Layer {
    fn name(&self) -> &'static str {
        "render"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let point2gauss = storage.as_slice::<f32>()?;
        let point2splat = self.point2splat.storage_and_layout().0;
        let point2splat = match point2splat.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
            _ => panic!(),
        };
        let pix2rgb = del_canvas::gaussian_splatting::rasterize(
            point2gauss,
            point2splat,
            &self.tile2jdx,
            &self.jdx2idx,
            &self.idx2point,
            self.img_shape,
            self.tile_size,
        );
        let shape = candle_core::Shape::from((self.img_shape.1, self.img_shape.0, 3));
        let storage = candle_core::WithDType::to_cpu_storage_owned(pix2rgb);
        Ok((storage, shape))
    }

    /// This function takes as argument the argument `arg` used in the forward pass, the result
    /// produced by the forward operation `res` and the gradient of the result `grad_res`.
    /// The function should return the gradient of the argument.
    fn bwd(
        &self,
        point2gauss: &Tensor,
        _pix2rgb: &Tensor,
        dw_pix2rgb: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let ndof_gauss = point2gauss.dims2()?.1;
        let point2gauss = point2gauss.storage_and_layout().0;
        let point2gauss = match point2gauss.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
            _ => panic!(),
        };
        let point2splat = self.point2splat.storage_and_layout().0;
        let point2splat = match point2splat.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
            _ => panic!(),
        };
        let dw_pix2rgb = dw_pix2rgb.storage_and_layout().0;
        let dw_pix2rgb = match dw_pix2rgb.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
            _ => panic!(),
        };
        let dw_point2gauss = del_canvas::gaussian_splatting::diff_point2gauss(
            point2gauss,
            point2splat,
            &self.tile2jdx,
            &self.jdx2idx,
            &self.idx2point,
            self.img_shape,
            self.tile_size,
            &self.mvp,
            dw_pix2rgb,
        );
        let num_point = self.point2splat.dims2()?.0;
        let dw_vtx2xyz = Tensor::from_vec(
            dw_point2gauss,
            candle_core::Shape::from((num_point, ndof_gauss)),
            &Cpu,
        )?;
        Ok(Some(dw_vtx2xyz))
    }
}

/*
fn rasterize_splats(
    point2gauss: &Tensor,
    point2splat: Tensor,
    cam: &Camera,
    tile2jdx: &[usize],
    jdx2idx: &[usize],
    idx2point: &[usize],
) -> anyhow::Result<Tensor> {
    println!("gaussian_naive");
    let point2gauss = point2gauss.storage_and_layout().0;
    let point2gauss = match point2gauss.deref() {
        candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
        _ => panic!(),
    };
    let point2splat = point2splat.storage_and_layout().0;
    let point2splat = match point2splat.deref() {
        candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
        _ => panic!(),
    };
    let img =  del_canvas::rasterize_gaussian_splatting::rasterize(
        point2gauss, point2splat,
        tile2jdx, jdx2idx, idx2point, cam.img_shape,
        TILE_SIZE);
    let img = Tensor::from_slice(&img, (cam.img_shape.1, cam.img_shape.0, 3), &Device::Cpu)?;
    Ok(img)
}
 */
