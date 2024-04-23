use std::ops::Deref;
use candle_core::{CpuStorage, Layout, Shape, Tensor};
use image::GenericImageView;

struct Layer {
    tri2vtx: candle_core::Tensor,
    vtx2xy: candle_core::Tensor,
    img_shape: (usize, usize), // (width, height)
    transform: nalgebra::Matrix3<f32>
}

impl candle_core::CustomOp1 for Layer {
    fn name(&self) -> &'static str {
        "render"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout)
        -> candle_core::Result<(CpuStorage, Shape)>
    {
        let (num_vtx, two) = self.vtx2xy.shape().dims2()?;
        assert_eq!(two, 2);
        let (num_vtx1, num_dim) = layout.shape().dims2()?;
        // dbg!(num_dim);
        assert_eq!(num_vtx, num_vtx1);
        let vtx2color = storage.as_slice::<f32>()?;
        let tri2vtx = self.tri2vtx.storage_and_layout().0;
        let tri2vtx = match tri2vtx.deref() {
            candle_core::Storage::Cpu(cpu_storage) => {cpu_storage.as_slice::<i64>()?},
            _ => panic!()
        };
        let vtx2xy = self.vtx2xy.storage_and_layout().0;
        let vtx2xy = match vtx2xy.deref() {
            candle_core::Storage::Cpu(cpu_storage) => {cpu_storage.as_slice::<f32>()?},
            _ => panic!()
        };
        let mut img = vec!(0f32; self.img_shape.0 * self.img_shape.1);
        del_canvas::trimsh2_vtxcolor(
            self.img_shape.0, self.img_shape.1, &mut img,
            tri2vtx, vtx2xy, vtx2color, &self.transform);
        let shape = candle_core::Shape::from(
            (self.img_shape.0, self.img_shape.1, num_dim));
        let storage = candle_core::WithDType::to_cpu_storage_owned(img);
        Ok((storage, shape))
    }

    /// This function takes as argument the argument `arg` used in the forward pass, the result
    /// produced by the forward operation `res` and the gradient of the result `grad_res`.
    /// The function should return the gradient of the argument.
    fn bwd(&self, vtx2color: &Tensor, pix2color: &Tensor, dw_pix2color: &Tensor)
        -> candle_core::Result<Option<Tensor>> {
        let (num_vtx, num_channels) = vtx2color.shape().dims2()?;
        let (height, width, _num_channels) = pix2color.shape().dims3()?;
        assert_eq!(num_channels, _num_channels);
        let tri2vtx = self.tri2vtx.storage_and_layout().0;
        let tri2vtx = match tri2vtx.deref() {
            candle_core::Storage::Cpu(cpu_tri2vtx) => {
                cpu_tri2vtx.as_slice::<i64>()? },
            _ => panic!()
        };
        let vtx2xy = self.vtx2xy.storage_and_layout().0;
        let vtx2xy = match vtx2xy.deref() {
            candle_core::Storage::Cpu(cpu_storage) => {cpu_storage.as_slice::<f32>()?},
            _ => panic!()
        };
        assert_eq!(vtx2xy.len(), num_vtx * 2);
        let dw_pix2color = dw_pix2color.storage_and_layout().0;
        let dw_pix2color = match dw_pix2color.deref() {
            candle_core::Storage::Cpu(cpu_storage) => {cpu_storage.as_slice::<f32>()?},
            _ => panic!()
        };
        assert_eq!(dw_pix2color.len(), height * width * num_channels);
        //
        let mut dw_vtx2color = vec!(0f32; num_vtx * num_channels);
        let transform_inv = self.transform.clone().try_inverse().unwrap();
        for i_h in 0..height {
            for i_w in 0..width {
                let p_xy = transform_inv * nalgebra::Vector3::<f32>::new(
                    i_w as f32, i_h as f32, 1.0f32);
                let p_xy = [p_xy[0] / p_xy[2], p_xy[1] / p_xy[2]];
                let Some((i_tri, r0, r1))
                    = del_msh::trimesh2::search_bruteforce_one_triangle_include_input_point(
                    &p_xy, tri2vtx, vtx2xy) else { continue; };
                let r2 = 1.0f32 - r0 - r1;
                let iv0 = tri2vtx[i_tri*3+0] as usize;
                let iv1 = tri2vtx[i_tri*3+1] as usize;
                let iv2 = tri2vtx[i_tri*3+2] as usize;
                for i_ch in 0..num_channels {
                    let dw_color = dw_pix2color[(i_h* width +i_w)* num_channels +i_ch];
                    dw_vtx2color[iv0* num_channels +i_ch] += dw_color*r0;
                    dw_vtx2color[iv1* num_channels +i_ch] += dw_color*r1;
                    dw_vtx2color[iv2* num_channels +i_ch] += dw_color*r2;
                }
            }
        }
        let dw_vtx2color = candle_core::Tensor::from_vec(
            dw_vtx2color,
            candle_core::Shape::from((num_vtx, num_channels)),
            &candle_core::Device::Cpu)?;
        Ok(Some(dw_vtx2color))
    }
}


#[test]
fn optimize_vtxcolor() -> anyhow::Result<()> {
    let img_trg = {
        let img_trg = image::open("tesla.png").unwrap();
        let (width, height) = img_trg.dimensions();
        let (width, height ) = (width as usize, height as usize);
        let img_trg = img_trg.grayscale().into_bytes();
        let img_trg: Vec<f32> = img_trg.iter().map(|&v| (v as f32) / 255.0f32 ).collect();
        assert_eq!(img_trg.len(), width * height);
        candle_core::Tensor::from_vec(
            img_trg,
            candle_core::Shape::from((height, width, 1)),
            &candle_core::Device::Cpu).unwrap()
    };
    let img_shape = (img_trg.dims3().unwrap().1, img_trg.dims3().unwrap().0);
    // transformation from xy to pixel coordinate
    let transform = nalgebra::Matrix3::<f32>::new(
        img_shape.0 as f32, 0., 0.,
        0., -(img_shape.1 as f32), img_shape.1 as f32,
        0., 0., 1.);
    let (tri2vtx, vtx2xyz)
        = del_msh::trimesh2_dynamic::meshing_from_polyloop2::<i64, f32>(
        &[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 0.03);
    let num_vtx = vtx2xyz.len() / 2;
    let vtx2xy = candle_core::Tensor::from_vec(
        vtx2xyz,
        candle_core::Shape::from((num_vtx, 2)),
        &candle_core::Device::Cpu).unwrap();
    let num_tri = tri2vtx.len() / 3;
    let tri2vtx = candle_core::Tensor::from_vec(
        tri2vtx,
        candle_core::Shape::from((num_tri, 3)),
        &candle_core::Device::Cpu).unwrap();
    let vtx2color = {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let vals: Vec<f32> = (0..num_vtx).map(|_| rng.gen::<f32>() ).collect();
        candle_core::Var::from_vec(
            vals,
            candle_core::Shape::from((num_vtx,1)),
            &candle_core::Device::Cpu).unwrap()
    };
    dbg!(&vtx2color.shape());
    for i_itr in 0..100 {
        let render = Layer { tri2vtx: tri2vtx.clone(), vtx2xy: vtx2xy.clone(), img_shape, transform };
        let img_out = vtx2color.apply_op1(render)?;
        dbg!(&img_out.shape());
        let diff = img_trg.sub(&img_out).unwrap().sqr()?.sum_all()?;
        dbg!(&diff.shape());
        dbg!(diff.to_vec0::<f32>().unwrap());
        let grad = diff.backward()?;
        let dw_vtx2color = grad.get(&vtx2color).unwrap();
        dbg!(dw_vtx2color.dims2().unwrap());
        {
            let img_out_vec: Vec<f32> = img_out.flatten_all()?.to_vec1()?;
            del_canvas::write_png_from_float_image(
                format!("target/foo_{}.png", i_itr),
                img_shape.0, img_shape.1, &img_out_vec);
        }
        let _ = vtx2color.set(&vtx2color.as_tensor().sub(&(dw_vtx2color*0.003)?)?);
    }
    Ok(())
}