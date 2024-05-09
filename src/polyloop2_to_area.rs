use std::ops::Deref;

use candle_core::{CpuStorage, Layout, Shape, Tensor};

#[derive(Clone, Copy)]
struct Layer {}

impl candle_core::CustomOp1 for crate::polyloop2_to_area::Layer {
    fn name(&self) -> &'static str {
        "polyloop_to_edgevector"
    }

    #[allow(clippy::identity_op)]
    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout)
               -> candle_core::Result<(CpuStorage, Shape)>
    {
        let (num_vtx, num_dim) = layout.shape().dims2()?;
        assert_eq!(num_dim, 2);
        let vtx2xy = storage.as_slice::<f32>()?;
        let mut area: f32 = 0.0f32;
        for i_edge in 0..num_vtx {
            let i0_vtx = i_edge;
            let i1_vtx = (i_edge + 1) % num_vtx;
            area += 0.5f32 * vtx2xy[i0_vtx * 2 + 0] * vtx2xy[i1_vtx * 2 + 1];
            area -= 0.5f32 * vtx2xy[i0_vtx * 2 + 1] * vtx2xy[i1_vtx * 2 + 0];
        }
        let shape = candle_core::Shape::from(());
        let storage = candle_core::WithDType::to_cpu_storage_owned(vec!(area));
        Ok((storage, shape))
    }

    /// This function takes as argument the argument `arg` used in the forward pass, the result
    /// produced by the forward operation `res` and the gradient of the result `grad_res`.
    /// The function should return the gradient of the argument.
    #[allow(clippy::identity_op)]
    fn bwd(&self, vtx2xy: &Tensor, _area: &Tensor, dw_area: &Tensor)
           -> candle_core::Result<Option<Tensor>> {
        let dw_area = dw_area.storage_and_layout().0;
        let dw_area = match dw_area.deref() {
            candle_core::Storage::Cpu(cpu_dw_area) => { cpu_dw_area.as_slice::<f32>()? }
            _ => panic!()
        };
        //
        let (num_vtx, two) = vtx2xy.shape().dims2()?;
        assert_eq!(two, 2);
        let vtx2xy = vtx2xy.storage_and_layout().0;
        let vtx2xy = match vtx2xy.deref() {
            candle_core::Storage::Cpu(cpu_vtx2xy) => { cpu_vtx2xy.as_slice::<f32>()? }
            _ => panic!()
        };
        //
        let mut dw_vtx2xy = vec!(0f32; num_vtx * 2);
        for i_edge in 0..num_vtx {
            let i0_vtx = i_edge;
            let i1_vtx = (i_edge + 1) % num_vtx;
            dw_vtx2xy[i0_vtx * 2 + 0] += 0.5f32 * vtx2xy[i1_vtx * 2 + 1] * dw_area[0];
            dw_vtx2xy[i1_vtx * 2 + 1] += 0.5f32 * vtx2xy[i0_vtx * 2 + 0] * dw_area[0];
            dw_vtx2xy[i0_vtx * 2 + 1] -= 0.5f32 * vtx2xy[i1_vtx * 2 + 0] * dw_area[0];
            dw_vtx2xy[i1_vtx * 2 + 0] -= 0.5f32 * vtx2xy[i0_vtx * 2 + 1] * dw_area[0];
        }
        let dw_vtx2xy = candle_core::Tensor::from_vec(
            dw_vtx2xy,
            candle_core::Shape::from((num_vtx, 2)),
            &candle_core::Device::Cpu)?;
        Ok(Some(dw_vtx2xy))
    }
}

#[test]
fn test_backward() -> anyhow::Result<()> {
    let num_vtx = 64;
    let vtx2xy = del_msh::polyloop2::from_circle(1.0, num_vtx);
    let vtx2xy = {
        candle_core::Var::from_slice(
            vtx2xy.as_slice(),
            candle_core::Shape::from((num_vtx, 2)),
            &candle_core::Device::Cpu).unwrap()
    };
    let render = crate::polyloop2_to_area::Layer {};
    let area0 = vtx2xy.apply_op1(render)?;
    {  // check area is circle
        let area0 = area0.to_vec0::<f32>()?;
        assert!((area0 - std::f32::consts::PI).abs() < 0.01);
    }
    { // add perturbation
        let rand0 = &candle_core::Tensor::randn(
            1f32, 1f32, (num_vtx, 2), &candle_core::Device::Cpu)?;
        vtx2xy.add(&rand0)?;
    }
    let area0 = vtx2xy.apply_op1(render)?;
    let grad = area0.backward()?;
    let area0 = area0.to_vec0::<f32>()?;
    let dw_vtx2xy = grad.get(&vtx2xy).unwrap();
    let dw_vtx2xy = dw_vtx2xy.flatten_all()?.to_vec1::<f32>()?;
    let eps = 1.0e-2f32;
    for i_vtx in 0..num_vtx {
        for i_dim in 0..2 {
            let mut vtx2xy1 = vtx2xy.clone().flatten_all()?.to_vec1::<f32>()?;
            vtx2xy1[i_vtx*2+i_dim] += eps;
            let vtx2xy1 = candle_core::Tensor::from_vec(
                vtx2xy1, (num_vtx, 2), &candle_core::Device::Cpu)?;
            let area1 = vtx2xy1.apply_op1(render)?;
            let area1 = area1.to_vec0::<f32>()?;
            let da0 = (area1-area0)/eps;
            let da1 = dw_vtx2xy[i_vtx*2+i_dim];
            assert!((da0-da1).abs()<1.0e-4);
        }
    }
    Ok(())
}

#[test]
fn area_constraint() -> anyhow::Result<()> {
    let num_vtx = 64;
    let mut vtx2xy = del_msh::polyloop2::from_circle(1.0, num_vtx);
    {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for mut vtx in vtx2xy.column_iter_mut() {
            vtx += nalgebra::Vector2::<f32>::new(rng.gen(), rng.gen());
        }
    }
    let vtx2xy = {
        candle_core::Var::from_slice(
            vtx2xy.as_slice(),
            candle_core::Shape::from((vtx2xy.ncols(), 2)),
            &candle_core::Device::Cpu).unwrap()
    };
    for iter in 0..200 {
        let render = crate::polyloop2_to_area::Layer {};
        let area = vtx2xy.apply_op1(render)?;
        dbg!(iter, area.to_vec0::<f32>()?);
        let area_sq = area.sqr()?;
        let grad = area_sq.backward()?;
        let dw_vtx2xyz = grad.get(&vtx2xy).unwrap();
        let _ = vtx2xy.set(&vtx2xy.as_tensor().sub(&(dw_vtx2xyz * 0.1)?)?);
        if iter % 10 == 0 {
            let vtx2xy: Vec<_> = vtx2xy.flatten_all()?.to_vec1::<f32>()?;
            let _ = del_msh::io_obj::save_vtx2xyz_as_polyloop(
                format!("target/polyloop_{}.obj", iter),
                &vtx2xy, 2);
        }
    }
    Ok(())
}