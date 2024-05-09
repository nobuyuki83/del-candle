use std::ops::Deref;
use candle_core::{CpuStorage, Layout, Shape, Tensor};

pub struct Layer {
}

impl candle_core::CustomOp1 for crate::polyloop_to_edgevector::Layer {
    fn name(&self) -> &'static str {
        "polyloop_to_edgevector"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout)
               -> candle_core::Result<(CpuStorage, Shape)>
    {
        let (num_vtx, num_dim) = layout.shape().dims2()?;
        let vtx2xy = storage.as_slice::<f32>()?;
        let mut edge2xy = vec!(0f32; num_vtx * num_dim);
        for i_edge in 0..num_vtx {
            let i0_vtx = i_edge;
            let i1_vtx = (i_edge+1) % num_vtx;
            for i_dim in 0..num_dim {
                edge2xy[i_edge*num_dim+i_dim] += vtx2xy[i1_vtx*num_dim+i_dim];
                edge2xy[i_edge*num_dim+i_dim] -= vtx2xy[i0_vtx*num_dim+i_dim];
            }
        }
        let shape = candle_core::Shape::from(
            (num_vtx, num_dim));
        let storage = candle_core::WithDType::to_cpu_storage_owned(edge2xy);
        Ok((storage, shape))
    }

    /// This function takes as argument the argument `arg` used in the forward pass, the result
    /// produced by the forward operation `res` and the gradient of the result `grad_res`.
    /// The function should return the gradient of the argument.
    fn bwd(&self, _vtx2xy: &Tensor, _edge2xy: &Tensor, dw_edge2xy: &Tensor)
        -> candle_core::Result<Option<Tensor>> {
        let (num_vtx, num_dim) = dw_edge2xy.shape().dims2()?;
        let dw_edge2xy = dw_edge2xy.storage_and_layout().0;
        let dw_edge2xy = match dw_edge2xy.deref() {
            candle_core::Storage::Cpu(cpu_tri2vtx) => { cpu_tri2vtx.as_slice::<f32>()? }
            _ => panic!()
        };
        let mut dw_vtx2xy = vec!(0f32; num_vtx * num_dim);
        for i_edge in 0..num_vtx {
            let i0_vtx = i_edge;
            let i1_vtx = (i_edge+1) % num_vtx;
            for i_dim in 0..num_dim {
                dw_vtx2xy[i1_vtx*num_dim+i_dim] += dw_edge2xy[i_edge*num_dim+i_dim];
                dw_vtx2xy[i0_vtx*num_dim+i_dim] -= dw_edge2xy[i_edge*num_dim+i_dim];
            }
        }
        let dw_vtx2xy = candle_core::Tensor::from_vec(
            dw_vtx2xy,
            candle_core::Shape::from((num_vtx, num_dim)),
            &candle_core::Device::Cpu)?;
        Ok(Some(dw_vtx2xy))

    }

}

#[test]
fn edge_length_constraint() ->  anyhow::Result<()> {
    let num_vtx = 16;
    let edge_length = 2.0f32 * std::f32::consts::PI / num_vtx as f32;
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
            candle_core::Shape::from((vtx2xy.ncols(),2)),
            &candle_core::Device::Cpu).unwrap()
    };
    for iter in 0..100 {
        let render = crate::polyloop_to_edgevector::Layer {};
        let edge2xy = vtx2xy.apply_op1(render)?;
        assert_eq!(edge2xy.shape(), vtx2xy.shape());
        {  // assert sum of all vectors are zero
            let sum = edge2xy.sum(0)?.sqr()?.sum_all()?.to_vec0::<f32>()?;
            assert!(sum < 1.0e-10);
        }
        let edge2length = edge2xy.sqr()?.sum(1)?.sqrt()?;
        dbg!(edge2length.shape());
        let edge2length_target = candle_core::Tensor::ones(
            num_vtx, candle_core::DType::F32, &candle_core::Device::Cpu)?
            .affine(edge_length as f64, 0.)?;
        let edge2length_diff = edge2length.sub(&edge2length_target)?.sqr()?.sum_all()?;
        dbg!(edge2length_diff.to_vec0::<f32>()?);
        let grad = edge2length_diff.backward()?;
        let dw_vtx2xyz = grad.get(&vtx2xy).unwrap();
        let _ = vtx2xy.set(&vtx2xy.as_tensor().sub(&(dw_vtx2xyz * 0.1)?)?);
        {
            let vtx2xy: Vec<_> = vtx2xy.flatten_all()?.to_vec1::<f32>()?;
            let _ = del_msh::io_obj::save_vtx2xyz_as_polyloop(
                format!("target/polyloop_{}.obj", iter),
                &vtx2xy, 2);
        }
        /*
        {
            let vtx2xy: Vec<_> = vtx2xy.flatten_all()?.to_vec1::<f32>()?;
            let vtx2xyz = del_msh::vtx2xyz::from_2d_to_3d(&vtx2xy);
            let vtx2xyz = del_msh::vtx2xyz::from_slice_to_nalgebra_matrix::<f32,3>(&vtx2xyz);
            let vtx2framex = del_msh::polyloop3::vtx2framex(vtx2xyz.as_slice());
            let (tri2vtxt, vtxt2xyz) = del_msh::polyloop3::to_trimesh3_torus(&vtx2xyz, &vtx2framex, 0.1f32, 32);
            let _ = del_msh::io_obj::save_tri_mesh_(
                format!("target/polyloop_{}.obj", iter),
                &tri2vtxt, &vtxt2xyz, 3);
        }
         */
    }
    Ok(())
}