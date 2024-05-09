struct Layer {
    tri2vtx: candle_core::Tensor,
}

impl candle_core::CustomOp1 for Layer {
    fn name(&self) -> &'static str {
        "trimesh3_to_tri2norm"
    }

    fn cpu_fwd(
        &self,
        storage: &candle_core::CpuStorage,
        layout: &candle_core::Layout)
        -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        let (_num_vtx, _three) = layout.shape().dims2()?;
        let vtx2xyz = storage.as_slice::<f32>()?;
        use std::ops::Deref;
        let tri2vtx = self.tri2vtx.storage_and_layout().0;
        let tri2vtx = match tri2vtx.deref() {
            candle_core::Storage::Cpu(cpu_tri2vtx) => { cpu_tri2vtx.as_slice::<i64>()? }
            _ => panic!(),
        };
        let tri2normal = del_msh::trimesh3::tri2normal(tri2vtx, vtx2xyz);
        let shape = candle_core::Shape::from(self.tri2vtx.shape().dims2()?);
        let storage = candle_core::WithDType::to_cpu_storage_owned(tri2normal);
        Ok((storage, shape))
    }

    /// This function takes as argument the argument `arg` used in the forward pass, the result
    /// produced by the forward operation `res` and the gradient of the result `grad_res`.
    /// The function should return the gradient of the argument.
    fn bwd(
        &self,
        vtx2xyz: &candle_core::Tensor,
        tri2nrm: &candle_core::Tensor,
        dw_tri2norm: &candle_core::Tensor) -> candle_core::Result<Option<candle_core::Tensor>> {
        let (num_vtx, _three0) = vtx2xyz.shape().dims2()?;
        let (_num_tri, _three1) = tri2nrm.shape().dims2()?;
        assert!(vtx2xyz.layout().is_contiguous());
        assert!(!vtx2xyz.layout().is_fortran_contiguous());
        use std::ops::Deref;
        let tri2vtx = self.tri2vtx.storage_and_layout().0;
        let tri2vtx = match tri2vtx.deref() {
            candle_core::Storage::Cpu(cpu_tri2vtx) => { cpu_tri2vtx.as_slice::<i64>()? }
            _ => panic!()
        };
        let vtx2xyz = vtx2xyz.storage_and_layout().0;
        let vtx2xyz = match vtx2xyz.deref() {
            candle_core::Storage::Cpu(cpu_vtx2xyz) => { cpu_vtx2xyz.as_slice::<f32>()? }
            _ => panic!()
        };
        let dw_tri2nrm = dw_tri2norm.storage_and_layout().0;
        let dw_tri2nrm = match dw_tri2nrm.deref() {
            candle_core::Storage::Cpu(dw_tr2nrm) => { dw_tr2nrm.as_slice::<f32>()? }
            _ => { panic!() }
        };
        let mut dw_vtx2xyz = vec!(0f32; num_vtx * 3);
        for (i_tri, node2vtx) in tri2vtx.chunks(3).enumerate() {
            let (i0, i1, i2)
                = (node2vtx[0] as usize, node2vtx[1] as usize, node2vtx[2] as usize);
            let p0 = del_geo::vec3::to_na(vtx2xyz, i0);
            let p1 = del_geo::vec3::to_na(vtx2xyz, i1);
            let p2 = del_geo::vec3::to_na(vtx2xyz, i2);
            let dw = del_geo::tri3::dw_normal(&p0, &p1, &p2);
            let dw_nrm = del_geo::vec3::to_na(dw_tri2nrm, i_tri);
            let q0 = dw_nrm.transpose() * dw[0];
            let q1 = dw_nrm.transpose() * dw[1];
            let q2 = dw_nrm.transpose() * dw[2];
            for i in 0..3 {
                dw_vtx2xyz[i0 * 3 + i] += q0[i];
                dw_vtx2xyz[i1 * 3 + i] += q1[i];
                dw_vtx2xyz[i2 * 3 + i] += q2[i];
            }
        }
        let dw_vtx2xyz = candle_core::Tensor::from_vec(
            dw_vtx2xyz,
            candle_core::Shape::from((num_vtx, 3)),
            &candle_core::Device::Cpu)?;
        Ok(Some(dw_vtx2xyz))
    }
}

#[test]
fn minimize_surface_area() -> anyhow::Result<()> {
    const MAJOR_RADIUS: f32 = 0.5;
    const MINOR_RADIUS: f32 = 0.2;
    let (tri2vtx, vtx2xyz)
        = del_msh::trimesh3_primitive::torus_yup::<i64, f32>(
        MAJOR_RADIUS, MINOR_RADIUS, 16, 16);
    dbg!("vtx_size", vtx2xyz.len()/3);
    let vtx2xyz = candle_core::Var::from_vec(
        vtx2xyz.clone(),
        candle_core::Shape::from((vtx2xyz.len() / 3, 3)),
        &candle_core::Device::Cpu).unwrap();
    let tnsr_tri2vtx = candle_core::Tensor::from_vec(
        tri2vtx.clone(),
        (tri2vtx.len() / 3, 3),
        &candle_core::Device::Cpu).unwrap();
    let mut prev_area = 0_f32;
    for itr in 0..100 {
        let ln = Layer { tri2vtx: tnsr_tri2vtx.clone() }; // cheap
        let tri2normal = vtx2xyz.apply_op1(ln)?;
        let area = (tri2normal.sqr()?.contiguous()?
            .sum_keepdim(1)?.sqrt()? * 0.5)?
            .sum_all()?;
        {
            let cur_area = area.to_vec0::<f32>().unwrap();
            // dbg!(cur_area, itr, prev_area * 0.9995);
            if itr == 0 {
                let smooth_area = MINOR_RADIUS * std::f32::consts::PI * 2.0
                    * MAJOR_RADIUS * std::f32::consts::PI * 2.0;
                assert!((smooth_area - cur_area).abs() < 0.09);
            } else {
                assert!(cur_area < prev_area * 0.9997);
            }
            prev_area = cur_area;
        }
        let grad_sum_area = area.backward()?;
        let dw_vtx2xyz = grad_sum_area.get(&vtx2xyz).unwrap();
        let _ = vtx2xyz.set(&vtx2xyz.as_tensor().sub(&(dw_vtx2xyz * 0.001)?)?);
    }
    Ok(())
}