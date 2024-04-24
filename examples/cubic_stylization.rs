fn main() -> anyhow::Result<()>{
    let num_vtx = 64;
    let edge_length = 2.0f32 * std::f32::consts::PI / num_vtx as f32;
    dbg!(edge_length);
    let mut vtx2xy = del_msh::polyloop2::from_circle(1.0, num_vtx);
    {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for mut vtx in vtx2xy.column_iter_mut() {
            vtx += nalgebra::Vector2::<f32>::new(rng.gen(), rng.gen());
        }
    }
    let vtx2xy = candle_core::Var::from_slice(
        vtx2xy.as_slice(),
        candle_core::Shape::from((vtx2xy.ncols(), 2)),
        &candle_core::Device::Cpu).unwrap();
    for iter in 0..300 {
        let polyloop_to_edgevector = del_candle::polyloop_to_edgevector::Layer {};
        let edge2xy = vtx2xy.apply_op1(polyloop_to_edgevector)?;
        let edge2nrm = {
            let x = edge2xy.get_on_dim(1, 0)?;
            let y = edge2xy.get_on_dim(1, 1)?;
            candle_core::Tensor::stack(&[(y * -1.)?,x], 1)?
        };
        let edge2len = edge2nrm.sqr().unwrap().sum(1)?.sqrt()?;
        let edge2len = candle_core::Tensor::stack(&[edge2len.clone(), edge2len.clone()], 1)?;
         let edge2unrm = edge2nrm.div(&edge2len)?;
        // dbg!(edge2unrm.sqr().unwrap().sum(1)?.sqrt()?.to_vec1::<f32>());
        let mut edge2unorm_trg = edge2unrm.flatten_all()?.to_vec1::<f32>()?;
        for unorm in edge2unorm_trg.chunks_mut(2) {
            let x0 = unorm[0];
            let y0 = unorm[1];
            if y0 > x0 && y0 > -x0 { unorm[0] = 0f32; unorm[1] = 1f32; }
            if y0 < x0 && y0 < -x0 { unorm[0] = 0f32; unorm[1] = -1f32; }
            if y0 > x0 && y0 < -x0 { unorm[0] = -1f32; unorm[1] = 0f32; }
            if y0 < x0 && y0 > -x0 { unorm[0] = 1f32; unorm[1] = 0f32; }
        }
        let edge2unorm_trg = candle_core::Tensor::from_slice(
            edge2unorm_trg.as_slice(),
            candle_core::Shape::from(vtx2xy.shape()),
            &candle_core::Device::Cpu).unwrap();
        let unorm_diff = edge2unrm.sub(&edge2unorm_trg)?.sqr()?.sum_all()?;
        let polyloop_to_diffcoord = del_candle::polyloop2_to_diffcoord::Layer {};
        let magdiffc = vtx2xy.apply_op1(polyloop_to_diffcoord)?.sqr().unwrap().sum_all()?;
        dbg!(unorm_diff.to_vec0::<f32>()?);
        let loss = if iter < 100 { (magdiffc*2.0)? }
        else{ (unorm_diff+magdiffc*0.3)? };
        let grad = loss.backward()?;
        let dw_vtx2xyz = grad.get(&vtx2xy).unwrap();
        let _ = vtx2xy.set(&vtx2xy.as_tensor().sub(&(dw_vtx2xyz * 0.005)?)?);
        if iter % 20 == 0 {
            let vtx2xy: Vec<_> = vtx2xy.flatten_all()?.to_vec1::<f32>()?;
            let hoge = del_msh::polyloop::to_cylinder_trimeshes(&vtx2xy, 2, 0.01);
            // let _ = del_msh::io_obj::save_polyloop_(format!("target/polyloop_{}.obj", iter), &vtx2xy, 2);
            let _ = del_msh::io_obj::save_tri_mesh_(format!("target/polyloop_{}.obj", iter), &hoge.0, &hoge.1, 3);


        }
    }
    Ok(())
}