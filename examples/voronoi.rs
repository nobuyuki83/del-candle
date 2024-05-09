fn main() -> anyhow::Result<()> {
    let vtxl2xy = vec!(
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0);
    let site2xy = del_msh::sampling::poisson_disk_sampling_from_polyloop2(
        &vtxl2xy, 0.15, 10);
    let site2xy = del_msh::vtx2xyz::from_array_of_nalgebra(&site2xy);
    let (site2vtxc2xy, site2vtxc2info)
        = del_msh::voronoi2::voronoi_cells(&vtxl2xy, &site2xy);
    let (site2idx, idx2vtxv, _vtxv2xy, vtxv2info)
        = del_msh::voronoi2::indexing(&site2vtxc2xy, &site2vtxc2info);
    let site2xy = {
        candle_core::Var::from_slice(
            &site2xy,
            candle_core::Shape::from((site2xy.len() / 2, 2)),
            &candle_core::Device::Cpu).unwrap()
    };
    let site2_to_voronoi2 = del_candle::site2_to_voronoi2::Layer {
        vtxl2xy,
        vtxv2info,
    };
    let vtxv2xy = site2xy.apply_op1(site2_to_voronoi2)?;
    {   // output to obj file
        let mut vtx2xy = vtxv2xy.clone().flatten_all()?.to_vec1::<f32>()?;
        let site2xy = site2xy.clone().flatten_all()?.to_vec1::<f32>()?;
        vtx2xy.extend(site2xy);
        let edge2vtxv = del_msh::edge2vtx::from_polygon_mesh(
            &site2idx, &idx2vtxv, vtx2xy.len() / 2);
        let _ = del_msh::io_obj::save_edge2vtx_vtx2xyz(
            "target/voronoi0.obj", &edge2vtxv, &vtx2xy, 2);
    }
    let loss = vtxv2xy.flatten_all()?.sum_all()?;
    let grad = loss.backward()?;
    Ok(())
}