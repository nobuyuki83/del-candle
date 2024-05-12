use del_canvas::canvas_gif::CanvasGif;

fn my_paint(
    canvas: &mut CanvasGif,
    transform_to_scr: &nalgebra::Matrix3::<f32>,
    vtxl2xy: &[f32],
    site2xy: &[f32],
    site2idx: &[usize],
    idx2vtxv: &[usize],
    vtxv2xy: &[f32])
{
    canvas.paint_polyloop(
        &vtxl2xy, &transform_to_scr,
        1., 1);
    for i_site in 0..site2xy.len() / 2 {
        let i_color = if i_site == 0 { 2 } else { 1 };
        canvas.paint_point(
            site2xy[i_site * 2 + 0], site2xy[i_site * 2 + 1], &transform_to_scr,
            3.0, i_color);
    }
    for i_site in 0..site2idx.len() - 1 {
        let num_vtx_in_site = site2idx[i_site + 1] - site2idx[i_site];
        for i0_vtx in 0..num_vtx_in_site {
            let i1_vtx = (i0_vtx + 1) % num_vtx_in_site;
            let i0 = idx2vtxv[site2idx[i_site] + i0_vtx];
            let i1 = idx2vtxv[site2idx[i_site] + i1_vtx];
            canvas.paint_line(
                vtxv2xy[i0 * 2 + 0],
                vtxv2xy[i0 * 2 + 1],
                vtxv2xy[i1 * 2 + 0],
                vtxv2xy[i1 * 2 + 1], &transform_to_scr, 0.7, 1);
        }
    }
}


fn main() -> anyhow::Result<()> {
    let mut canvas = del_canvas::canvas_gif::CanvasGif::new(
        "target/area_opt.gif",
        (300, 300),
        &vec!(0xffffff, 0x000000, 0xff0000));
    let transform_to_scr = nalgebra::Matrix3::<f32>::new(
        canvas.width as f32 * 0.8, 0., canvas.width as f32 * 0.1,
        0., -(canvas.height as f32) * 0.8, canvas.height as f32 * 0.9,
        0., 0., 1.);
    let vtxl2xy = vec!(
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0);
    let site2xy = del_msh::sampling::poisson_disk_sampling_from_polyloop2(
        &vtxl2xy, 0.15, 10);
    let site2xy = candle_core::Var::from_slice(
        &site2xy, candle_core::Shape::from((site2xy.len() / 2, 2)), &candle_core::Device::Cpu).unwrap();
    for _iter in 0..200 {
        let (vtxv2xy, voronoi_info) = del_candle::voronoi2::voronoi(&vtxl2xy, &site2xy);
        let polygonmesh2_to_areas = del_candle::polygonmesh2_to_areas::Layer {
            elem2idx: voronoi_info.site2idx.clone(),
            idx2vtx: voronoi_info.idx2vtxv.clone(),
        };
        let site2areas = vtxv2xy.apply_op1(polygonmesh2_to_areas)?;
        dbg!(&site2areas);
        use candle_core::IndexOp;
        let loss_area = site2areas.i(0)?.affine(-1.0, 0.0)?;
        let polygonmesh2_to_cogs = del_candle::polygonmesh2_to_cogs::Layer {
            elem2idx: voronoi_info.site2idx.clone(),
            idx2vtx: voronoi_info.idx2vtxv.clone(),
        };
        let site2cogs = vtxv2xy.apply_op1(polygonmesh2_to_cogs)?;
        assert_eq!(site2cogs.shape(), site2cogs.shape());
        let loss_lloyd = site2xy.sub(&site2cogs)?.sqr().unwrap().sum_all()?;
        let loss = (loss_area + loss_lloyd.affine(0.1, 0.0)?)?;
        let grad = loss.backward().unwrap();
        let dw_site2xy = grad.get(&site2xy).unwrap();
        let _ = site2xy.set(&site2xy.as_tensor().sub(&(dw_site2xy * 0.01)?)?);
        {
            canvas.clear(0);
            let site2xy = site2xy.flatten_all()?.to_vec1::<f32>()?;
            let vtxv2xy = vtxv2xy.flatten_all()?.to_vec1::<f32>()?;
            my_paint(&mut canvas, &transform_to_scr,
                     &vtxl2xy,
                     &site2xy, &voronoi_info.site2idx, &voronoi_info.idx2vtxv, &vtxv2xy);
            canvas.write();
        }
    }


    Ok(())
}