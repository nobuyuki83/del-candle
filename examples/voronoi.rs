use del_canvas::canvas_gif::Canvas;

use del_candle::voronoi2::VoronoiInfo;

fn my_paint(
    canvas: &mut Canvas,
    transform_to_scr: &[f32; 9],
    vtxl2xy: &[f32],
    site2xy: &[f32],
    voronoi_info: &VoronoiInfo,
    vtxv2xy: &[f32],
) {
    del_canvas::rasterize_polygon::stroke(
        &mut canvas.data,
        canvas.width,
        &vtxl2xy,
        &transform_to_scr,
        1.6,
        1,
    );
    for i_site in 0..site2xy.len() / 2 {
        del_canvas::rasterize_circle::fill(
            &mut canvas.data,
            canvas.width,
            &[site2xy[i_site * 2 + 0], site2xy[i_site * 2 + 1]],
            &transform_to_scr,
            3.0,
            1,
        );
    }
    let site2idx = &voronoi_info.site2idx;
    let idx2vtxv = &voronoi_info.idx2vtxv;
    for i_site in 0..site2idx.len() - 1 {
        let num_vtx_in_site = site2idx[i_site + 1] - site2idx[i_site];
        for i0_vtx in 0..num_vtx_in_site {
            let i1_vtx = (i0_vtx + 1) % num_vtx_in_site;
            let i0 = idx2vtxv[site2idx[i_site] + i0_vtx];
            let i1 = idx2vtxv[site2idx[i_site] + i1_vtx];
            del_canvas::rasterize_line::draw_dda_with_transformation(
                &mut canvas.data,
                canvas.width,
                &[vtxv2xy[i0 * 2 + 0], vtxv2xy[i0 * 2 + 1]],
                &[vtxv2xy[i1 * 2 + 0], vtxv2xy[i1 * 2 + 1]],
                &transform_to_scr,
                1,
            );
        }
    }
}

fn main() -> anyhow::Result<()> {
    let vtxl2xy = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let vtxl2xy = vec![
        0.0, 0.0, 1.0, 0.0, 1.0, 0.2, 0.1, 0.2, 0.2, 0.5, 1.0, 0.5, 1.0, 1.0, 0.0, 1.0,
    ];
    let mut reng = rand::thread_rng();
    let site2xy =
        del_msh::sampling::poisson_disk_sampling_from_polyloop2(&vtxl2xy, 0.1, 50, &mut reng);
    // dbg!(&site2room);
    // del_canvas from here
    let mut canvas = {
        del_canvas::canvas_gif::Canvas::new(
            "target/area_opt.gif",
            (300, 300),
            &vec![0xffffff, 0x000000],
        )
    };
    let transform_to_scr: [f32; 9] = nalgebra::Matrix3::<f32>::new(
        canvas.width as f32 * 0.8,
        0.,
        canvas.width as f32 * 0.1,
        0.,
        -(canvas.height as f32) * 0.8,
        canvas.height as f32 * 0.9,
        0.,
        0.,
        1.,
    )
    .as_slice()
    .try_into()
    .unwrap();
    // ---------------------
    // candle from here
    let site2xy = candle_core::Var::from_slice(
        &site2xy,
        candle_core::Shape::from((site2xy.len() / 2, 2)),
        &candle_core::Device::Cpu,
    )
    .unwrap();
    let adamw_params = candle_nn::ParamsAdamW {
        lr: 0.05,
        ..Default::default()
    };
    use candle_nn::Optimizer;
    let mut optimizer = candle_nn::AdamW::new(vec![site2xy.clone()], adamw_params)?;
    for _iter in 0..400 {
        let (vtxv2xy, voronoi_info) =
            del_candle::voronoi2::voronoi(&vtxl2xy, &site2xy, |_i_site| true);
        let loss_lloyd = del_candle::voronoi2::loss_lloyd(
            &voronoi_info.site2idx,
            &voronoi_info.idx2vtxv,
            &site2xy,
            &vtxv2xy,
        )?;
        /*
        let polygonmesh2_to_cogs = del_candle::polygonmesh2_to_cogs::Layer {
            elem2idx: voronoi_info.site2idx.clone(),
            idx2vtx: voronoi_info.idx2vtxv.clone(),
        };
        let site2cogs = vtxv2xy.apply_op1(polygonmesh2_to_cogs)?;
        let loss_lloyd = site2xy.sub(&site2cogs)?.sqr().unwrap().sum_all()?;
         */
        let loss = loss_lloyd;
        dbg!(loss.to_vec0::<f32>()?);
        optimizer.backward_step(&loss)?;
        // ----------------
        canvas.clear(0);
        my_paint(
            &mut canvas,
            &transform_to_scr,
            &vtxl2xy,
            &site2xy.flatten_all()?.to_vec1::<f32>()?,
            &voronoi_info,
            &vtxv2xy.flatten_all()?.to_vec1::<f32>()?,
        );
        canvas.write();
    }

    Ok(())
}
