use candle_core::Tensor;
use del_canvas::canvas_gif::CanvasGif;
use del_candle::voronoi2::VoronoiInfo;

fn my_paint(
    canvas: &mut CanvasGif,
    transform_to_scr: &nalgebra::Matrix3::<f32>,
    vtxl2xy: &[f32],
    site2xy: &[f32],
    voronoi_info: &VoronoiInfo,
    vtxv2xy: &[f32],
    site2room: &[usize],
    edge2vtxv_wall: &[usize])
{
    del_canvas::paint_pixcenter::polyloop(
        &mut canvas.data, canvas.width,
        &vtxl2xy, &transform_to_scr,
        1.6, 1);
    for i_site in 0..site2xy.len() / 2 {
        let i_room = site2room[i_site];
        // if i_room == usize::MAX { continue; }
        let i_color: u8 = if i_room == usize::MAX {1} else { (i_room+2).try_into().unwrap() };
        del_canvas::paint_pixcenter::point(
            &mut canvas.data, canvas.width,
            &[site2xy[i_site * 2 + 0], site2xy[i_site * 2 + 1]],
            &transform_to_scr,
            3.0, i_color);
    }
    let site2idx = &voronoi_info.site2idx;
    let idx2vtxv = &voronoi_info.idx2vtxv;
    for i_site in 0..site2idx.len() - 1 {
        let num_vtx_in_site = site2idx[i_site + 1] - site2idx[i_site];
        for i0_vtx in 0..num_vtx_in_site {
            let i1_vtx = (i0_vtx + 1) % num_vtx_in_site;
            let i0 = idx2vtxv[site2idx[i_site] + i0_vtx];
            let i1 = idx2vtxv[site2idx[i_site] + i1_vtx];
            del_canvas::dda::line(
                &mut canvas.data, canvas.width,
                &[vtxv2xy[i0 * 2 + 0], vtxv2xy[i0 * 2 + 1]],
                &[vtxv2xy[i1 * 2 + 0], vtxv2xy[i1 * 2 + 1]],
                &transform_to_scr, 1);
        }
    }
    for i_edge in 0..edge2vtxv_wall.len() / 2 {
        let i0_vtxv = edge2vtxv_wall[i_edge*2+0];
        let i1_vtxv = edge2vtxv_wall[i_edge*2+1];
        del_canvas::paint_pixcenter::line(
            &mut canvas.data, canvas.width,
            &[vtxv2xy[i0_vtxv*2+0], vtxv2xy[i0_vtxv*2+1]],
            &[vtxv2xy[i1_vtxv*2+0], vtxv2xy[i1_vtxv*2+1]],
            &transform_to_scr, 1.6, 1);
    }
}

fn edge2vtvx_wall(
    voronoi_info: &VoronoiInfo,
    site2room: &[usize]) -> Vec<usize>
{
    let site2idx = &voronoi_info.site2idx;
    let idx2vtxv = &voronoi_info.idx2vtxv;
    let mut edge2vtxv = vec!(0usize;0);
    for i_site in 0..site2idx.len() - 1 {
        let i_room = site2room[i_site];
        let num_vtx_in_site = site2idx[i_site + 1] - site2idx[i_site];
        for i0_vtx in 0..num_vtx_in_site {
            let i1_vtx = (i0_vtx + 1) % num_vtx_in_site;
            let idx = site2idx[i_site] + i0_vtx;
            let i0_vtxv = idx2vtxv[idx];
            let i1_vtxv = idx2vtxv[site2idx[i_site] + i1_vtx];
            let j_site = voronoi_info.idx2elem[idx];
            if j_site == usize::MAX { continue; }
            if i_site >= j_site { continue; }
            let j_room = site2room[j_site];
            if i_room == j_room { continue; }
            edge2vtxv.push(i0_vtxv);
            edge2vtxv.push(i1_vtxv);
        }
    }
    edge2vtxv
}

fn room2area(
    site2room: &[usize],
    site2idx: &[usize], idx2vtxv: &[usize], vtxv2xy: &candle_core::Tensor)
    -> candle_core::Result<candle_core::Tensor>
{
    let polygonmesh2_to_areas = del_candle::polygonmesh2_to_areas::Layer {
        elem2idx: Vec::<usize>::from(site2idx),
        idx2vtx: Vec::<usize>::from(idx2vtxv),
    };
    let site2areas = vtxv2xy.apply_op1(polygonmesh2_to_areas)?;
    let site2areas = site2areas.reshape((site2areas.dim(0).unwrap(), 1))?; // change shape to use .mutmul()
    //
    let num_room = site2room.iter().filter(|&v|*v!=usize::MAX).max().unwrap() + 1;
    let num_site = site2room.len();
    let sum_sites_for_rooms = {
        let mut sum_sites_for_rooms = vec!(0f32; num_site * num_room);
        for i_site in 0..num_site {
            let i_room = site2room[i_site];
            if i_room == usize::MAX { continue; }
            sum_sites_for_rooms[i_room * num_site + i_site] = 1f32;
        }
        candle_core::Tensor::from_slice(
            &sum_sites_for_rooms,
            candle_core::Shape::from((num_room, num_site)),
            &candle_core::Device::Cpu)?
    };
    sum_sites_for_rooms.matmul(&site2areas)
}

fn remove_site_too_close(
    site2room: &mut [usize],
    site2xy: &candle_core::Tensor)
{
    assert_eq!(site2room.len(), site2xy.dims2().unwrap().0);
    let num_site = site2room.len();
    let site2xy = site2xy.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for i_site in 0..num_site {
        let i_room = site2room[i_site];
        if i_room == usize::MAX { continue; }
        let p_i = del_geo::vec2::to_na(&site2xy, i_site);
        for j_site in (i_site+1)..num_site {
            let j_room = site2room[j_site];
            if j_room == usize::MAX { continue; }
            if i_room != j_room { continue; }
            let p_j = del_geo::vec2::to_na(&site2xy, j_site);
            if (p_i-p_j).norm() < 0.05 {
                site2room[j_site] = usize::MAX;
            }
        }
    }

}


fn main() -> anyhow::Result<()> {
    let mut canvas = del_canvas::canvas_gif::CanvasGif::new(
        "target/area_opt.gif",
        (300, 300),
        &vec!(0xffffff, 0x000000, 0xff0000, 0x009900));
    let transform_to_scr = nalgebra::Matrix3::<f32>::new(
        canvas.width as f32 * 0.8, 0., canvas.width as f32 * 0.1,
        0., -(canvas.height as f32) * 0.8, canvas.height as f32 * 0.9,
        0., 0., 1.);
    let vtxl2xy = vec!(
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0);
    /*
    let site2xy = del_msh::sampling::poisson_disk_sampling_from_polyloop2(
        &vtxl2xy, 0.15, 10);
     */
    let site2xy: Vec<f32> = vec!(
        0.11, 0.1,
        0.8, 0.13,
        0.31, 0.5,
        0.5, 0.52,
        0.8, 0.6,
        0.8, 0.83,
        0.1, 0.81
    );
    let site2xy = candle_core::Var::from_slice(
        &site2xy,
        candle_core::Shape::from((site2xy.len() / 2, 2)), &candle_core::Device::Cpu).unwrap();
    //
    let mut site2room: Vec<usize> = vec!(1, 1, 0, 0, 0, 0, 0);
    assert_eq!(site2room.len(), site2xy.dims2()?.0);
    let room2area_trg = candle_core::Tensor::from_vec(
        vec!(0.5f32, 0.5f32),
        candle_core::Shape::from((2, 1)),
        &candle_core::Device::Cpu).unwrap();
    for _iter in 0..400 {
        let (vtxv2xy, voronoi_info) = del_candle::voronoi2::voronoi(
            &vtxl2xy, &site2xy, &site2room);
        let edge2vtxv_wall = edge2vtvx_wall(&voronoi_info, &site2room);
        let loss_cubic = del_candle::cubic_stylization::from_edge2vtx(&vtxv2xy, &edge2vtxv_wall)?;
        let loss_area = crate::room2area(
            &site2room,
            &voronoi_info.site2idx, &voronoi_info.idx2vtxv, &vtxv2xy)?
            .sub(&room2area_trg)?.sqr()?.sum_all()?;
        //
        /*
        let polygonmesh2_to_cogs = del_candle::polygonmesh2_to_cogs::Layer {
            elem2idx: voronoi_info.site2idx.clone(),
            idx2vtx: voronoi_info.idx2vtxv.clone(),
        };
        let site2cogs = vtxv2xy.apply_op1(polygonmesh2_to_cogs)?;
        let loss_lloyd = site2xy.sub(&site2cogs)?.sqr().unwrap().sum_all()?;
        //
        let loss = (loss_area + loss_lloyd.affine(0.5, 0.0)? + loss_cubic)?;
         */
        let loss = (loss_cubic + loss_area)?;
        dbg!(loss.to_vec0::<f32>()?);
        // let loss = (loss_lloyd + loss_cubic)?;
        let grad = loss.backward().unwrap();
        let dw_site2xy = grad.get(&site2xy).unwrap();
        let _ = site2xy.set(&site2xy.as_tensor().sub(&(dw_site2xy * 0.05)?)?);
        // dbg!(&site2room);
        // remove_site_too_close(&mut site2room, &site2xy);
        //
        canvas.clear(0);
        my_paint(&mut canvas, &transform_to_scr,
                 &vtxl2xy,
                 &site2xy.flatten_all()?.to_vec1::<f32>()?,
                 &voronoi_info,
                 &vtxv2xy.flatten_all()?.to_vec1::<f32>()?,
                 &site2room,
                 &edge2vtxv_wall);
        canvas.write();
    }


    Ok(())
}